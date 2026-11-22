"""DVIS++ の online 版(Zhang ら TPAMI 2025、"DVIS++: Improved Decoupled Framework for Universal Video Segmentation")。exp00502。

参考: 公式のコード(zhang-tao-whu/DVIS_Plus の `DVIS_Plus/dvis_Plus/tracker.py` の ReferringTracker_noiser、`noiser.py` の Noiser、
`meta_architecture.py` の DVIS_Plus_online、`video_mask2former_transformer_decoder.py` の ..._dvisPlus、
`configs/dvis_Plus/VIPSeg/DVIS_Plus_Online_R50.yaml`)。コードは写さず、規則を読み取って書いた。

DVIS(dvis.py)からの違い(どれも公式の DVIS++ と同じ):
    - segmenter: CTVIS(ctvis.py。reid の head を持ち、動画の区間で対照学習した Mask2Former)を固定して使う
    - 追跡の部の入力: segmenter のクエリの埋め込み(LayerNorm の後)と reid の埋め込みをつなげた 512 次元(公式の REID_BRANCH: True で
      hidden_channel = 256 × 2)。キーと値は LayerNorm の前の値と reid をつなげたもの
    - referring cross-attention: 問い = ref_proj(前のフレームの最後の層の出力)(3 層の MLP)、キー・値 = 今のフレームの埋め込み
      (LayerNorm の前、並べ替える前)。最初のフレームは、1 層目の問いが ref_proj(今の埋め込み)、2 層目からは ref_proj(前の層の出力)
    - クラスの head: [ref_proj の出力(そのフレームの参照), LayerNorm した層の出力] をつなげた 1,024 次元から(Linear)
    - マスクの特徴: segmenter のマスクを作る特徴に 1×1 の畳み込み(mask_feature_proj)をかけてから使う
    - noiser(学習だけ。公式の VIPSeg の設定は NOISE_MODE "wa"、NOISE_RATIO 0.5): 2 フレーム目から、確率 0.5 で照合を使わず、
      今のクエリを無作為に並べ替えたものとの重み付きの平均(重みはクエリごとに 0〜1 の一様乱数)を 1 層目の初期値にする。並べ替え
      (次のフレームの照合の基準・学習の前半の割り当てに使う並び)は、重みが 0.5 より小さいクエリは並べ替えた先、それ以外はそのまま
      (公式の _wa_noise_forward と同じ。重複することがある)。照合の誤りを学習の中で作り、追跡の部にそれを直させるため
    - 対照学習(USE_CL): 公式の VIPSeg の設定は False なので使わない

MOT(動画全体での物体の ID の管理・識別)の形: DVIS と同じ **推論の照合 + 学習する追跡の部**。クエリの枠 = ID。照合に使う埋め込みに、
対照学習した reid の埋め込みが入る。noiser で照合の誤りを学習中に作るので、照合が誤ったとき(隠れた後の再出現の取り違えなど)に
追跡の部が前のフレームの物体の情報から同じ物体を探し直すように学習する。区別できる物体はクエリの数まで。

形の記号: B = 区間の数、Q = クエリの数、D = 追跡の部の次元(512)、K = クラスの数、h・w = マスクの縦・横。
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from segmentation.instance_segmentation.lib_common.models.mask2former import D_MODEL, Mask2Former
from segmentation.instance_segmentation.lib_common.models.video.dvis import DVISOnline, TrackerLayer
from segmentation.instance_segmentation.lib_common.models.video.minvis import match_embeds


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    """3 層の MLP(層の間は ReLU。公式の MLP と同じ)。"""
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_hidden), nn.ReLU(inplace=True),
                         nn.Linear(d_hidden, d_out))


def wa_noise(cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """noiser の "wa"(公式の Noiser._wa_noise_forward と同じ規則)。

    Args:
        cur: 今のフレームのクエリ (Q, D)(LayerNorm の前 + reid)。

    Returns:
        (並べ替え (Q,), 1 層目の初期値 (Q, D))。
    """
    q = cur.shape[0]
    perm = np.random.permutation(q)
    w = torch.rand(q, 1, device=cur.device, dtype=cur.dtype)
    init = cur * w + cur[torch.as_tensor(perm, device=cur.device)] * (1 - w)
    idx = np.arange(q)
    low = (w[:, 0] < 0.5).cpu().numpy()
    idx[low] = perm[low]
    return torch.as_tensor(idx, device=cur.device), init


class ReferringTrackerPlus(nn.Module):
    """DVIS++ の追跡の部(上の違い)。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        layers: 層の数(公式は 6)。
        noise_ratio: 学習中に noiser を使う確率(公式の VIPSeg は 0.5)。
    """

    def __init__(self, num_classes: int, layers: int = 6, noise_ratio: float = 0.5):
        super().__init__()
        d = 2 * D_MODEL  # クエリの埋め込み + reid の埋め込み
        self.layers = nn.ModuleList([TrackerLayer(d) for _ in range(layers)])
        self.decoder_norm = nn.LayerNorm(d)
        self.class_embed = nn.Linear(2 * d, num_classes + 1)
        self.mask_embed = _mlp(d, d, D_MODEL)
        self.ref_proj = _mlp(d, d, d)
        for layer in self.ref_proj:  # 公式と同じ c2_xavier_fill
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.zeros_(layer.bias)
        self.mask_feature_proj = nn.Conv2d(D_MODEL, D_MODEL, 1)
        self.noise_ratio = noise_ratio

    def step(self, frame: dict, state: dict | None, last_only: bool = False) -> tuple[list, dict, torch.Tensor]:
        """1 フレーム分(dvis.py の ReferringTracker.step と同じ呼び方)。

        Args:
            frame: segmenter の出力 {"embeds", "embeds_raw", "reid": (B, Q, C), "mask_feat": (B, C, h, w), ...}。
            state: 前のフレームの {"embeds": 並べ替えた後の照合の埋め込み (B, Q, D), "out": 最後の層の出力 (B, Q, D)} か None。
            last_only: 予測を最後の層だけ出す。

        Returns:
            (層ごとの予測 [(クラスの値, マスクの値)], 次の state, 並べ替え (B, Q))。
        """
        cur = torch.cat([frame["embeds"], frame["reid"]], dim=-1)  # 照合に使う (B, Q, D)
        key = torch.cat([frame["embeds_raw"], frame["reid"]], dim=-1)  # キー・値・初期値に使う (B, Q, D)
        b, q = cur.shape[:2]
        if state is None:  # 区間・動画の最初のフレーム(照合は自分どうし = そのまま。noiser は使わない)
            idx = torch.arange(q, device=cur.device).expand(b, q)
            init, ref = key, None
        else:
            idxs, inits = [], []
            for i in range(b):  # 区間ごと(公式は GPU 1 枚あたり 1 区間)
                if self.training and random.random() < self.noise_ratio:
                    ii, init_i = wa_noise(key[i])
                else:
                    ii = match_embeds(state["embeds"][i].detach(), cur[i].detach())
                    init_i = key[i][ii]
                idxs.append(ii); inits.append(init_i)
            idx, init = torch.stack(idxs), torch.stack(inits)
            ref = self.ref_proj(state["out"])  # 問い(全部の層で同じ)
        outs, x = [], init
        for layer in self.layers:
            query = self.ref_proj(key if not outs else x) if ref is None else ref
            x = layer(x, query, key, key)
            outs.append(x)
        reference = self.ref_proj(key) if ref is None else ref  # クラスの head に足す、このフレームの参照
        mask_feat = self.mask_feature_proj(frame["mask_feat"])
        preds = [self.predict(x, reference, mask_feat) for x in (outs[-1:] if last_only else outs)]
        new_state = {"embeds": torch.gather(cur, 1, idx[..., None].expand_as(cur)), "out": outs[-1]}
        return preds, new_state, idx

    def predict(self, x: torch.Tensor, reference: torch.Tensor, mask_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: x・reference (B, Q, D)、mask_feat (B, C, h, w)(射影の後)。Returns: (クラスの値 (B, Q, K+1), マスクの値 (B, Q, h, w))。"""
        x = self.decoder_norm(x)
        logits = self.class_embed(torch.cat([reference, x], dim=-1))
        return logits, torch.einsum("bqc,bchw->bqhw", self.mask_embed(x), mask_feat)


def load_ctvis_segmenter(path: str) -> Mask2Former:
    """CTVIS の学習(exp00502 の train.py --arch ctvis)で保存した重みから、reid の head を持つ Mask2Former を作って読む。

    Args:
        path: runs/{学習の名前}/best.pt(同じ場所の config.json から backbone・クエリの数・pixel decoder・学習のセットを読む)。
    """
    from segmentation.instance_segmentation.lib_common.video_data import load_video_set
    cfg = json.loads((Path(path).parent / "config.json").read_text())
    n_classes = len(load_video_set(cfg["train_set"])["categories"])
    model = Mask2Former(n_classes, cfg["backbone"], cfg["num_queries"], pretrained=False, pixel_decoder=cfg["pixel_decoder"],
                        reid_head=True)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({k.removeprefix("segmenter."): v for k, v in state.items()})
    return model


class DVISPlusOnline(DVISOnline):
    """DVIS++ の online 版(固定した CTVIS の segmenter + ReferringTrackerPlus)。学習・推論は DVISOnline と同じ。

    Args:
        segmenter: 学習済みの CTVIS の Mask2Former(reid_head=True)。
        num_classes: クラスの数。
        tracker_layers: 追跡の部の層の数(公式は 6)。
        noise_ratio: noiser の確率(公式の VIPSeg は 0.5)。
    """

    def __init__(self, segmenter: Mask2Former, num_classes: int, tracker_layers: int = 6, noise_ratio: float = 0.5):
        assert segmenter.reid_embed is not None, "DVIS++ の segmenter は reid の head を持つ CTVIS"
        super().__init__(segmenter, num_classes, tracker=ReferringTrackerPlus(num_classes, tracker_layers, noise_ratio))
