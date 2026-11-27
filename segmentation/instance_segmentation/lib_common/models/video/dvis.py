"""DVIS の online 版(Zhang ら ICCV 2023、"DVIS: Decoupled Video Instance Segmentation Framework")。exp00502。

参考: 公式のコード(zhang-tao-whu/DVIS の `dvis/video_dvis_modules.py` の ReferringTracker・ReferringCrossAttentionLayer、
`dvis/meta_architecture.py` の DVIS_online、`configs/VIPSeg/DVIS_Online_R50_720p.yaml`)。コードは写さず、規則を読み取って書いた。

作り(セグメンテーションと追跡を分ける):
    1. セグメンテーションの部(segmenter): 画像の Mask2Former。先に MinVIS と同じくフレームの画像で学習し、DVIS の学習では
       重みを固定する(公式と同じ。eval のまま、勾配なし)。フレームごとに、最後の decoder 層のクエリの埋め込み(LayerNorm の後)と
       マスクを作る特徴を出す
    2. 追跡の部(referring tracker、6 層。公式の TRACKER.DECODER_LAYERS): フレームごとに
         a. 今のフレームの segmenter のクエリを、前のフレームの(並べ替えた後の)クエリと cos の近さで照合して並べ替える(MinVIS と同じ)
         b. 各層: referring cross-attention(問い = 前のフレームの追跡の部の最後の層の出力、キー・値 = 今のフレームの segmenter の
            クエリ(並べ替える前)、残差に足すもの = 1 層目は並べ替えた今のクエリ、2 層目からは前の層の出力)→ self-attention → FFN
            (どれも残差 + LayerNorm の後置き。Mask2Former の decoder と同じ部品。dropout 0)
         c. 区間・動画の最初のフレームは、問いもキーも今のフレームのクエリ(前のフレームがないため)
       各層の出力に LayerNorm → クラス(Linear、K + 1)とマスクの埋め込み(3 層の MLP)を出し、segmenter のマスクを作る特徴との内積をマスクにする。
       クラスとマスクの head は追跡の部が新しく持つ(segmenter の head は使わない)
    MOT(動画全体での物体の ID の管理・識別)の形: **推論の照合 + 学習する追跡の部**。クエリの枠 = ID。照合(a)で前のフレームの枠に
    そろえ、追跡の部が前のフレームの物体の情報を問いにして、今のフレームの全部のクエリから同じ物体を探し直す(照合の誤りを直す)。
    学習は VidEoMT と同じ割り当て(物体が最初に写ったフレームのクエリにずっと割り当て、隠れているフレームもクラスの正解をその物体にする。
    losses/video_consistent_criterion.py)なので、隠れても同じ枠が物体を持ち続け、新しい物体は空いた枠に入るように学習する。
    推論には規則はなく、照合と学習した振る舞いだけ。区別できる物体はクエリの数まで。学習の区間(5 フレーム)より長い隠れは学習に出てこない。
    照合(a)の基準は直前のフレームだけで記憶はなく、隠れている間はその枠の基準が物体のないクエリに置き換わるので、再出現で同じ枠に
    戻れるかは追跡の部の状態(前のフレームの最後の層の出力)しだい。消えた物体の枠を新しい物体が取るのを防ぐ規則・ID を終える仕組みはない。

公式との違い(私の判断。理由つき):
    - 照合(a)の向き: 公式の DVIS(初版)の match_embds は cos の行列の向きが MinVIS と逆で、並べ替えが逆の置換になる。後継の DVIS++
      (同じ著者、zhang-tao-whu/DVIS_Plus の Noiser.match_embds)で MinVIS と同じ向きに直っているので、そちらに合わせる
    - 照合はまとめの中の区間ごとに行う(公式は GPU 1 枚あたり 1 区間なので、先頭の区間だけで照合している。同じ意味)

学習の手順(公式と同じ): segmenter を固定し、追跡の部だけを学習する。学習の前半(公式は MAX_ITER // 2 回)は、割り当てを segmenter の
予測(追跡の部の並びに並べ替えたもの)で決め、全部の層で使う。後半は追跡の部の層ごとの予測で割り当てる。

形の記号: B = 区間の数、T = フレームの数、Q = クエリの数、C = クエリの次元(256)、K = クラスの数、h・w = マスクの縦・横(入力の 1/4)。
"""
import json
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from segmentation.instance_segmentation.lib_common.models.mask2former import Mask2Former
from segmentation.instance_segmentation.lib_common.models.video.minvis import match_embeds


def _xavier(module: nn.Module) -> None:
    """2 次元以上の重みを xavier_uniform で初期化する(公式の SelfAttentionLayer・CrossAttentionLayer・FFNLayer と同じ)。

    Args:
        module: 初期化する層。
    """
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class TrackerLayer(nn.Module):
    """追跡の部の 1 層: referring cross-attention → self-attention → FFN(どれも残差 + LayerNorm の後置き)。

    Args:
        d: クエリの次元。
        nhead: 注意の頭の数。
        ffn: FFN の中間の次元。
    """

    def __init__(self, d: int = 256, nhead: int = 8, ffn: int = 2048):
        super().__init__()
        self.cross = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, d))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        _xavier(self.cross); _xavier(self.self_attn); _xavier(self.ffn)

    def forward(self, identity: torch.Tensor, query: torch.Tensor, memory: torch.Tensor,
                value: torch.Tensor | None = None) -> torch.Tensor:
        """1 層進める。

        Args:
            identity: 残差に足すもの (B, Q, C)。
            query: 問い (B, Q, C)。
            memory: キー (B, Q, C)(value がなければ値も)。
            value: 値 (B, Q, C) か None(DVIS++ はキーと値を分ける)。

        Returns:
            (B, Q, C)。
        """
        value = memory if value is None else value
        x = self.norms[0](identity + self.cross(query, memory, value, need_weights=False)[0])  # referring cross-attention
        x = self.norms[1](x + self.self_attn(x, x, x, need_weights=False)[0])
        return self.norms[2](x + self.ffn(x))


class ReferringTracker(nn.Module):
    """追跡の部(上の 2)。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        d: クエリの次元(segmenter のクエリの次元と同じにする。公式は Mask2Former の HIDDEN_DIM = 256)。
        layers: 層の数(公式は 6)。
    """

    def __init__(self, num_classes: int, d: int = 256, layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([TrackerLayer(d) for _ in range(layers)])
        self.decoder_norm = nn.LayerNorm(d)
        self.class_embed = nn.Linear(d, num_classes + 1)
        self.mask_embed = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.ReLU(inplace=True),
                                        nn.Linear(d, d))

    def step(self, frame: dict, state: dict | None, last_only: bool = False) -> tuple[list, dict, torch.Tensor]:
        """1 フレーム分。

        Args:
            frame: 今のフレームの segmenter の出力 {"embeds": クエリの埋め込み (B, Q, C), "mask_feat": (B, C, h, w), ...}。
            state: 前のフレームの {"embeds": 並べ替えた後の segmenter のクエリ (B, Q, C), "out": 追跡の部の最後の層の出力 (B, Q, C)}
                (区間・動画の最初のフレームは None)。
            last_only: 予測を最後の層だけ出す(推論)。

        Returns:
            (層ごとの予測 [(クラスの値 (B, Q, K+1), マスクの値 (B, Q, h, w))], 次のフレームへ渡す state,
             並べ替え (B, Q)(枠 q に入る今のクエリの番号))。
        """
        outs, state, idx = self._layers(frame["embeds"], state)
        return [self.predict(x, frame["mask_feat"]) for x in (outs[-1:] if last_only else outs)], state, idx

    def _layers(self, cur: torch.Tensor, state: dict | None) -> tuple[list[torch.Tensor], dict, torch.Tensor]:
        """追跡の部の層(ファイルの先頭の 2a〜c)。

        Args:
            cur: 今のフレームの segmenter のクエリの埋め込み (B, Q, C)。
            state: 前のフレームの state(step を参照)か None。

        Returns:
            (層ごとの出力(LayerNorm の前), 次の state, 並べ替え (B, Q))。
        """
        b, q = cur.shape[:2]
        if state is None:
            idx = torch.arange(q, device=cur.device).expand(b, q)
            cur_sorted, ref = cur, None
        else:
            idx = torch.stack([match_embeds(state["embeds"][i].detach(), cur[i].detach()) for i in range(b)])  # (B, Q)
            cur_sorted, ref = torch.gather(cur, 1, idx[..., None].expand_as(cur)), state["out"]
        outs, x = [], cur_sorted
        for j, layer in enumerate(self.layers):
            # 最初のフレームは問いも今の層の入力(1 層目は今のクエリ)。2 フレーム目からは問い = 前のフレームの最後の層の出力
            x = layer(x, x if ref is None else ref, cur)
            outs.append(x)
        return outs, {"embeds": cur_sorted, "out": outs[-1]}, idx

    def predict(self, x: torch.Tensor, mask_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """層の出力からクラスとマスクを出す。

        Args:
            x: 層の出力 (B, Q, C)。
            mask_feat: マスクを作る特徴 (B, C, h, w)。

        Returns:
            (クラスの値 (B, Q, K+1), マスクの値 (B, Q, h, w))。
        """
        x = self.decoder_norm(x)
        return self.class_embed(x), torch.einsum("bqc,bchw->bqhw", self.mask_embed(x), mask_feat)


def load_segmenter(path: str, video_set: str) -> Mask2Former:
    """画像の学習(exp00101 の train.py・exp00502 の train_image.py)で保存した Mask2Former の重みを、同じ設定で作って読む。

    Args:
        path: runs/{学習の名前}/best.pt(同じ場所の config.json から backbone・クエリの数・pixel decoder を読む)。
        video_set: 使う動画のセットの名前(出力のクラスの番号をこのセットの並びにする。画像のセットと動画のセットではクラスの並びが
            違うことがあるため。image_adapter.reorder_classes)。

    Returns:
        重みを読んだ Mask2Former。
    """
    from segmentation.instance_segmentation.lib_common.models.video.image_adapter import reorder_classes, video_class_order
    cfg = json.loads((Path(path).parent / "config.json").read_text())
    n_classes = len(json.loads(Path(cfg["train_json"]).read_text())["categories"])  # 学習のデータのクラスの数
    # pretrained=False: backbone の学習済みの重みは要らない(すぐ下で、学習した全体の重みで上書きする)
    model = Mask2Former(n_classes, cfg["backbone"], cfg["num_queries"], pretrained=False, pixel_decoder=cfg["pixel_decoder"],
                        mask_pos=cfg["mask_pos"])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return reorder_classes(model, "mask2former", video_class_order(cfg["train_json"], video_set))


class DVISOnline(nn.Module):
    """DVIS の online 版(固定した segmenter + 追跡の部)。

    Args:
        segmenter: 学習済みの画像のモデル(重みを固定する)。公式は Mask2Former。GTR 系も ImageSegmenter で包めば使える(追跡の部は
            segmenter のクエリの埋め込み(GTR: 最後の層の出力のクエリ)とマスクを作る特徴を使い、次元はどちらも dim であること)。
            GTR 系はクラスが sigmoid なので、学習の前半の割り当てのクラスの費用は −sigmoid の確率(match_cost の activation)。
        num_classes: クラスの数。
        tracker_layers: 追跡の部の層の数(公式は 6)。
        tracker: 追跡の部(None なら ReferringTracker。DVIS++ は ReferringTrackerPlus、CAVIS は CAVISTracker を渡す)。
        dim: segmenter のクエリとマスクを作る特徴の次元(追跡の部の次元にする。公式と同じく segmenter の次元に合わせる。
            Mask2Former は 256、GTR 系は学習の設定の dim)。
    """

    def __init__(self, segmenter: nn.Module, num_classes: int, tracker_layers: int = 6, tracker: nn.Module | None = None,
                 dim: int = 256):
        super().__init__()
        from segmentation.instance_segmentation.lib_common.models.video.image_adapter import as_segmenter
        # segmenter: Mask2Former(そのまま)か、GTR 系などを image_adapter.ImageSegmenter で包んだもの(同じ形の出力にそろえる)
        self.segmenter = as_segmenter(segmenter)
        for p in self.segmenter.parameters():
            p.requires_grad_(False)
        # 追跡の部(DVIS++ は dvis_plus.py の ReferringTrackerPlus を渡す。step(frame, state, last_only) の呼び方は同じ)
        self.tracker = tracker if tracker is not None else ReferringTracker(num_classes, d=dim, layers=tracker_layers)

    def train(self, mode: bool = True):
        """学習の状態を切り替える(segmenter は学習中も eval のまま。公式と同じく BatchNorm・dropout を学習の動きにしない)。

        Args:
            mode: True で学習の状態。

        Returns:
            self。
        """
        super().train(mode)
        self.segmenter.eval()
        return self

    @torch.no_grad()
    def _segment(self, frames: torch.Tensor) -> dict:
        """固定した segmenter をフレームにかける。

        Args:
            frames: (M, 3, S, S)。

        Returns:
            最後の層の {"logits", "masks", "embeds", "embeds_raw", "mask_feat"(, "reid")}(勾配なし)。
        """
        out = self.segmenter(frames)
        self.activation_seg = out["activation"]  # segmenter のクラスの値の付け方(学習の前半の割り当てに使う)
        keys = ("logits", "masks", "embeds", "embeds_raw", "mask_feat") + (("reid",) if "reid" in out else ())
        return {k: out[k].detach() for k in keys}

    def forward(self, clips: torch.Tensor) -> tuple[list[dict], dict]:
        """学習: 区間を時刻の順に処理する(区間の最初のフレームは、動画の最初と同じく前の状態なし。公式と同じ)。

        Args:
            clips: (B, T, 3, S, S)。

        Returns:
            (層ごとの {"logits": (B, T, Q, K+1), "masks": (B, Q, T, h, w)} の並び,
             segmenter の最後の層の予測を追跡の部の並びに並べ替えたもの {"logits", "masks"}(同じ形。学習の前半の割り当てに使う))。
        """
        b, t = clips.shape[:2]
        seg = self._segment(rearrange(clips, "b t c h w -> (b t) c h w"))
        seg = {k: rearrange(v, "(b t) ... -> b t ...", b=b) for k, v in seg.items()}  # (B, T, ...)
        state, per_layer, idxs = None, None, []
        for i in range(t):
            preds, state, idx = self.tracker.step({k: v[:, i] for k, v in seg.items()}, state)
            per_layer = [[p] for p in preds] if per_layer is None else [acc + [p] for acc, p in zip(per_layer, preds)]
            idxs.append(idx)
        result = [{"logits": torch.stack([p[0] for p in ps], dim=1),  # (B, T, Q, K+1)
                   "masks": torch.stack([p[1] for p in ps], dim=2)}  # (B, Q, T, h, w)
                  for ps in per_layer]
        idx = torch.stack(idxs, dim=1)  # (B, T, Q)
        seg_logits = torch.gather(seg["logits"], 2, idx[..., None].expand_as(seg["logits"]))
        seg_masks = torch.gather(seg["masks"], 2, idx[..., None, None].expand_as(seg["masks"]))
        return result, {"logits": seg_logits, "masks": rearrange(seg_masks, "b t q h w -> b q t h w"),
                        "activation": self.activation_seg}

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor, chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """1 本の動画の全フレームを時刻の順に処理する(online。segmenter は chunk 枚ずつまとめ、追跡の部の状態はつなげる)。

        Args:
            frames: (T, 3, S, S)。
            chunk: segmenter でまとめて処理するフレームの数。

        Returns:
            (フレームごとのクラスの値 (T, Q, K+1), マスクの値 (Q, T, h, w))(追跡の部の最後の層。クエリの枠 = ID)。
        """
        logits, masks, state = [], [], None
        for s in range(0, len(frames), chunk):
            seg = self._segment(frames[s: s + chunk])
            for i in range(len(seg["embeds"])):
                preds, state, _ = self.tracker.step({k: v[i: i + 1] for k, v in seg.items()}, state, last_only=True)
                logits.append(preds[-1][0][0]); masks.append(preds[-1][1][0])
        return torch.stack(logits), torch.stack(masks, dim=1)  # (T, Q, K+1), (Q, T, h, w)
