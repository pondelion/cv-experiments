"""CAVIS の online 版(Lee ら ICCV 2025、"Context-Aware Video Instance Segmentation")。exp00502。

参考: 公式のコード(Seung-Hun-Lee/CAVIS の `cavis/video_cavis_modules.py` の CAVIS_Tracker・ReferringCrossAttentionLayer_CAVIS、
`cavis/cavis.py` の CAVIS_segmenter・CAVIS_online、`cavis/video_mask2former_transformer_decoder.py`、
`configs/VIPSeg/CAVIS_Segmenter_R50.yaml`・`CAVIS_Online_R50.yaml`)。コードは写さず、規則を読み取って書いた。

作り(DVIS と同じく、セグメンテーションと追跡を分ける。2 段階で学習する):
    1. segmenter(CAVISSeg): 画像の Mask2Former に「文脈の特徴」(最後の層の予測マスクの境界の周りの平均の特徴。
       models/mask2former.py の context_head)と、[クエリの埋め込み; 文脈の特徴] から作る reid の埋め込みを足し、動画の区間
       (公式の VIPSeg は 3 フレーム、前後 2 フレームから選ぶ)で学習する。損失は losses/cavis_criterion.py(画像の損失 + 文脈の
       対照学習 + 原型の対照学習)。推論の照合は [クエリの埋め込み; reid の埋め込み](2 × 256 次元)の cos(公式の post_processing)
    2. 追跡の部(CAVISTracker、6 層): segmenter を固定し、フレームごとに
         a. 今のフレームの [クエリの埋め込み; reid の埋め込み] を、前のフレームの(並べ替えた後の)同じものと cos で照合して並べ替える
         b. 各層: referring cross-attention(キー = 今のフレームの [埋め込み; reid] を Linear(512 → 256)に通したもの(文脈つきのキー)、
            値 = 今のフレームのクエリ(LayerNorm の前)、残差に足すもの = 1 層目は並べ替えた今のクエリの埋め込み、2 層目からは前の層の
            出力)→ self-attention → FFN(どれも残差 + LayerNorm の後置き、dropout 0)
            問い: 2 フレーム目からは、前のフレームの「文脈つきの問い」([前のフレームの最後の層の出力; その枠に照合した前のフレームの
            reid の埋め込み])を同じ Linear に通したもの(全部の層で同じ)。区間・動画の最初のフレームは、1 層目は今のフレームの
            文脈つきのキー、2 層目からは [前の層の出力; 今のフレームの reid の埋め込み(並べ替える前)] を Linear に通したもの
         c. 最後の層の出力と今のフレームのクエリ(LayerNorm の前)を cos で照合し、各枠に今のフレームの reid の埋め込みを並べて、
            次のフレームの文脈つきの問いにする(論文の式 (10))
       各層の出力に LayerNorm → クラス(Linear、K + 1)とマスクの埋め込み(3 層の MLP)、segmenter のマスクを作る特徴との内積をマスクにする
    MOT(動画全体での物体の ID の管理・識別)の形: **推論の照合 + 学習する追跡の部**(DVIS と同じ)。評価の ID はクエリの枠の番号。
    大事なのは「再出現した物体が前と同じ枠に戻り、新しい物体はまだ使っていない枠に入る」ことで、その仕組みは次のとおり(公式のコードで確認):
        - 再出現: 規則はなく、学習した振る舞いだけ。照合(a)の基準は直前のフレーム(並べ替えた後の segmenter の埋め込み)だけで、
          過去の記憶はない。隠れている間、その枠の基準は毎フレーム「その枠に照合した今のクエリ」(物体のないクエリ)に置き換わるので、
          照合(a)だけでは戻れない。戻れるかは追跡の部の状態(前のフレームの最後の層の出力 + 文脈)が隠れている間も物体を表し続け、
          再出現したフレームで今の全部のクエリからその物体を探し直せるかによる。学習の割り当て(物体が最初に写ったフレームで枠を決め、
          隠れているフレームもその枠に物体のクラスの正解を付ける。losses/video_consistent_criterion.py。公式の
          VideoHungarianMatcher_Consistent)がこれを教える。ただし学習の区間は 5 フレームなので、それより長い隠れは学習に出てこない
        - 新しい物体: 規則はなく、学習した振る舞いだけ。学習の割り当ては、区間の中で前の物体に使った枠を後から出た物体に使わない
          ので、空いた枠に入るように学ぶ。推論には、消えた物体の枠を新しい物体が取るのを防ぐ規則はない(取れば 1 つの ID に 2 物体が
          入り、動画の AP の誤りになる)。枠を空きに戻す・ID を終える仕組みもない
        - 区別できる物体はクエリの数まで
    学習の前半(公式は MAX_ITER // 2)は割り当てを segmenter の予測(追跡の部の並びに並べ替えたもの)で決める。推論は DVIS と同じく、
    フレームごとのクラスの値を平均してから softmax。公式の VIPSeg の設定は追跡の部の対照学習(USE_CL)を使わないので、それも使わない。

公式との違い(私の判断。理由つき):
    - 照合(a・c)の向き: 公式の match_embds は DVIS の初版と同じく cos の行列の向きが逆で、並べ替えが逆の置換になる(DVIS++ で直っている)。
      dvis.py と同じく正しい向き(minvis.match_embeds)にする
    - 照合はまとめの中の区間ごとに行う(公式は GPU 1 枚あたり 1 区間で、先頭の区間だけで照合している。同じ意味)

形の記号: B = 区間の数、T = フレームの数、Q = クエリの数、C = クエリの次元(256)、K = クラスの数、h・w = マスクの縦・横(入力の 1/4)。
"""
import json
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from segmentation.instance_segmentation.lib_common.models.mask2former import Mask2Former
from segmentation.instance_segmentation.lib_common.models.video.dvis import DVISOnline, TrackerLayer
from segmentation.instance_segmentation.lib_common.models.video.minvis import MinVIS, match_embeds


class CAVISSeg(nn.Module):
    """CAVIS の segmenter(文脈の特徴を持つ Mask2Former を、動画の区間で学習する)。

    Args:
        segmenter: Mask2Former(reid_head=True, context_head=True)。
    """

    def __init__(self, segmenter: Mask2Former):
        super().__init__()
        assert segmenter.context_filter_size is not None, "CAVIS の segmenter は context_head=True で作る"
        self.segmenter = segmenter

    def forward(self, clips: torch.Tensor) -> tuple[list[dict], None]:
        """学習: 区間の全フレームを画像としてまとめて Mask2Former にかける。

        Args:
            clips: (B, T, 3, S, S)。

        Returns:
            (層ごとの予測(画像は (b t) の順に B·T 枚。最後の要素に "reid"・"mask_feat"), None)。
        """
        return self.segmenter(rearrange(clips, "b t c h w -> (b t) c h w")), None

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor, chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """隣のフレームのクエリを [クエリの埋め込み; reid の埋め込み] で照合する(公式の CAVIS_segmenter.post_processing)。

        Args:
            frames: (T, 3, S, S)。
            chunk: segmenter でまとめて処理するフレームの数。

        Returns:
            (フレームごとのクラスの値 (T, Q, K+1), マスクの値 (Q, T, h, w))(クエリの枠 = ID)。
        """
        return MinVIS(self.segmenter, chunk, embed_key="embeds_ctx").predict_video(frames)


class CAVISTracker(nn.Module):
    """CAVIS の追跡の部(上の 2)。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        d: クエリの次元(segmenter と同じ 256)。
        layers: 層の数(公式は 6)。
    """

    def __init__(self, num_classes: int, d: int = 256, layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([TrackerLayer(d) for _ in range(layers)])
        self.ctx_query_embed = nn.Linear(2 * d, d)  # 文脈つきの問い・キー(公式の ctx_query_embed。初期化は nn.Linear の既定)
        self.decoder_norm = nn.LayerNorm(d)
        self.class_embed = nn.Linear(d, num_classes + 1)
        self.mask_embed = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.ReLU(inplace=True),
                                        nn.Linear(d, d))

    def step(self, frame: dict, state: dict | None, last_only: bool = False) -> tuple[list, dict, torch.Tensor]:
        """1 フレーム分(dvis.ReferringTracker.step と同じ呼び方)。

        Args:
            frame: 今のフレームの segmenter の出力 {"embeds": (B, Q, C)(LayerNorm の後), "embeds_raw": (B, Q, C)(前),
                "reid": (B, Q, C), "mask_feat": (B, C, h, w)}。
            state: 前のフレームの {"embeds": 並べ替えた後の [埋め込み; reid] (B, Q, 2C), "ctx_query": 文脈つきの問い (B, Q, 2C)}
                (区間・動画の最初のフレームは None)。
            last_only: 予測を最後の層だけ出す(推論)。

        Returns:
            (層ごとの予測 [(クラスの値 (B, Q, K+1), マスクの値 (B, Q, h, w))], 次の state, 並べ替え (B, Q)(枠 q に入る今のクエリの番号))。
        """
        e, value, reid = frame["embeds"], frame["embeds_raw"], frame["reid"]
        both = torch.cat([e, reid], -1)  # (B, Q, 2C)
        key = self.ctx_query_embed(both)  # (B, Q, C) 文脈つきのキー
        b, q = e.shape[:2]
        if state is None:
            idx = torch.arange(q, device=e.device).expand(b, q)
            fixed_query = None
        else:
            idx = torch.stack([match_embeds(state["embeds"][i].detach(), both[i].detach()) for i in range(b)])  # (B, Q)
            fixed_query = self.ctx_query_embed(state["ctx_query"])
        gather = lambda x: torch.gather(x, 1, idx[..., None].expand_as(x))
        outs, x = [], gather(e)
        for j, layer in enumerate(self.layers):
            if fixed_query is not None:
                query = fixed_query
            else:  # 最初のフレーム: 1 層目は今のフレームの文脈つきのキー、2 層目からは [前の層の出力; 今の reid(並べ替える前)]
                query = key if j == 0 else self.ctx_query_embed(torch.cat([x, reid], -1))
            x = layer(x, query, key, value)
            outs.append(x)
        # c. 最後の層の出力の各枠に、今のフレームのクエリ(LayerNorm の前)を照合し、その reid の埋め込みを並べる(式 (10))
        ctx_idx = torch.stack([match_embeds(outs[-1][i].detach(), value[i].detach()) for i in range(b)])
        ctx_sorted = torch.gather(reid, 1, ctx_idx[..., None].expand_as(reid))
        state = {"embeds": gather(both), "ctx_query": torch.cat([outs[-1], ctx_sorted], -1)}
        preds = [self.predict(o, frame["mask_feat"]) for o in (outs[-1:] if last_only else outs)]
        return preds, state, idx

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


def load_cavis_segmenter(path: str) -> Mask2Former:
    """CAVIS の segmenter の学習(exp00502 の train.py --arch cavis_seg)の重みから Mask2Former を作って読む。

    Args:
        path: runs/{学習の名前}/best.pt(同じ場所の config.json から backbone・クエリの数・pixel decoder・学習のセットを読む)。

    Returns:
        重みを読んだ Mask2Former(reid の head と文脈の特徴の head つき)。
    """
    from segmentation.instance_segmentation.lib_common.video_data import load_video_set
    cfg = json.loads((Path(path).parent / "config.json").read_text())
    n_classes = len(load_video_set(cfg["train_set"])["categories"])
    model = Mask2Former(n_classes, cfg["backbone"], cfg["num_queries"], pretrained=False, pixel_decoder=cfg["pixel_decoder"],
                        reid_head=True, context_head=True)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({k.removeprefix("segmenter."): v for k, v in state.items()})
    return model


class CAVISOnline(DVISOnline):
    """CAVIS の online 版(固定した CAVIS の segmenter + CAVISTracker)。学習・推論の手順は DVISOnline と同じ。

    Args:
        segmenter: 学習済みの CAVIS の segmenter(load_cavis_segmenter)。重みを固定する。
        num_classes: クラスの数。
        tracker_layers: 追跡の部の層の数(公式は 6)。
    """

    def __init__(self, segmenter: Mask2Former, num_classes: int, tracker_layers: int = 6):
        assert getattr(segmenter, "context_filter_size", None) is not None, "CAVIS の segmenter(context_head=True)を渡す"
        super().__init__(segmenter, num_classes, tracker=CAVISTracker(num_classes, layers=tracker_layers))
