"""Mask2Former × VidEoMT 型(画像の Mask2Former を初期値にし、VidEoMT と同じくクエリを次のフレームに引き継いで、全体を動画の区間で
学習する)。exp00502。

土台は画像の段階の Mask2Former(MinVIS の学習の重み。models/mask2former.py)。フレーム間の情報のやり取りは、VidEoMT(公式
tue-mps/videomt の videomt_online)と同じくクエリだけで、規則も VidEoMT と同じにする:
    - 最初のフレーム: Mask2Former の学習するクエリ q(query_feat)をそのまま decoder に入れる
    - 2 フレーム目から: 「前のフレームの decoder の最後の層の出力のクエリを線形の層(query_updater)に通したもの + q」を入れる
      (VidEoMT の q = query_updater(prev) + q と同じ形)。Q 個のクエリ全部を渡し、クエリの番号が動画を通した ID になる
次のフレームへ渡すクエリは、decoder の最後の層の出力の LayerNorm(decoder_norm)の前の値(公式の VidEoMT の NORM_QUERIES の
既定 False と同じく、正規化の前の値)。

MOT(動画全体での物体の ID の管理・識別)の形: **学習して身に付ける形**(VidEoMT と同じ)。クエリの番号 = ID で、推論に規則はない。
    学習の割り当て・損失: VidEoMT と同じ(losses/video_consistent_criterion.py。物体を区間の中で最初に写ったフレームでだけ、
    まだ物体を持たないクエリにハンガリアン法で割り当て、その組を区間の最後まで使う。隠れている間の正解はその物体のクラスと空のマスク)。
    補助の損失は Mask2Former の全部の予測(最初のクエリ + decoder の各層。画像の Mask2Former の学習と同じ)にかける。
    推論(VidEoMT と同じ): 全部のフレームの Q 個のクエリの予測を出し、クエリごとにフレームで平均したクラスの値から、動画ごとに
    上位 100 個を出す(video_infer.video_predictions)。

形の記号: B = 区間の数、T = フレームの数、Q = クエリの数、C = クエリの次元、S = 入力の一辺、h・w = マスクの縦・横(入力の 1/4)、
K = クラスの数。
"""
import torch
from einops import repeat
from torch import nn


class M2FVidEoMT(nn.Module):
    """Mask2Former × VidEoMT 型。

    Args:
        segmenter: 画像の段階の Mask2Former(学習した重みを読み、クラスの並びを動画のセットにそろえたもの)。全体を学習する。
    """

    activation = "softmax"  # 評価(video_infer.video_predictions)でのクラスの値の付け方(Mask2Former と同じ)

    def __init__(self, segmenter: nn.Module):
        super().__init__()
        self.segmenter = segmenter
        c = segmenter.query_feat.weight.shape[1]  # クエリの次元 C
        self.query_updater = nn.Linear(c, c)  # 前のフレームのクエリ → 今のフレームのクエリに足す分(VidEoMT と同じ)

    @classmethod
    def from_run(cls, path: str, video_set: str) -> "M2FVidEoMT":
        """画像の Mask2Former の学習の重みから作る。

        Args:
            path: 画像の学習の重み(runs/{名前}/best.pt。同じ場所の config.json で作り方を読む)。
            video_set: 学習の動画のセット(出力のクラスの番号をこのセットの並びにする)。

        Returns:
            M2FVidEoMT(Mask2Former の部分は学習した重み、query_updater は初期の値)。
        """
        from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter
        seg = ImageSegmenter.from_run(path, video_set)
        assert seg.arch == "mask2former", f"Mask2Former の重みではない: {seg.arch}"
        return cls(seg.model)

    def step(self, images: torch.Tensor, prev: torch.Tensor | None) -> tuple[list[dict], torch.Tensor]:
        """1 フレーム分の予測を作り、次のフレームに渡すクエリを返す。

        Args:
            images: このフレームの画像 (B, 3, S, S)。
            prev: 前のフレームから渡すクエリ (B, Q, C)。動画の最初のフレームでは None。

        Returns:
            (Mask2Former の予測の並び(最初のクエリ + decoder の各層。各要素は {"logits": (B, Q, K+1), "masks": (B, Q, h, w)}),
             次のフレームに渡すクエリ (B, Q, C)(最後の層の出力の LayerNorm の前の値))。
        """
        q = repeat(self.segmenter.query_feat.weight, "q c -> b q c", b=images.shape[0])  # (B, Q, C)
        if prev is not None:
            q = self.query_updater(prev) + q  # 前のフレームのクエリを足す(VidEoMT と同じ)
        outs = self.segmenter(images, init_q=q)
        return outs, outs[-1]["embeds_raw"]

    def forward(self, clips: torch.Tensor, prev: torch.Tensor | None = None) -> tuple[list[dict], torch.Tensor]:
        """学習の区間を時刻の順に処理し、各フレームの予測を作る(区間の最初のフレームは、動画の最初と同じく前のクエリなし)。

        Args:
            clips: 区間の画像 (B, T, 3, S, S)。
            prev: 区間の最初のフレームに渡すクエリ (B, Q, C)。動画の最初から始めるときは None。

        Returns:
            (層ごとの {"logits": (B, T, Q, K+1), "masks": (B, Q, T, h, w)} の並び(最後の要素が最終の予測),
             区間の最後のフレームの、次に渡すクエリ (B, Q, C))。
        """
        per_frame = []
        for t in range(clips.shape[1]):
            outs, prev = self.step(clips[:, t], prev)
            per_frame.append(outs)
        res = []
        for layer in range(len(per_frame[0])):
            res.append({"logits": torch.stack([f[layer]["logits"] for f in per_frame], dim=1),  # (B, T, Q, K+1)
                        "masks": torch.stack([f[layer]["masks"] for f in per_frame], dim=2)})  # (B, Q, T, h, w)
        return res, prev

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """推論: 1 本の動画の全フレームの予測を作る(online。クエリの番号が動画を通した ID)。

        Args:
            frames: 動画の全フレームの画像 (T, 3, S, S)。

        Returns:
            (各フレームのクラスの値 (T, Q, K+1), 各フレームのマスクの値 (Q, T, h, w)(sigmoid の前))。
        """
        logits, masks, prev = [], [], None
        for t in range(len(frames)):
            outs, prev = self.step(frames[t: t + 1], prev)
            logits.append(outs[-1]["logits"][0])  # (Q, K+1)
            masks.append(outs[-1]["masks"][0])  # (Q, h, w)
        return torch.stack(logits), torch.stack(masks, dim=1)  # (T, Q, K+1), (Q, T, h, w)
