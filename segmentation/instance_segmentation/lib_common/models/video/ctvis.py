"""CTVIS(Ying ら ICCV 2023、"CTVIS: Consistent Training for Online Video Instance Segmentation")。exp00502。

参考: 公式のコード(DVIS++ に入っている版。zhang-tao-whu/DVIS_Plus の `DVIS_DAQ/dvis_Plus/ctvis.py` の CTMinVIS、
`configs/dvis_Plus/VIPSeg/CTVIS_r50.yaml`)。

作り: 画像の Mask2Former に reid の head(models/mask2former.py の reid_head)を足し、動画の区間(5 フレーム。公式の
SAMPLING_FRAME_RANGE 10 なので、基準のフレームの前後 10 フレームから飛び飛びに選ぶ)で学習する。損失は
losses/ctvis_criterion.py(フレームごとの画像の損失 + 学習用の記憶のバンクを使う reid の対照学習)。DVIS++・DVIS-DAQ の segmenter にもなる。

MOT(動画全体での物体の ID の管理・識別)の形: **推論の照合(学習した埋め込み)**。クエリの枠 = ID。推論は MinVIS と同じく隣のフレームの
クエリをハンガリアン法で照合するが、照合に使う埋め込みは、対照学習で「同じ物体は近く・違う物体は遠く」なるように学習した reid の
埋め込み(公式の CTMinVIS.post_processing と同じ)。対照学習の正例は、前のフレームの埋め込みや、類似度で重みを付けて混ぜた過去の埋め込み
(記憶)なので、隠れた後の再出現でも過去の埋め込みに近くなるように学習する。推論の照合は隣のフレームどうしだけで、記憶のバンクは
学習だけに使う(公式のこの版と同じ)。区別できる物体はクエリの数まで。
"""
import torch
from einops import rearrange
from torch import nn

from segmentation.instance_segmentation.lib_common.models.mask2former import Mask2Former
from segmentation.instance_segmentation.lib_common.models.video.minvis import MinVIS


class CTVIS(nn.Module):
    """CTVIS(reid の head を持つ Mask2Former を、動画の区間で学習する)。

    Args:
        segmenter: reid の head を持つ Mask2Former(reid_head=True)。
    """

    def __init__(self, segmenter: Mask2Former):
        super().__init__()
        assert segmenter.reid_embed is not None, "CTVIS の segmenter は reid_head=True で作る"
        self.segmenter = segmenter

    def forward(self, clips: torch.Tensor) -> tuple[list[dict], None]:
        """学習: 区間の全フレームを画像としてまとめて Mask2Former にかける。

        Args:
            clips: (B, T, 3, S, S)。

        Returns:
            (Mask2Former の層ごとの予測(画像は (b t) の順に B·T 枚。最後の要素に "reid"), None)。
        """
        return self.segmenter(rearrange(clips, "b t c h w -> (b t) c h w")), None

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor, chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """隣のフレームのクエリを reid の埋め込みで照合する(MinVIS と同じ形で返す)。

        Args:
            frames: (T, 3, S, S)。
            chunk: segmenter でまとめて処理するフレームの数。

        Returns:
            (フレームごとのクラスの値 (T, Q, K+1), マスクの値 (Q, T, h, w))(クエリの枠 = ID)。
        """
        return MinVIS(self.segmenter, chunk, embed_key="reid").predict_video(frames)
