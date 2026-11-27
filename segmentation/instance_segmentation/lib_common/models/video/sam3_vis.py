"""SAM 3 型の動画インスタンスセグメンテーション(決まったクラス)。exp00502。

参考: SAM 3 の論文(Carion ら 2025、"SAM 3: Segment Anything with Concepts")と公式のコード(facebookresearch/sam3)。
SAM 3 は、画像の検出の部分(DETR 系の decoder + MaskFormer のマスクの head。テキストのプロンプトで概念を与える)と、SAM 2 と同じ
記憶の追跡の部分が、1 つの backbone を共有する。学習は、検出の部分を学習した後、backbone を固定して追跡の部分を SAM 2 と同じく
学習する(論文の 4 段目)。推論は、毎フレーム、追跡している物体を伝搬し、検出とマスクの IoU で照合して、新しい物体を足し、
消えた物体を除く。

ここでの作り(決まった 8 クラスにするための置き換えと、exp の条件):
    - backbone: 検出の部分と追跡の部分で共有する(SAM 3 と同じ)。ViT-B/16(DINOv3、ViT-Adapter で 4 段)
    - 検出の部分: 学習済みの Mask2Former(VIPSeg のフレームで学習した 8 クラスの画像のモデル。MinVIS ViT-B/16 の画像のモデル)。
      DETR 系の decoder + MaskFormer のマスクの head で、SAM 3 の検出の部分の、テキストのプロンプトでクラスを決める部分を 8 クラスの
      分類に置き換えた形。presence token は、各クラスの確率が同じ役をするので使わない。SAM 3 の検出の部分の学習(論文の 2・3 段目)に
      当たる学習は済んでいるので、固定する
    - 追跡の部分: SAM 2 と同じ部品(models/video/sam2/model.py。画像の encoder の幹は検出の部分の backbone を共有し、neck 以降を持つ)。
      backbone と検出の部分を固定して、SAM 2 と同じ学習の手順で学習する(論文の 4 段目。models/video/sam2/training.py)
    - 推論: SAM 3 型の組み立て(models/video/sam2/tracker.py の SAM3StyleVIS)。AP の評価の前に予測を確信度の閾値で切らないよう、
      公式の検出の閾値と新しい物体の閾値は使わず、同時に追跡する物体の数の上限 100 と、動画ごとに上位 100 個の出力にする
    - 学習中の評価・最良の重みの選び方: ほかの方式と同じく dev の動画の AP(video_train.evaluate_video_set)
"""
import json
from pathlib import Path

import torch

from segmentation.instance_segmentation.lib_common.models.video.dvis import load_segmenter
from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter
from segmentation.instance_segmentation.lib_common.models.video.sam2.model import SAM2
from segmentation.instance_segmentation.lib_common.models.video.sam2.tracker import SAM3StyleVIS


class SAM3VIS(SAM3StyleVIS):
    """SAM 3 型の動画インスタンスセグメンテーション(ファイルの先頭の作り)。

    Args:
        detector_run: 検出の部分にする画像の Mask2Former の学習の重み(runs/{名前}/best.pt)。
        video_set: 学習の動画のセット(検出の部分のクラスの番号をこのセットの並びにする)。
        size: 入力の一辺。
    """

    def __init__(self, detector_run: str, video_set: str, size: int = 512):
        det = load_segmenter(detector_run, video_set)
        cfg = json.loads((Path(detector_run).parent / "config.json").read_text())
        sam2 = SAM2(cfg["backbone"], size, pretrained=False)
        sam2.image_encoder.trunk = det.backbone  # backbone を共有する(SAM 3 と同じ)
        for p in det.parameters():  # backbone と検出の部分は固定(SAM 3 の論文の 4 段目)
            p.requires_grad_(False)
        super().__init__(sam2, ImageSegmenter(det, "mask2former"))

    def train(self, mode: bool = True):
        """学習の状態を切り替える(固定した backbone と検出の部分は学習中も eval のまま。BatchNorm の統計を変えない)。

        Args:
            mode: True で学習の状態。

        Returns:
            self。
        """
        super().train(mode)
        self.detector.eval()
        return self
