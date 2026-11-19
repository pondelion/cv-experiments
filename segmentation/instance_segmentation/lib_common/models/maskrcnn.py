"""Mask R-CNN(古典の方式。ほかの方式と比べるための参考の値)。

torchvision の改良版の Mask R-CNN(`maskrcnn_resnet50_fpn_v2` と同じ構成。detectron2 の new baselines と同じく、FPN・RPN の 2 層の
畳み込み・box の head(畳み込み 4 層 + 全結合 1 層)・マスクの head(畳み込み 4 層)に BN を入れたもの)を使い、backbone だけを
timm の ResNet に差し替える(ほかの方式と同じ timm の ImageNet の事前学習の重み。backbone 以外はゼロから学習する)。
timm の ResNet は torchvision と層の名前(layer1〜layer4)が同じなので、torchvision の BackboneWithFPN にそのまま渡せる。

使い方はほかの方式と少し違う:
- 学習: model(images, targets) が損失の内訳の dict を返す(torchvision の Mask R-CNN は損失をモデルの中で計算する)。
  MaskRCNNLoss がそれを足して、ほかの方式の損失と同じ (損失, 内訳) の形にする
- 推論: model(images) が画像ごとの {"boxes", "labels", "scores", "masks"} の並びを返す(postprocess.maskrcnn_to_predictions)

入力の画像は、データの変換(transforms)で正規化と 512 × 512 への拡大縮小・埋めが済んでいるので、torchvision の中の変換
(GeneralizedRCNNTransform)では何もしない設定にする(平均 0・標準偏差 1、一辺 512 のまま)。

形の記号: B = 画像の数、S = 入力の一辺、T = 画像の中の物体の数、N = 予測の数
"""
import timm
import torch
from torch import nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, RPNHead, _default_anchorgen
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.ops import masks_to_boxes
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class MaskRCNNSeg(nn.Module):
    """timm の ResNet を backbone にした Mask R-CNN(torchvision の v2 の構成)。

    Args:
        num_classes: クラスの数(背景を含めない)。
        backbone: timm の ResNet のモデル名。
        size: 入力の一辺 S(torchvision の中の変換で大きさを変えないように渡す)。
        pretrained: backbone に事前学習の重みを読むか。
    """

    def __init__(self, num_classes: int, backbone: str = "resnet50.a1_in1k", size: int = 512, pretrained: bool = True):
        super().__init__()
        resnet = timm.create_model(backbone, pretrained=pretrained)
        body = BackboneWithFPN(resnet, {f"layer{i}": str(i - 1) for i in range(1, 5)},  # stride 4・8・16・32 の 4 段
                               [resnet.feature_info[i]["num_chs"] for i in range(1, 5)], 256,
                               extra_blocks=LastLevelMaxPool(), norm_layer=nn.BatchNorm2d)
        anchors = _default_anchorgen()
        self.model = MaskRCNN(
            body, num_classes=num_classes + 1,  # torchvision は背景をクラス 0 にする
            rpn_anchor_generator=anchors,
            rpn_head=RPNHead(256, anchors.num_anchors_per_location()[0], conv_depth=2),
            box_head=FastRCNNConvFCHead((256, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d),
            mask_head=MaskRCNNHeads(256, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d),
            min_size=size, max_size=size, image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0])

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        """学習では損失を、推論では予測を返す。

        Args:
            images: 正規化した画像 (B, 3, S, S)。
            targets: 学習のときの正解。画像ごとに {"masks": (T, S, S) の 0/1, "labels": (T,)(クラスの ID − 1)}。

        Returns:
            学習: 損失の内訳 {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"}。
            推論: 画像ごとの {"boxes": (N, 4) 入力の画素の x0, y0, x1, y1, "labels": (N,)(1 始まり), "scores": (N,),
                  "masks": (N, 1, S, S) の確率}。
        """
        if not self.training:
            return self.model(list(images))
        tv_targets = []
        for t in targets:
            boxes = masks_to_boxes(t["masks"]).float() if len(t["masks"]) else t["masks"].new_zeros((0, 4), dtype=torch.float32)
            boxes[:, 2:] += 1  # masks_to_boxes は端の画素の位置(含む)。枠は画素の外の端までにする(幅 1 画素の物体も幅 1 になる)
            tv_targets.append({"boxes": boxes, "labels": t["labels"] + 1, "masks": t["masks"]})
        return self.model(list(images), tv_targets)


class MaskRCNNLoss(nn.Module):
    """torchvision の Mask R-CNN が返す損失の内訳を足し、ほかの方式の損失と同じ (損失, 内訳) の形にする。"""

    def forward(self, outs: dict, targets: list[dict]) -> tuple[torch.Tensor, dict]:
        """損失を足す。

        Args:
            outs: MaskRCNNSeg の学習のときの出力(損失の内訳)。
            targets: 正解(使わない。ほかの方式の損失と呼び方をそろえるため)。

        Returns:
            (損失の合計, 内訳 {名前: 値})。
        """
        return sum(outs.values()), {k: float(v.detach()) for k, v in outs.items()}
