"""1 段(stride 16)の特徴しか出さない ViT から、stride 8・16・32 の 3 段の特徴を作る adapter(DEIMv2 の DINOv3STAs と同じ作り)。

作り(DEIMv2 の `engine/backbone/dinov3_adapter.py` を参考にした自作):
1. ViT の途中の 3 つの層(既定は 4・8・12 層目 = 番号 3・7・11)の出力を、最後の LayerNorm を通して (B, D, H/16, W/16) で取り出す
   (timm の `forward_intermediates`)。浅い層ほど細かい段に使い、bilinear で H/8・H/16・H/32 に拡大・縮小する
2. 画像から小さい畳み込みの枝(STA: spatial tuning adapter)で、細かい位置の情報を持つ stride 8・16・32 の特徴を作る
   (ViT はパッチ 16 なので、それより細かい形の情報が足りない分を補う)
3. 段ごとに 1 と 2 をチャンネルの向きにつなげ、1×1 の畳み込み + BN で hidden_dim にそろえる

STA と融合の層は学習済みの重みがないので、学習率は backbone(ViT)の倍率をかけずにほかの部分と同じにする(DEIMv2 と同じ。
train.py の gtr_seg の AdamW の組分けで、名前が `backbone.sta.`・`backbone.fuse.` の重みを backbone から外す)。

形の記号: B = 画像の数、D = ViT の幅、c = STA の基本のチャンネル(既定 16)、C = hidden_dim、H・W = 入力の高さ・幅。
"""
from collections import OrderedDict
from types import SimpleNamespace

import timm
import torch
import torch.nn.functional as F
from torch import nn


def conv_bn(cin: int, cout: int, act_before: bool) -> nn.Sequential:
    """(GELU →)3×3・stride 2 の畳み込み → BN(STA の 1 段)。

    Args:
        cin: 入力のチャンネル。
        cout: 出力のチャンネル。
        act_before: 畳み込みの前に GELU を入れるか。

    Returns:
        nn.Sequential(BN の名前は weight decay を 0 にする組分けのため `bn`)。
    """
    layers = [("act", nn.GELU())] if act_before else []
    layers += [("conv", nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False)), ("bn", nn.BatchNorm2d(cout))]
    return nn.Sequential(OrderedDict(layers))


class SpatialTuningAdapter(nn.Module):
    """画像から stride 8・16・32 の細かい特徴を作る小さい畳み込みの枝(DEIMv2 の SpatialPriorModulev2 と同じ構成)。

    Args:
        c: 基本のチャンネル(出力は stride 8 で 2c、16・32 で 4c)。
    """

    def __init__(self, c: int = 16, with_s4: bool = False):
        super().__init__()
        self.stem = nn.Sequential(OrderedDict([  # (B, 3, H, W) -> (B, c, H/4, W/4)
            ("conv", nn.Conv2d(3, c, 3, stride=2, padding=1, bias=False)), ("bn", nn.BatchNorm2d(c)),
            ("act", nn.GELU()), ("pool", nn.MaxPool2d(3, stride=2, padding=1))]))
        self.s8 = conv_bn(c, 2 * c, act_before=False)  # -> (B, 2c, H/8, W/8)
        self.s16 = conv_bn(2 * c, 4 * c, act_before=True)  # -> (B, 4c, H/16, W/16)
        self.s32 = conv_bn(4 * c, 4 * c, act_before=True)  # -> (B, 4c, H/32, W/32)
        self.with_s4 = with_s4  # stem の出力(stride 4)も返す(stride 4 の段を使う Mask2Former 用)
        self.channels = ([c] if with_s4 else []) + [2 * c, 4 * c, 4 * c]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Args: x: (B, 3, H, W)。Returns: [(B, 2c, H/8, W/8), (B, 4c, H/16, W/16), (B, 4c, H/32, W/32)]
        (with_s4 のときは先頭に (B, c, H/4, W/4))。"""
        f4 = self.stem(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        return ([f4] if self.with_s4 else []) + [f8, f16, self.s32(f16)]


class ViTAdapter(nn.Module):
    """ViT の途中の層と STA から、stride 8・16・32 の 3 段の特徴を作る(timm の features_only の backbone と同じ使い方)。

    Args:
        name: timm の ViT の名前(パッチ 16)。
        pretrained: ViT の学習済みの重みを読むか。
        img_size: 入力の一辺(位置の埋め込みをこの大きさで作る)。
        layers: 取り出す層の番号(0 始まり。浅い順に stride 8・16・32 に使う)。
        sta_channels: STA の基本のチャンネル。
        hidden_dim: 出力のチャンネル(None なら ViT の幅)。
        vit: timm にない ViT を使うとき、作った ViT(`forward_intermediates` を持つ timm の形。name・pretrained・img_size は使わない)。
        norm: 取り出す層の出力に、ViT の最後の LayerNorm を通すか(DINOv3 の get_intermediate_layers の既定と同じく True。
            最後の LayerNorm を持たない DEIMv2 の ViT-Tiny では False)。
        strides: 出す段の stride。既定は stride 8・16・32 の 3 段。(4, 8, 16, 32) にすると、STA の stem(stride 4)と ViT の
            もう 1 つの浅い層から stride 4 の段も作る(stride 4 からの 4 段を使う Mask2Former 用。layers は 4 つ必要。
            ViT-Adapter と同じく、12 層なら番号 2・5・8・11)。
    """

    def __init__(self, name: str, pretrained: bool = True, img_size: int = 512, layers: tuple[int, ...] | None = None,
                 sta_channels: int = 16, hidden_dim: int | None = None, vit: nn.Module | None = None, norm: bool = True,
                 strides: tuple[int, ...] = (8, 16, 32)):
        super().__init__()
        self.vit = vit if vit is not None else timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size)
        strides = tuple(strides)
        if strides not in ((8, 16, 32), (4, 8, 16, 32)):
            raise ValueError(f"strides は (8, 16, 32) か (4, 8, 16, 32) だけ: {strides}")
        if layers is None:
            layers = (3, 7, 11) if len(strides) == 3 else (2, 5, 8, 11)
        if len(layers) != len(strides):
            raise ValueError(f"layers({layers})と strides({strides})の数が違う")
        self.layers, self.norm = list(layers), norm
        d = self.vit.embed_dim
        hidden_dim = hidden_dim or d
        self.sta = SpatialTuningAdapter(sta_channels, with_s4=len(strides) == 4)
        self.fuse = nn.ModuleList(nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(d + c, hidden_dim, 1, bias=False)), ("bn", nn.BatchNorm2d(hidden_dim))]))
            for c in self.sta.channels)
        # timm の features_only の backbone と同じ属性(GTRSeg・Mask2Former が出力のチャンネル・入力の正規化を読む)
        self.feature_info = SimpleNamespace(channels=lambda: [hidden_dim] * len(strides), reduction=lambda: list(strides))
        self.pretrained_cfg = self.vit.pretrained_cfg

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """ViT の層で gradient checkpointing を使う(STA と融合の層は軽いので使わない)。"""
        self.vit.set_grad_checkpointing(enable)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Args: x: (B, 3, H, W)。Returns: 段ごとの [(B, C, H/s, W/s)](既定は s = 8・16・32、strides が 4 段なら 4・8・16・32)。"""
        sem = self.vit.forward_intermediates(x, indices=self.layers, norm=self.norm, output_fmt="NCHW",
                                             intermediates_only=True)  # 段の数 × (B, D, H/16, W/16)
        detail = self.sta(x)
        outs = []
        for f, d, fuse in zip(sem, detail, self.fuse):
            f = F.interpolate(f, size=d.shape[-2:], mode="bilinear", align_corners=False)  # -> (B, D, H/s, W/s)
            outs.append(fuse(torch.cat([f, d], dim=1)))  # (B, D + c_s, H/s, W/s) -> (B, C, H/s, W/s)
        return outs
