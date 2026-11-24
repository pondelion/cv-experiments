"""YOLO26-seg の方式の自作のモデル(論文 arXiv 2606.03748 を参考に、backbone を timm の CNN にしたもの。Ultralytics のコードは
AGPL なので写さず、論文と、YOLO 系で公開されている構成の考え方から作る)。

構成:
    backbone(timm、features_only。stride 8・16・32 の段)
    → neck(PAN: 上からの経路と下からの経路で 3 段を混ぜる。ブロックは C3k2、最上段は SPPF と C2PSA(自己注意))
    → 段ごとの head(クラス・枠・マスクの係数)。枠は reg_max = 1(DFL なし。中心から 4 辺までの距離を直接出す)
      head は 2 組: 1 対多(o2m。学習だけ)と 1 対 1(o2o。推論はこちらだけで NMS なし)。o2o の head は特徴を detach して受ける
    → マスク: 3 段を足した特徴(F_proto = P3 + Up(1×1(P4)) + Up(1×1(P5)))から、M 枚の原型のマスク(prototype、stride 4)を作り、
      物体ごとの係数との線形結合でマスクにする(YOLACT 系)。学習のときだけ、F_proto からクラスごとの意味のマスクも出す(補助の損失)

形の記号: B = 画像の数、C = チャンネル、h・w = 特徴の縦・横、A = 全部の段の位置(アンカー)の数、K = クラスの数、
         M = 原型のマスクの数、S = 入力の一辺
"""
import copy

import timm
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

STRIDES = (8, 16, 32)
NUM_PROTOS = 32


class ConvBNSiLU(nn.Sequential):
    """畳み込み + BatchNorm + SiLU(YOLO 系の基本の層)。

    Args:
        cin: 入力のチャンネル。
        cout: 出力のチャンネル。
        k: 窓の大きさ。
        s: stride。
        g: groups(cin と同じなら depthwise)。
    """

    def __init__(self, cin: int, cout: int, k: int = 1, s: int = 1, g: int = 1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False), nn.BatchNorm2d(cout), nn.SiLU())


class Bottleneck(nn.Module):
    """3×3 の畳み込み 2 つと残差(C3k2 の中の部品)。

    Args:
        c: チャンネル。
    """

    def __init__(self, c: int):
        super().__init__()
        self.cv1, self.cv2 = ConvBNSiLU(c, c, 3), ConvBNSiLU(c, c, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差つきで 2 つの畳み込みを通す。

        Args:
            x: (B(画像の数), C(チャンネル), h, w)。

        Returns:
            (B, C, h, w)。
        """
        return x + self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """C3k2(c3k=False の形): 1×1 で 2 つに分け、片方に Bottleneck を n 回重ね、途中の出力も全部つなげて 1×1 で混ぜる。

    Args:
        cin: 入力のチャンネル。
        cout: 出力のチャンネル。
        n: Bottleneck の数。
    """

    def __init__(self, cin: int, cout: int, n: int = 1):
        super().__init__()
        self.c = cout // 2
        self.cv1 = ConvBNSiLU(cin, 2 * self.c)
        self.blocks = nn.ModuleList(Bottleneck(self.c) for _ in range(n))
        self.cv2 = ConvBNSiLU((2 + n) * self.c, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """2 つに分けて片方に Bottleneck を重ね、途中の出力も全部つなげて混ぜる。

        Args:
            x: (B(画像の数), cin, h, w)。

        Returns:
            (B, cout, h, w)。
        """
        y = list(self.cv1(x).chunk(2, dim=1))  # 2 × (B, c, h, w)
        for blk in self.blocks:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, dim=1))  # (B, (2+n)·c, h, w) -> (B, cout, h, w)


class SPPF(nn.Module):
    """5×5 の max pooling を 3 回重ね、途中の結果をつなげて広い範囲の情報を混ぜる。

    Args:
        c: 入力・出力のチャンネル。
    """

    def __init__(self, c: int):
        super().__init__()
        self.cv1, self.cv2 = ConvBNSiLU(c, c // 2), ConvBNSiLU(c // 2 * 4, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """max pooling を 3 回重ね、途中の結果をつなげて混ぜる。

        Args:
            x: (B(画像の数), C(チャンネル), h, w)。

        Returns:
            (B, C, h, w)。
        """
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(F.max_pool2d(y[-1], 5, 1, 2))
        return self.cv2(torch.cat(y, dim=1))  # (B, 4·C/2, h, w) -> (B, C, h, w)


class PSABlock(nn.Module):
    """自己注意(値に 3×3 の depthwise 畳み込みの位置の情報を足す)+ 1×1 の畳み込みの FFN、どちらも残差。

    Args:
        c: チャンネル。
        heads: 注意の頭の数。
    """

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.pe = nn.Conv2d(c, c, 3, 1, 1, groups=c)
        self.proj = nn.Conv2d(c, c, 1)
        self.ffn = nn.Sequential(ConvBNSiLU(c, 2 * c), nn.Conv2d(2 * c, c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """自己注意と FFN を、どちらも残差つきで通す。

        Args:
            x: (B(画像の数), C(チャンネル), h, w)。

        Returns:
            (B, C, h, w)。
        """
        h, w = x.shape[-2:]
        q, k, v = rearrange(self.qkv(x), "b (three n d) h w -> three b n (h w) d", three=3, n=self.heads)
        a = F.scaled_dot_product_attention(q, k, v)  # (B, n, hw, d)
        a = rearrange(a, "b n (h w) d -> b (n d) h w", h=h, w=w)  # -> (B, C, h, w)
        x = x + self.proj(a + self.pe(rearrange(v, "b n (h w) d -> b (n d) h w", h=h, w=w)))
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """1×1 で 2 つに分け、片方に PSABlock を通してつなげ直す(最上段の特徴に広い範囲の関係を入れる)。

    Args:
        c: 入力・出力のチャンネル。
        n: PSABlock の数。
    """

    def __init__(self, c: int, n: int = 1):
        super().__init__()
        self.cv1, self.cv2 = ConvBNSiLU(c, c), ConvBNSiLU(c, c)
        self.blocks = nn.Sequential(*(PSABlock(c // 2, max(1, c // 128)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """2 つに分け、片方だけ PSABlock を通してつなげ直す。

        Args:
            x: (B(画像の数), C(チャンネル), h, w)。

        Returns:
            (B, C, h, w)。
        """
        a, b = self.cv1(x).chunk(2, dim=1)  # 2 × (B, C/2, h, w)
        return self.cv2(torch.cat([a, self.blocks(b)], dim=1))


class Neck(nn.Module):
    """PAN: backbone の stride 8・16・32 の特徴を、上から下・下から上の 2 回混ぜて、3 段の特徴にする。

    Args:
        cins: backbone の段のチャンネル (C3, C4, C5)。
        widths: 出力の段のチャンネル (P3, P4, P5)。
        n: C3k2 の Bottleneck の数。
    """

    def __init__(self, cins: tuple[int, int, int], widths: tuple[int, int, int] = (256, 512, 512), n: int = 2):
        super().__init__()
        c3, c4, c5 = widths
        self.lat3, self.lat4 = ConvBNSiLU(cins[0], c3), ConvBNSiLU(cins[1], c4)
        self.top = nn.Sequential(ConvBNSiLU(cins[2], c5), SPPF(c5), C2PSA(c5))
        self.td4 = C3k2(c5 + c4, c4, n)
        self.td3 = C3k2(c4 + c3, c3, n)
        self.down3, self.bu4 = ConvBNSiLU(c3, c3, 3, 2), C3k2(c3 + c4, c4, n)
        self.down4, self.bu5 = ConvBNSiLU(c4, c4, 3, 2), C3k2(c4 + c5, c5, n)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """3 段の特徴を混ぜる。

        Args:
            feats: backbone の [(B, C3, S/8, S/8), (B, C4, S/16, S/16), (B, C5, S/32, S/32)]。

        Returns:
            [P3 (B, c3, S/8, S/8), P4 (B, c4, S/16, S/16), P5 (B, c5, S/32, S/32)]。
        """
        x3, x4, p5 = self.lat3(feats[0]), self.lat4(feats[1]), self.top(feats[2])
        t4 = self.td4(torch.cat([F.interpolate(p5, size=x4.shape[-2:], mode="nearest"), x4], dim=1))
        p3 = self.td3(torch.cat([F.interpolate(t4, size=x3.shape[-2:], mode="nearest"), x3], dim=1))
        p4 = self.bu4(torch.cat([self.down3(p3), t4], dim=1))
        p5 = self.bu5(torch.cat([self.down4(p4), p5], dim=1))
        return [p3, p4, p5]


class Head(nn.Module):
    """段ごとのクラス・枠・マスクの係数の head(段ごとに別の重み)。

    Args:
        widths: 段のチャンネル。
        num_classes: クラスの数 K。
    """

    def __init__(self, widths: tuple[int, ...], num_classes: int):
        super().__init__()
        c_box, c_cls, c_coef = max(16, widths[0] // 4), max(widths[0], min(num_classes, 100)), max(widths[0] // 4, NUM_PROTOS)
        self.box = nn.ModuleList(nn.Sequential(ConvBNSiLU(c, c_box, 3), ConvBNSiLU(c_box, c_box, 3), nn.Conv2d(c_box, 4, 1))
                                 for c in widths)
        self.cls = nn.ModuleList(nn.Sequential(ConvBNSiLU(c, c, 3, g=c), ConvBNSiLU(c, c_cls),
                                               ConvBNSiLU(c_cls, c_cls, 3, g=c_cls), ConvBNSiLU(c_cls, c_cls),
                                               nn.Conv2d(c_cls, num_classes, 1)) for c in widths)
        self.coef = nn.ModuleList(nn.Sequential(ConvBNSiLU(c, c_coef, 3), ConvBNSiLU(c_coef, c_coef, 3),
                                                nn.Conv2d(c_coef, NUM_PROTOS, 1)) for c in widths)
        for box, cls, s in zip(self.box, self.cls, STRIDES):  # 最初の値(YOLO 系の慣例): 距離は 1 stride、クラスの確率は小さく
            nn.init.constant_(box[-1].bias, 1.0)
            nn.init.constant_(cls[-1].bias, -torch.log(torch.tensor((640 / s) ** 2 / 5 * num_classes)).item())

    def forward(self, feats: list[torch.Tensor]) -> dict:
        """段ごとの出力を、全部の段の位置を並べた形にする。

        Args:
            feats: [P3, P4, P5]。

        Returns:
            {"logits": (B, A, K), "ltrb": (B, A, 4) 中心から左・上・右・下の辺までの距離(stride の単位),
             "coef": (B, A, M)}。
        """
        cat = lambda mods: torch.cat([rearrange(m(f), "b c h w -> b (h w) c") for m, f in zip(mods, feats)], dim=1)
        return {"logits": cat(self.cls), "ltrb": cat(self.box), "coef": cat(self.coef)}


class Proto(nn.Module):
    """3 段の特徴を足して(F_proto)、原型のマスク(stride 4)と、学習のときの意味のマスク(stride 8)を作る。

    Args:
        widths: 段のチャンネル。
        num_classes: クラスの数 K。
        c: 原型を作る層のチャンネル。
    """

    def __init__(self, widths: tuple[int, int, int], num_classes: int, c: int = 256):
        super().__init__()
        self.proj = nn.ModuleList(nn.Conv2d(w, widths[0], 1) for w in widths[1:])
        self.make = nn.Sequential(ConvBNSiLU(widths[0], c, 3), nn.Upsample(scale_factor=2, mode="nearest"),
                                  ConvBNSiLU(c, c, 3), ConvBNSiLU(c, c), nn.Conv2d(c, NUM_PROTOS, 1))
        self.semantic = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """原型のマスクを作る。

        Args:
            feats: [P3, P4, P5]。

        Returns:
            (原型 (B, M, S/4, S/4), 学習のときはクラスごとの意味のマスクの値 (B, K, S/8, S/8)、推論では None)。
        """
        f = feats[0] + sum(F.interpolate(p(x), size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
                           for p, x in zip(self.proj, feats[1:]))  # (B, c3, S/8, S/8)
        return self.make(f), (self.semantic(f) if self.training else None)


class YOLOSeg(nn.Module):
    """YOLO26-seg の方式の自作のモデル。

    Args:
        num_classes: クラスの数 K。
        backbone: timm のモデル名。
        pretrained: backbone に timm の事前学習の重みを読むか。
        widths: neck の出力の段 (P3, P4, P5) のチャンネル(既定は公式の l の版と同じ。公式の s は (128, 256, 512)、
            n は (64, 128, 256))。
        depth: neck の C3k2 の Bottleneck の数(既定は l の版の 2。s・n は 1)。
        proto: 原型のマスクを作る層のチャンネル(既定は l の版の 256。s は 128、n は 64)。
    """

    def __init__(self, num_classes: int = 8, backbone: str = "resnet50.a1_in1k", pretrained: bool = True,
                 widths: tuple[int, int, int] = (256, 512, 512), depth: int = 2, proto: int = 256):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4))
        self.neck = Neck(tuple(self.backbone.feature_info.channels()), tuple(widths), depth)
        self.o2m = Head(widths, num_classes)
        self.o2o = copy.deepcopy(self.o2m)
        self.proto = Proto(widths, num_classes, proto)

    def forward(self, images: torch.Tensor) -> dict:
        """予測する。

        Args:
            images: (B, 3, S, S)。

        Returns:
            {"o2o": Head の出力, "o2m": Head の出力(学習のときだけ), "protos": (B, M, S/4, S/4),
             "semantic": (B, K, S/8, S/8)(学習のときだけ), "anchors": (A, 2) 位置の中心の画素の x, y, "strides": (A,)}。
        """
        feats = self.neck(self.backbone(images))
        protos, semantic = self.proto(feats)
        out = {"o2o": self.o2o([f.detach() for f in feats]), "protos": protos, "semantic": semantic}
        if self.training:
            out["o2m"] = self.o2m(feats)
        anchors, strides = [], []
        for f, s in zip(feats, STRIDES):
            h, w = f.shape[-2:]
            ys, xs = torch.meshgrid(torch.arange(h, device=f.device), torch.arange(w, device=f.device), indexing="ij")
            anchors.append(rearrange(torch.stack([xs, ys], dim=-1), "h w two -> (h w) two").float().add(0.5).mul(s))
            strides.append(torch.full((h * w,), float(s), device=f.device))
        out["anchors"], out["strides"] = torch.cat(anchors), torch.cat(strides)  # (A, 2), (A,)
        return out


def decode_boxes(ltrb: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """中心から 4 辺までの距離(stride の単位)を、画素の x0, y0, x1, y1 の枠にする。

    Args:
        ltrb: (..., A, 4)。
        anchors: (A, 2) 位置の中心の画素の x, y。
        strides: (A,)。

    Returns:
        (..., A, 4)。
    """
    d = ltrb.float() * strides[:, None]  # (..., A, 4) 画素の距離
    return torch.cat([anchors - d[..., :2], anchors + d[..., 2:]], dim=-1)
