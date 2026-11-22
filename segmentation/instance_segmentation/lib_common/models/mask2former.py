"""Mask2Former(Cheng ら CVPR 2022)系のインスタンスセグメンテーションのモデル(スクラッチ実装)。

構成:
    backbone:          timm の画像のモデル(features_only)。stride 4・8・16・32 の 4 段の特徴
    pixel decoder:     FPN(上の段から足し合わせる)。論文の多重解像度の deformable な注意の版は exp00102 で入れる
                       出力: マスクを作る特徴(stride 4、256 次元)と、decoder が注意を向ける 3 段の特徴(stride 32・16・8)
    transformer decoder: 学習するクエリ(物体の候補)NUM_QUERIES 個。各層は
                       マスクで絞った cross-attention(前の層の予測マスクの外は見ない)→ self-attention → FFN。
                       3 段の特徴を順番に使い、3 周で 9 層
    予測:              各層の出力から、クラス(クラスの数 + 物体なし)と、マスク(クエリの埋め込み × マスクの特徴の内積)を出す。
                       学習では全層(と最初のクエリ)の予測に損失を掛ける(補助の損失)

形の記号(このファイルで共通):
    B = 画像の数、S = 入力の画像の一辺の画素数、Q = クエリの数、C = 特徴の次元(D_MODEL = 256)、
    H・W = 特徴の縦・横の画素数(段ごとに違う)、N = H × W(特徴の画素を 1 列に並べた数)、K = クラスの数、
    h = 注意の頭の数(nhead)
"""
import math

import timm
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from segmentation.instance_segmentation.lib_common.backbones import create_feature_backbone

D_MODEL = 256


def sine_pos(x: torch.Tensor, temperature: float = 10000.0) -> torch.Tensor:
    """2 次元の位置の埋め込み(DETR の sine 型)。画素の縦・横の位置を、周期の違う sin・cos の値の並びで表す。

    Args:
        x: 特徴 (B(画像の数), C(特徴の次元), H(縦の画素数), W(横の画素数))。形と型だけを使う。

    Returns:
        x と同じ形 (B, C, H, W) の位置の埋め込み。チャンネルの前半 C/2 が縦の位置、後半 C/2 が横の位置。
    """
    b, c, h, w = x.shape
    # 位置を 0〜2π に並べる: (H, W)
    y_embed = repeat(torch.arange(1, h + 1, device=x.device, dtype=torch.float32), "h -> h w", w=w) / h * 2 * math.pi
    x_embed = repeat(torch.arange(1, w + 1, device=x.device, dtype=torch.float32), "w -> h w", h=h) / w * 2 * math.pi
    n = c // 2
    dim_t = temperature ** (2 * (torch.arange(n, device=x.device) // 2) / n)  # (C/2,) 周期
    px = rearrange(x_embed, "h w -> h w 1") / dim_t  # (H, W) -> (H, W, C/2)
    py = rearrange(y_embed, "h w -> h w 1") / dim_t  # (H, W) -> (H, W, C/2)
    # 偶数番目に sin、奇数番目に cos を交互に並べる: (H, W, C/4, 2) -> (H, W, C/2)
    px = rearrange(torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], dim=-1), "h w c two -> h w (c two)")
    py = rearrange(torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], dim=-1), "h w c two -> h w (c two)")
    pos = torch.cat([py, px], dim=-1)  # (H, W, C)
    return repeat(pos, "h w c -> b c h w", b=b).to(x.dtype)  # (H, W, C) -> (B, C, H, W)


class PixelDecoder(nn.Module):
    """FPN: backbone の 4 段の特徴を、上の段(粗い)から順に足し合わせる。

    Args:
        in_channels: backbone の 4 段(stride 4・8・16・32 の順)のチャンネルの数。
    """

    def __init__(self, in_channels: list[int]):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, D_MODEL, 1) for c in in_channels])
        self.output = nn.ModuleList([nn.Sequential(nn.Conv2d(D_MODEL, D_MODEL, 3, padding=1, bias=False),
                                                   nn.GroupNorm(32, D_MODEL), nn.ReLU(inplace=True)) for _ in in_channels])
        self.mask_proj = nn.Conv2d(D_MODEL, D_MODEL, 1)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """4 段の特徴を合わせる。

        Args:
            feats: stride 4・8・16・32 の順の特徴。i 段目は (B(画像の数), C_i(その段のチャンネルの数), S/s_i, S/s_i)
                (S = 入力の一辺、s_i = その段の stride)。

        Returns:
            (マスクを作る特徴 (B, C(=256), S/4, S/4)、
             decoder が注意を向ける特徴 [stride 32 の (B, C, S/32, S/32), stride 16 の (B, C, S/16, S/16),
             stride 8 の (B, C, S/8, S/8)])。
        """
        x = None
        outs = []
        for i in range(len(feats) - 1, -1, -1):  # 粗い段から
            lat = self.lateral[i](feats[i])  # (B, C_i, H, W) -> (B, C, H, W)
            # 1 つ上の(粗い)段の結果を、この段の大きさに拡大して足す: (B, C, H/2, W/2) -> (B, C, H, W)
            x = lat if x is None else lat + F.interpolate(x, size=lat.shape[-2:], mode="nearest")
            outs.append(self.output[i](x))  # (B, C, H, W)
        # outs: stride 32, 16, 8, 4 の順
        return self.mask_proj(outs[3]), outs[:3]


def conv_gn_relu(cin: int, cout: int, k: int, relu: bool = True) -> nn.Sequential:
    """畳み込み(bias なし)+ GroupNorm(+ ReLU)。重みは kaiming の一様分布(a = 1)で初期化する。

    Args:
        cin: 入力のチャンネルの数。
        cout: 出力のチャンネルの数。
        k: カーネルの大きさ(1 か 3)。
        relu: 最後に ReLU を掛けるか。

    Returns:
        (B, cin, H, W) -> (B, cout, H, W) の層。
    """
    conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
    nn.init.kaiming_uniform_(conv.weight, a=1)
    return nn.Sequential(conv, nn.GroupNorm(32, cout), *([nn.ReLU(inplace=True)] if relu else []))


class M2FPixelDecoder(nn.Module):
    """公式の Mask2Former の FPN(MaskFormer の pixel decoder。論文の表 4(e) の FPN)と同じ構成の FPN。

    `PixelDecoder`(簡略版)との違い:
        - 最上段(stride 32)は横からの 1×1 を通さず、3×3 の畳み込み + GN + ReLU だけ
        - 横からの 1×1 の畳み込みに GN を付ける
        - 下の段へ渡すのは、3×3 の畳み込み + GN + ReLU を通した後の値(段ごとに非線形の層が重なる)
        - マスクを作る特徴の射影は 3×3 の畳み込み
        - 畳み込みの重みは kaiming の一様分布(a = 1)、bias は 0

    Args:
        in_channels: backbone の 4 段(stride 4・8・16・32 の順)のチャンネルの数。
    """

    def __init__(self, in_channels: list[int]):
        super().__init__()
        self.top = conv_gn_relu(in_channels[-1], D_MODEL, 3)
        self.lateral = nn.ModuleList([conv_gn_relu(c, D_MODEL, 1, relu=False) for c in in_channels[:-1]])
        self.output = nn.ModuleList([conv_gn_relu(D_MODEL, D_MODEL, 3) for _ in in_channels[:-1]])
        self.mask_proj = nn.Conv2d(D_MODEL, D_MODEL, 3, padding=1)
        nn.init.kaiming_uniform_(self.mask_proj.weight, a=1)
        nn.init.zeros_(self.mask_proj.bias)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """4 段の特徴を合わせる。

        Args:
            feats: stride 4・8・16・32 の順の特徴。i 段目は (B, C_i, S/s_i, S/s_i)(s_i = その段の stride)。

        Returns:
            (マスクを作る特徴 (B, C(=256), S/4, S/4)、
             decoder が注意を向ける特徴 [stride 32 の (B, C, S/32, S/32), stride 16 の (B, C, S/16, S/16),
             stride 8 の (B, C, S/8, S/8)])。
        """
        y = self.top(feats[-1])  # (B, C_4, S/32, S/32) -> (B, C, S/32, S/32)
        outs = [y]
        for i in range(len(feats) - 2, -1, -1):  # stride 16・8・4 の順
            lat = self.lateral[i](feats[i])  # (B, C_i, H, W) -> (B, C, H, W)
            # 1 つ上の段の出力(3×3 を通した後)を、この段の大きさに拡大して足し、3×3 を通す: (B, C, H, W)
            y = self.output[i](lat + F.interpolate(y, size=lat.shape[-2:], mode="nearest"))
            outs.append(y)
        # outs: stride 32, 16, 8, 4 の順
        return self.mask_proj(outs[3]), outs[:3]


class MultiScaleDeformableAttention(nn.Module):
    """多重解像度の deformable な注意(Deformable DETR の考え方。pixel decoder の encoder 用)。

    各トークンが、自分の位置(参照点)の近くの点を段ごとに points 個ずつ双線形に取り、学習する重みで足す。
    点の位置 = 参照点 + 予測したずれ / その段の (幅, 高さ)(ずれは段の画素の単位)。

    Args:
        heads: 頭の数。
        levels: 段の数。
        points: 1 頭・1 段あたりの点の数。
    """

    def __init__(self, heads: int = 8, levels: int = 3, points: int = 4):
        super().__init__()
        self.heads, self.levels, self.points = heads, levels, points
        self.value_proj = nn.Linear(D_MODEL, D_MODEL)
        self.sampling_offsets = nn.Linear(D_MODEL, heads * levels * points * 2)
        self.attention_weights = nn.Linear(D_MODEL, heads * levels * points)
        self.output_proj = nn.Linear(D_MODEL, D_MODEL)
        # 最初の点の置き方: 頭ごとに等しい角度で向きを分け、正方形の辺の上(大きい方の成分が 1)の i 番目の点を i 倍の遠さに置く
        nn.init.zeros_(self.sampling_offsets.weight)
        angle = torch.linspace(0, 2 * math.pi, heads + 1)[:-1]
        direction = torch.stack([angle.cos(), angle.sin()], -1)
        direction = direction / direction.abs().amax(-1, keepdim=True)  # (heads, 2)
        rank = torch.arange(1, points + 1).float()  # (points,)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(rearrange(
                torch.einsum("hx,p->hpx", direction, rank)[:, None].expand(heads, levels, points, 2), "h l p x -> (h l p x)"))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for lin in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, query: torch.Tensor, ref: torch.Tensor, value: torch.Tensor,
                shapes: list[tuple[int, int]]) -> torch.Tensor:
        """注意の出力を作る。

        Args:
            query: (B, N, C)(N = 全部の段のトークンの数。位置の埋め込みを足したもの)。
            ref: 参照点 (B, N, 2)(0〜1 の x, y)。
            value: (B, N, C)(位置の埋め込みを足さないもの)。
            shapes: 段ごとの (h, w)(value の並びの順)。

        Returns:
            (B, N, C)。
        """
        b, n, _ = query.shape
        v = rearrange(self.value_proj(value), "b n (h c) -> b h c n", h=self.heads)  # (B, N, C) -> (B, heads, c, N)
        v = [rearrange(x, "b h c (y w) -> (b h) c y w", y=hh) for x, (hh, _) in
             zip(v.split([hh * ww for hh, ww in shapes], dim=-1), shapes)]  # 段ごとに (B·heads, c, h_l, w_l)
        off = rearrange(self.sampling_offsets(query), "b n (h l p two) -> b n h l p two",
                        h=self.heads, l=self.levels, p=self.points)
        attn = rearrange(self.attention_weights(query), "b n (h lp) -> b n h lp", h=self.heads).softmax(-1)
        attn = rearrange(attn, "b n h (l p) -> (b h) 1 n l p", l=self.levels)  # 段と点をまとめて softmax した重み
        scale = torch.tensor([[w, hh] for hh, w in shapes], device=query.device, dtype=off.dtype)  # (L, 2) 段の (幅, 高さ)
        loc = rearrange(ref, "b n two -> b n 1 1 1 two") + off / scale[:, None]  # (B, N, heads, L, P, 2) 0〜1 の位置
        grid = rearrange(2 * loc - 1, "b n h l p two -> (b h) l n p two")  # grid_sample の座標(−1〜1)
        out = sum((F.grid_sample(v[l], grid[:, l].to(v[l].dtype), mode="bilinear", padding_mode="zeros",
                                 align_corners=False) * attn[..., l, :].to(v[l].dtype)).sum(-1)
                  for l in range(self.levels))  # 段ごとに (B·heads, c, N, P) を重みで足して (B·heads, c, N)
        return self.output_proj(rearrange(out, "(b h) c n -> b n (h c)", b=b))


class DeformableEncoderLayer(nn.Module):
    """pixel decoder の encoder の 1 層: 多重解像度の deformable な注意 → FFN(どちらも残差と後置きの LayerNorm)。

    Args:
        ffn: FFN の中間の次元(公式は 1024)。
        heads: 頭の数。
        points: 1 頭・1 段あたりの点の数。
    """

    def __init__(self, ffn: int = 1024, heads: int = 8, points: int = 4):
        super().__init__()
        self.attn = MultiScaleDeformableAttention(heads, 3, points)
        self.norm1, self.norm2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, D_MODEL))

    def forward(self, x: torch.Tensor, pos: torch.Tensor, ref: torch.Tensor, shapes: list[tuple[int, int]]) -> torch.Tensor:
        """1 層を通す。

        Args:
            x: 全部の段のトークン (B(画像の数), N(トークンの数), C(特徴の次元))。
            pos: 位置と段の埋め込み (B, N, C)。
            ref: 参照点 (B, N, 2)(0〜1 の x, y)。
            shapes: 段ごとの (h, w)(x の並びの順)。

        Returns:
            (B, N, C)。
        """
        x = self.norm1(x + self.attn(x + pos, ref, x, shapes))
        return self.norm2(x + self.ffn(x))


class DeformablePixelDecoder(nn.Module):
    """公式の Mask2Former の pixel decoder(MSDeformAttnPixelDecoder)を参考にした版。

    stride 8・16・32 の段を 1×1 の畳み込み + GroupNorm で C チャンネルにし、3 段の全部の画素をトークンにして
    deformable な注意の encoder を enc_layers 層通す(段をまたいで情報を混ぜる)。stride 4 の段は FPN と同じく、
    encoder の stride 8 の出力を拡大して足し、3×3 の畳み込みに通す。マスクの特徴はその 1×1 の畳み込み。
    返り値の形は PixelDecoder と同じ。

    Args:
        in_channels: backbone の 4 段(stride 4・8・16・32 の順)のチャンネルの数。
        enc_layers: encoder の層の数(公式は 6)。
    """

    def __init__(self, in_channels: list[int], enc_layers: int = 6):
        super().__init__()
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(c, D_MODEL, 1), nn.GroupNorm(32, D_MODEL))
                                        for c in in_channels[1:])  # stride 8・16・32
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.zeros_(proj[0].bias)
        self.level_embed = nn.Parameter(torch.randn(3, D_MODEL))  # 段の埋め込み(公式と同じく標準正規分布で初期化)
        self.layers = nn.ModuleList(DeformableEncoderLayer() for _ in range(enc_layers))
        self.lateral = nn.Sequential(nn.Conv2d(in_channels[0], D_MODEL, 1, bias=False), nn.GroupNorm(32, D_MODEL))
        self.output = nn.Sequential(nn.Conv2d(D_MODEL, D_MODEL, 3, padding=1, bias=False), nn.GroupNorm(32, D_MODEL),
                                    nn.ReLU(inplace=True))
        self.mask_proj = nn.Conv2d(D_MODEL, D_MODEL, 1)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """backbone の特徴から、マスクを作る特徴と、decoder が注意を向ける 3 段の特徴を作る。

        Args:
            feats: stride 4・8・16・32 の順の特徴。i 段目は (B(画像の数), C_i(その段のチャンネルの数), S/s_i, S/s_i)
                (S = 入力の一辺、s_i = その段の stride)。

        Returns:
            (マスクを作る特徴 (B, C, S/4, S/4)、[stride 32 の (B, C, S/32, S/32), stride 16 の (B, C, S/16, S/16),
             stride 8 の (B, C, S/8, S/8)])。
        """
        srcs = [proj(f) for proj, f in zip(self.input_proj, feats[1:])][::-1]  # stride 32, 16, 8 の順に (B, C, h_l, w_l)
        shapes = [tuple(s.shape[-2:]) for s in srcs]
        x = torch.cat([rearrange(s, "b c h w -> b (h w) c") for s in srcs], dim=1)  # (B, N, C)
        pos = torch.cat([rearrange(sine_pos(s), "b c h w -> b (h w) c") + self.level_embed[l] for l, s in enumerate(srcs)],
                        dim=1)  # (B, N, C) 位置 + 段の埋め込み
        ref = torch.cat([rearrange(torch.stack(torch.meshgrid((torch.arange(w, device=x.device) + 0.5) / w,
                                                             (torch.arange(h, device=x.device) + 0.5) / h,
                                                             indexing="xy"), -1), "h w two -> (h w) two")
                         for h, w in shapes])[None].expand(x.shape[0], -1, -1)  # (B, N, 2) 自分の位置(0〜1 の x, y)
        for layer in self.layers:
            x = layer(x, pos, ref, shapes)
        outs = [rearrange(z, "b (h w) c -> b c h w", h=h) for z, (h, w) in
                zip(x.split([h * w for h, w in shapes], dim=1), shapes)]  # 段ごとに (B, C, h_l, w_l)
        fine = self.output(self.lateral(feats[0]) + F.interpolate(outs[-1], size=feats[0].shape[-2:], mode="bilinear",
                                                                   align_corners=False))  # (B, C, S/4, S/4)
        return self.mask_proj(fine), outs


class UnetBlock(nn.Module):
    """UNet の decoder の 1 段(smp の DecoderBlock と同じ形): 2 倍に拡大 → skip とつなげる → 3×3 の畳み込み + BN + ReLU を 2 回。

    Args:
        cin: 下から来る(粗い段の)チャンネル。
        cskip: skip(encoder の同じ解像度の段)のチャンネル。
        cout: 出力のチャンネル。
    """

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        conv = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(conv(cin + cskip, cout), conv(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """1 段上げる。

        Args:
            x: 下から来る(粗い段の)特徴 (B(画像の数), cin, h, w)。
            skip: encoder の同じ解像度の段の特徴 (B, cskip, 2h, 2w)。

        Returns:
            (B, cout, 2h, 2w)。
        """
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")  # (B, cin, h, w) -> (B, cin, 2h, 2w)
        return self.conv(torch.cat([x, skip], dim=1))


class UnetPixelDecoder(nn.Module):
    """UNet の decoder(smp の UnetDecoder の形)を Mask2Former の pixel decoder にしたもの。

    stride 32 の特徴から、UnetBlock で stride 16・8・4 へ順に上げる(各段で encoder の同じ解像度の段を skip としてつなげる)。
    Mask2Former が使う特徴は、stride 32(encoder の出力)・16・8 の 3 段(decoder が注意を向ける)と、stride 4(マスクの特徴)。
    既定はどれも 1×1 の畳み込みで C チャンネルにそろえる。gn_proj=True では、公式の FPN と同じく 3×3 の畳み込み + GN + ReLU で
    そろえる(マスクの特徴はその後に 1×1 の畳み込み)。返り値の形は PixelDecoder と同じ。

    Args:
        in_channels: backbone の 4 段(stride 4・8・16・32 の順)のチャンネルの数。
        decoder_channels: UnetBlock の出力のチャンネル(stride 16・8・4 の順。既定は smp の既定の先頭 3 つ)。
        gn_proj: C チャンネルにそろえる層を 3×3 の畳み込み + GN + ReLU にするか。
    """

    def __init__(self, in_channels: list[int], decoder_channels: tuple[int, int, int] = (256, 128, 64),
                 gn_proj: bool = False):
        super().__init__()
        cins = [in_channels[3]] + list(decoder_channels[:-1])  # 下から来るチャンネル
        self.blocks = nn.ModuleList(UnetBlock(cin, cskip, cout) for cin, cskip, cout in
                                    zip(cins, in_channels[2::-1], decoder_channels))  # skip は stride 16・8・4 の段
        level_channels = [in_channels[3]] + list(decoder_channels[:2])  # decoder に渡す stride 32・16・8 の段
        if gn_proj:
            self.proj = nn.ModuleList(conv_gn_relu(c, D_MODEL, 3) for c in level_channels)
            self.mask_proj = nn.Sequential(conv_gn_relu(decoder_channels[2], D_MODEL, 3), nn.Conv2d(D_MODEL, D_MODEL, 1))
        else:
            self.proj = nn.ModuleList(nn.Conv2d(c, D_MODEL, 1) for c in level_channels)
            self.mask_proj = nn.Conv2d(decoder_channels[2], D_MODEL, 1)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """backbone の特徴から、マスクを作る特徴と、decoder が注意を向ける 3 段の特徴を作る。

        Args:
            feats: stride 4・8・16・32 の順の特徴。i 段目は (B(画像の数), C_i(その段のチャンネルの数), S/s_i, S/s_i)
                (S = 入力の一辺、s_i = その段の stride)。

        Returns:
            (マスクを作る特徴 (B, C, S/4, S/4)、[stride 32 の (B, C, S/32, S/32), stride 16 の (B, C, S/16, S/16),
             stride 8 の (B, C, S/8, S/8)])。
        """
        x, levels = feats[3], [feats[3]]  # stride 32 から
        for block, skip in zip(self.blocks, feats[2::-1]):  # skip: stride 16, 8, 4
            x = block(x, skip)
            levels.append(x)  # stride 16, 8, 4 の順に足していく
        outs = [proj(f) for proj, f in zip(self.proj, levels[:3])]  # stride 32, 16, 8 を (B, C, h_l, w_l) に
        return self.mask_proj(levels[3]), outs


class DecoderLayer(nn.Module):
    """マスクで絞った cross-attention → self-attention → FFN(どれも残差 + LayerNorm、論文と同じ後置きの正規化)。

    Args:
        nhead: 注意の頭の数。
        ffn: FFN の中間の次元。
    """

    def __init__(self, nhead: int = 8, ffn: int = 2048):
        super().__init__()
        self.cross = nn.MultiheadAttention(D_MODEL, nhead, batch_first=True)
        self.self_attn = nn.MultiheadAttention(D_MODEL, nhead, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, D_MODEL))
        self.norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(3)])

    def forward(self, q: torch.Tensor, qpos: torch.Tensor, mem: torch.Tensor, mpos: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        """クエリを 1 層分更新する。

        Args:
            q: クエリ (B(画像の数), Q(クエリの数), C(特徴の次元))。
            qpos: クエリの位置の埋め込み (B, Q, C)。
            mem: 注意を向ける画像の特徴 (B, N(特徴の画素の数 = H × W), C)。
            mpos: 画像の特徴の位置の埋め込み (B, N, C)。
            attn_mask: cross-attention で見ない画素 (B × h(注意の頭の数), Q, N) の bool(True = 見ない)。

        Returns:
            更新したクエリ (B, Q, C)。
        """
        # need_weights=False: 注意の重みを返させない(返させると速い fused の計算が使われず、遅くメモリも増える)
        q = self.norms[0](q + self.cross(q + qpos, mem + mpos, mem, attn_mask=attn_mask, need_weights=False)[0])
        q = self.norms[1](q + self.self_attn(q + qpos, q + qpos, q, need_weights=False)[0])
        return self.norms[2](q + self.ffn(q))


class Mask2Former(nn.Module):
    """Mask2Former 系のモデル。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        backbone: timm のモデル名。
        num_queries: クエリの数。
        dec_layers: transformer decoder の層の数(3 段の特徴を順番に使う)。
        nhead: decoder の注意の頭の数。
        pretrained: backbone に timm の事前学習の重みを読むか。
        pixel_decoder: "fpn"(PixelDecoder。簡略版)/ "fpn_m2f"(M2FPixelDecoder。公式の FPN と同じ構成)/
            "msdeform"(DeformablePixelDecoder。公式と同じ構成)/
            "unet"(UnetPixelDecoder。UNet の decoder)/ "unet_gn"・"unet_gn_wide"(UNet の確かめ用。上の decoders の注)。
        mask_pos: マスクを作る画素の特徴に位置の情報を足すか("none": 足さない / "sine": sine の位置の埋め込みを
            1×1 の畳み込み(重みは 0 から)に通して足す)。
        reid_head: 最後の層のクエリから reid の埋め込みを作る head を持つか(CTVIS・DVIS++ の segmenter。動画の exp00502)。
        context_head: CAVIS の segmenter(公式の VideoMultiScaleMaskedTransformerDecoder_cavis)にするか。reid_head も要る。
            最後の層の予測マスクの境界の画素で、周り(context_filter_size × context_filter_size)の平均の特徴を集めた「文脈の特徴」を
            作り、reid の head の入力を [クエリの埋め込み; 文脈の特徴](2 × 256 次元)にする(_context 参照)。
        context_filter_size: 文脈の特徴の平均をとる窓の一辺(公式の VIPSeg の設定は 9)。
    """

    def __init__(self, num_classes: int = 20, backbone: str = "resnet50.a1_in1k", num_queries: int = 100,
                 dec_layers: int = 9, nhead: int = 8, pretrained: bool = True, pixel_decoder: str = "fpn",
                 mask_pos: str = "none", reid_head: bool = False, context_head: bool = False, context_filter_size: int = 9):
        super().__init__()
        # stride 4・8・16・32 の 4 段(R50 では今までと同じ段の番号 (1, 2, 3, 4)。ConvNeXt などは番号が違うので stride で選ぶ。
        # 素の ViT は adapter で 4 段を作る。backbones/__init__.py の create_feature_backbone)
        self.backbone = create_feature_backbone(backbone, pretrained, strides=(4, 8, 16, 32))
        decoders = {"fpn": PixelDecoder, "fpn_m2f": M2FPixelDecoder, "msdeform": DeformablePixelDecoder,
                    "unet": UnetPixelDecoder,
                    # UNet の確かめ用: 渡す特徴を 3×3 + GN + ReLU でそろえる / さらに UNet の幅を全部 256 にする
                    "unet_gn": lambda c: UnetPixelDecoder(c, gn_proj=True),
                    "unet_gn_wide": lambda c: UnetPixelDecoder(c, (256, 256, 256), gn_proj=True)}
        self.pixel_decoder = decoders[pixel_decoder](self.backbone.feature_info.channels())
        # マスクを作る画素の特徴に位置の情報を足すか(畳み込みだけの pixel decoder では、同じ見た目の物体どうしを
        # 場所で見分ける情報が画素の特徴にないので)。重みを 0 から始め、最初は足さないのと同じ計算にする
        self.mask_pos = None
        if mask_pos == "sine":
            self.mask_pos = nn.Conv2d(D_MODEL, D_MODEL, 1)
            nn.init.zeros_(self.mask_pos.weight)
            nn.init.zeros_(self.mask_pos.bias)
        elif mask_pos != "none":
            raise ValueError(f"知らない mask_pos: {mask_pos}")
        self.level_embed = nn.Embedding(3, D_MODEL)
        self.query_feat = nn.Embedding(num_queries, D_MODEL)
        self.query_pos = nn.Embedding(num_queries, D_MODEL)
        self.layers = nn.ModuleList([DecoderLayer(nhead) for _ in range(dec_layers)])
        self.decoder_norm = nn.LayerNorm(D_MODEL)
        self.class_embed = nn.Linear(D_MODEL, num_classes + 1)
        self.mask_embed = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(inplace=True), nn.Linear(D_MODEL, D_MODEL),
                                        nn.ReLU(inplace=True), nn.Linear(D_MODEL, D_MODEL))
        # reid の head(CTVIS・DVIS++ の segmenter。公式の VideoMultiScaleMaskedTransformerDecoder_dvisPlus の reid_embed と同じ):
        # 最後の層のクエリの埋め込み(LayerNorm の後)から、フレーム間の照合と対照学習に使う埋め込みを作る 3 層の MLP。
        # 初期化は公式と同じ c2_xavier_fill(kaiming_uniform(a=1)、bias 0)
        # CAVIS(context_head)は公式の MLP(hidden_dim × 2 → 256 → 256 → 256)と同じく、1 層目の入力が 2 × 256 次元
        self.reid_embed = None
        assert reid_head or not context_head, "context_head は reid_head と一緒に使う(CAVIS の segmenter)"
        self.context_filter_size = context_filter_size if context_head else None
        if reid_head:
            d_in = 2 * D_MODEL if context_head else D_MODEL
            self.reid_embed = nn.Sequential(nn.Linear(d_in, D_MODEL), nn.ReLU(inplace=True), nn.Linear(D_MODEL, D_MODEL),
                                            nn.ReLU(inplace=True), nn.Linear(D_MODEL, D_MODEL))
            for layer in self.reid_embed:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=1)
                    nn.init.zeros_(layer.bias)
        self.nhead = nhead

    def _predict(self, q: torch.Tensor, mask_feat: torch.Tensor, next_size) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """クエリから、クラスの値・マスクの値と、次の層の注意のマスクを作る。

        Args:
            q: クエリ (B(画像の数), Q(クエリの数), C(特徴の次元))。
            mask_feat: マスクを作る特徴 (B, C, S/4, S/4)(S = 入力の一辺)。
            next_size: 次の層が注意を向ける特徴の (H, W)。

        Returns:
            (クラスの値 (B, Q, K+1(クラスの数 + 物体なし))、マスクの値 (B, Q, S/4, S/4)、
             次の層の注意のマスク (B × h(注意の頭の数), Q, N(= H × W)) の bool(True = 見ない))。
        """
        q = self.decoder_norm(q)
        logits = self.class_embed(q)  # (B, Q, C) -> (B, Q, K+1)
        # クエリの埋め込みと、各画素の特徴の内積がマスクの値: (B, Q, C) × (B, C, S/4, S/4) -> (B, Q, S/4, S/4)
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_embed(q), mask_feat)
        attn = F.interpolate(masks, size=next_size, mode="bilinear", align_corners=False)  # -> (B, Q, H, W)
        attn = rearrange(attn.sigmoid() < 0.5, "b q h w -> b q (h w)")  # (B, Q, H, W) -> (B, Q, N)。True = 予測マスクの外(見ない)
        attn[attn.all(-1)] = False  # 全部が外のクエリは、全体を見せる(論文と同じ)
        # 注意の頭ごとに同じマスクを使う(MultiheadAttention は画像と頭をまとめた形を求める): (B, Q, N) -> (B × h, Q, N)
        attn = repeat(attn, "b q n -> (b h) q n", h=self.nhead).detach()
        return logits, masks, attn

    def forward(self, images: torch.Tensor) -> list[dict]:
        """画像から、全層の予測を出す。

        Args:
            images: 入力の画像 (B(画像の数), 3(RGB), S(一辺の画素数), S)。

        Returns:
            予測の並び(最初のクエリ + 各層。最後の要素が最終の予測)。各要素は
            {"logits": (B, Q(クエリの数), K+1(クラスの数 + 物体なし)), "masks": (B, Q, S/4, S/4)}。
        """
        mask_feat, multi = self.pixel_decoder(self.backbone(images))
        if self.mask_pos is not None:  # (B, C, S/4, S/4) に、sine の位置の埋め込みを 1×1 の畳み込みに通して足す
            mask_feat = mask_feat + self.mask_pos(sine_pos(mask_feat))
        mems, poss, sizes = [], [], []
        for lvl, f in enumerate(multi):  # stride 32, 16, 8
            sizes.append(f.shape[-2:])
            poss.append(rearrange(sine_pos(f), "b c h w -> b (h w) c"))  # (B, C, H, W) -> (B, N, C)
            mems.append(rearrange(f, "b c h w -> b (h w) c") + self.level_embed.weight[lvl])  # (B, C, H, W) -> (B, N, C)
        b = images.shape[0]
        q = repeat(self.query_feat.weight, "q c -> b q c", b=b)  # (Q, C) -> (B, Q, C)
        qpos = repeat(self.query_pos.weight, "q c -> b q c", b=b)  # (Q, C) -> (B, Q, C)
        logits, masks, attn = self._predict(q, mask_feat, sizes[0])
        outs = [{"logits": logits, "masks": masks}]
        for i, layer in enumerate(self.layers):
            lvl = i % 3
            q = layer(q, qpos, mems[lvl], poss[lvl], attn)
            logits, masks, attn = self._predict(q, mask_feat, sizes[(i + 1) % 3])
            outs.append({"logits": logits, "masks": masks})
        outs[-1]["embeds"] = self.decoder_norm(q)  # 最後の層のクエリの埋め込み (B, Q, C)(MinVIS のフレーム間の照合・DVIS の入力)
        outs[-1]["embeds_raw"] = q  # 同じクエリの LayerNorm の前の値 (B, Q, C)(DVIS++ の追跡の部のキー・値)
        outs[-1]["mask_feat"] = mask_feat  # マスクを作る特徴 (B, C, S/4, S/4)(DVIS の追跡の部がマスクを作るのに使う)
        if self.context_filter_size is not None:  # CAVIS: reid の埋め込みは [クエリの埋め込み; 文脈の特徴] から作る
            outs[-1]["reid"] = self.reid_embed(torch.cat([outs[-1]["embeds"], self._context(masks, mask_feat)], -1))
            # フレーム間の照合の埋め込み (B, Q, 2C)(公式の pred_embds = [LayerNorm の後のクエリ; reid の埋め込み])
            outs[-1]["embeds_ctx"] = torch.cat([outs[-1]["embeds"], outs[-1]["reid"]], -1)
        elif self.reid_embed is not None:  # reid の埋め込み (B, Q, C)(CTVIS の対照学習とフレーム間の照合)
            outs[-1]["reid"] = self.reid_embed(outs[-1]["embeds"])
        return outs

    def _context(self, masks: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
        """CAVIS の文脈の特徴(公式の get_context_queries、論文の式 (6))。

        1. 最後の層の予測マスク(> 0、勾配を切る)の境界の画素: 3×3 のラプラシアン([-1 … 8 … -1]、外は 0 で埋める)が正の画素
           (マスクの中で、周りの 8 画素のどれかがマスクの外か画像の外)
        2. マスクを作る特徴に k × k の平均の窓をかけ、境界の画素で平均する(境界がなければ 0)
        公式の窓は畳み込みの重み (C, C, k, k) を全部 1/k² にしたもので、各出力の次元が「全部の入力の次元の和」の k × k の平均になる
        (出力の C 次元は全部同じ値)。ここでは同じ値を、次元の和に k × k の平均(外は 0 として数える)をかけて計算する(同じ計算で速い)。
        Args: masks (B, Q, h, w)、mask_feat (B, C, h, w)。Returns: (B, Q, C)。
        """
        k = self.context_filter_size
        m = (masks.detach() > 0).to(mask_feat.dtype)
        lap = m.new_tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view(1, 1, 3, 3)
        bdy = (F.conv2d(rearrange(m, "b q h w -> (b q) 1 h w"), lap, padding=1) > 0).to(mask_feat.dtype)
        bdy = rearrange(bdy, "(b q) 1 h w -> b q (h w)", b=m.shape[0])  # (B, Q, h·w) 境界の画素
        x = F.avg_pool2d(mask_feat.sum(1, keepdim=True), k, stride=1, padding=k // 2, count_include_pad=True)  # (B, 1, h, w)
        ctx = (bdy * rearrange(x, "b 1 h w -> b 1 (h w)")).sum(-1) / (bdy.sum(-1) + 1e-6)  # (B, Q)
        return ctx[..., None].expand(-1, -1, mask_feat.shape[1])  # (B, Q) -> (B, Q, C)(全部の次元が同じ値。公式と同じ)
