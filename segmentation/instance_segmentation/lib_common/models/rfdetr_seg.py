"""RF-DETR-Seg(Robinson ら ICLR 2026、公式のコード roboflow/rf-detr 1.12.0)の方式の自作(スクラッチ実装)。

公式のコード(vendor/rf-detr の models/lwdetr.py・transformer.py・heads/segmentation.py・backbone/projector.py・
ops/modules/ms_deform_attn.py)を読んで、同じ計算になるように一から書いた(公式の速さのための工夫は写していない)。
backbone は timm のモデル(既定は公式との比較と同じ resnet50.a1_in1k)。部品の対応は exp00101 の README の
「自作の RF-DETR-Seg 系の設計」の表。

構成:
    特徴:      backbone の stride 8・16・32 の段を stride 16 にそろえてチャンネルの向きにつなげ、C2f + LayerNorm で 1 段(P4)にする
    two-stage: P4 の各画素を候補の枠にし、組ごとに、クラスの値の最大が上位 Q 個の画素から最初の枠を作る(勾配は止める)
    decoder:   内容は学習する埋め込み、枠は two-stage の枠 + 学習する差分。各層は組ごとの self-attention →
               deformable な cross-attention(枠の中を点で見る)→ FFN。枠(参照点)は層ごとに更新しない(公式の lite_refpoint_refine)
    head:      層ごとにクラス(sigmoid)・枠(最初の枠からの差分)・マスク(画素の特徴 · クエリの特徴 + バイアス)
    学習では Q 個のクエリを G 組(Group DETR)持ち、組ごとに正解を割り当てる。推論は 1 組目だけを使う

形の記号(このファイルで共通):
    B = 画像の数、S = 入力の一辺、C = 特徴の次元(D_MODEL = 256)、h・w = P4 の縦・横(S/16)、N = h × w、
    G = クエリの組の数(学習 13、推論 1)、Q = 1 組のクエリの数(100)、K = クラスの数、
    H = 注意の頭の数、P = deformable な注意の 1 頭あたりの点の数、mh・mw = マスクの縦・横(S/4)
"""
import math

import timm
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

D_MODEL = 256


class ChannelLayerNorm(nn.Module):
    """(B, C, h, w) のチャンネルの向きの LayerNorm(公式の backbone/projector.py の LayerNorm と同じ)。

    Args:
        channels: チャンネルの数 C。
        eps: 0 で割らないための小さい値。
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """正規化する。

        Args:
            x: (B, C, h, w)。

        Returns:
            (B, C, h, w)。
        """
        return rearrange(self.norm(rearrange(x, "b c h w -> b h w c")), "b h w c -> b c h w")


class ConvLNSiLU(nn.Sequential):
    """畳み込み(bias なし)→ チャンネルの LayerNorm → SiLU(公式の ConvX を layer_norm=True・act="silu" で使ったもの)。

    Args:
        cin, cout: 入力・出力のチャンネルの数。
        k: 畳み込みの大きさ。
    """

    def __init__(self, cin: int, cout: int, k: int):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), ChannelLayerNorm(cout), nn.SiLU(inplace=True))


class C2f(nn.Module):
    """YOLOv8 系の C2f のブロック(公式の backbone/projector.py の C2f。shortcut なし)。

    1×1 で 2 つに分け、片方を 3×3 の畳み込み 2 つの組に n 回通し、途中の結果を全部つなげて 1×1 で cout にする。

    Args:
        cin, cout: 入力・出力のチャンネルの数。
        n: 3×3 の組の数(公式は 3)。
    """

    def __init__(self, cin: int, cout: int, n: int = 3):
        super().__init__()
        self.c = cout // 2
        self.cv1 = ConvLNSiLU(cin, 2 * self.c, 1)
        self.m = nn.ModuleList(nn.Sequential(ConvLNSiLU(self.c, self.c, 3), ConvLNSiLU(self.c, self.c, 3)) for _ in range(n))
        self.cv2 = ConvLNSiLU((2 + n) * self.c, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴を変える。

        Args:
            x: (B, cin, h, w)。

        Returns:
            (B, cout, h, w)。
        """
        y = list(self.cv1(x).split(self.c, dim=1))  # 2 つの (B, c, h, w)
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))  # (B, (2+n)·c, h, w) -> (B, cout, h, w)


class Features(nn.Module):
    """backbone(timm)の stride 8・16・32 の段を stride 16 にそろえてつなげ、C2f + LayerNorm で 256 チャンネルの P4 にする。

    公式では backbone(timm の差し替え)が段を stride 16 にそろえ(vendor の timm_backbone.py)、projector(MultiScaleProjector、
    倍率 1 の 1 段)がつなげて C2f に通す。その 2 つをまとめたもの。

    Args:
        backbone: timm のモデル名。
        out_indices: 使う段(features_only の番号)。
        pretrained: timm の事前学習の重みを読むか。
    """

    def __init__(self, backbone: str = "resnet50.a1_in1k", out_indices: tuple[int, ...] = (2, 3, 4), pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.fuse = C2f(sum(self.backbone.feature_info.channels()), D_MODEL, n=3)
        self.norm = ChannelLayerNorm(D_MODEL)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """P4 を作る。

        Args:
            images: (B, 3, S, S)。

        Returns:
            P4 (B, C, h, w)(h = S/16)。
        """
        size = (images.shape[-2] // 16, images.shape[-1] // 16)
        feats = [f if f.shape[-2:] == size else
                 F.interpolate(f, size=size, mode="bilinear", align_corners=False, antialias=f.shape[-1] > size[1])
                 for f in self.backbone(images)]  # 段ごとに (B, C_i, S/s_i, S/s_i) -> (B, C_i, h, w)
        return self.norm(self.fuse(torch.cat(feats, dim=1)))  # (B, ΣC_i, h, w) -> (B, C, h, w)


class MLP(nn.Module):
    """全結合の層を重ねたもの(間は ReLU。公式の MLP と同じ)。

    Args:
        din, dhidden, dout: 入力・中間・出力の次元。
        layers: 層の数。
    """

    def __init__(self, din: int, dhidden: int, dout: int, layers: int):
        super().__init__()
        dims = [din] + [dhidden] * (layers - 1) + [dout]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通す。

        Args:
            x: (..., din)。

        Returns:
            (..., dout)。
        """
        for i, layer in enumerate(self.layers):
            x = layer(x) if i == len(self.layers) - 1 else F.relu(layer(x))
        return x


def sine_embed_boxes(boxes: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """枠 (cx, cy, w, h) の sine の位置の埋め込み(DAB-DETR 系。公式の gen_sineembed_for_position と同じ式・並び)。

    Args:
        boxes: 0〜1 の枠 (B, M, 4)(M = クエリの数)。
        dim: 1 つの座標あたりの次元。

    Returns:
        (B, M, 4 × dim)。並びは y, x, w, h の順(各座標は sin・cos を交互に並べる)。
    """
    dim_t = torch.arange(dim, device=boxes.device, dtype=torch.float32)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)  # (dim,) 周期
    parts = []
    for i in (1, 0, 2, 3):  # y, x, w, h の順
        v = rearrange(boxes[..., i].float() * 2 * math.pi, "b m -> b m 1") / dim_t  # (B, M) -> (B, M, dim)
        v = torch.stack([v[..., 0::2].sin(), v[..., 1::2].cos()], dim=-1)  # (B, M, dim/2, 2)
        parts.append(rearrange(v, "b m d two -> b m (d two)"))  # -> (B, M, dim)
    return torch.cat(parts, dim=-1).to(boxes.dtype)  # (B, M, 4·dim)


class DeformableAttention(nn.Module):
    """1 段の特徴の deformable な注意(公式の MSDeformAttn を n_levels=1 で使ったもの)。

    クエリごと・頭ごとに P 個の点の位置(枠の中心からのずれ)と重みを予測し、特徴をその点で双線形に取って重みで足す。
    点の位置 = 枠の中心 + ずれ / P × 枠の幅・高さ × 0.5。

    Args:
        heads: 頭の数 H。
        points: 1 頭あたりの点の数 P。
    """

    def __init__(self, heads: int = 16, points: int = 2):
        super().__init__()
        self.heads, self.points = heads, points
        self.sampling_offsets = nn.Linear(D_MODEL, heads * points * 2)
        self.attention_weights = nn.Linear(D_MODEL, heads * points)
        self.value_proj = nn.Linear(D_MODEL, D_MODEL)
        self.output_proj = nn.Linear(D_MODEL, D_MODEL)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """公式と同じ初期化: ずれは頭ごとに放射状(i 番目の点は i+1 倍)、重みは 0(最初は点を等しく足す)。"""
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.heads, dtype=torch.float32) * (2 * math.pi / self.heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)  # (H, 2)
        grid = grid / grid.abs().max(-1, keepdim=True)[0]
        grid = repeat(grid, "h xy -> h p xy", p=self.points) * rearrange(torch.arange(1, self.points + 1).float(), "p -> 1 p 1")
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(rearrange(grid, "h p xy -> (h p xy)"))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query: torch.Tensor, boxes: torch.Tensor, value_map: torch.Tensor) -> torch.Tensor:
        """注意を計算する。

        Args:
            query: クエリ(位置の埋め込みを足したもの)(B, M, C)(M = クエリの数)。
            boxes: 参照の枠 (B, M, 4)(0〜1 の cx, cy, w, h)。
            value_map: 注意を向ける特徴 (B, C, h, w)。

        Returns:
            (B, M, C)。
        """
        b, m, _ = query.shape
        value = self.value_proj(rearrange(value_map, "b c h w -> b h w c"))  # (B, h, w, C)
        value = rearrange(value, "b h w (n d) -> (b n) d h w", n=self.heads)  # -> (B·H, C/H, h, w)
        offsets = rearrange(self.sampling_offsets(query), "b m (n p xy) -> b m n p xy", n=self.heads, p=self.points)
        weights = rearrange(self.attention_weights(query), "b m (n p) -> b m n p", n=self.heads).softmax(-1)  # (B, M, H, P)
        center = rearrange(boxes[..., :2], "b m xy -> b m 1 1 xy")
        wh = rearrange(boxes[..., 2:], "b m xy -> b m 1 1 xy")
        loc = center + offsets / self.points * wh * 0.5  # (B, M, H, P, 2) 0〜1 の点の位置
        grid = rearrange(2 * loc - 1, "b m n p xy -> (b n) m p xy")  # grid_sample は −1〜1、(B·H, M, P, 2)
        sampled = F.grid_sample(value, grid.to(value.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)  # (B·H, C/H, M, P)
        w = rearrange(weights, "b m n p -> (b n) 1 m p").to(sampled.dtype)
        out = rearrange((sampled * w).sum(-1), "(b n) d m -> b m (n d)", b=b)  # (B·H, C/H, M) -> (B, M, C)
        return self.output_proj(out)


class DecoderLayer(nn.Module):
    """組ごとの self-attention → deformable な cross-attention → FFN(後置きの LayerNorm。公式の TransformerDecoderLayer)。

    Args:
        groups: 学習のときのクエリの組の数 G(組どうしは self-attention で見ない)。
        sa_heads: self-attention の頭の数。
        ca_heads: cross-attention の頭の数。
        points: cross-attention の 1 頭あたりの点の数。
        ffn: FFN の中間の次元。
    """

    def __init__(self, groups: int, sa_heads: int = 8, ca_heads: int = 16, points: int = 2, ffn: int = 2048):
        super().__init__()
        self.groups = groups
        self.self_attn = nn.MultiheadAttention(D_MODEL, sa_heads, batch_first=True)
        self.cross_attn = DeformableAttention(ca_heads, points)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, D_MODEL))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)

    def forward(self, tgt: torch.Tensor, query_pos: torch.Tensor, boxes: torch.Tensor, value_map: torch.Tensor) -> torch.Tensor:
        """クエリを 1 層分更新する。

        Args:
            tgt: クエリの内容 (B, G·Q, C)(推論では G = 1)。
            query_pos: クエリの位置の埋め込み (B, G·Q, C)。
            boxes: 参照の枠 (B, G·Q, 4)。
            value_map: P4 (B, C, h, w)。

        Returns:
            (B, G·Q, C)。
        """
        g = self.groups if self.training else 1
        q = tgt + query_pos
        # 組ごとに self-attention(組を画像の向きに並べ直す): (B, G·Q, C) -> (G·B, Q, C)
        q_g = rearrange(q, "b (g q) c -> (g b) q c", g=g)
        v_g = rearrange(tgt, "b (g q) c -> (g b) q c", g=g)
        sa = self.self_attn(q_g, q_g, v_g, need_weights=False)[0]
        tgt = self.norm1(tgt + rearrange(sa, "(g b) q c -> b (g q) c", g=g))  # -> (B, G·Q, C)
        tgt = self.norm2(tgt + self.cross_attn(tgt + query_pos, boxes, value_map))
        return self.norm3(tgt + self.ffn(tgt))


class DepthwiseConvBlock(nn.Module):
    """ConvNeXt 型のブロック(MLP なし。公式の heads/segmentation.py の DepthwiseConvBlock): 3×3 の depthwise 畳み込み →
    LayerNorm → Linear → GELU、残差。

    Args:
        dim: チャンネルの数。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通す。

        Args:
            x: (B, C, mh, mw)。

        Returns:
            (B, C, mh, mw)。
        """
        y = rearrange(self.dwconv(x), "b c h w -> b h w c")
        y = F.gelu(self.pwconv(self.norm(y)))
        return x + rearrange(y, "b h w c -> b c h w")


class MLPBlock(nn.Module):
    """LayerNorm → 4 倍に広げる MLP(GELU)、残差(公式の heads/segmentation.py の MLPBlock)。

    Args:
        dim: 次元。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通す。

        Args:
            x: (..., dim)。

        Returns:
            (..., dim)。
        """
        return x + self.mlp(self.norm(x))


class SegmentationHead(nn.Module):
    """マスクの head(公式の heads/segmentation.py の SegmentationHead)。

    P4 を入力の 1/4 に拡大し、decoder の層ごとに DepthwiseConvBlock を 1 つずつ通した特徴(画素の特徴)と、クエリを
    MLPBlock に通した特徴の内積 + バイアス(1 つ)をマスクの値にする。学習では全部のクエリのマスクを作らず、
    画素の特徴とクエリの特徴を返す(割り当てたクエリの分だけ損失でマスクにする。全部作るとメモリに載らないため)。

    Args:
        layers: decoder の層の数(ブロックの数)。
    """

    def __init__(self, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(DepthwiseConvBlock(D_MODEL) for _ in range(layers))
        self.spatial_proj = nn.Conv2d(D_MODEL, D_MODEL, 1)
        self.query_block = MLPBlock(D_MODEL)
        self.query_proj = nn.Linear(D_MODEL, D_MODEL)
        self.bias = nn.Parameter(torch.zeros(1))

    def parts(self, p4: torch.Tensor, queries: list[torch.Tensor], image_size: tuple[int, int],
              skip_blocks: bool = False) -> list[dict]:
        """層ごとのマスクの材料(画素の特徴・クエリの特徴・バイアス)を作る。

        Args:
            p4: (B, C, h, w)。
            queries: 層ごとのクエリ (B, M, C) の並び(M = クエリの数)。
            image_size: 入力の (縦, 横)。
            skip_blocks: True ならブロックを通さない(two-stage の encoder の出力のマスク。queries は 1 つ)。

        Returns:
            層ごとの {"spatial": (B, C, mh, mw), "query": (B, M, C), "bias": (1,)}。
        """
        spatial = F.interpolate(p4, size=(image_size[0] // 4, image_size[1] // 4), mode="bilinear", align_corners=False)
        out = []
        for i, q in enumerate(queries):
            if not skip_blocks:
                spatial = self.blocks[i](spatial)
            out.append({"spatial": self.spatial_proj(spatial), "query": self.query_proj(self.query_block(q)), "bias": self.bias})
        return out


def masks_from_parts(parts: dict) -> torch.Tensor:
    """マスクの材料から、全部のクエリのマスクの値を作る。

    Args:
        parts: SegmentationHead.parts の 1 つ。

    Returns:
        (B, M, mh, mw)。
    """
    return torch.einsum("bchw,bmc->bmhw", parts["spatial"], parts["query"]) + parts["bias"]


def refine_boxes(ref: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """枠を差分で直す(公式の bbox_reparam): 中心 = 差分 × 幅・高さ + 中心、幅・高さ = exp(差分) × 幅・高さ。

    Args:
        ref: 元の枠 (..., 4)(cx, cy, w, h)。
        delta: 差分 (..., 4)。

    Returns:
        直した枠 (..., 4)。
    """
    return torch.cat([delta[..., :2] * ref[..., 2:] + ref[..., :2], delta[..., 2:].exp() * ref[..., 2:]], dim=-1)


class RFDETRSeg(nn.Module):
    """RF-DETR-Seg の方式の自作のモデル。

    Args:
        num_classes: クラスの数 K(物体なしは含めない。クラスは sigmoid)。
        backbone: timm のモデル名。
        num_queries: 1 組のクエリの数 Q。
        groups: 学習のときのクエリの組の数 G(Group DETR)。
        dec_layers: decoder の層の数。
        pretrained: backbone に timm の事前学習の重みを読むか。
    """

    def __init__(self, num_classes: int = 8, backbone: str = "resnet50.a1_in1k", num_queries: int = 100, groups: int = 13,
                 dec_layers: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_queries, self.groups = num_queries, groups
        self.features = Features(backbone, pretrained=pretrained)
        # two-stage: 組ごとの Linear + LayerNorm と、クラス・枠の head(最初は decoder の head と同じ値)
        self.enc_output = nn.ModuleList(nn.Linear(D_MODEL, D_MODEL) for _ in range(groups))
        self.enc_output_norm = nn.ModuleList(nn.LayerNorm(D_MODEL) for _ in range(groups))
        self.query_feat = nn.Embedding(num_queries * groups, D_MODEL)
        self.refpoint_embed = nn.Embedding(num_queries * groups, 4)
        nn.init.zeros_(self.refpoint_embed.weight)
        self.ref_point_head = MLP(2 * D_MODEL, D_MODEL, D_MODEL, 2)
        self.layers = nn.ModuleList(DecoderLayer(groups) for _ in range(dec_layers))
        self.decoder_norm = nn.LayerNorm(D_MODEL)
        self.class_embed = nn.Linear(D_MODEL, num_classes)
        self.bbox_embed = MLP(D_MODEL, D_MODEL, 4, 3)
        prior = 0.01  # 最初の確率が 0.01 になるバイアス(focal 系の損失の慣例。公式と同じ)
        nn.init.constant_(self.class_embed.bias, -math.log((1 - prior) / prior))
        nn.init.zeros_(self.bbox_embed.layers[-1].weight)
        nn.init.zeros_(self.bbox_embed.layers[-1].bias)
        self.enc_class_embed = nn.ModuleList(_copy(self.class_embed) for _ in range(groups))
        self.enc_bbox_embed = nn.ModuleList(_copy(self.bbox_embed) for _ in range(groups))
        self.seg_head = SegmentationHead(dec_layers)

    def _proposals(self, b: int, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """P4 の各画素を候補の枠にする(中心 = 画素の中心、幅・高さ = 0.05。公式の gen_encoder_output_proposals)。

        Args:
            b: 画像の数。
            h, w: P4 の縦・横。
            device: 置く装置。

        Returns:
            (候補の枠 (B, N, 4)、使える候補か (B, N, 1) の bool(0.01〜0.99 の中))。
        """
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        center = torch.stack([(xs + 0.5) / w, (ys + 0.5) / h], dim=-1)  # (h, w, 2)
        props = torch.cat([center, torch.full_like(center, 0.05)], dim=-1)  # (h, w, 4)
        props = repeat(props, "h w four -> b (h w) four", b=b)  # -> (B, N, 4)
        valid = ((props > 0.01) & (props < 0.99)).all(-1, keepdim=True)  # (B, N, 1)
        return props.masked_fill(~valid, 0.0), valid

    def forward(self, images: torch.Tensor) -> dict:
        """予測する。

        Args:
            images: (B, 3, S, S)。

        Returns:
            {"layers": 層ごとの {"logits": (B, G·Q, K), "boxes": (B, G·Q, 4), "mask_parts": SegmentationHead.parts の 1 つ},
             "enc": two-stage の encoder の出力(同じ形)}。G は学習では groups、推論では 1。枠は 0〜1 の cx, cy, w, h。
        """
        b = images.shape[0]
        image_size = (images.shape[-2], images.shape[-1])
        p4 = self.features(images)  # (B, C, h, w)
        memory = rearrange(p4, "b c h w -> b (h w) c")  # (B, N, C)
        props, valid = self._proposals(b, p4.shape[-2], p4.shape[-1], p4.device)  # (B, N, 4), (B, N, 1)
        memory = memory.masked_fill(~valid, 0.0)
        g = self.groups if self.training else 1
        q = self.num_queries
        refs, enc_tgts, enc_logits, enc_boxes = [], [], [], []
        for i in range(g):  # 組ごとに上位 Q 個の画素を選ぶ
            pre = self.enc_output[i](memory)  # (B, N, C)
            logits_all = self.enc_class_embed[i](self.enc_output_norm[i](pre))  # (B, N, K)
            idx = logits_all.max(-1).values.topk(min(q, pre.shape[1]), dim=1).indices  # (B, Q)
            sel = torch.gather(pre, 1, repeat(idx, "b q -> b q c", c=pre.shape[-1]))  # (B, N, C) -> (B, Q, C)
            tgt_i = self.enc_output_norm[i](sel)
            prop_i = torch.gather(props, 1, repeat(idx, "b q -> b q four", four=4))  # (B, Q, 4)
            box_i = refine_boxes(prop_i, self.enc_bbox_embed[i](tgt_i))  # (B, Q, 4)
            refs.append(box_i.detach())
            enc_tgts.append(tgt_i)
            enc_logits.append(self.enc_class_embed[i](tgt_i))
            enc_boxes.append(box_i)
        ref_ts = torch.cat(refs, dim=1)  # (B, G·Q, 4)
        enc_tgt = torch.cat(enc_tgts, dim=1)  # (B, G·Q, C)
        # decoder のクエリ: 内容は学習する埋め込み、枠は two-stage の枠を学習する差分で直したもの
        tgt = repeat(self.query_feat.weight[: g * q], "m c -> b m c", b=b)
        ref = refine_boxes(ref_ts, repeat(self.refpoint_embed.weight[: g * q], "m four -> b m four", b=b))  # (B, G·Q, 4)
        query_pos = self.ref_point_head(sine_embed_boxes(ref, D_MODEL // 2))  # (B, G·Q, 4·128) -> (B, G·Q, C)
        hs = []
        for layer in self.layers:  # 参照の枠は層ごとに更新しない(公式の lite_refpoint_refine)
            tgt = layer(tgt, query_pos, ref, p4)
            hs.append(self.decoder_norm(tgt))
        mask_parts = self.seg_head.parts(p4, hs, image_size)
        layers = [{"logits": self.class_embed(h), "boxes": refine_boxes(ref, self.bbox_embed(h)), "mask_parts": mp}
                  for h, mp in zip(hs, mask_parts)]
        enc = {"logits": torch.cat(enc_logits, dim=1), "boxes": torch.cat(enc_boxes, dim=1),
               "mask_parts": self.seg_head.parts(p4, [enc_tgt], image_size, skip_blocks=True)[0]}
        return {"layers": layers, "enc": enc}


def _copy(module: nn.Module) -> nn.Module:
    """同じ値の重みを持つ、別の重みのモジュールを作る(two-stage の head の最初の値を decoder の head と同じにするため)。

    Args:
        module: 元のモジュール。

    Returns:
        複製。
    """
    import copy
    return copy.deepcopy(module)
