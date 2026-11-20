"""GTR-Seg(arXiv 2609.26590、公式のコード github.com/Intellindust-AI-Lab/GTR、MIT)の方式の自作のモデル。

論文と公式のコード(engine/gtr/hybrid_encoder.py の GTREncoder、det/decoder.py の GTRTransformer、det/denoising.py、
det/segmentation_head.py、configs/seg)を参考に、仕組みと設定の値を読み取って自分で設計して書いたもの(公式の関数を
1 つずつ写してはいない。同じ計算になることはテストで確かめる)。backbone は公式のゲート付き線形注意の ViT ではなく、
比較の条件に合わせて timm の CNN(R50)にする。

名前の注記: GTR の論文の中心は backbone(ゲート付き線形注意の ViT)で、このモデルはそれを含まない。使っているのは backbone より
後ろ(D-FINE・DEIM 系の decoder と損失、GTR の論文の組み合わせと設定)だけで、encoder="hybrid"(exp00103 の A 以降の既定)では
encoder も RT-DETR の hybrid encoder になる。実態は「RT-DETR / D-FINE 系のインスタンスセグメンテーション」に近い。名前は
これまでの結果との対応のためそのまま使う。

構成:
    backbone(timm、stride 8・16・32 の段)
    → encoder(GTREncoder): 出力の stride 8・16・32 のそれぞれで、backbone の全部の段をその stride に拡大縮小してつなげ、
      C2f + LayerNorm で 256 チャンネルにする(公式は 1 段の ViT の 3 つの層を 2 倍・1 倍・0.5 倍にする。R50 は段ごとに
      stride が違うので、補間で合わせる)
    → decoder(D-FINE 系、4 層):
      - two-stage: 3 段の全部の位置を候補の枠(中心 = 画素の中心、幅・高さ = 0.05 × 2^段)にし、クラスの値の最大が上位 300 個を
        クエリにする。学習では 3 組(Group DETR。組ごとに候補を選ぶ head が別)
      - denoising: 正解の枠とクラスにノイズを入れたクエリ(最大 100 個)を足して、元に戻すことを学習させる(学習だけ)
      - 層: self-attention(組ごと) → 多重解像度の deformable な cross-attention(8 ヘッド、段ごとに 3・6・3 点) → Gate
        (残差の代わりに、2 つの入力をゲートで混ぜる) → FFN
      - 枠: 最初の層で枠を出し(pre)、以後は 4 辺までの距離を 33 個の値の分布で表して層ごとに直していく(FDR)。
        クラスの値に、分布の尖り方から求めた枠の確からしさを足す(LQE)
    → マスク: RF-DETR-Seg と同じ head(encoder の stride 8 の特徴を入力の 1/4 に拡大し、層ごとのクエリとの内積)

形の記号: B = 画像の数、C = チャンネル(256)、N = 3 段の全部の位置の数、Q = 1 組のクエリの数(300)、G = 組の数、
         D = denoising のクエリの数、M = decoder に入れるクエリの数(D + G·Q)、K = クラスの数、R = 33(分布の値の数)、
         S = 入力の一辺
"""
import copy
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torchvision.ops import box_convert

from segmentation.instance_segmentation.lib_common.backbones import create_feature_backbone
from segmentation.instance_segmentation.lib_common.models.hybrid_encoder import HybridEncoder
from segmentation.instance_segmentation.lib_common.models.rfdetr_seg import C2f, ChannelLayerNorm, SegmentationHead

D_MODEL = 256
REG_MAX = 32  # 分布の区切りの数(値は REG_MAX + 1 個)
REG_SCALE, UP = 4.0, 0.5
STRIDES = (8, 16, 32)
NUM_POINTS = (3, 6, 3)  # cross-attention で段ごとに見る点の数


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """sigmoid の逆(0〜1 の値をロジットにする)。

    Args:
        x: 0〜1 の値(任意の形)。
        eps: 0 と 1 の近くで発散しないための下限。

    Returns:
        x と同じ形のロジット。
    """
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def distance_bins(reg_max: int = REG_MAX, up: float = UP, reg_scale: float = REG_SCALE, device=None) -> torch.Tensor:
    """分布の reg_max + 1 個の値が表す「辺のずれ」(D-FINE の W(n))。

    0 を中心に左右対称で、中心から k 番目は a^k − 1(a は等比の比。0 の近くは細かく、遠くほど粗い)。
    内側の端(中心から reg_max/2 − 1 番目)が up × reg_scale になるように a を決め、両端だけは 2 × up × reg_scale にする
    (大きくずれた枠も表せるように)。

    Args:
        reg_max: 分布の区切りの数(値は reg_max + 1 個)。
        up: 内側の端の大きさを決める倍率。
        reg_scale: ずれの単位(参照の枠の幅・高さ / reg_scale)。
        device: 置く装置。

    Returns:
        (reg_max + 1,) の値(小さい順)。
    """
    edge = up * reg_scale
    k = torch.arange(1, reg_max // 2, device=device, dtype=torch.float32)  # 1, ..., reg_max/2 − 1
    inner = (edge + 1) ** (2 * k / (reg_max - 2)) - 1  # a = (edge + 1)^(2 / (reg_max − 2))
    outer = torch.full((1,), 2 * edge, device=device)
    return torch.cat([-outer, -inner.flip(0), inner.new_zeros(1), inner, outer])


def distance2bbox(ref: torch.Tensor, distance: torch.Tensor, reg_scale: float = REG_SCALE) -> torch.Tensor:
    """参照の枠と 4 辺のずれから枠を作る。

    辺の位置 = 参照の中心 ∓ (reg_scale / 2 + ずれ) × 参照の幅・高さ / reg_scale(ずれ 0 なら参照の枠そのもの)。

    Args:
        ref: 参照の枠 (..., 4) cx, cy, w, h。
        distance: (..., 4) 左・上・右・下のずれ。

    Returns:
        (..., 4) cx, cy, w, h。
    """
    center, unit = ref[..., :2], ref[..., 2:] / reg_scale  # (..., 2)
    lt = center - (reg_scale / 2 + distance[..., :2]) * unit
    rb = center + (reg_scale / 2 + distance[..., 2:]) * unit
    return box_convert(torch.cat([lt, rb], -1), "xyxy", "cxcywh")


class MLP(nn.Module):
    """全結合の層を重ねたもの(最後の層の後は活性化なし)。

    Args:
        din, dhidden, dout: 入力・途中・出力の次元。
        layers: 層の数。
        act: 活性化("silu" / "relu")。
    """

    def __init__(self, din: int, dhidden: int, dout: int, layers: int, act: str = "silu"):
        super().__init__()
        dims = [din] + [dhidden] * (layers - 1) + [dout]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = nn.SiLU() if act == "silu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """全結合の層を順に通す。

        Args:
            x: (..., din)。

        Returns:
            (..., dout)。
        """
        for i, layer in enumerate(self.layers):
            x = layer(x) if i == len(self.layers) - 1 else self.act(layer(x))
        return x


class Encoder(nn.Module):
    """GTREncoder: 出力の段ごとに、backbone の全部の段をその stride に合わせてつなげ、C2f + LayerNorm で 256 チャンネルにする。

    Args:
        cins: backbone の段のチャンネル。
        n: C2f の Bottleneck の数(公式の num_blocks)。
        dim: 出力の特徴の次元(既定は D_MODEL)。
    """

    def __init__(self, cins: list[int], n: int = 3, dim: int = D_MODEL):
        super().__init__()
        self.stages = nn.ModuleList(nn.Sequential(C2f(sum(cins), dim, n=n), ChannelLayerNorm(dim)) for _ in STRIDES)

    def forward(self, feats: list[torch.Tensor], image_size: tuple[int, int]) -> list[torch.Tensor]:
        """3 段の特徴を作る。

        Args:
            feats: backbone の段の特徴 [(B, C_i, S/s_i, S/s_i)]。
            image_size: 入力の (縦, 横)。

        Returns:
            [(B, C, S/8, S/8), (B, C, S/16, S/16), (B, C, S/32, S/32)]。
        """
        outs = []
        for stage, s in zip(self.stages, STRIDES):
            size = (image_size[0] // s, image_size[1] // s)
            x = [f if f.shape[-2:] == size else
                 F.interpolate(f, size=size, mode="bilinear", align_corners=False, antialias=f.shape[-1] > size[1])
                 for f in feats]  # 段ごとに (B, C_i, h_i, w_i) -> (B, C_i, h, w)
            outs.append(stage(torch.cat(x, dim=1)))  # (B, ΣC_i, h, w) -> (B, C, h, w)
        return outs


class MSDeformableAttention(nn.Module):
    """多重解像度の deformable な注意(D-FINE の版。値と出力の射影はなく、段ごとに点の数が違う)。

    クエリごとに、参照の枠の中心から(枠の大きさに比例した)ずれの位置の特徴を、段ごとに決まった数の点で双線形に取り、
    学習する重みで足す。

    Args:
        heads: 頭の数。
        points: 段ごとの点の数。
        offset_scale: ずれの大きさの倍率。
        dim: クエリの次元(既定は D_MODEL)。
    """

    def __init__(self, heads: int = 8, points: tuple[int, ...] = NUM_POINTS, offset_scale: float = 0.5,
                 dim: int = D_MODEL):
        super().__init__()
        self.heads, self.points, self.offset_scale = heads, list(points), offset_scale
        total = sum(points)
        self.register_buffer("point_scale", torch.tensor([1 / n for n in points for _ in range(n)]))  # (P,)
        self.sampling_offsets = nn.Linear(dim, heads * total * 2)
        self.attention_weights = nn.Linear(dim, heads * total)
        # 最初の点の置き方(Deformable DETR 系の考え方): 頭ごとに等しい角度で向きを分け、その向きの正方形の辺の上
        # (大きい方の成分が 1)に、段の中の i 番目の点を i 倍の遠さで置く。ずれは問い合わせによらない(重み 0)。
        nn.init.zeros_(self.sampling_offsets.weight)
        angle = torch.linspace(0, 2 * math.pi, heads + 1)[:-1]  # (heads,)
        direction = torch.stack([angle.cos(), angle.sin()], -1)
        direction = direction / direction.abs().amax(-1, keepdim=True)  # (heads, 2) 正方形の辺の上
        rank = torch.cat([torch.arange(1, n + 1) for n in points]).float()  # (P,) 段の中の何番目か
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(rearrange(torch.einsum("hx,p->hpx", direction, rank), "h p x -> (h p x)"))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

    def forward(self, query: torch.Tensor, ref: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
        """注意の出力を作る。

        Args:
            query: (B, M, C)。
            ref: 参照の枠 (B, M, 4) cx, cy, w, h(0〜1)。
            values: 段ごとの特徴 [(B·heads, C/heads, h_l, w_l)]。

        Returns:
            (B, M, C)。
        """
        b, m, _ = query.shape
        p = sum(self.points)
        off = rearrange(self.sampling_offsets(query), "b m (h p two) -> b m h p two", h=self.heads, p=p)
        attn = rearrange(self.attention_weights(query), "b m (h p) -> b m h p", h=self.heads).softmax(-1)
        loc = ref[:, :, None, None, :2] + off * self.point_scale[:, None] * ref[:, :, None, None, 2:] * self.offset_scale
        grid = rearrange(2 * loc - 1, "b m h p two -> (b h) m p two")  # grid_sample の座標(−1〜1)
        sampled = [F.grid_sample(v, g.to(v.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
                   for v, g in zip(values, grid.split(self.points, dim=2))]  # 段ごとに (B·h, c, M, P_l)
        sampled = torch.cat(sampled, dim=-1)  # (B·h, c, M, P)
        w = rearrange(attn, "b m h p -> (b h) 1 m p")
        out = (sampled * w.to(sampled.dtype)).sum(-1)  # (B·h, c, M)
        return rearrange(out, "(b h) c m -> b m (h c)", b=b)


class Gate(nn.Module):
    """2 つの入力をゲート(sigmoid)で混ぜて LayerNorm する(D-FINE の残差の代わり)。最初は 0.5 ずつ。

    Args:
        dim: 入力の次元(既定は D_MODEL)。
    """

    def __init__(self, dim: int = D_MODEL):
        super().__init__()
        self.gate = nn.Linear(2 * dim, 2 * dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)  # sigmoid(0) = 0.5(公式の bias_init_with_prob(0.5))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """2 つの入力をゲートで混ぜる。

        Args:
            x1: 1 つ目の入力(残差の側)(B(画像の数), M(クエリの数), C(特徴の次元))。
            x2: 2 つ目の入力(注意の出力の側)(B, M, C)。

        Returns:
            (B, M, C)。
        """
        g1, g2 = self.gate(torch.cat([x1, x2], -1)).sigmoid().chunk(2, dim=-1)
        return self.norm(g1 * x1 + g2 * x2)


class DecoderLayer(nn.Module):
    """decoder の 1 層。self-attention は、学習では denoising のクエリどうしと、検出のクエリの組ごとに分けて行う。

    Args:
        groups: 学習のときのクエリの組の数 G。
        ffn: FFN の途中の次元。
        heads: 頭の数。
        dim: クエリの次元(既定は D_MODEL)。
    """

    def __init__(self, groups: int, ffn: int = 1024, heads: int = 8, dim: int = D_MODEL):
        super().__init__()
        self.groups = groups
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = MSDeformableAttention(heads, dim=dim)
        self.gateway = Gate(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.SiLU(), nn.Linear(ffn, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tgt: torch.Tensor, ref: torch.Tensor, values: list[torch.Tensor], pos: torch.Tensor,
                num_dn: int, dn_mask: torch.Tensor | None) -> torch.Tensor:
        """1 層を通す。

        Args:
            tgt: クエリ (B, M, C)。先頭の num_dn 個が denoising のクエリ。
            ref: 参照の枠 (B, M, 4)。
            values: 段ごとの特徴(MSDeformableAttention.forward の values)。
            pos: クエリの位置の埋め込み (B, M, C)。
            num_dn: denoising のクエリの数 D。
            dn_mask: denoising のクエリどうしの注意の禁止 (D, D)(True = 見ない)。

        Returns:
            (B, M, C)。
        """
        b = tgt.shape[0]
        q = tgt + pos
        g = self.groups if self.training else 1
        nw = not getattr(self, "sdpa", False)  # sdpa なら注意の重みを返さない(速い scaled_dot_product_attention を使う。計算は同じ)
        if self.training and g > 1:
            outs = []
            if num_dn:
                outs.append(self.self_attn(q[:, :num_dn], q[:, :num_dn], tgt[:, :num_dn], attn_mask=dn_mask,
                                           need_weights=nw)[0])
            det = lambda x: rearrange(x[:, num_dn:], "b (g q) c -> (g b) q c", g=g)  # 組を画像の向きに並べる
            o = self.self_attn(det(q), det(q), det(tgt), need_weights=nw)[0]
            outs.append(rearrange(o, "(g b) q c -> b (g q) c", g=g))
            sa = torch.cat(outs, dim=1)
        else:
            sa = self.self_attn(q, q, tgt, need_weights=nw)[0]
        tgt = self.norm1(tgt + sa)
        tgt = self.gateway(tgt, self.cross_attn(tgt + pos, ref, values))
        return self.norm2((tgt + self.ffn(tgt)).clamp(-65504, 65504))


class LQE(nn.Module):
    """分布の上位 4 つの確率とその平均から、枠の確からしさを求めてクラスの値に足す(最初は 0 を足す)。"""

    def __init__(self, k: int = 4, hidden: int = 64):
        super().__init__()
        self.k = k
        self.reg_conf = MLP(4 * (k + 1), hidden, 1, 2, act="relu")
        nn.init.zeros_(self.reg_conf.layers[-1].weight)
        nn.init.zeros_(self.reg_conf.layers[-1].bias)

    def forward(self, scores: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        """分布の尖り方から求めた枠の確からしさを、クラスの値に足す。

        Args:
            scores: クラスの値 (B(画像の数), M(クエリの数), K(クラスの数))。
            corners: 4 辺のずれの分布の値 (B, M, 4·R)(R = 分布の値の数)。

        Returns:
            (B, M, K)。
        """
        prob = rearrange(corners, "b m (four r) -> b m four r", four=4).softmax(-1)
        top = prob.topk(self.k, dim=-1).values  # (B, M, 4, k)
        stat = rearrange(torch.cat([top, top.mean(-1, keepdim=True)], -1), "b m four k -> b m (four k)")
        return scores + self.reg_conf(stat)


def denoising_queries(targets: dict, num_classes: int, class_embed: nn.Embedding, groups: int, num_dn: int = 100,
                      label_noise: float = 0.5, box_noise: float = 1.0) -> tuple | None:
    """denoising のクエリ(DN-DETR・DINO の contrastive denoising の考え方。数の決め方は公式の GTR と同じ)。

    1 まとまり = 画像ごとの正解の枠に、小さいずれ(各辺を幅・高さの半分 × 0〜1 倍)を入れた正例 T 個と、大きいずれ(1〜2 倍)を
    入れた負例 T 個(T = 1 枚の画像の正解の数の最大。足りない分は埋め)。クラスは label_noise / 2 の確率でランダムに替える。
    まとまりを num_dn // T 個 × 組の数だけ並べ、クエリは自分のまとまりの中だけを見る(ほかのまとまりの答えを見ないように)。

    Args:
        targets: {"labels": (T_all,), "boxes": (T_all, 4) cx, cy, w, h, "counts": 画像ごとの物体の数}。
        num_classes: クラスの数 K(K 番目を「なし」の埋め込みにする)。
        class_embed: クラスの埋め込み (K + 1, C)。
        groups: クエリの組の数 G。
        num_dn: denoising のクエリの数の目安。
        label_noise: クラスを替える確率の 2 倍。
        box_noise: 枠のずれの大きさの倍率。

    Returns:
        (クエリの内容 (B, D, C), 枠のロジット (B, D, 4), 見てはいけない組み合わせ (D, D) の bool, meta)。正解がなければ None。
        meta = {"num_dn": D, "num_group": 1 組あたりのまとまりの数, "positive": 画像ごとの正例のクエリの番号
        (まとまりの順、まとまりの中は正解の順)}。
    """
    counts = targets["counts"]
    t = max(counts, default=0)
    if t == 0:
        return None
    b, device = len(counts), targets["labels"].device
    chunks = max(num_dn // t, 1) * groups  # まとまりの数(全部の組)
    # 画像ごとの正解を (B, T) に詰める(足りない所は「なし」のクラスと 0 の枠)
    slot = torch.cat([torch.arange(n, device=device) for n in counts])  # (T_all,) 画像の中での番号
    owner = torch.repeat_interleave(torch.arange(b, device=device), torch.tensor(counts, device=device))
    cls = torch.full((b, t), num_classes, dtype=torch.long, device=device)
    box = torch.zeros(b, t, 4, device=device)
    cls[owner, slot], box[owner, slot] = targets["labels"], targets["boxes"]
    real = cls < num_classes  # (B, T) 埋めた所ではない
    # (B, まとまり, 正例・負例, T) の形でずれを入れる
    shape = (b, chunks, 2, t)
    cls = repeat(cls, "b t -> b c two t", c=chunks, two=2).clone()
    swap = (torch.rand(shape, device=device) < label_noise / 2) & real[:, None, None]
    cls[swap] = torch.randint(0, num_classes, (int(swap.sum()),), device=device)
    scale = torch.tensor([0.0, 1.0], device=device).view(1, 1, 2, 1, 1)  # 正例 0〜1 倍、負例 1〜2 倍
    amount = (torch.rand(*shape, 4, device=device) + scale) * (torch.randint(0, 2, (*shape, 4), device=device) * 2 - 1)
    half_wh = repeat(box[..., 2:] / 2, "b t two -> b 1 1 t (r two)", r=2) * box_noise  # (B, 1, 1, T, 4) 辺ごとの単位
    xyxy = (box_convert(box, "cxcywh", "xyxy")[:, None, None] + amount * half_wh).clamp(0, 1)
    noisy = box_convert(xyxy, "xyxy", "cxcywh").abs()
    # 自分のまとまりの中だけを見る: まとまりの番号が違う組み合わせを禁止
    chunk_id = torch.arange(chunks, device=device).repeat_interleave(2 * t)  # (D,)
    mask = chunk_id[:, None] != chunk_id[None, :]
    positive = [(torch.arange(chunks, device=device)[:, None] * 2 * t + torch.arange(n, device=device)).reshape(-1)
                for n in counts]  # 画像ごとに、まとまり c の正例 j は c·2T + j 番目
    flat = lambda x: rearrange(x, "b c two t ... -> b (c two t) ...")
    return class_embed(flat(cls)), inverse_sigmoid(flat(noisy)), mask, \
        {"num_dn": chunks * 2 * t, "num_group": chunks // groups, "positive": positive}


class GTRSeg(nn.Module):
    """GTR-Seg の方式の自作のモデル。

    Args:
        num_classes: クラスの数 K。
        backbone: timm のモデル名。
        num_queries: 1 組のクエリの数 Q。
        groups: 学習のときの組の数 G。
        dec_layers: decoder の層の数。
        ffn: decoder の FFN の途中の次元(公式の GTR-M は 1024)。
        num_dn: denoising のクエリの数の目安。
        pretrained: backbone に timm の事前学習の重みを読むか。
        encoder: "c2f"(`Encoder`。公式の GTR と同じ)/ "hybrid"(RT-DETR の hybrid encoder。ECInsSeg と同じ。exp00103 の A)。
        grad_checkpoint: backbone とマスクの head のブロックで gradient checkpointing を使うか(途中の値を保存せず、勾配を
            計算するときに計算し直す。計算の結果は同じで、GPU のメモリが減り、少し遅くなる)。
        dim: encoder・decoder・マスクの head の特徴の次元(既定は D_MODEL = 256。公式の GTR-Seg の M は 256、S は 192)。
        enc_expansion: hybrid encoder の CSPBlock の幅の倍率(公式の expansion)。
        enc_depth: encoder のブロックの繰り返しの数(hybrid の RepBlock・c2f の Bottleneck の数。公式の round(3 × depth_mult))。
        enc_ffn: hybrid encoder の AIFI の FFN の中間の次元。
        mask_stride: マスクを作る解像度の、入力に対する縮小の倍率(既定は公式と同じ 4。公式の mask_downsample_ratio)。
        mask_dim: マスクを作る画素の特徴の次元(None なら dim。exp00104 の E2。rfdetr_seg.SegmentationHead)。
        enc_block: hybrid encoder の融合のブロックの中の 3×3 のブロック("rep" / "dw"。exp00104 の E3)。
        sdpa_self_attn: decoder の自己注意で注意の重みを返さず、速い scaled_dot_product_attention を使う(計算は同じ。既定は今まで
            どおり False。exp00105 の精度重視の学習を速くするため)。
    """

    def __init__(self, num_classes: int = 8, backbone: str = "resnet50.a1_in1k", num_queries: int = 300, groups: int = 3,
                 dec_layers: int = 4, ffn: int = 1024, num_dn: int = 100, pretrained: bool = True, encoder: str = "c2f",
                 grad_checkpoint: bool = False, dim: int = D_MODEL, enc_expansion: float = 0.75, enc_depth: int = 3,
                 enc_ffn: int = 1024, mask_stride: int = 4, mask_dim: int | None = None, enc_block: str = "rep",
                 sdpa_self_attn: bool = False):
        super().__init__()
        self.num_classes, self.num_queries, self.groups, self.num_dn = num_classes, num_queries, groups, num_dn
        # stride 8・16・32 の段を選ぶ(段の番号は backbone によって違う。R50 では (2, 3, 4)。exp00105)
        # 入力の大きさに位置の情報が固定された backbone(EfficientFormerV2)は、入力 512 に合わせて作り、位置のバイアスを補間して読む
        self.backbone = create_feature_backbone(backbone, pretrained)
        cins = self.backbone.feature_info.channels()
        # 入力はデータの変換で ImageNet の平均・標準偏差で正規化してある。事前学習の正規化が違う backbone(平均・標準偏差 0.5 など)は、
        # backbone の入口でその正規化に直す(同じなら何もしない。学習の重みには入れない)
        cfg = self.backbone.pretrained_cfg
        mean_in, std_in = torch.tensor((0.485, 0.456, 0.406)), torch.tensor((0.229, 0.224, 0.225))
        mean_bb, std_bb = torch.tensor(cfg.get("mean", mean_in.tolist())), torch.tensor(cfg.get("std", std_in.tolist()))
        self.renorm = None
        if not (torch.allclose(mean_bb, mean_in, atol=1e-3) and torch.allclose(std_bb, std_in, atol=1e-3)):
            scale = std_in / std_bb  # x_bb = (x · std_in + mean_in − mean_bb) / std_bb = x · scale + shift
            self.register_buffer("renorm_scale", scale.view(1, 3, 1, 1), persistent=False)
            self.register_buffer("renorm_shift", ((mean_in - mean_bb) / std_bb).view(1, 3, 1, 1), persistent=False)
            self.renorm = True
        self.encoder = Encoder(cins, enc_depth, dim) if encoder == "c2f" else \
            HybridEncoder(cins, enc_expansion, enc_depth, ffn=enc_ffn, dim=dim, block=enc_block)
        prior = -math.log((1 - 0.01) / 0.01)  # 最初のクラスの確率 0.01
        self.enc_score_head = nn.ModuleList(nn.Linear(dim, num_classes) for _ in range(groups))
        self.enc_bbox_head = nn.ModuleList(MLP(dim, dim, 4, 3) for _ in range(groups))
        for s, bb in zip(self.enc_score_head, self.enc_bbox_head):
            nn.init.xavier_uniform_(s.weight)
            nn.init.constant_(s.bias, prior)
            for layer in bb.layers[:-1]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.zeros_(bb.layers[-1].weight)
            nn.init.zeros_(bb.layers[-1].bias)
        self.denoising_class_embed = nn.Embedding(num_classes + 1, dim, padding_idx=num_classes)
        nn.init.normal_(self.denoising_class_embed.weight[:-1])
        self.query_pos_head = MLP(4, dim, dim, 3)
        for layer in self.query_pos_head.layers:
            nn.init.xavier_uniform_(layer.weight)
        self.pre_bbox_head = MLP(dim, dim, 4, 3)
        nn.init.zeros_(self.pre_bbox_head.layers[-1].weight)
        nn.init.zeros_(self.pre_bbox_head.layers[-1].bias)
        self.layers = nn.ModuleList(DecoderLayer(groups, ffn, dim=dim) for _ in range(dec_layers))
        for layer in self.layers:
            layer.sdpa = sdpa_self_attn
        self.score_head = nn.ModuleList(nn.Linear(dim, num_classes) for _ in range(dec_layers))
        self.bbox_head = nn.ModuleList(MLP(dim, dim, 4 * (REG_MAX + 1), 3) for _ in range(dec_layers))
        for s, bb in zip(self.score_head, self.bbox_head):
            nn.init.constant_(s.bias, prior)
            nn.init.zeros_(bb.layers[-1].weight)
            nn.init.zeros_(bb.layers[-1].bias)
        self.lqe = nn.ModuleList(LQE() for _ in range(dec_layers))
        self.seg_head = SegmentationHead(dec_layers, dim, mask_stride, mask_dim)
        self.seg_head.checkpoint = grad_checkpoint
        self.backbone.set_grad_checkpointing(grad_checkpoint)
        self.register_buffer("project", distance_bins(), persistent=False)  # (R,)

    def _anchors(self, shapes: list[tuple[int, int]], device) -> tuple[torch.Tensor, torch.Tensor]:
        """two-stage の候補の枠: 段ごとの全部の画素の中心に、段が 1 つ粗くなるごとに 2 倍の大きさの正方形の枠を置く。

        Args:
            shapes: 段ごとの (h, w)(stride 8・16・32 の順)。
            device: 置く装置。

        Returns:
            (枠のロジット (1, N, 4)、使えない所は inf, 使える候補か (1, N, 1))。
        """
        out = []
        for lvl, (h, w) in enumerate(shapes):
            ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            xy = (torch.stack([xs, ys], -1).float() + 0.5) / torch.tensor([w, h], device=device)  # (h, w, 2)
            wh = torch.full_like(xy, 0.05 * 2.0 ** lvl)
            out.append(rearrange(torch.cat([xy, wh], -1), "h w four -> (h w) four"))
        anchors = torch.cat(out)[None]  # (1, N, 4)
        valid = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        return torch.where(valid, torch.log(anchors / (1 - anchors)), torch.full_like(anchors, float("inf"))), valid

    def integral(self, corners: torch.Tensor) -> torch.Tensor:
        """4 辺のずれの分布を、ずれの期待値にする。

        Args:
            corners: 分布の値 (..., 4·R)(R = 分布の値の数。softmax の前)。

        Returns:
            (..., 4) 左・上・右・下のずれ。
        """
        prob = rearrange(corners, "... (four r) -> ... four r", four=4).softmax(-1)
        return prob @ self.project.to(prob.dtype)

    def forward(self, images: torch.Tensor, targets: dict | None = None) -> dict:
        """予測する。

        Args:
            images: (B, 3, S, S)。
            targets: 学習のとき、denoising のための正解({"labels", "boxes" (cx, cy, w, h、0〜1), "counts"})。

        Returns:
            {"layers": 層ごとの {"logits": (B, G·Q, K), "boxes": (B, G·Q, 4), "corners": (B, G·Q, 4·R),
                                 "refs": (B, G·Q, 4), "mask_parts"},
             "pre": 最初の層の前の枠の {"logits", "boxes", "mask_parts"}, "enc": two-stage の {"logits", "boxes"},
             "dn_layers"・"dn_pre"・"dn_meta": denoising の同じ形の出力(学習で正解があるとき)}。
            推論では G = 1 で、"layers" は最後の層だけ。
        """
        b = images.shape[0]
        image_size = tuple(images.shape[-2:])
        x = images * self.renorm_scale + self.renorm_shift if self.renorm else images  # 正規化が違う backbone だけ直す
        feats = self.encoder(self.backbone(x), image_size)  # 3 × (B, C, h_l, w_l)
        shapes = [tuple(f.shape[-2:]) for f in feats]
        memory = torch.cat([rearrange(f, "b c h w -> b (h w) c") for f in feats], dim=1)  # (B, N, C)
        anchors, valid = self._anchors(shapes, images.device)
        memory = memory * valid.to(memory.dtype)
        g = self.groups if self.training else 1
        contents, refs, enc_logits, enc_boxes = [], [], [], []
        for i in range(g):  # 組ごとに上位 Q 個の位置をクエリにする
            logits = self.enc_score_head[i](memory)  # (B, N, K)
            idx = logits.max(-1).values.topk(self.num_queries, dim=1).indices  # (B, Q)
            pick = lambda x: torch.gather(x, 1, repeat(idx, "b q -> b q c", c=x.shape[-1]))
            mem = pick(memory)  # (B, Q, C)
            box_unact = self.enc_bbox_head[i](mem) + pick(anchors.expand(b, -1, -1))  # (B, Q, 4)
            enc_logits.append(pick(logits))
            enc_boxes.append(box_unact.sigmoid())
            contents.append(mem.detach())
            refs.append(box_unact.detach())
        tgt, ref_unact = torch.cat(contents, 1), torch.cat(refs, 1)  # (B, G·Q, C), (B, G·Q, 4)
        dn = denoising_queries(targets, self.num_classes, self.denoising_class_embed, g, self.num_dn) \
            if self.training and targets is not None and self.num_dn > 0 else None
        num_dn, dn_mask = 0, None
        if dn is not None:
            tgt, ref_unact = torch.cat([dn[0], tgt], 1), torch.cat([dn[1], ref_unact], 1)
            num_dn, dn_mask = dn[3]["num_dn"], dn[2]
        values = [rearrange(f, "b (h c) y x -> (b h) c y x", h=self.layers[0].cross_attn.heads) for f in feats]
        ref = ref_unact.sigmoid()
        pos = self.query_pos_head(ref).clamp(-10, 10)
        out_detach, corners_prev, hidden, layers = 0, 0, [], []
        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, ref, values, pos, num_dn, dn_mask)
            if i == 0:  # 最初の層の出力で枠を出し、以後の層の距離の基準にする
                pre_boxes = (self.pre_bbox_head(tgt) + inverse_sigmoid(ref)).sigmoid()
                pre_logits = self.score_head[0](tgt)
                ref_init = pre_boxes.detach()
            corners = self.bbox_head[i](tgt + out_detach) + corners_prev  # (B, M, 4·R) 前の層の分布に足す
            boxes = distance2bbox(ref_init, self.integral(corners))
            hidden.append(tgt)  # マスクの head は層ごとにブロックを 1 つずつ通すので、推論でも全部の層を使う
            if self.training or i == len(self.layers) - 1:
                logits = self.lqe[i](self.score_head[i](tgt), corners)
                layers.append({"logits": logits, "boxes": boxes, "corners": corners, "refs": ref_init})
            # 参照の枠は層ごとに直すが、位置の埋め込みは最初の枠のまま(公式の GTR と同じ)
            corners_prev, ref, out_detach = corners, boxes.detach(), tgt.detach()
        parts = self.seg_head.parts(feats[0], hidden, image_size, last_only=not self.training)  # 推論は最後の層の分だけ
        for l, p in zip(layers, parts):
            l["mask_parts"] = p
        split = lambda x: (x[:, :num_dn], x[:, num_dn:])
        out = {"layers": [], "dn_layers": []}
        for l in layers:
            dn_l, det_l = {}, {}
            for k, v in l.items():
                if k == "mask_parts":
                    dq, qq = split(v["query"])
                    dn_l[k], det_l[k] = v | {"query": dq}, v | {"query": qq}
                else:
                    dn_l[k], det_l[k] = split(v)
            out["layers"].append(det_l)
            out["dn_layers"].append(dn_l)
        if self.training:
            p0 = parts[0]
            (dpl, ppl), (dpb, ppb), (dpq, ppq) = split(pre_logits), split(pre_boxes), split(p0["query"])
            out["pre"] = {"logits": ppl, "boxes": ppb, "mask_parts": p0 | {"query": ppq}}
            out["dn_pre"] = {"logits": dpl, "boxes": dpb, "mask_parts": p0 | {"query": dpq}}
            out["enc"] = {"logits": torch.cat(enc_logits, 1), "boxes": torch.cat(enc_boxes, 1)}
            out["dn_meta"] = dn[3] if dn is not None else None
        return out
