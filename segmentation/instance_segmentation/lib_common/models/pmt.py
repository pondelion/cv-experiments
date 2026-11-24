"""PMT(Plain Mask Transformer、arXiv 2603.25398、CVPR 2026 Workshop)の自作の実装。exp00103 の E。

参考: 論文と公式のコード(tue-mps/pmt、MIT。`image/models/pmt.py`・`pmd.py`・`layers.py`・`utils.py`、
`image/training/lightning_module.py`・`warmup_cosine_schedule.py`、`image/configs/coco/panoptic/pmt_s_640.yaml`)。
コードは写さず、構成と設定の値を読み取って書いた。

構成:
    encoder: 事前学習した ViT(既定は EoMT と同じ ViT-S/16 `deit3_small_patch16_224.fb_in1k`)。固定する(重みを学習しない)。
    横のつながり: ViT の等間隔の層(ViT-S は 2・5・8・11 番目)の出力に ViT の最後の LayerNorm を通し、層ごとに BatchNorm →
        2 層の MLP(次元を半分にして戻す。残差あり)→ チャンネルごとの倍率(1 で初期化)を掛けて足し合わせる。
    decoder(PMD): 初めから学習する素直な transformer の層 6 つ(前置きの LayerNorm、自己注意(鍵の射影は bias なし)、
        LayerScale(1 で初期化)、MLP(GELU))。クエリ 200 個を画像のトークンの前につなげ、まとめて自己注意で処理する。
        位置は画像のトークンにだけ RoPE(DINOv3 の 2 次元の RoPE。学習のときは座標を 1/2〜2 倍にランダムに拡大縮小)で入れる。
    予測と学習中の注意のマスク: EoMT と同じ(各層の前と最後で予測し、層ごとにマスクを使う確率を学習の進み方に合わせて 1 → 0)。

出力の形は自作の Mask2Former・EoMT と同じ({"logits", "masks"} の並び)。

形の記号: B = 画像の数、Q = クエリの数、P = ViT の先頭のトークンの数、h・w = パッチの縦・横の数、N = h × w、C = ViT の次元、
d = 注意の 1 頭の次元、K = クラスの数、S = 入力の一辺。
"""
import math

import timm
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from segmentation.instance_segmentation.lib_common.models.eomt import ScaleBlock

# 公式の pmt_s_640.yaml の注意のマスクを使う確率を下げる区間(全部の更新の回数 73,910 回に対する割合)。
# 1 番目の層は最初の 3,000 回、ほかは 0.32〜0.52・0.44〜0.64・0.56〜0.76・0.68〜0.88・0.80〜1.0
ANNEAL_WINDOWS = [(0.0, 3000 / 73910), (23651 / 73910, 38433 / 73910), (32520 / 73910, 47302 / 73910),
                  (41389 / 73910, 56171 / 73910), (50258 / 73910, 65040 / 73910), (59128 / 73910, 1.0)]


def rope_cos_sin(h: int, w: int, head_dim: int, device, training: bool, theta: float = 100.0
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """DINOv3 の 2 次元の RoPE の cos・sin(パッチの中心の座標を -1〜1 にし、縦・横それぞれに head_dim / 4 個の周波数)。

    学習のときは、座標全体を 1/2〜2 倍(対数で一様)にランダムに拡大縮小する(公式の rescale=2.0)。

    Args:
        h: パッチの縦の数。
        w: パッチの横の数。
        head_dim: 注意の 1 頭の次元 d。
        device: 作る場所。
        training: 学習のときか。
        theta: 周波数の基準(公式は 100)。

    Returns:
        (cos, sin)。どちらも (N, d)。
    """
    inv_freq = 1 / theta ** torch.arange(0, 1, 4 / head_dim, device=device, dtype=torch.float32)  # (d/4,)
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1  # (h,) 中心の縦の座標
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1  # (w,)
    coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # (h, w, 2) の (縦, 横)
    coords = rearrange(coords, "h w two -> (h w) two")  # (N, 2)
    if training:
        coords = coords * torch.empty(1, device=device).uniform_(-math.log(2.0), math.log(2.0)).exp()
    angles = rearrange(2 * math.pi * coords[:, :, None] * inv_freq, "n two f -> n (two f)")  # (N, d/2)
    angles = torch.cat([angles, angles], dim=-1)  # (N, d)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """後ろの次元を前半・後半に分け、(−後半, 前半) にする(RoPE の回転)。

    Args:
        x: (..., d)。

    Returns:
        (..., d)。
    """
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class DecoderLayer(nn.Module):
    """PMD の 1 層(DINOv3 の ViT の層と同じ形): 前置きの LayerNorm → 自己注意 → LayerScale の残差 → LayerNorm → MLP → LayerScale の残差。

    Args:
        dim: 次元 C。
        heads: 注意の頭の数。
        ffn: MLP の中間の次元。
    """

    def __init__(self, dim: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ls1, self.ls2 = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.ones(dim))
        self.mlp = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """1 層を通す。RoPE は最後の N 個のトークン(画像のトークン)の問いと鍵にだけ掛ける。

        Args:
            x: (B, T, C)(T = Q + P + N)。
            mask: 注意のマスク (B, 1, T, T)(True = 見る)か None。
            cos: RoPE の cos (N, d)。
            sin: RoPE の sin (N, d)。

        Returns:
            (B, T, C)。
        """
        y = self.norm1(x)
        q, k, v = [rearrange(f(y), "b t (h d) -> b h t d", h=self.heads) for f in (self.q, self.k, self.v)]
        n = cos.shape[0]
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = torch.cat([q[:, :, :-n], q[:, :, -n:] * cos + rotate_half(q[:, :, -n:]) * sin], dim=2)
        k = torch.cat([k[:, :, :-n], k[:, :, -n:] * cos + rotate_half(k[:, :, -n:]) * sin], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, 頭の数, T, d)
        x = x + self.ls1 * self.o(rearrange(y, "b h t d -> b t (h d)"))
        return x + self.ls2 * self.mlp(self.norm2(x))


class PMT(nn.Module):
    """PMT の自作のモデル。

    Args:
        num_classes: クラスの数 K(物体なしを含めない)。
        backbone: timm の ViT の名前(固定して使う)。
        num_queries: クエリの数 Q(公式は 200)。
        num_blocks: decoder の層の数(公式は 6)。
        layers: 横のつながりに使う ViT の層の番号(公式の ViT-S は 2・5・8・11)。
        ffn: decoder の MLP の中間の次元(公式の ViT-S は 384)。
        img_size: 入力の一辺。
        pretrained: timm の事前学習の重みを読むか。
    """

    def __init__(self, num_classes: int = 8, backbone: str = "deit3_small_patch16_224.fb_in1k", num_queries: int = 200,
                 num_blocks: int = 6, layers: tuple[int, ...] = (2, 5, 8, 11), ffn: int = 384, img_size: int = 512,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.backbone.requires_grad_(False)  # encoder は固定する
        vit = self.backbone
        dim, patch, heads = vit.embed_dim, vit.patch_embed.patch_size[0], vit.blocks[0].attn.num_heads
        self.num_queries, self.prefix, self.layers, self.heads = num_queries, vit.num_prefix_tokens, list(layers), heads
        self.lateral_bn = nn.ModuleList(nn.BatchNorm1d(dim) for _ in layers)
        self.lateral = nn.ModuleList(nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim))
                                     for _ in layers)
        self.scalers = nn.ParameterList(nn.Parameter(torch.ones(dim)) for _ in layers)
        self.query = nn.Embedding(num_queries, dim)
        self.blocks = nn.ModuleList(DecoderLayer(dim, heads, ffn) for _ in range(num_blocks))
        self.decoder_norm = nn.LayerNorm(dim)
        self.upscale = nn.Sequential(*[ScaleBlock(dim) for _ in range(max(1, int(math.log2(patch)) - 2))])  # 16 → 4
        self.mask_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.class_head = nn.Linear(dim, num_classes + 1)
        self.windows = ANNEAL_WINDOWS if num_blocks == len(ANNEAL_WINDOWS) else \
            [(i / num_blocks, (i + 1) / num_blocks) for i in range(num_blocks)]
        self.register_buffer("mask_probs", torch.ones(num_blocks))

    def train(self, mode: bool = True):
        """学習・評価の切り替え。固定した encoder は常に評価のときの動きにする。

        Args:
            mode: True で学習の状態。

        Returns:
            self。
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def set_progress(self, progress: float) -> None:
        """学習の進み方(0〜1)から、層ごとの注意のマスクを使う確率を決める(区間の中では (1 − 進み)^0.9 で下げる)。

        Args:
            progress: 今までの更新の回数 / 全部の回数。
        """
        for i, (s, e) in enumerate(self.windows):
            t = min(max((progress - s) / (e - s), 0.0), 1.0)
            self.mask_probs[i] = (1.0 - t) ** 0.9

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        """固定した ViT の指定の層の出力を、横のつながりで 1 つにまとめる。

        Args:
            images: (B, 3, S, S)。

        Returns:
            (B, P + N, C)。
        """
        vit = self.backbone
        with torch.no_grad():
            x = vit.norm_pre(vit._pos_embed(vit.patch_embed(images)))
            feats = []
            for i, block in enumerate(vit.blocks):
                x = block(x)
                if i in self.layers:
                    feats.append(vit.norm(x))  # (B, P + N, C)
        out = 0
        for f, bn, lat, sc in zip(feats, self.lateral_bn, self.lateral, self.scalers):
            f = rearrange(bn(rearrange(f, "b t c -> b c t")), "b c t -> b t c")  # チャンネルごとの BatchNorm
            out = out + (f + lat(f)) * sc
        return out

    def _predict(self, x: torch.Tensor, hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """decoder の LayerNorm を通したトークンから、クラスとマスクを出す。

        Args:
            x: (B, Q + P + N, C)。
            hw: パッチの (縦, 横) の数。

        Returns:
            (クラス (B, Q, K+1), マスク (B, Q, 4h, 4w))。
        """
        x = self.decoder_norm(x)
        q = x[:, :self.num_queries]
        img = rearrange(x[:, self.num_queries + self.prefix:], "b (h w) c -> b c h w", h=hw[0], w=hw[1])
        return self.class_head(q), torch.einsum("bqc,bchw->bqhw", self.mask_head(q), self.upscale(img))

    @torch.compiler.disable
    def _attn_mask(self, masks: torch.Tensor, n_tokens: int, hw: tuple[int, int], prob: torch.Tensor) -> torch.Tensor:
        """クエリが予測のマスクの外の画像のトークンを見ないようにする注意のマスク(EoMT と同じ。True = 見る)。

        Args:
            masks: 予測のマスク (B, Q, 4h, 4w)。
            n_tokens: トークンの数 Q + P + N。
            hw: パッチの (縦, 横) の数。
            prob: この層でマスクを使う確率(0 次元)。

        Returns:
            (B, 1, T, T) の bool。
        """
        b, q = masks.shape[:2]
        allow = torch.ones(b, n_tokens, n_tokens, dtype=torch.bool, device=masks.device)
        inside = rearrange(F.interpolate(masks, size=hw, mode="bilinear") > 0, "b q h w -> b q (h w)")
        free = torch.rand(b, q, device=masks.device) > prob
        allow[:, :q, q + self.prefix:] = inside | free[..., None]
        return rearrange(allow, "b n m -> b 1 n m")

    def forward(self, images: torch.Tensor) -> list[dict]:
        """画像から、decoder の各層の前の予測と最後の予測を出す。

        Args:
            images: 正規化した画像 (B, 3, S, S)。

        Returns:
            予測の並び(num_blocks + 1 個。最後の要素が最終の予測)。各要素は {"logits": (B, Q, K+1), "masks": (B, Q, S/4, S/4)}。
        """
        hw = (images.shape[-2] // self.backbone.patch_embed.patch_size[0],
              images.shape[-1] // self.backbone.patch_embed.patch_size[1])
        x = self._features(images)  # (B, P + N, C)
        x = torch.cat([repeat(self.query.weight, "q c -> b q c", b=x.shape[0]), x], dim=1)  # (B, Q + P + N, C)
        cos, sin = rope_cos_sin(hw[0], hw[1], x.shape[-1] // self.heads, x.device, self.training)
        outs = []
        for i, block in enumerate(self.blocks):
            logits, masks = self._predict(x, hw)
            outs.append({"logits": logits, "masks": masks})
            mask = self._attn_mask(masks.detach(), x.shape[1], hw, self.mask_probs[i])
            x = block(x, mask, cos, sin)
        logits, masks = self._predict(x, hw)
        outs.append({"logits": logits, "masks": masks})
        return outs
