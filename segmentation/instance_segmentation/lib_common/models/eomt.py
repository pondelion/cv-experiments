"""EoMT(Kerssies ら CVPR 2025、"Your ViT is Secretly an Image Segmentation Model")の自作の実装。exp00103 の C。

参考: 論文と公式のコード(tue-mps/eomt、MIT。`models/eomt.py`・`models/scale_block.py`・`training/lightning_module.py`・
`configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml`)。コードは写さず、構成と設定の値を読み取って書いた。

構成:
    ViT(timm。既定は R50 と同じ程度の大きさの ViT-S/16 `deit3_small_patch16_224.fb_in1k`)の最後の num_blocks 層(ViT-S は 3)の前で、
    学習するクエリ Q 個を画像のトークンの前につなげ、そこからはクエリと画像のトークンを ViT の自己注意でまとめて処理する
    (pixel decoder も transformer decoder もない)。
    予測: ViT の最後の LayerNorm を通したクエリから、クラス(Linear。K + 1)と、マスクの埋め込み(3 層の MLP)を出し、画像のトークンを
    ScaleBlock(2×2 の転置畳み込みで 2 倍 → GELU → 3×3 の depthwise → LayerNorm)で stride 4 まで上げた特徴との内積をマスクにする。
    学習中の注意のマスク(mask annealing): クエリを入れた各層の前でも予測を出し(補助の損失にも使う)、その層ではクエリが
    予測のマスクの外の画像のトークンを見ないようにする。層ごとに「マスクを使う確率」を学習の進み方に合わせて 1 から 0 へ下げ
    (0.9 乗)、学習の最後にはマスクなし(推論はマスクなしの普通の ViT と同じ)にする。

出力の形は自作の Mask2Former(`mask2former.py`)と同じ({"logits", "masks"} の並び)なので、損失(Mask2FormerCriterion)と
推論(postprocess.to_predictions)はそのまま使う。

形の記号: B = 画像の数、Q = クエリの数、P = ViT の先頭のトークン(クラスのトークン)の数、h・w = パッチの縦・横の数、
N = h × w、C = ViT の次元、K = クラスの数、S = 入力の一辺。
"""
import math

import timm
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import LayerNorm2d
from torch import nn


class ScaleBlock(nn.Module):
    """特徴を 2 倍に上げる: 2×2 の転置畳み込み(stride 2)→ GELU → 3×3 の depthwise の畳み込み → LayerNorm。

    Args:
        dim: チャンネルの数。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴を 2 倍に上げる。

        Args:
            x: (B, C, h, w)。

        Returns:
            (B, C, 2h, 2w)。
        """
        return self.norm(self.dw(F.gelu(self.up(x))))


def annealing_windows(num_blocks: int) -> list[tuple[float, float]]:
    """クエリを入れた層ごとの、注意のマスクを使う確率を 1 から 0 に下げる区間(学習の進み方の割合)。

    公式の設定(ViT-S の 3 層: 0.2〜0.4・0.5〜0.7・0.8〜1.0、ViT-L の 4 層: 0.2〜0.4・0.4〜0.6・0.6〜0.8・0.8〜1.0)と同じく、
    始まりを 0.2〜0.8 に等しい間隔で並べ、それぞれ学習の 0.2 の長さで下げる(前の層から順に外す)。

    Args:
        num_blocks: クエリを入れた層の数。

    Returns:
        層ごとの (始まり, 終わり)。
    """
    starts = [0.2 + 0.6 * i / (num_blocks - 1) for i in range(num_blocks)] if num_blocks > 1 else [0.8]
    return [(s, s + 0.2) for s in starts]


class EoMT(nn.Module):
    """EoMT の自作のモデル。

    Args:
        num_classes: クラスの数 K(物体なしを含めない)。
        backbone: timm の ViT の名前。
        num_queries: クエリの数 Q(公式は 200)。
        num_blocks: クエリを入れる最後の層の数(公式は ViT-S・B で 3、L で 4)。
        img_size: 入力の一辺(ViT の位置の埋め込みをこの大きさに合わせる)。
        pretrained: timm の事前学習の重みを読むか。
        quality_head: iFAN のマスクの質の head(全部の層・クエリで共有の Linear 1 つ。sigmoid の前の値を出す)を足すか。
    """

    def __init__(self, num_classes: int = 8, backbone: str = "deit3_small_patch16_224.fb_in1k", num_queries: int = 200,
                 num_blocks: int = 3, img_size: int = 512, pretrained: bool = True, quality_head: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=img_size)
        vit = self.backbone
        dim, patch = vit.embed_dim, vit.patch_embed.patch_size[0]
        self.num_queries, self.num_blocks, self.prefix = num_queries, num_blocks, vit.num_prefix_tokens
        self.windows = annealing_windows(num_blocks)
        self.register_buffer("mask_probs", torch.ones(num_blocks))  # 層ごとの注意のマスクを使う確率
        self.query = nn.Embedding(num_queries, dim)
        self.class_head = nn.Linear(dim, num_classes + 1)
        self.mask_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.upscale = nn.Sequential(*[ScaleBlock(dim) for _ in range(max(1, int(math.log2(patch)) - 2))])  # 16 → 4
        self.quality = nn.Linear(dim, 1) if quality_head else None

    def set_progress(self, progress: float) -> None:
        """学習の進み方(0〜1)から、層ごとの注意のマスクを使う確率を決める(区間の中では (1 − 進み)^0.9 で下げる)。

        Args:
            progress: 今までの更新の回数 / 全部の回数。
        """
        for i, (s, e) in enumerate(self.windows):
            t = min(max((progress - s) / (e - s), 0.0), 1.0)
            self.mask_probs[i] = (1.0 - t) ** 0.9

    def _predict(self, x: torch.Tensor, hw: tuple[int, int]) -> dict:
        """ViT の LayerNorm を通したトークンから、クラスとマスク(と、あればマスクの質)を出す。

        Args:
            x: クエリ + 先頭のトークン + 画像のトークン (B, Q + P + N, C)。
            hw: パッチの (縦, 横) の数。

        Returns:
            {"logits": (B, Q, K+1), "masks": (B, Q, 4h, 4w)}(quality_head なら "quality": (B, Q) も)。
        """
        x = self.backbone.norm(x)
        q = x[:, :self.num_queries]  # (B, Q, C)
        img = rearrange(x[:, self.num_queries + self.prefix:], "b (h w) c -> b c h w", h=hw[0], w=hw[1])  # (B, C, h, w)
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_head(q), self.upscale(img))  # (B, Q, 4h, 4w)
        out = {"logits": self.class_head(q), "masks": masks}
        if self.quality is not None:
            out["quality"] = self.quality(q)[..., 0]  # (B, Q)
        return out

    @torch.compiler.disable
    def _attn_mask(self, masks: torch.Tensor, n_tokens: int, hw: tuple[int, int], prob: torch.Tensor) -> torch.Tensor:
        """クエリが予測のマスクの外の画像のトークンを見ないようにする注意のマスク(True = 見る)。

        クエリごとに、確率 1 − prob でマスクを使わない(全部見る)。クエリから先頭のトークン・クエリへ、画像のトークンからは
        どこへも制限しない。

        Args:
            masks: 予測のマスク (B, Q, 4h, 4w)。
            n_tokens: トークンの数 Q + P + N。
            hw: パッチの (縦, 横) の数。
            prob: この層でマスクを使う確率(0 次元)。

        Returns:
            (B, 1, Q + P + N, Q + P + N) の bool(頭の分は広げて使う)。
        """
        b, q = masks.shape[:2]
        allow = torch.ones(b, n_tokens, n_tokens, dtype=torch.bool, device=masks.device)
        inside = rearrange(F.interpolate(masks, size=hw, mode="bilinear") > 0, "b q h w -> b q (h w)")  # (B, Q, N)
        free = torch.rand(b, q, device=masks.device) > prob  # (B, Q) この層ではマスクを使わないクエリ
        allow[:, :q, q + self.prefix:] = inside | free[..., None]
        return rearrange(allow, "b n m -> b 1 n m")

    def _block(self, block: nn.Module, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """timm の ViT の Block を、注意のマスクを渡せる形で計算する(自己注意と MLP の残差。LayerScale があれば掛ける)。

        Args:
            block: timm の Block(attn・norm1・norm2・mlp・ls1・ls2 を持つ)。
            x: トークン (B, T, C)。
            mask: 注意のマスク (B, 1, T, T)(True = 見る)か None。

        Returns:
            (B, T, C)。
        """
        attn = block.attn
        qkv = rearrange(attn.qkv(block.norm1(x)), "b t (three h d) -> three b h t d", three=3, h=attn.num_heads)
        q, k, v = attn.q_norm(qkv[0]), attn.k_norm(qkv[1]), qkv[2]  # 3 × (B, 頭の数, T, d)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, 頭の数, T, d)
        y = attn.proj(rearrange(y, "b h t d -> b t (h d)"))  # (B, T, C)
        x = x + block.ls1(y)
        return x + block.ls2(block.mlp(block.norm2(x)))

    def forward(self, images: torch.Tensor) -> list[dict]:
        """画像から、クエリを入れた各層の前の予測と、最後の予測を出す。

        Args:
            images: 正規化した画像 (B, 3, S, S)。

        Returns:
            予測の並び(num_blocks + 1 個。最後の要素が最終の予測)。各要素は
            {"logits": (B, Q, K+1), "masks": (B, Q, S/4, S/4)}。
        """
        vit = self.backbone
        x = vit.norm_pre(vit._pos_embed(vit.patch_embed(images)))  # (B, P + N, C)
        hw = (images.shape[-2] // vit.patch_embed.patch_size[0], images.shape[-1] // vit.patch_embed.patch_size[1])
        first = len(vit.blocks) - self.num_blocks
        outs = []
        for i, block in enumerate(vit.blocks):
            mask = None
            if i == first:  # クエリを画像のトークンの前につなげる: (B, P + N, C) -> (B, Q + P + N, C)
                x = torch.cat([repeat(self.query.weight, "q c -> b q c", b=x.shape[0]), x], dim=1)
            if i >= first:
                outs.append(self._predict(x, hw))
                mask = self._attn_mask(outs[-1]["masks"].detach(), x.shape[1], hw, self.mask_probs[i - first])
            x = self._block(block, x, mask)
        outs.append(self._predict(x, hw))
        return outs
