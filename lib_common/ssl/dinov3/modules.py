"""DINOv3 の自己教師ありの学習の部品: timm の ViT を、iBOT の「隠したパッチ」を入れられる形で包むものと、DINO の head。

公式(facebookresearch/dinov3 の `dinov3/models/vision_transformer.py`・`dinov3/layers/dino_head.py`)と同じ働きを、timm の
ViT の上に自分で書いたもの(コードは写していない)。backbone は timm の全部の層が同じ解像度の ViT(VisionTransformer。
DINOv3 の ViT は timm では Eva。RoPE)を、そのまま使う。

形の記号: B = 画像の数、N = パッチの数、P = 先頭のトークン(クラスのトークン + register のトークン)の数、C = ViT の次元、
K = head の出力の次元(prototype の数)。
"""
import timm
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.init import trunc_normal_


class MaskableViT(nn.Module):
    """timm の ViT を包み、パッチの埋め込みの直後に、隠すパッチを学習するトークン(mask token)に置き換えられるようにする。

    入力の大きさは切り抜きごとに違う(大きい切り抜き 224、小さい切り抜き 96)ので、timm の dynamic_img_size で、位置の埋め込みを
    その大きさに合わせる。

    Args:
        name: timm の ViT の名前(VisionTransformer か Eva)。
        pretrained: 学習済みの重みを読むか(一から学習するなら False)。
        drop_path_rate: 層を確率的に飛ばす割合(公式の ViT-L の ImageNet-1K の設定は 0.3)。
        **kwargs: timm.create_model に渡すほかの引数。
    """

    def __init__(self, name: str, pretrained: bool = False, drop_path_rate: float = 0.0, **kwargs):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True,
                                     drop_path_rate=drop_path_rate, **kwargs)
        if getattr(self.vit, "rope_mixed", False):
            raise NotImplementedError("層ごとに違う RoPE(rope_mixed)には対応していない")
        self.embed_dim = self.vit.embed_dim
        self.num_prefix = self.vit.num_prefix_tokens
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))  # 公式と同じく 0 で始める

    @staticmethod
    def _qk(attn: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """注意の層の q・k を、層の入力 x (B, T, C) から計算する(timm の Attention・EvaAttention の両方。RoPE をかける前)。

        Returns:
            (q, k)。どちらも (B, 頭の数, T, 頭の次元)。
        """
        b, n, _ = x.shape
        if getattr(attn, "qkv", None) is not None:
            if getattr(attn, "q_bias", None) is not None:  # Eva: q・v だけの bias(k は 0)
                qkv = F.linear(x, attn.qkv.weight, torch.cat((attn.q_bias, attn.k_bias, attn.v_bias)))
            else:
                qkv = attn.qkv(x)
            q, k, _ = rearrange(qkv, "b n (three h d) -> three b h n d", three=3, h=attn.num_heads).unbind(0)
        else:  # Eva の q・k・v が別の線形の層の版
            q = rearrange(attn.q_proj(x), "b n (h d) -> b h n d", h=attn.num_heads)
            k = rearrange(attn.k_proj(x), "b n (h d) -> b h n d", h=attn.num_heads)
        return attn.q_norm(q), attn.k_norm(k)

    def forward(self, x: torch.Tensor, masks: torch.Tensor | None = None, return_last_qk: bool = False) -> dict:
        """Args:
            x: (B, 3, H, W)。
            masks: (B, N) の bool(True のパッチを mask token に置き換える。None なら置き換えない)。
            return_last_qk: 最後の層の q・k と [CLS] からの注意も返す(DoRA が物体を見つけ・追うのに使う)。

        Returns:
            {"cls": (B, C), "storage": (B, P − 1, C)(register のトークン), "patch": (B, N, C)}(どれも最後の LayerNorm の後)。
            return_last_qk のときは、加えて {"q": (B, N, C), "k": (B, N, C)(パッチの分。頭をつなげた形), "cls_attn": (B, 頭の数, N)
            ([CLS] から各パッチへの注意。softmax は全部のトークンで取る)}。
        """
        v = self.vit
        t = v.patch_embed(x)  # (B, h, w, C)(dynamic_img_size のとき)
        if masks is not None:
            m = rearrange(masks, "b (h w) -> b h w 1", h=t.shape[1])
            t = torch.where(m, self.mask_token.to(t.dtype), t)
        pe = v._pos_embed(t)  # 先頭のトークンをつなげ、位置の埋め込みを足す
        t, rope = pe if isinstance(pe, tuple) else (pe, None)  # Eva は RoPE の値も返す
        if getattr(v, "patch_drop", None) is not None:  # Eva は None のことがある
            t = v.patch_drop(t)
        t = v.norm_pre(t)
        extra = {}
        for i, blk in enumerate(v.blocks):
            if return_last_qk and i == len(v.blocks) - 1:
                q, k = self._qk(blk.attn, blk.norm1(t))  # (B, 頭, T, d)
                attn = (q[:, :, :1] @ k.transpose(-1, -2) * blk.attn.scale).softmax(-1)  # (B, 頭, 1, T) [CLS] からの注意
                p = self.num_prefix
                extra = {"q": rearrange(q[:, :, p:], "b h n d -> b n (h d)"), "k": rearrange(k[:, :, p:], "b h n d -> b n (h d)"),
                         "cls_attn": attn[:, :, 0, p:]}
            t = blk(t, rope=rope) if rope is not None else blk(t)
        t = v.norm(t)
        return {"cls": t[:, 0], "storage": t[:, 1: self.num_prefix], "patch": t[:, self.num_prefix:], **extra}


class DINOHead(nn.Module):
    """DINO・iBOT の head: MLP(既定 3 層、中間 2048)→ L2 正規化 → 重みの正規化のない最後の線形の層(prototype の数 K)。

    Args:
        in_dim: 入力の次元 C。
        out_dim: prototype の数 K(公式は 65536)。
        hidden_dim: MLP の中間の次元(公式は 2048)。
        bottleneck_dim: 最後の線形の層の前の次元(公式は 256)。
        nlayers: MLP の層の数(公式は 3)。
    """

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048, bottleneck_dim: int = 256, nlayers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (max(nlayers, 1) - 1) + [bottleneck_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        """公式と同じ初期値(線形の層の重みは標準偏差 0.02 の切断正規分布、bias は 0)。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(M, C) -> (M, K)。"""
        x = F.normalize(self.mlp(x), dim=-1, eps=1e-6 if x.dtype == torch.float16 else 1e-12)
        return self.last_layer(x)
