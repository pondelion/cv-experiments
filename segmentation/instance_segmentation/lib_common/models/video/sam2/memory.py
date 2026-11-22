"""SAM 2 の記憶の encoder と記憶の注意。exp00502。

参考: 公式のコード(facebookresearch/sam2 の `sam2/modeling/memory_encoder.py`・`memory_attention.py`、設定 `sam2.1_hiera_b+.yaml`)。
コードは写さず、計算を読み取って書いた。設定は公式の SAM 2.1 と同じ:
    - 記憶の encoder: マスク(sigmoid の後)を 3×3・stride 2 の畳み込み 4 回で 1/16 に縮め(1 → 4 → 16 → 64 → 256 チャンネル、
      各回 LayerNorm2d + GELU、最後に 1×1)、画像の特徴(1×1 の射影)に足し、ConvNeXt の block 2 つ(7×7 depthwise、layer scale 1e-6)で
      混ぜ、1×1 で 64 次元にする。位置の埋め込みは 64 次元の sine
    - 記憶の注意: 4 層。各層は前置きの LayerNorm で、self-attention(RoPE、頭 1)→ 記憶への cross-attention(RoPE、頭 1、
      キー・値 64 次元、キーにだけ位置の埋め込み、物体のポインタのトークンには RoPE をかけない)→ FFN(2,048、ReLU)。dropout 0.1。
      入力に 0.1 × 位置の埋め込みを足し、最後に LayerNorm
    - 公式は 1 つの層を作って deepcopy で並べる(get_clones)ので、層の重みの初期値はどれも同じ。ここも同じにする

形の記号: B = まとめの数、C = 次元(256)、M = 記憶の次元(64)、H・W = 特徴の縦・横(入力の 1/16)、N = 記憶のトークンの数。
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from segmentation.instance_segmentation.lib_common.models.video.sam2.blocks import LayerNorm2d, PositionEmbeddingSine, RoPEAttention


class MaskDownSampler(nn.Module):
    """マスクを stride 2 の畳み込みで 1/16 に縮める(チャンネルは毎回 4 倍)。"""

    def __init__(self, dim: int = 256, kernel_size: int = 3, stride: int = 2, padding: int = 1, total_stride: int = 16):
        super().__init__()
        layers, cin = [], 1
        n = 0
        while stride ** n < total_stride:
            n += 1
        assert stride ** n == total_stride
        for _ in range(n):
            cout = cin * stride ** 2
            layers += [nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding), LayerNorm2d(cout), nn.GELU()]
            cin = cout
        layers.append(nn.Conv2d(cin, dim, 1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CXBlock(nn.Module):
    """ConvNeXt の block(7×7 depthwise → LayerNorm → Linear 4 倍 → GELU → Linear → layer scale、残差)。"""

    def __init__(self, dim: int = 256, kernel_size: int = 7, padding: int = 3, layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1, self.act, self.pwconv2 = nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.dwconv(x)).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(y)))
        return x + y.permute(0, 3, 1, 2)


class MemoryEncoder(nn.Module):
    """画像の特徴とマスクから、1 フレームの記憶(特徴と位置の埋め込み)を作る。"""

    def __init__(self, dim: int = 256, out_dim: int = 64):
        super().__init__()
        self.mask_downsampler = MaskDownSampler(dim)
        self.pix_feat_proj = nn.Conv2d(dim, dim, 1)
        block = CXBlock(dim)
        self.fuser = nn.ModuleList(copy.deepcopy(block) for _ in range(2))
        self.out_proj = nn.Conv2d(dim, out_dim, 1)
        self.position_encoding = PositionEmbeddingSine(out_dim)

    def forward(self, pix_feat: torch.Tensor, masks: torch.Tensor, skip_mask_sigmoid: bool = False):
        """Args: pix_feat (B, C, H, W)、masks (B, 1, 16H, 16W)(値。skip_mask_sigmoid なら 0〜1)。

        Returns: (記憶の特徴 (B, M, H, W), 位置の埋め込み (B, M, H, W))。
        """
        if not skip_mask_sigmoid:
            masks = torch.sigmoid(masks)
        x = self.pix_feat_proj(pix_feat) + self.mask_downsampler(masks)
        for layer in self.fuser:
            x = layer(x)
        x = self.out_proj(x)
        return x, self.position_encoding(x).to(x.dtype)


class MemoryAttentionLayer(nn.Module):
    """記憶の注意の 1 層(前置きの LayerNorm)。"""

    def __init__(self, dim: int = 256, mem_dim: int = 64, ffn: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = RoPEAttention(dim, 1, dropout=dropout)
        self.cross_attn_image = RoPEAttention(dim, 1, dropout=dropout, kv_in_dim=mem_dim, rope_k_repeat=True)
        self.linear1, self.linear2 = nn.Linear(dim, ffn), nn.Linear(ffn, dim)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout, self.dropout1, self.dropout2, self.dropout3 = (nn.Dropout(dropout) for _ in range(4))

    def forward(self, tgt, memory, pos, query_pos, num_k_exclude_rope: int = 0):
        """Args: tgt (B, H·W, C)、memory・pos (B, N, M)、query_pos (B, H·W, C)(使わない: 公式の pos_enc_at_attn・
        pos_enc_at_cross_attn_queries が False)、num_k_exclude_rope: 記憶の最後の物体のポインタのトークンの数。"""
        t2 = self.norm1(tgt)
        tgt = tgt + self.dropout1(self.self_attn(t2, t2, t2))
        t2 = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.cross_attn_image(t2, memory + pos, memory, num_k_exclude_rope=num_k_exclude_rope))
        t2 = self.norm3(tgt)
        return tgt + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(t2)))))


class MemoryAttention(nn.Module):
    """今のフレームの画像の特徴を、記憶(過去のフレームの記憶の特徴 + 物体のポインタ)に注意させて直す(4 層)。"""

    def __init__(self, dim: int = 256, mem_dim: int = 64, num_layers: int = 4):
        super().__init__()
        layer = MemoryAttentionLayer(dim, mem_dim)
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens: int = 0) -> torch.Tensor:
        """Args: curr・curr_pos (B, H·W, C)、memory・memory_pos (B, N, M)。Returns: (B, H·W, C)。"""
        x = curr + 0.1 * curr_pos
        for layer in self.layers:
            x = layer(x, memory, memory_pos, curr_pos, num_obj_ptr_tokens)
        return self.norm(x)
