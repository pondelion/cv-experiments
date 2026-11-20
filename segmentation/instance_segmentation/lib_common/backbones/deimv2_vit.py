"""DEIMv2 の ViT-Tiny(DINOv3 の ViT-S から蒸留した ViT。DEIMv2-S の backbone)を、timm の Eva で作って重みを読む。

DEIMv2 の ViT-Tiny(公式の `engine/backbone/vit_tiny.py`)の構成: 幅 192・12 層・3 頭、パッチ 16、cls token 1 つ(register の
token なし)、RoPE(基数 100、座標は縦・横それぞれ 0〜1 に正規化。DINOv3 と同じ)、qkv に q・k・v 全部のバイアス、layer scale なし、
LayerNorm の eps 1e-6、最後の LayerNorm なし。timm の Eva の DINOv3 の形(`rope_type="dinov3"`)の設定で同じ構成になる。

違いは 1 つだけ: timm の Eva は k のバイアスを 0 に固定する(学習しない buffer)。DEIMv2 の重みは k のバイアスも持つ(0 ではない)ので、
各層の k のバイアスを学習する重み(nn.Parameter)に置き換えてから読む。

重みは公式の README の ViT-Tiny のリンク(Google Drive)から入手し、リポジトリ直下の `vendor/deimv2/vitt_distill.pt` に置く。
"""
from functools import partial
from pathlib import Path

import torch
from timm.layers import LayerNorm
from timm.models.eva import Eva
from torch import nn

WEIGHTS = Path(__file__).resolve().parents[4] / "vendor" / "deimv2" / "vitt_distill.pt"  # リポジトリ直下からの場所


def create_deimv2_vit_tiny(pretrained: bool = True, img_size: int = 512) -> Eva:
    """DEIMv2 の ViT-Tiny を作る。

    Args:
        pretrained: DEIMv2 の蒸留した重みを読むか。
        img_size: 入力の一辺(RoPE は入力の大きさから毎回作るので、重みは大きさによらない)。

    Returns:
        timm の Eva(`forward_intermediates` で途中の層を取り出せる)。最後の LayerNorm は DEIMv2 にないので使わないこと
        (adapter で norm=False にする)。
    """
    vit = Eva(img_size=img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, qkv_bias=True, init_values=None,
              rope_type="dinov3", rope_temperature=100, rope_rotate_half=True, use_rot_pos_emb=True, use_abs_pos_emb=False,
              num_reg_tokens=0, use_fc_norm=False, num_classes=0, dynamic_img_size=True,
              norm_layer=partial(LayerNorm, eps=1e-6))
    for blk in vit.blocks:  # k のバイアスを学習する重みにする(timm は 0 に固定する buffer)
        attn = blk.attn
        del attn.k_bias
        attn.k_bias = nn.Parameter(torch.zeros_like(attn.q_bias))
    vit.pretrained_cfg = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}  # DEIMv2 の学習の正規化
    if pretrained:
        src = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        dst = {}
        for k, v in src.items():
            if k.endswith("attn.qkv.bias"):  # (3·192,) を q・k・v に分ける
                q, kb, vb = v.chunk(3)
                base = k[: -len("qkv.bias")]
                dst[base + "q_bias"], dst[base + "k_bias"], dst[base + "v_bias"] = q, kb, vb
            elif k != "rope_embed.periods":  # RoPE の周期は設定から同じ値が作られる(1 〜 100^(15/16))
                dst[k] = v
        missing, unexpected = vit.load_state_dict(dst, strict=False)
        assert not unexpected and all(k.startswith("norm.") for k in missing), (missing, unexpected)
    return vit
