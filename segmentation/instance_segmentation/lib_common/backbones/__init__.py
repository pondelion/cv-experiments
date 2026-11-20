"""インスタンスセグメンテーションの方式で共通に使う backbone の作り方(timm の上に、timm だけでは足りない部分を足す)。

- `create_feature_backbone`: stride 8・16・32 の 3 段の特徴を出す timm の backbone を作る。学習のときの解像度に位置の情報が
  固定されている backbone(EfficientFormerV2)は、入力の大きさに合わせて作り、学習済みの位置のバイアスを補間して読む。
  全部の段が同じ解像度の backbone(素の ViT・EVA・BEiT など。timm の features_only の各段の縮小の倍率が全部同じもの)は、DEIMv2 と
  同じ作りの adapter(`vit_adapter.py`)で 3 段にする。
  `deimv2_vit_tiny` は、timm にない DEIMv2 の ViT-Tiny(`deimv2_vit.py`)を同じ adapter で使う
"""
import timm
import torch
import torch.nn.functional as F
from torch import nn

STRIDES = (8, 16, 32)


def _needs_img_size(name: str) -> bool:
    """作るときに入力の大きさを渡す必要がある backbone か(注意の位置のバイアスの表の大きさが入力の大きさで決まるもの)。

    Args:
        name: timm のモデルの名前。

    Returns:
        渡す必要があれば True。
    """
    return name.startswith("efficientformerv2")


def _resize_attention_biases(src: dict, dst: dict) -> dict:
    """学習済みの重みの位置のバイアス(`attention_biases`)を、作ったモデルの大きさに補間する。

    EfficientFormerV2 の `attention_biases` は (頭の数 H, r·r) の表で、縦・横のずれ(0〜r−1)ごとの値を持つ
    (`attention_bias_idxs` で注意の行列の各組に配る)。入力が大きいと r が大きくなるので、(H, r, r) の画像と見て bicubic で
    (H, r', r') に補間する(LeViT・EfficientFormer の系統で解像度を変えるときと同じやり方)。

    Args:
        src: 学習済みの重み(既定の入力の大きさのモデルの state_dict)。
        dst: 作ったモデルの state_dict(大きさの見本)。

    Returns:
        dst と同じ大きさにした重み。大きさが違って補間しないもの(`attention_bias_idxs` などの buffer)は dst の値を使う。
    """
    out = {}
    for k, v in src.items():
        want = dst[k]
        if v.shape == want.shape:
            out[k] = v
        elif k.endswith("attention_biases"):
            h, n = v.shape
            r, r2 = round(n ** 0.5), round(want.shape[1] ** 0.5)
            grid = v.reshape(1, h, r, r)  # (H, r·r) -> (1, H, r, r)
            grid = F.interpolate(grid, size=(r2, r2), mode="bicubic", align_corners=False)  # -> (1, H, r', r')
            out[k] = grid.reshape(h, r2 * r2)  # -> (H, r'·r')
        else:
            out[k] = want  # 入力の大きさから作る番号の表(学習しない値)
    return out


def create_feature_backbone(name: str, pretrained: bool = True, img_size: int = 512) -> nn.Module:
    """stride 8・16・32 の 3 段の特徴を出す timm の backbone を作る。

    Args:
        name: timm のモデルの名前。
        pretrained: 学習済みの重みを読むか。
        img_size: 入力の一辺(位置の情報が入力の大きさで決まる backbone だけで使う)。

    Returns:
        features_only の timm のモデル(forward は 3 段の (B, C_l, H/s_l, W/s_l) の並び)。
    """
    from segmentation.instance_segmentation.lib_common.backbones.vit_adapter import ViTAdapter
    if name == "deimv2_vit_tiny":  # DEIMv2 の ViT-Tiny(timm にない。deimv2_vit.py)。最後の LayerNorm を持たない
        from segmentation.instance_segmentation.lib_common.backbones.deimv2_vit import create_deimv2_vit_tiny
        return ViTAdapter(name, vit=create_deimv2_vit_tiny(pretrained, img_size), norm=False)
    size_kw = {"img_size": img_size} if _needs_img_size(name) else {}
    reductions = timm.create_model(name, pretrained=False, features_only=True, **size_kw).feature_info.reduction()
    if len(set(reductions)) == 1:  # 全部の段が同じ解像度(素の ViT・EVA・BEiT など)は、adapter で 3 段にする(vit_adapter.py)
        return ViTAdapter(name, pretrained, img_size)
    out_indices = tuple(reductions.index(s) for s in STRIDES)  # 段の番号は backbone によって違う(R50 では (2, 3, 4))
    if not size_kw:
        return timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
    model = timm.create_model(name, pretrained=False, features_only=True, out_indices=out_indices, **size_kw)
    if pretrained:  # 既定の入力の大きさで学習済みの重みを読み、位置のバイアスだけを補間して移す(名前は同じ features_only の形)
        src = timm.create_model(name, pretrained=True, features_only=True, out_indices=out_indices).state_dict()
        model.load_state_dict(_resize_attention_biases(src, model.state_dict()), strict=True)
    return model
