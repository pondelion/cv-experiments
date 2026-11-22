"""自己教師ありの学習(`lib_common/ssl/` の DINOv3・DoRA・V-JEPA 2.1)で保存した timm の ViT の重みを、後の学習のモデルの ViT に読む。

保存した重みは、事前学習の入力の大きさ(224 などの)の位置の埋め込みを持つので、読む先の ViT の大きさ(512 など)に補間する
(RoPE の ViT は位置の埋め込みを持たないので、そのまま読める)。
"""
import torch
from timm.layers import resample_abs_pos_embed
from torch import nn


def load_ssl_vit(vit: nn.Module, path: str) -> None:
    """timm の ViT に、自己教師ありの学習の重みを読む(位置の埋め込みは読む先の格子に補間する)。

    Args:
        vit: 読む先の timm の ViT(VisionTransformer か Eva)。
        path: `teacher_backbone.pt` などの、timm の ViT の state_dict。画像の段階の学習済みのモデルの `best.pt`(GTR 系の
            ViTAdapter。名前が `backbone.vit.` で始まる)なら、その ViT の部分だけを読む。

    Raises:
        RuntimeError: 読む先にあって重みにない、または重みにあって読む先にないパラメータがあるとき(head は除く)。
    """
    sd = torch.load(path, map_location="cpu")
    prefix = "backbone.vit."
    if any(k.startswith(prefix) for k in sd):  # 画像の段階のモデル全体の重み → ViT の部分だけ
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if "pos_embed" in sd and getattr(vit, "pos_embed", None) is not None and sd["pos_embed"].shape != vit.pos_embed.shape:
        npt = 0 if getattr(vit, "no_embed_class", False) else vit.num_prefix_tokens
        n_old = sd["pos_embed"].shape[1] - npt
        old = int(round(n_old ** 0.5))
        sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=vit.patch_embed.grid_size, old_size=(old, old),
                                                 num_prefix_tokens=npt)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("head")]
    if missing or unexpected:
        raise RuntimeError(f"重みが合わない: 足りない {missing[:5]}…({len(missing)} 個)、余分 {unexpected[:5]}…({len(unexpected)} 個)")
