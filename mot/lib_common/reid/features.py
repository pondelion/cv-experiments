"""見た目の特徴の計算: 物体の枠の切り出しの画像を、特徴を出すモデルに通す(mot/exp00201 の F1・F2)。

切り出し: 枠を画像の内側に切り詰め、縦 × 横 = crop_size(既定 256 × 128。人の ReID の標準の縦長の比)に引き伸ばし、ImageNet の平均・
標準偏差で正規化する(FastReID・timm の学習済みの重みと同じ正規化)。
F2(汎用の学習済みの特徴、学習なし): timm の DINOv3 の ViT に通し、最後の層の LayerNorm の後の CLS のトークンを特徴にする。

形の記号: B = 切り出しの数、H・W = 切り出しの縦・横、C = 特徴の次元。
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop(image: Image.Image, box: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    """枠を切り出し、決まった大きさに引き伸ばして正規化する。

    Args:
        image: フレームの画像(RGB)。
        box: 枠 (左上の x, 左上の y, 幅, 高さ)(画素)。
        size: 切り出しの (縦, 横)。

    Returns:
        正規化した切り出し (3, H, W) の float32。
    """
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(image.width), x + w), min(float(image.height), y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:  # 画像の外の枠: 1 画素の領域にする(特徴は意味を持たないが、並びを保つ)
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
    patch = image.crop((x0, y0, x1, y1)).resize((size[1], size[0]), Image.BILINEAR)
    a = (np.asarray(patch, dtype=np.float32) / 255.0 - MEAN) / STD  # (H, W, 3)
    return np.ascontiguousarray(a.transpose(2, 0, 1))  # (H, W, 3) -> (3, H, W)


class DINOv3Feature(torch.nn.Module):
    """F2: 学習し直さない DINOv3 の ViT の CLS のトークン。

    Args:
        name: timm のモデルの名前(学習済みの重みのタグつき。例 "vit_base_patch16_dinov3.lvd1689m")。
        crop_size: 切り出しの (縦, 横)(ViT の入力の大きさ。パッチの一辺 16 の倍数)。
    """

    def __init__(self, name: str = "vit_base_patch16_dinov3.lvd1689m", crop_size: tuple[int, int] = (256, 128)):
        super().__init__()
        import timm
        self.crop_size = crop_size
        self.vit = timm.create_model(name, pretrained=True, num_classes=0, img_size=crop_size).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴を出す。

        Args:
            x: 切り出し (B, 3, H, W)。

        Returns:
            特徴 (B, C)(最後の LayerNorm の後の CLS のトークン)。
        """
        return self.vit.forward_features(x)[:, 0]


@torch.no_grad()
def extract(model: torch.nn.Module, items: list[tuple[Path, tuple[float, float, float, float]]], crop_size: tuple[int, int],
            batch: int = 64, device: str = "cuda") -> np.ndarray:
    """切り出しの並びの特徴をまとめて計算する(同じ画像の切り出しは画像を 1 回だけ読む)。

    Args:
        model: 切り出し (B, 3, H, W) → 特徴 (B, C) のモデル。
        items: (フレームの画像のパス, 枠) の並び。
        crop_size: 切り出しの (縦, 横)。
        batch: まとめて通す切り出しの数。
        device: 計算する装置。

    Returns:
        特徴 (N, C) の float32(items と同じ順)。
    """
    feats, buf, cache = [], [], {}
    for path, box in items:
        if path not in cache:
            cache = {path: Image.open(path).convert("RGB")}  # 並びは画像の順なので、直前の 1 枚だけ持つ
        buf.append(crop(cache[path], box, crop_size))
        if len(buf) == batch:
            feats.append(model(torch.from_numpy(np.stack(buf)).to(device)).float().cpu().numpy())
            buf = []
    if buf:
        feats.append(model(torch.from_numpy(np.stack(buf)).to(device)).float().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 1), dtype=np.float32)
