"""マスクの変換(0/1 の画像 ⇔ COCO 形式の RLE)。

RLE(run-length encoding)は、マスクを「0 が何画素続き、次に 1 が何画素続き…」という長さの列で表す圧縮形式。
COCO 形式の正解・予測のマスクはこの形で保存する。
"""
import numpy as np
from pycocotools import mask as mask_utils


def mask_to_rle(mask: np.ndarray) -> dict:
    """0/1 のマスクを、JSON に書ける COCO 形式の RLE にする。

    Args:
        mask: (H, W) の bool または 0/1 の配列。

    Returns:
        {"size": [H, W], "counts": str} の辞書。
    """
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))  # pycocotools は列優先の並びを要求する
    rle["counts"] = rle["counts"].decode("ascii")  # bytes → str(JSON に書くため)
    return rle


def rle_to_mask(rle: dict) -> np.ndarray:
    """COCO 形式の RLE をマスクに戻す。

    Args:
        rle: {"size": [H, W], "counts": ...} の辞書。

    Returns:
        (H(縦), W(横)) の bool のマスク。
    """
    return mask_utils.decode(rle).astype(bool)


def mask_bbox(mask: np.ndarray) -> list[float]:
    """マスクを囲む枠を返す。

    Args:
        mask: (H(縦), W(横)) の bool または 0/1 の配列。

    Returns:
        COCO 形式の枠 [x, y, 幅, 高さ](空のマスクは [0, 0, 0, 0])。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = xs.min(), ys.min()
    return [float(x0), float(y0), float(xs.max() - x0 + 1), float(ys.max() - y0 + 1)]
