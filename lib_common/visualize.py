"""予測・正解の可視化と、結果のページ(GitHub Pages)に載せる見本の画像の保存。

GitHub に置くので、画像は長い辺を MAX_SIDE 画素に縮め、JPEG(品質 JPEG_QUALITY)で保存して小さくする。
"""
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 480  # 正解と予測がどれくらい合っているかが大まかにわかれば十分なので、小さめにする
JPEG_QUALITY = 70

# 物体ごとに色を変えるための色の並び(見分けやすい 20 色、RGB)
PALETTE = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
    (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
    (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
], dtype=np.uint8)


def draw_instances(image: np.ndarray, masks: list[np.ndarray], labels: list[str], alpha: float = 0.45) -> np.ndarray:
    """画像に、物体ごとのマスク(半透明の色)・輪郭・ラベルを描く。

    Args:
        image: (H, W, 3) の RGB の uint8 の画像。
        masks: 物体ごとの (H, W) の bool のマスク。後のものほど上に描く。
        labels: 物体ごとのラベルの文字列(例 "dog 0.93")。英数字のみ(OpenCV の文字の描画は日本語に対応していない)。
        alpha: マスクの色の濃さ。

    Returns:
        描いた (H, W, 3) の RGB の uint8 の画像。
    """
    out = image.astype(np.float32)
    for i, mask in enumerate(masks):
        color = PALETTE[i % len(PALETTE)].astype(np.float32)
        out[mask] = out[mask] * (1 - alpha) + color * alpha
    out = out.astype(np.uint8)
    scale = max(image.shape[:2]) / MAX_SIDE  # 縮めた後も線・文字が同じ太さに見えるようにする
    thick = max(1, round(2 * scale))
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = tuple(int(c) for c in PALETTE[i % len(PALETTE)])
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thick)
        ys, xs = np.nonzero(mask)
        if len(xs):
            x, y = int(xs.min()), max(int(ys.min()), int(14 * scale))
            font = 0.45 * scale
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
            cv2.rectangle(out, (x, y - th - 2), (x + tw + 2, y + 2), color, -1)
            cv2.putText(out, label, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def side_by_side(panels: list[tuple[str, np.ndarray]], side: int = 360, title_h: int = 24) -> np.ndarray:
    """複数の画像を、見出しを付けて横に並べた 1 枚の画像にする(例: 元の画像 | 正解 | 予測)。

    Args:
        panels: (見出し, (H, W, 3) の RGB の uint8 の画像) の並び。見出しは英数字のみ(OpenCV の文字の描画は日本語に対応していない)。
        side: 各画像の長い辺をこの画素数に縮める。
        title_h: 見出しの帯の高さ(画素)。

    Returns:
        並べた画像 (title_h + 各画像の縦の最大, 各画像の横の合計 + 間の 4 画素, 3) の RGB の uint8。
    """
    gap = 4
    resized = []
    for _, img in panels:
        s = side / max(img.shape[:2])
        resized.append(cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA))
    h = title_h + max(r.shape[0] for r in resized)
    w = sum(r.shape[1] for r in resized) + gap * (len(resized) - 1)
    out = np.full((h, w, 3), 255, np.uint8)  # 白の地
    x = 0
    for (title, _), r in zip(panels, resized):
        out[title_h: title_h + r.shape[0], x: x + r.shape[1]] = r
        cv2.putText(out, title, (x + 4, title_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        x += r.shape[1] + gap
    return out


def save_jpeg(image: np.ndarray, path: Path, max_side: int | None = MAX_SIDE) -> None:
    """RGB の画像を、長い辺 max_side 画素に縮めて JPEG で保存する。

    Args:
        image: (H(縦), W(横), 3(RGB)) の uint8。
        path: 保存先。
        max_side: 長い辺の上限(None なら縮めない。side_by_side で並べた画像など、すでに縮めてあるとき)。
    """
    h, w = image.shape[:2]
    s = max_side / max(h, w) if max_side else 1.0
    if s < 1:
        image = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
