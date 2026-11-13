"""切り抜いた物体を背景に合成し、正解のインスタンスマスクを作る(① 生成だけ の ①-a。認識モデルは使わない)。

正解のマスクは、貼った物体の形そのもの。物体が重なった所は、後から貼った物体が上になり、下の物体のマスクからその部分を除く。
見えている部分が小さくなりすぎた物体は、正解から外し、画像からも消す(貼らない)。

境目の処理(どれも決まった計算で、正解のマスクは変えない):
    - 色のにじみの除去: 物体の輪郭の近くでは、背景の強い色(グリーンバックなら緑)を、弱い色の大きい方までに抑える
    - 輪郭のぼかし: 貼るときの透明度だけを 1 画素ぶんぼかす(正解のマスクは、ぼかす前の形)
"""
import random
from dataclasses import dataclass

import cv2
import numpy as np
from einops import rearrange


@dataclass
class ObjectCutout:
    """切り抜いた物体 1 つ。

    Attributes:
        name: クラス名。
        image: 物体を囲む枠で切り出した (h, w, 3) の RGB の画像。
        mask: 同じ枠の (h, w) の bool のマスク。
        bg_color: 生成したときの背景の色(色のにじみの除去に使う)。
    """
    name: str
    image: np.ndarray
    mask: np.ndarray
    bg_color: np.ndarray


def crop_object(name: str, image: np.ndarray, mask: np.ndarray, bg_color) -> ObjectCutout:
    """生成した画像とマスクから、物体を囲む枠で切り出す。

    Args:
        name: クラス名。
        image: 生成した画像 (H(縦), W(横), 3(RGB)) の uint8。
        mask: 物体の画素 (H, W) の bool。
        bg_color: 生成したときの背景の色 (3,)。

    Returns:
        物体を囲む枠 (h, w) で切り出した ObjectCutout。
    """
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return ObjectCutout(name, image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), np.asarray(bg_color, np.float32))


def despill(obj: ObjectCutout, band: int = 3) -> np.ndarray:
    """輪郭から band 画素以内で、背景の強い色を弱い色の大きい方までに抑えた画像を返す(物体の内側の色は変えない)。

    Args:
        obj: 切り抜いた物体。
        band: 輪郭から内側に何画素までを直すか。

    Returns:
        直した画像 (h(縦), w(横), 3(RGB)) の uint8(obj.image と同じ形)。
    """
    img = obj.image.astype(np.float32)
    bg = obj.bg_color
    mid = (bg.max() + bg.min()) / 2
    strong, weak = [i for i in range(3) if bg[i] > mid], [i for i in range(3) if bg[i] <= mid]
    if not weak:  # 白などの背景(強い色がない)では何もしない
        return obj.image
    inner = cv2.erode(obj.mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=band).astype(bool)
    edge = obj.mask & ~inner  # 輪郭の近くの帯
    cap = img[..., weak].max(axis=-1)  # (h, w) 弱い色の大きい方
    for c in strong:
        img[..., c] = np.where(edge, np.minimum(img[..., c], cap), img[..., c])
    return img.clip(0, 255).astype(np.uint8)


@dataclass
class Placement:
    """物体を置く場所と大きさ(画像の画素の単位)。"""
    obj: ObjectCutout
    x: int  # 左上の位置
    y: int
    scale: float


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """2 つの枠の IoU(重なりの面積 / 合わせた面積)。

    Args:
        a, b: 枠 (x0, y0, x1, y1)(画素)。

    Returns:
        IoU(0〜1)。
    """
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def random_layout(objects: list[ObjectCutout], size: int, rng: random.Random, min_height: float = 0.08,
                  max_height: float = 0.35, max_width: float = 0.6, max_overlap: float = 0.2,
                  tries: int = 20) -> list[Placement]:
    """物体の置き方を決める。画像の下の方ほど手前とみなして大きくし、手前の物体を後から(上に)貼る。

    物体が大きすぎる・重なりすぎると不自然になる(15〜60% の高さでは、電車などが画像の 4 割を占めて貼り付けたように見えた)ので、
    高さ・幅に上限を付け、ほかの物体の枠との重なり(IoU)が max_overlap を超える置き方は、置き直す(tries 回でだめなら置かない)。

    Args:
        objects: 置く物体。
        size: 画像の一辺の画素数。
        rng: 乱数。
        min_height, max_height: 物体の高さの、画像の高さに対する割合の範囲(奥 → 手前)。
        max_width: 物体の幅の、画像の幅に対する割合の上限。
        max_overlap: ほかの物体の枠との IoU の上限。
        tries: 1 つの物体を置き直す回数の上限。

    Returns:
        置けた物体の置き方の並び(奥から順。置けなかった物体は入らない)。
    """
    placements, boxes = [], []
    for obj in objects:
        for _ in range(tries):
            depth = rng.random()  # 0 = 奥、1 = 手前
            height = size * (min_height + (max_height - min_height) * depth) * rng.uniform(0.8, 1.2)
            scale = min(height / obj.image.shape[0], max_width * size / obj.image.shape[1])
            h, w = round(obj.image.shape[0] * scale), round(obj.image.shape[1] * scale)
            bottom = round(size * (0.55 + 0.45 * depth))  # 手前ほど物体の下端が画像の下の方
            y = min(max(bottom - h, 0), size - h)
            x = rng.randint(0, max(size - w, 0))
            box = (x, y, x + w, y + h)
            if all(_box_iou(box, b) <= max_overlap for b in boxes):
                placements.append((depth, Placement(obj, x, y, scale)))
                boxes.append(box)
                break
    return [p for _, p in sorted(placements, key=lambda t: t[0])]  # 奥から順に貼る


def compose(background: np.ndarray, placements: list[Placement], min_visible: float = 0.3,
            min_pixels: int = 200) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """背景に物体を順に貼り、合成した画像と、物体ごとの正解のマスク(見えている部分)を返す。

    Args:
        background: (H, W, 3) の RGB の背景の画像。
        placements: 貼る順(奥から)の置き方。
        min_visible: 見えている部分の面積が、物体の面積のこの割合より小さくなった物体は外す。
        min_pixels: 見えている部分がこの画素数より小さくなった物体は外す。

    Returns:
        (画像, [(クラス名, (H, W) の bool のマスク), ...])。外した物体は画像にも貼らない。
    """
    fg, alpha, gts = compose_layers(background.shape[:2], placements, min_visible, min_pixels)
    return blend(background, fg, alpha), gts


def blend(background: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """背景に、物体の層(compose_layers の fg・alpha)を重ねる: 画像 = 背景 × (1 − alpha) + fg。

    Args:
        background: 背景 (H(縦), W(横), 3(RGB)) の uint8。
        fg: 透明度を掛けた物体の色 (H, W, 3) の float32。
        alpha: 透明度 (H, W, 1) の float32(0〜1)。

    Returns:
        合成した画像 (H, W, 3) の uint8。
    """
    return (background.astype(np.float32) * (1 - alpha) + fg).clip(0, 255).astype(np.uint8)


def compose_layers(shape: tuple[int, int], placements: list[Placement], min_visible: float = 0.3,
                   min_pixels: int = 200, return_kept: bool = False):
    """物体だけの層を作る。どんな背景にも blend(背景, fg, alpha) で重ねられる(背景を後から作り直すとき用)。

    Args:
        shape: 画像の (高さ, 幅)。
        placements, min_visible, min_pixels: compose と同じ。
        return_kept: True なら、正解に残った物体の番号も返す。

    Returns:
        (fg, alpha, 正解)。fg は (H, W, 3) の float32 で、透明度を掛けた物体の色(物体のない所は 0)。
        alpha は (H, W, 1) の float32 の透明度(0〜1)。正解は compose と同じ。
        return_kept が True のときは、4 つ目に、正解に残った物体の placements での番号のリスト(正解と同じ順)も返す。
    """
    H, W = shape
    full_masks, layers = [], []  # 物体ごとの、画像全体の大きさのマスクと、(色, 透明度)
    for p in placements:
        h, w = round(p.obj.image.shape[0] * p.scale), round(p.obj.image.shape[1] * p.scale)
        img = cv2.resize(despill(p.obj), (w, h), interpolation=cv2.INTER_AREA)
        m = cv2.resize(p.obj.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((H, W), bool)
        color = np.zeros((H, W, 3), np.uint8)
        ys, xs = slice(p.y, p.y + h), slice(p.x, p.x + w)
        full[ys, xs] = m[: H - p.y, : W - p.x]
        color[ys, xs] = img[: H - p.y, : W - p.x]
        full_masks.append(full)
        layers.append(color)

    # 見えている部分 = 自分のマスク − 後から(上に)貼る物体のマスク
    visible = []
    for i, m in enumerate(full_masks):
        above = np.zeros_like(m)
        for later in full_masks[i + 1:]:
            above |= later
        visible.append(m & ~above)
    keep = [i for i, (m, v) in enumerate(zip(full_masks, visible))
            if v.sum() >= min_pixels and v.sum() >= min_visible * m.sum()]

    # 奥から順に、透明度を 1 画素ぶんぼかして重ねる。fg・alpha は「背景 × (1 − alpha) + fg」になるように積み上げる
    fg = np.zeros((H, W, 3), np.float32)
    total = np.zeros((H, W, 1), np.float32)
    for i in keep:
        a = rearrange(cv2.GaussianBlur(full_masks[i].astype(np.float32), (3, 3), 0), "h w -> h w 1")  # (H, W) -> (H, W, 1)
        fg = fg * (1 - a) + layers[i].astype(np.float32) * a
        total = total * (1 - a) + a
    # 外した物体の分、上の物体に隠されていた部分が見えるので、見えている部分を残った物体だけで作り直す
    gts = []
    for k, i in enumerate(keep):
        above = np.zeros((H, W), bool)
        for j in keep[k + 1:]:
            above |= full_masks[j]
        gts.append((placements[i].obj.name, full_masks[i] & ~above))
    return (fg, total, gts, keep) if return_kept else (fg, total, gts)
