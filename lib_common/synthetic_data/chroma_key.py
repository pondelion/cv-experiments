"""無地の背景で生成した物体を、背景の色との差という決まった規則で抜き出す(① 生成だけ の ①-a。学習済みの認識モデルは使わない)。

規則:
    1. 画像の端の画素の色の中央値を、背景の色とする
    2. 背景の色との色の差(RGB の距離)が threshold より大きい画素を物体とする
    3. 小さな点(ノイズ)を消し、穴を埋める
物体の影・背景の色に近い色の物体・輪郭の数画素は、この規則の誤差になる。

あわせて、抜き出しに使えるかを判断する値(どれも認識モデルを使わない)を返す。
"""
import cv2
import numpy as np

THRESHOLD = 40.0  # 背景の色との差の閾値(RGB の距離、0〜441)


def key_mask(image: np.ndarray, threshold: float = THRESHOLD) -> tuple[np.ndarray, dict]:
    """背景の色との差で物体のマスクを作り、マスクの状態を表す値も返す。

    Args:
        image: (H, W, 3) の RGB の uint8 の画像。
        threshold: 背景の色との差の閾値。

    Returns:
        (mask, stats)。mask は (H, W) の bool。stats は次の値:
            bg_color: 背景の色(端の画素の中央値、RGB)
            bg_spread: 端の画素の、背景の色との差の 90% 点(小さいほど背景の色がそろっている)
            area_ratio: 物体の面積 ÷ 画像の面積
            n_components: 物体のまとまりの数(面積が全体の 1% 以上のもの)
            main_component_ratio: 一番大きいまとまりの面積 ÷ 物体の面積(1 に近いほど 1 つにまとまっている)
            touches_border: 物体が画像の端に触れているか(途中で切れている目安)
    """
    h, w = image.shape[:2]
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]]).astype(np.float32)  # (端の画素の数, 3)
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(image.astype(np.float32) - bg, axis=-1)  # (H, W) 背景の色との差
    mask = (diff > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 小さな点を消す
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # 小さな穴を埋める
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = cc_stats[1:, cv2.CC_STAT_AREA]
    total = int(mask.sum())
    stats = {
        "bg_color": [round(float(c)) for c in bg],
        "bg_spread": float(np.percentile(np.linalg.norm(border - bg, axis=-1), 90)),
        "area_ratio": total / (h * w),
        "n_components": int((areas >= 0.01 * h * w).sum()),
        "main_component_ratio": float(areas.max() / total) if total else 0.0,
        "touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
    }
    return mask.astype(bool), stats


def dominance_key_mask(image: np.ndarray, ratio: float = 0.4) -> tuple[np.ndarray, dict]:
    """背景の色の強い色(グリーンバックなら緑、マゼンタなら赤と青)が、弱い色よりどれだけ強いかで、背景と物体を分ける
    (クロマキーの基本のやり方)。

    影は暗くても色の偏りが残るので、RGB の距離で分ける key_mask と違い、影を背景に戻せる
    (マゼンタの背景で RGB の距離を使うと、鉢植えの 10 枚のうち 4 枚で、画像の 2〜4% の影が物体に入った)。
    代わりに、背景と同じ色に偏った物体(グリーンバックでの鉢植えの葉など)は背景と判定されやすい。

    規則: 背景の色の 3 つの値のうち、一番大きい値と一番小さい値の中間より大きいものを「強い色」、残りを「弱い色」とする。
    画素の「強い色の小さい方 − 弱い色の大きい方」が、背景の色の同じ値の ratio 倍より大きければ背景。
    グリーンバックでは「緑 − 赤と青の大きい方」、マゼンタでは「赤と青の小さい方 − 緑」になる。

    Args:
        image: (H, W, 3) の RGB の uint8 の画像。
        ratio: 背景と判定する強さの割合(0〜1。小さいほど、薄い緑・暗い緑まで背景にする)。

    Returns:
        (mask, stats)。stats は key_mask と同じ形。
    """
    h, w = image.shape[:2]
    img = image.astype(np.float32)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])  # (端の画素の数, 3)
    bg = np.median(border, axis=0)
    mid = (bg.max() + bg.min()) / 2
    strong = [i for i in range(3) if bg[i] > mid]  # 背景の色の強い色の番号(緑なら [1]、マゼンタなら [0, 2])
    weak = [i for i in range(3) if bg[i] <= mid]
    dominance = img[..., strong].min(axis=-1) - img[..., weak].max(axis=-1)  # (H, W) 強い色が弱い色よりどれだけ強いか
    bg_dominance = bg[strong].min() - bg[weak].max()
    mask = (dominance <= ratio * bg_dominance).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = cc_stats[1:, cv2.CC_STAT_AREA]
    total = int(mask.sum())
    stats = {
        "bg_color": [round(float(x)) for x in bg],
        "bg_spread": float(np.percentile(np.linalg.norm(border - bg, axis=-1), 90)),
        "area_ratio": total / (h * w),
        "n_components": int((areas >= 0.01 * h * w).sum()),
        "main_component_ratio": float(areas.max() / total) if total else 0.0,
        "touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
    }
    return mask.astype(bool), stats


def key_for_background(image: np.ndarray, background: str) -> tuple[np.ndarray, dict]:
    """背景の色に合った規則で抜き出す(色のついた背景は色の偏りの強さ、白は RGB の距離)。

    Args:
        image: 生成した画像 (H(縦), W(横), 3(RGB)) の uint8。
        background: 背景の色の名前(prompts.BACKGROUNDS のキー)。

    Returns:
        (物体のマスク (H, W) の bool, マスクの状態を表す値)(key_mask・dominance_key_mask と同じ)。
    """
    return key_mask(image) if background == "white" else dominance_key_mask(image)


def is_usable(stats: dict, max_bg_spread: float = 20.0, min_area: float = 0.02, max_area: float = 0.7) -> bool:
    """抜き出したマスクを、切り貼りの物体として使えるかの判定(決まった規則)。

    背景の色がそろっていて、物体が 1 つにまとまり、画像の端で切れておらず、大きさが極端でないもの。

    Args:
        stats: key_mask・dominance_key_mask が返したマスクの状態を表す値。
        max_bg_spread: 背景の色のばらつきの上限。
        min_area: 物体の面積の、画像の面積に対する割合の下限。
        max_area: 同じ割合の上限。

    Returns:
        使えるなら True。
    """
    return (stats["bg_spread"] <= max_bg_spread and stats["n_components"] == 1 and stats["main_component_ratio"] >= 0.95
            and not stats["touches_border"] and min_area <= stats["area_ratio"] <= max_area)
