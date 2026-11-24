"""閾値つきの 1 対 1 の割り当てと、枠の IoU。

割り当ては ByteTrack・DeepSORT 系の公式の linear_assignment(lap.lapjv(extend_cost=True, cost_limit=閾値))と同じ問題を
scipy の linear_sum_assignment で解く: 費用の行列 C (n, m) に、「割り当てない」を表す枠を足した (n + m) × (n + m) の行列
    [[C, 対角だけ 閾値 / 2 の (n, n)], [対角だけ 閾値 / 2 の (m, m), 0 の (m, n)]]
で最小の費用の 1 対 1 を求め、C の中で選ばれた組を割り当てにする(組の費用が閾値を超えると、両方を割り当てない方が安くなる)。
lap の拡張と同じ作り方なので、解は同じ最小の費用になる。ただし費用が同じ組がいくつもあるとき(ほぼ同じ検出が重なるときなど)に
どれを選ぶかは lap と違うことがあるので、lap が入っていれば lap を使う(公式の ByteTrack は lap だけを使う)。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e9  # 選べない組の費用


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """費用が閾値以下になる組だけを使う 1 対 1 の割り当て。

    Args:
        cost: 費用 (n, m)。
        thresh: 組にしてよい費用の上限。

    Returns:
        (組 (k, 2) の (行, 列), 割り当てなかった行, 割り当てなかった列)。
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int), np.arange(n), np.arange(m)
    try:  # 公式と同じ lap があれば使う(費用が同じ組がいくつもあるとき、どれを選ぶかまで公式と同じになる)
        import lap
        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
        pairs = np.array([[i, j] for i, j in enumerate(x) if j >= 0], dtype=int).reshape(-1, 2)
        return pairs, np.nonzero(x < 0)[0], np.nonzero(y < 0)[0]
    except ImportError:
        pass
    big = np.full((n + m, n + m), _BIG)
    big[:n, :m] = np.where(cost <= thresh, cost, _BIG)
    big[:n, m:][np.arange(n), np.arange(n)] = thresh / 2
    big[n:, :m][np.arange(m), np.arange(m)] = thresh / 2
    big[n:, m:] = 0
    rows, cols = linear_sum_assignment(big)
    pairs = np.array([(r, c) for r, c in zip(rows, cols) if r < n and c < m and cost[r, c] <= thresh], dtype=int).reshape(-1, 2)
    return pairs, np.setdiff1d(np.arange(n), pairs[:, 0]), np.setdiff1d(np.arange(m), pairs[:, 1])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """枠どうしの IoU(ByteTrack の公式が使う cython_bbox.bbox_overlaps と同じく、幅・高さ・重なりに +1 をする画素の数え方)。

    Args:
        a: (n, 4) の (左, 上, 右, 下)。
        b: (m, 4) の (左, 上, 右, 下)。

    Returns:
        (n, m)。
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt + 1, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0] + 1) * (x[:, 3] - x[:, 1] + 1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)
