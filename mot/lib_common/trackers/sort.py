"""SORT(Bewley ら ICIP 2016、"Simple Online and Realtime Tracking")。

参考: 公式のコード(abewley/sort の `sort.py`)。コードは写さず、規則を読み取って書いた。

MOT の仕組み(動画全体での物体の ID の管理):
    - 対応付け: 各物体の枠をカルマンフィルタで今のフレームに予測し、予測した枠と検出の枠の IoU を費用にしたハンガリアン法で 1 対 1 に
      組にする(IoU が iou_threshold(0.3)未満の組は捨てる)。手がかりは枠の位置と大きさだけ
    - 見失った物体の保持(再出現): 照合できなかった物体も、max_age(1)フレームまでは残し、カルマンフィルタの予測を続ける。既定の
      max_age 1 では、1 フレームより長く隠れた物体は消え、再び写っても新しい ID になる
    - 新しい物体(track の開始): どの物体とも照合できなかった検出から、すぐ新しい ID を始める。出力するのは、min_hits(3)回続けて
      照合できた物体(動画の最初の min_hits フレームは、照合できた物体を全部出す)
    - 消えた物体(track の終了): 照合できないフレームが max_age を超えた物体を消す
出力: 照合できた物体の、カルマンフィルタで直した後の枠。

公式との違い: 公式は渡された検出を全部使う(検出を確信度で選ばない)。ここでは、確信度の低い検出が多い自作の画像のモデルに合わせて、
確信度の下限 det_thresh を足して val の役のセットで選ぶ(det_thresh 0 が公式と同じ)。
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """枠どうしの IoU(SORT・OC-SORT の公式の iou_batch と同じく、画素の +1 をしない連続の座標の数え方)。

    Args:
        a: (n, 4) の (左, 上, 右, 下)。
        b: (m, 4) の (左, 上, 右, 下)。

    Returns:
        (n, m) の IoU。
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    union = area(a)[:, None] + area(b)[None, :] - inter
    # 幅か高さが 0 の枠どうし(画像のモデルの後処理で、負の幅・高さを 0 にした枠)は和が 0 になるので、IoU を 0 にする
    # (公式はこの場合 0 ÷ 0 になる。そのままだとハンガリアン法が値の壊れた行列で止まる)
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def xyxy_to_z(b: np.ndarray) -> np.ndarray:
    """枠 (左, 上, 右, 下) を、カルマンフィルタの観測 (中心 x, 中心 y, 面積 s, 縦横比 r = 幅 ÷ 高さ) にする。

    Args:
        b: (4,) 以上(5 個目以降は使わない)。

    Returns:
        (4,)。
    """
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / float(h)])


def x_to_xyxy(x: np.ndarray) -> np.ndarray:
    """カルマンフィルタの状態の先頭 4 個 (中心 x, 中心 y, 面積, 縦横比) を、枠 (左, 上, 右, 下) にする。

    Args:
        x: (4,) 以上。

    Returns:
        (4,)。
    """
    # 公式の convert_x_to_bbox と同じ式。面積 × 縦横比が負なら幅が NaN になり、その物体は予測が壊れたとして消される(update の中)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class BoxKalman:
    """SORT・OC-SORT の枠のカルマンフィルタ(公式は filterpy の KalmanFilter に、下の値を入れて使う)。

    状態は 7 次元 (中心 x, 中心 y, 面積 s, 縦横比 r, x の速度, y の速度, 面積の速度)。縦横比は一定とみなす(速度なし)。観測は先頭 4 個。
    雑音の大きさ: filterpy の既定(観測の雑音 R・状態の雑音 Q・共分散 P がどれも単位行列)に、公式と同じ倍率をかける
    (R の面積・縦横比 ×10、P の速度 ×1000 してから全体 ×10、Q の面積の速度と速度の部分 ×0.01)。

    Args:
        z: 最初の観測 (4,)。
    """

    def __init__(self, z: np.ndarray):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0  # 1 フレームで 位置 += 速度
        self.H = np.eye(4, 7)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0  # 速度は最初わからないので大きくする
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7)
        self.x[:4] = z

    def predict(self) -> None:
        """1 フレーム先を予測する(面積が負にならないように、予測で面積が 0 以下になるときは面積の速度を 0 にする。公式と同じ)。"""
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """観測で直す(filterpy の update と同じ式。共分散は Joseph の形で更新する)。

        Args:
            z: 観測 (4,)。
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        ikh = np.eye(7) - K @ self.H
        self.P = ikh @ self.P @ ikh.T + K @ self.R @ K.T


@dataclass
class SortConfig:
    """SORT の設定(公式の既定の値)。

    Attributes:
        det_thresh: 使う検出の確信度の下限(公式にはない。0 なら全部使う)。
        max_age: 照合できないまま残すフレームの数。
        min_hits: 出力するまでに続けて照合できる回数。
        iou_threshold: 組にしてよい IoU の下限。
    """
    det_thresh: float = 0.0
    max_age: int = 1
    min_hits: int = 3
    iou_threshold: float = 0.3


class SortTrack:
    """1 つの物体(1 つの ID)。

    Args:
        box: 最初の検出の枠 (左, 上, 右, 下)。
        score: 最初の検出の確信度。
        track_id: ID。
        extra: 検出に付けた情報(マスクなど)。
    """

    def __init__(self, box: np.ndarray, score: float, track_id: int, extra: dict):
        self.kf = BoxKalman(xyxy_to_z(box))
        self.track_id, self.score, self.extra = track_id, score, extra
        self.time_since_update, self.hits, self.hit_streak, self.age = 0, 0, 0, 0

    def predict(self) -> np.ndarray:
        """1 フレーム先を予測し、予測した枠を返す(照合できないフレームがあれば、続けて照合できた回数を 0 に戻す)。

        Returns:
            予測した枠 (左, 上, 右, 下)。
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_xyxy(self.kf.x)

    def update(self, box: np.ndarray, score: float, extra: dict) -> None:
        """照合した検出で直す。

        Args:
            box: 検出の枠 (左, 上, 右, 下)。
            score: 検出の確信度。
            extra: 検出に付けた情報。
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(xyxy_to_z(box))
        self.score, self.extra = score, extra

    @property
    def box(self) -> np.ndarray:
        """出力する枠 (左, 上, 幅, 高さ)(カルマンフィルタの状態から作る。公式の get_state と同じ)。"""
        b = x_to_xyxy(self.kf.x)
        return np.r_[b[:2], b[2:] - b[:2]]


def associate(dets: np.ndarray, trks: np.ndarray, iou_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """検出と予測した枠を IoU で 1 対 1 に組にする(公式の associate_detections_to_trackers)。

    IoU が閾値を超える組が、どの検出・どの物体についても 1 つ以下なら、その組をそのまま使う。そうでなければ −IoU を費用にした
    ハンガリアン法で組にする。最後に IoU が閾値未満の組を捨てる。

    Args:
        dets: 検出の枠 (n, 4) の (左, 上, 右, 下)。
        trks: 予測した枠 (m, 4)。
        iou_threshold: 組にしてよい IoU の下限。

    Returns:
        (組 (k, 2) の (検出, 物体), 組にならなかった検出, 組にならなかった物体)。
    """
    if len(trks) == 0 or len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.arange(len(trks))
    iou = iou_xyxy(dets, trks)
    a = (iou > iou_threshold).astype(int)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        pairs = np.stack(np.nonzero(a), axis=1)
    else:
        pairs = lap_assignment(-iou)
    return split_pairs(pairs, iou, iou_threshold, len(dets), len(trks))


def lap_assignment(cost: np.ndarray) -> np.ndarray:
    """最小の費用の 1 対 1 の割り当て(SORT・OC-SORT の公式の linear_assignment と同じ)。

    公式は lap が入っていれば lap.lapjv(extend_cost=True)を使い、なければ scipy の linear_sum_assignment を使う。lapjv の
    extend_cost は、行と列の数が違う行列に費用 0 の行・列を足して正方形にして解く。そのため、どの列とも費用が 0 以上になる行
    (IoU が 0 の組しかない検出など)は、足した列(費用 0)に割り当てられて「組なし」になる。scipy は行と列の少ない方を全部どれかと
    組にするので、組の選び方と、組にならなかったものの並びが lap と違うことがある(新しい ID の番号の順が変わる)。公式と同じ結果に
    するため、lap があれば lap を使う。

    Args:
        cost: 費用 (n, m)。

    Returns:
        組 (k, 2) の (行, 列)(行の順)。
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0], dtype=int).reshape(-1, 2)
    except ImportError:
        return np.stack(linear_sum_assignment(cost), axis=1)


def split_pairs(pairs: np.ndarray, iou: np.ndarray, iou_threshold: float, n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """割り当てた組から IoU が閾値未満の組を外し、組にならなかったものを公式と同じ順に並べる(SORT・OC-SORT の公式と同じ)。

    組にならなかった検出の並びは、どの組にも入らなかった検出(番号の順)の後に、IoU が足りずに外した組の検出(組の順)を続ける。
    新しい ID はこの順に振るので、順が違うと同じフレームで始まる物体どうしの ID の番号が入れ替わる。

    Args:
        pairs: 割り当てた組 (k, 2) の (検出, 物体)。
        iou: IoU (n, m)。
        iou_threshold: 組にしてよい IoU の下限。
        n: 検出の数。
        m: 物体の数。

    Returns:
        (残した組 (k', 2), 組にならなかった検出, 組にならなかった物体)。
    """
    pairs = pairs.reshape(-1, 2)
    good = [p for p in pairs if iou[p[0], p[1]] >= iou_threshold]
    bad = [p for p in pairs if iou[p[0], p[1]] < iou_threshold]
    u_det = [d for d in range(n) if d not in pairs[:, 0]] + [int(p[0]) for p in bad]
    u_trk = [t for t in range(m) if t not in pairs[:, 1]] + [int(p[1]) for p in bad]
    return np.array(good, dtype=int).reshape(-1, 2), np.array(u_det, dtype=int), np.array(u_trk, dtype=int)


class SortTracker:
    """SORT の追跡(1 本の動画ごとに作り直す)。ByteTracker と同じ呼び方(update・keep_output)にする。

    Args:
        cfg: 設定。
        fps: 動画のフレームの速さ(SORT は使わない。呼び方をそろえるため)。
    """

    def __init__(self, cfg: SortConfig, fps: float = 30.0):
        self.cfg, self.tracks, self.frame_count, self.next_id = cfg, [], 0, 1

    def update(self, boxes: np.ndarray, scores: np.ndarray, extras: list[dict] | None = None) -> list[SortTrack]:
        """1 フレーム分を処理する(公式の Sort.update)。

        Args:
            boxes: 検出の枠 (N, 4) の (左, 上, 幅, 高さ)。
            scores: 検出の確信度 (N,)。
            extras: 検出ごとに持たせる情報か None。

        Returns:
            このフレームで出力する物体の並び。
        """
        c = self.cfg
        self.frame_count += 1
        extras = extras if extras is not None else [{} for _ in range(len(boxes))]
        keep = np.nonzero(scores >= c.det_thresh)[0] if len(scores) else np.empty(0, dtype=int)
        dets = np.array([np.r_[boxes[i][:2], boxes[i][:2] + boxes[i][2:]] for i in keep]).reshape(-1, 4)
        pred = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        ok = ~np.isnan(pred).any(1)  # 予測が壊れた物体を除く(公式と同じ)
        self.tracks, pred = [t for t, k in zip(self.tracks, ok) if k], pred[ok]
        pairs, u_det, _ = associate(dets, pred, c.iou_threshold)
        for d, t in pairs:
            self.tracks[t].update(dets[d], float(scores[keep[d]]), extras[keep[d]])
        for d in u_det:
            self.tracks.append(SortTrack(dets[d], float(scores[keep[d]]), self.next_id, extras[keep[d]]))
            self.next_id += 1
        out = [t for t in self.tracks if t.time_since_update < 1 and (t.hit_streak >= c.min_hits or self.frame_count <= c.min_hits)]
        self.tracks = [t for t in self.tracks if t.time_since_update <= c.max_age]
        return out

    def keep_output(self, t: SortTrack) -> bool:
        """出力に書くか(SORT の公式は枠の大きさ・縦横比で選ばないので、いつも書く)。

        Args:
            t: 物体。

        Returns:
            いつも True。
        """
        return True
