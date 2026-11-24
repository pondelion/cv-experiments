"""ByteTrack(Zhang ら ECCV 2022、"ByteTrack: Multi-Object Tracking by Associating Every Detection Box")。

参考: 公式のコード(ifzhang/ByteTrack の `yolox/tracker/byte_tracker.py`・`kalman_filter.py`・`matching.py`、
`yolox/evaluators/mot_evaluator.py` の結果の書き出し、`tools/track.py` の既定の値)。コードは写さず、規則を読み取って書いた。

MOT の仕組み(動画全体での物体の ID の管理):
    - 対応付け: 枠の IoU(と検出の確信度)。追跡している物体はカルマンフィルタ(kalman.py)で今のフレームの位置を予測し、
      費用 = 1 − IoU(1 段目と、確定していない物体の照合は IoU × 確信度で弱める fuse_score)の閾値つきのハンガリアン法で決める。
      段階は 3 つ: (1) 確信度の高い検出(> track_thresh)と、追跡している物体 + 見失った物体、(2) 1 段目で残った「追跡している」
      物体と、確信度の低い検出(0.1 < 確信度 < track_thresh。隠れて確信度の下がった物体を拾う。費用の閾値 0.5)、
      (3) 前のフレームで始めたばかりで確定していない物体と、1 段目で残った確信度の高い検出(閾値 0.7)
    - 見失った物体の保持(再出現): 照合できなかった物体は「見失った」状態にして track_buffer × fps / 30 フレーム残す。残している間も
      カルマンフィルタで位置を予測し続け(高さの速度は 0 にする)、1 段目で確信度の高い検出と照合できたら同じ ID に戻す。
      手がかりは予測した枠の IoU だけなので、長く隠れて予測の位置がずれると戻れない
    - 新しい物体(track の開始): どの物体とも照合できなかった確信度の高い検出のうち、確信度 ≥ track_thresh + 0.1 のものから新しい ID を
      始める。次のフレームで照合できたら確定し、出力する(動画の最初のフレームは、始めた時点で確定)
    - 消えた物体(track の終了): 見失ってから保持のフレームの数を超えた物体、確定する前に照合できなかった物体を終える。
      追跡している物体と見失った物体の枠がほぼ重なる(IoU > 0.85)ときは、続いた時間の短い方を消す(公式の remove_duplicate_stracks)
出力(公式の mot_evaluator と同じ): 確定して追跡している物体の、カルマンフィルタで直した枠。面積 ≤ min_box_area と、
横長(幅 ÷ 高さ > 1.6。歩行者のデータのとき)の枠は書かない。

公式との違い: 公式の評価(mot_evaluator)は MOT17 の test の動画ごとに track_thresh・track_buffer を手で変えている。ここでは動画ごとに
変えず、val の役のセットで 1 組を選ぶ(mot/README.md の方針)。
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mot.lib_common.trackers.assignment import box_iou, linear_assignment
from mot.lib_common.trackers.kalman import KalmanFilter


class State(Enum):
    """物体の状態。"""
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class ByteTrackConfig:
    """ByteTrack の設定(既定は公式の tools/track.py の値)。

    Attributes:
        track_thresh: 確信度の高い検出の閾値(新しい物体の開始は + 0.1 以上)。
        low_thresh: 確信度の低い検出の下限(公式は 0.1 で固定)。
        match_thresh: 1 段目の照合の費用(1 − IoU × 確信度)の上限。
        second_thresh: 2 段目の照合の費用(1 − IoU)の上限。
        unconfirmed_thresh: 確定していない物体の照合の費用の上限。
        track_buffer: 見失った物体を残すフレームの数(30 fps のとき。fps に比例して変える)。
        fuse_score: 1 段目と確定していない物体の照合の費用に確信度を掛けるか(公式は MOT20 だけ使わない)。
        min_box_area: 出力する枠の面積の下限。
        max_aspect: 出力する枠の幅 ÷ 高さの上限(None なら制限しない。公式は歩行者のデータで 1.6)。
        duplicate_iou: 追跡している物体と見失った物体の枠が重なりすぎとみなす IoU(公式は 1 − 0.15)。
    """
    track_thresh: float = 0.6
    low_thresh: float = 0.1
    match_thresh: float = 0.9
    second_thresh: float = 0.5
    unconfirmed_thresh: float = 0.7
    track_buffer: int = 30
    fuse_score: bool = True
    min_box_area: float = 100.0
    max_aspect: float | None = 1.6
    duplicate_iou: float = 0.85


def tlwh_to_xyah(b: np.ndarray) -> np.ndarray:
    """(左, 上, 幅, 高さ) → (中心 x, 中心 y, 幅 ÷ 高さ, 高さ)。

    Args:
        b: (4,)。

    Returns:
        (4,)。
    """
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2, b[2] / b[3], b[3]])


@dataclass(eq=False)  # 物体は同じものかどうかを id(オブジェクトの同一性)で比べる
class Track:
    """1 つの物体(1 つの ID)。

    Attributes:
        tlwh: 検出の枠 (左, 上, 幅, 高さ)(カルマンフィルタを始める前の値)。
        score: 最後に照合した検出の確信度。
        track_id: ID(始めたときに振る)。
        mean: カルマンフィルタの平均 (8,)。
        cov: カルマンフィルタの共分散 (8, 8)。
        state: 状態。
        is_activated: 確定したか。
        frame_id: 最後に照合したフレーム。
        start_frame: 始めたフレーム。
    """
    tlwh: np.ndarray
    score: float
    track_id: int = -1
    mean: np.ndarray | None = None
    cov: np.ndarray | None = None
    state: State = State.TRACKED
    is_activated: bool = False
    frame_id: int = 0
    start_frame: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def box(self) -> np.ndarray:
        """今の枠 (左, 上, 幅, 高さ)(カルマンフィルタを始めた後はその平均から作る)。"""
        if self.mean is None:
            return self.tlwh.copy()
        cx, cy, a, h = self.mean[:4]
        return np.array([cx - a * h / 2, cy - h / 2, a * h, h])

    @property
    def tlbr(self) -> np.ndarray:
        """今の枠 (左, 上, 右, 下)。"""
        b = self.box
        return np.r_[b[:2], b[:2] + b[2:]]


class ByteTracker:
    """ByteTrack の追跡(1 本の動画ごとに作り直す)。

    Args:
        cfg: 設定。
        fps: 動画のフレームの速さ(見失った物体を残すフレームの数に使う)。
    """

    def __init__(self, cfg: ByteTrackConfig, fps: float = 30.0):
        self.cfg, self.kf = cfg, KalmanFilter()
        self.max_lost = int(fps / 30.0 * cfg.track_buffer)
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.removed_ids: set[int] = set()  # 前のフレームまでに終えた物体の ID(公式の removed_stracks)
        self.frame_id, self.next_id = 0, 1

    def _predict(self, tracks: list[Track]) -> None:
        """物体の位置を 1 フレーム先に予測する(見失った物体は高さの速度を 0 にしてから。公式と同じ)。

        Args:
            tracks: 物体の並び(その場で書き換える)。
        """
        if not tracks:
            return
        mean = np.stack([t.mean.copy() for t in tracks])
        mean[[i for i, t in enumerate(tracks) if t.state != State.TRACKED], 7] = 0
        mean, cov = self.kf.predict(mean, np.stack([t.cov for t in tracks]))
        for t, m, c in zip(tracks, mean, cov):
            t.mean, t.cov = m, c

    def _cost(self, tracks: list[Track], dets: list[Track], fuse: bool) -> np.ndarray:
        """照合の費用 1 − IoU(fuse なら 1 − IoU × 検出の確信度)。

        Args:
            tracks: 物体の並び。
            dets: 検出の並び。
            fuse: 確信度を掛けるか。

        Returns:
            (物体の数, 検出の数)。
        """
        iou = box_iou(np.array([t.tlbr for t in tracks]).reshape(-1, 4), np.array([d.tlbr for d in dets]).reshape(-1, 4))
        if fuse and len(dets):
            iou = iou * np.array([d.score for d in dets])[None, :]
        return 1 - iou

    def _update(self, t: Track, d: Track, refind: bool) -> None:
        """物体を検出で直す(公式の update・re_activate)。

        Args:
            t: 物体。
            d: 照合した検出。
            refind: 見失った物体を戻すときは True。
        """
        t.mean, t.cov = self.kf.update(t.mean, t.cov, tlwh_to_xyah(d.tlwh))
        t.state, t.is_activated, t.frame_id, t.score = State.TRACKED, True, self.frame_id, d.score
        t.extra = d.extra

    def update(self, boxes: np.ndarray, scores: np.ndarray, extras: list[dict] | None = None) -> list[Track]:
        """1 フレーム分を処理する(公式の BYTETracker.update)。

        Args:
            boxes: 検出の枠 (N, 4) の (左, 上, 幅, 高さ)(元の画像の画素)。
            scores: 検出の確信度 (N,)。
            extras: 検出ごとに持たせる情報(マスクなど。出力の物体の extra に写す)か None。

        Returns:
            このフレームで出力する物体(確定して追跡しているもの)の並び。
        """
        c = self.cfg
        self.frame_id += 1
        extras = extras if extras is not None else [{} for _ in range(len(boxes))]
        make = lambda idx: [Track(np.asarray(boxes[i], float), float(scores[i]), extra=extras[i]) for i in idx]
        high = make(np.nonzero(scores > c.track_thresh)[0])
        low = make(np.nonzero((scores > c.low_thresh) & (scores < c.track_thresh))[0])
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        activated, refound, newly_lost, removed = [], [], [], []
        # 1 段目: 追跡している物体 + 見失った物体 と 確信度の高い検出
        pool = confirmed + [t for t in self.lost if t not in confirmed]
        self._predict(pool)
        pairs, u_track, u_det = linear_assignment(self._cost(pool, high, c.fuse_score), c.match_thresh)
        for ti, di in pairs:
            t = pool[ti]
            (activated if t.state == State.TRACKED else refound).append(t)
            self._update(t, high[di], refind=t.state != State.TRACKED)
        # 2 段目: 残った「追跡している」物体 と 確信度の低い検出
        rest = [pool[i] for i in u_track if pool[i].state == State.TRACKED]
        pairs, u_rest, _ = linear_assignment(self._cost(rest, low, False), c.second_thresh)
        for ti, di in pairs:
            activated.append(rest[ti])
            self._update(rest[ti], low[di], refind=False)
        for i in u_rest:
            rest[i].state = State.LOST
            newly_lost.append(rest[i])
        # 3 段目: 確定していない物体 と 1 段目で残った確信度の高い検出
        high = [high[i] for i in u_det]
        pairs, u_unc, u_det = linear_assignment(self._cost(unconfirmed, high, c.fuse_score), c.unconfirmed_thresh)
        for ti, di in pairs:
            activated.append(unconfirmed[ti])
            self._update(unconfirmed[ti], high[di], refind=False)
        for i in u_unc:
            unconfirmed[i].state = State.REMOVED
            removed.append(unconfirmed[i])
        # 新しい物体
        for i in u_det:
            d = high[i]
            if d.score < c.track_thresh + 0.1:
                continue
            d.track_id, self.next_id = self.next_id, self.next_id + 1
            d.mean, d.cov = self.kf.initiate(tlwh_to_xyah(d.tlwh))
            d.state, d.is_activated, d.frame_id, d.start_frame = State.TRACKED, self.frame_id == 1, self.frame_id, self.frame_id
            activated.append(d)
        # 見失って保持のフレームの数を超えた物体を終える
        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_lost:
                t.state = State.REMOVED
                removed.append(t)
        # 状態の並びを更新する(公式の joint_stracks・sub_stracks と同じ順)。見失った物体の並びから除くのは、前のフレームまでに
        # 終えた物体だけ(公式は、このフレームで終えた物体を除いた後で終えた物体の記録に足す)。そのため、このフレームで終えた物体は
        # 次のフレームまで見失った物体の並びに残り、次のフレームの 1 段目の照合で同じ ID に戻ることがある(公式と同じ動き)
        tracked = [t for t in self.tracked if t.state == State.TRACKED]
        for t in activated + refound:
            if t not in tracked:
                tracked.append(t)
        lost = [t for t in self.lost if t not in tracked] + newly_lost
        lost = [t for t in lost if t.track_id not in self.removed_ids]
        self.removed_ids.update(t.track_id for t in removed)
        self.tracked, self.lost = self._remove_duplicates(tracked, lost)
        return [t for t in self.tracked if t.is_activated]

    def _remove_duplicates(self, tracked: list[Track], lost: list[Track]) -> tuple[list[Track], list[Track]]:
        """追跡している物体と見失った物体の枠がほぼ重なる組で、続いた時間の短い方を消す(公式の remove_duplicate_stracks)。

        Args:
            tracked: 追跡している物体。
            lost: 見失った物体。

        Returns:
            (残した追跡している物体, 残した見失った物体)。
        """
        if not tracked or not lost:
            return tracked, lost
        iou = box_iou(np.array([t.tlbr for t in tracked]), np.array([t.tlbr for t in lost]))
        drop_a, drop_b = set(), set()
        for p, q in zip(*np.nonzero(iou > self.cfg.duplicate_iou)):
            if tracked[p].frame_id - tracked[p].start_frame > lost[q].frame_id - lost[q].start_frame:
                drop_b.add(q)
            else:
                drop_a.add(p)
        return [t for i, t in enumerate(tracked) if i not in drop_a], [t for i, t in enumerate(lost) if i not in drop_b]

    def keep_output(self, t: Track) -> bool:
        """出力に書くか(公式の mot_evaluator の面積と縦横比の条件)。

        Args:
            t: 物体。

        Returns:
            書くなら True。
        """
        w, h = t.box[2:]
        return w * h > self.cfg.min_box_area and (self.cfg.max_aspect is None or w / h <= self.cfg.max_aspect)
