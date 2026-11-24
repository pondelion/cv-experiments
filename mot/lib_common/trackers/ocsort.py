"""OC-SORT(Cao ら CVPR 2023、"Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking")。

参考: 公式のコード(noahcao/OC_SORT の `trackers/ocsort_tracker/ocsort.py`・`association.py`・`kalmanfilter.py`)。コードは写さず、
規則を読み取って書いた。SORT(sort.py)のカルマンフィルタ・IoU を土台に、次の 3 つを足す:
    - OCM(observation-centric momentum): 照合の費用に、動きの向きの一致を足す。物体の動きの向き(過去の観測どうしを結ぶ向き)と、
      物体の delta_t フレーム前の観測から検出への向きが近いほど費用を下げる。重みは inertia × 検出の確信度
    - OCR(observation-centric recovery): 1 段目で組にならなかった物体を、予測した枠ではなく最後に観測した枠で、残った検出ともう一度
      IoU で照合する(止まった物体や、予測がずれた物体を拾う)
    - ORU(observation-centric re-update): 見失っていた物体を再び照合できたとき、見失う前の状態に戻し、見失っていた間のフレームの枠を
      最後の観測と今の観測の間の直線で補って(中心・幅・高さを等間隔に)、カルマンフィルタの予測と更新をやり直す

MOT の仕組み(動画全体での物体の ID の管理):
    - 対応付け: 確信度が det_thresh を超える検出と、カルマンフィルタで予測した枠を、IoU + 動きの向きの一致(OCM)の費用で
      ハンガリアン法により組にする(IoU が iou_threshold(0.3)未満の組は捨てる)。残った物体と検出を、最後に観測した枠の IoU でもう
      一度組にする(OCR)
    - 見失った物体の保持(再出現): 照合できなかった物体は max_age(30)フレームまで残し、予測を続ける。最後に観測した枠でも照合する
      ので、止まって隠れた物体は同じ ID に戻りやすい。戻ったときに、見失っていた間の状態を直線で補ってやり直す(ORU)
    - 新しい物体(track の開始): どの物体とも組にならなかった検出から、新しい ID を始める。出力するのは、min_hits(3)回続けて
      照合できた物体(動画の最初の min_hits フレームは、照合できた物体を全部出す)
    - 消えた物体(track の終了): 照合できないフレームが max_age を超えた物体を消す
出力: 照合できた物体の、最後に観測した枠(検出の枠。カルマンフィルタの状態ではない)。公式の評価と同じく、面積が min_box_area 以下
と、横長(幅 ÷ 高さ > 1.6。歩行者のデータのとき)の枠は書かない。

設定は公式の既定の値(MOT17・DanceTrack の実行の設定と同じ。BYTE の照合(use_byte)は使わない)。det_thresh だけを val の役のセットで選ぶ。
"""
import copy
from dataclasses import dataclass

import numpy as np

from mot.lib_common.trackers.sort import iou_xyxy, lap_assignment, split_pairs, x_to_xyxy, xyxy_to_z

PLACEHOLDER = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])  # まだ観測がないことを表す枠(公式と同じ)


@dataclass
class OCSortConfig:
    """OC-SORT の設定(公式の既定の値)。

    Attributes:
        det_thresh: 使う検出の確信度の下限(これを超える検出だけを使う)。
        max_age: 照合できないまま残すフレームの数。
        min_hits: 出力するまでに続けて照合できる回数。
        iou_threshold: 組にしてよい IoU の下限。
        delta_t: 動きの向きを測るときに遡るフレームの数。
        inertia: 照合の費用の、動きの向きの一致の項の重み。
        min_box_area: 出力する枠の面積の下限。
        max_aspect: 出力する枠の幅 ÷ 高さの上限(None なら制限しない)。
    """
    det_thresh: float = 0.6
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    delta_t: int = 3
    inertia: float = 0.2
    min_box_area: float = 100.0
    max_aspect: float | None = 1.6


class ORUKalman:
    """ORU のできる枠のカルマンフィルタ(公式の KalmanFilterNew の、OC-SORT が使う部分。式と雑音の値は SORT と同じ)。

    観測がないフレーム(update(None))が始まったところで状態を保存し(freeze)、再び観測が来たら保存した状態に戻して、見失っていた間の
    観測を直線で補って更新し直す(unfreeze)。

    Args:
        z: 最初の観測 (中心 x, 中心 y, 面積, 縦横比)。
    """

    def __init__(self, z: np.ndarray):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7)
        self.x[:4] = z
        self.history_obs: list = []  # フレームごとの観測(観測がないフレームは None)
        self.observed = False
        self.saved: dict | None = None

    def predict(self) -> None:
        """1 フレーム先を予測する。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _correct(self, z: np.ndarray) -> None:
        """観測で直す(filterpy の update と同じ式)。

        Args:
            z: 観測 (4,)。
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        ikh = np.eye(7) - K @ self.H
        self.P = ikh @ self.P @ ikh.T + K @ self.R @ K.T

    def update(self, z: np.ndarray | None) -> None:
        """観測で直す。観測がない(None)なら直さず、見失い始めたところで状態を保存する。見失っていた後の観測なら ORU をしてから直す。

        Args:
            z: 観測 (4,) か None。
        """
        self.history_obs.append(z)
        if z is None:
            if self.observed:
                self.saved = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "saved"}
            self.observed = False
            return
        if not self.observed:
            self._unfreeze()
        self.observed = True
        self._correct(z)

    def _unfreeze(self) -> None:
        """ORU: 保存した状態に戻し、見失っていた間の観測を直線で補って、予測と更新をやり直す(公式の unfreeze)。

        補うのは (中心 x, 中心 y, 幅, 高さ) の空間で、最後の観測から今の観測までを等間隔に分ける。最後に補う観測は今の観測と同じで、
        その後に今の観測での通常の更新がもう一度かかる(公式と同じ)。
        """
        if self.saved is None:
            return
        new_history = list(self.history_obs)
        for k, v in self.saved.items():
            setattr(self, k, v)
        self.saved = None
        self.history_obs = self.history_obs[:-1]  # 保存したときに足していた None を除く
        seen = [i for i, z in enumerate(new_history) if z is not None]
        i1, i2 = seen[-2], seen[-1]
        x1, y1, s1, r1 = new_history[i1]
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        x2, y2, s2, r2 = new_history[i2]
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        gap = i2 - i1
        dx, dy, dw, dh = (x2 - x1) / gap, (y2 - y1) / gap, (w2 - w1) / gap, (h2 - h1) / gap
        for i in range(gap):
            w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
            self.update(np.array([x1 + (i + 1) * dx, y1 + (i + 1) * dy, w * h, w / float(h)]))
            if i != gap - 1:
                self.predict()


def speed_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """枠 a の中心から枠 b の中心への向き(長さ 1 の (縦, 横))。

    Args:
        a: 枠 (左, 上, 右, 下, …)。
        b: 枠 (左, 上, 右, 下, …)。

    Returns:
        (2,) の (縦の成分, 横の成分)。
    """
    d = np.array([(b[1] + b[3]) / 2 - (a[1] + a[3]) / 2, (b[0] + b[2]) / 2 - (a[0] + a[2]) / 2])
    return d / (np.sqrt((d ** 2).sum()) + 1e-6)


class OCSortTrack:
    """1 つの物体(1 つの ID)。公式の KalmanBoxTracker と同じく、最初の検出は観測の記録に入れない(最初の観測の記録は次の照合から)。

    Args:
        det: 最初の検出 (左, 上, 右, 下, 確信度)。
        track_id: ID。
        delta_t: 動きの向きを測るときに遡るフレームの数。
        extra: 検出に付けた情報(マスクなど)。
    """

    def __init__(self, det: np.ndarray, track_id: int, delta_t: int, extra: dict):
        self.kf = ORUKalman(xyxy_to_z(det))
        self.track_id, self.delta_t, self.extra, self.score = track_id, delta_t, extra, float(det[4])
        self.time_since_update, self.hits, self.hit_streak, self.age = 0, 0, 0, 0
        self.last_observation = PLACEHOLDER.copy()
        self.observations: dict[int, np.ndarray] = {}  # 照合したときの age → 検出
        self.velocity: np.ndarray | None = None

    def predict(self) -> np.ndarray:
        """1 フレーム先を予測し、予測した枠を返す(SORT と同じ)。

        Returns:
            予測した枠 (左, 上, 右, 下)。
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_xyxy(self.kf.x)

    def update(self, det: np.ndarray | None, extra: dict | None = None) -> None:
        """照合した検出で直す(None なら照合できなかったフレーム)。照合したときは、動きの向きも測り直す。

        Args:
            det: 検出 (左, 上, 右, 下, 確信度) か None。
            extra: 検出に付けた情報。
        """
        if det is None:
            self.kf.update(None)
            return
        if self.last_observation.sum() >= 0:  # 前の観測がある: delta_t フレーム前(なければそれに近い過去)の観測からの向き
            prev = next((self.observations[self.age - dt] for dt in range(self.delta_t, 0, -1)
                         if self.age - dt in self.observations), self.last_observation)
            self.velocity = speed_direction(prev, det)
        self.last_observation = det
        self.observations[self.age] = det
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score, self.extra = float(det[4]), extra if extra is not None else {}
        self.kf.update(xyxy_to_z(det))

    def k_previous_obs(self) -> np.ndarray:
        """動きの向きの一致を測るときの、過去の観測(delta_t フレーム前から近い順に探し、なければ最後の観測。公式の k_previous_obs)。

        Returns:
            (5,) の検出(観測がなければ PLACEHOLDER)。
        """
        if not self.observations:
            return PLACEHOLDER.copy()
        for dt in range(self.delta_t, 0, -1):
            if self.age - dt in self.observations:
                return self.observations[self.age - dt]
        return self.observations[max(self.observations)]

    @property
    def box(self) -> np.ndarray:
        """出力する枠 (左, 上, 幅, 高さ)(最後に観測した枠。観測がまだなければカルマンフィルタの状態)。"""
        b = x_to_xyxy(self.kf.x) if self.last_observation.sum() < 0 else self.last_observation[:4]
        return np.r_[b[:2], b[2:] - b[:2]]


def associate_ocm(dets: np.ndarray, trks: np.ndarray, iou_threshold: float, velocities: np.ndarray, previous_obs: np.ndarray,
                  inertia: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 段目の照合: IoU と動きの向きの一致(OCM)を足した値が大きい組を選ぶ(公式の associate)。

    動きの向きの一致: 物体の過去の観測(previous_obs)から各検出への向きと、物体の動きの向き(velocities)のなす角 θ から、
    (π/2 − θ) / π(向きが同じで +0.5、逆で −0.5)を作り、inertia と検出の確信度をかける。過去の観測がない物体は 0。

    Args:
        dets: 検出 (n, 5) の (左, 上, 右, 下, 確信度)。
        trks: 予測した枠 (m, 4)。
        iou_threshold: 組にしてよい IoU の下限。
        velocities: 物体の動きの向き (m, 2) の (縦, 横)。
        previous_obs: 物体の過去の観測 (m, 5)(観測がなければ PLACEHOLDER)。
        inertia: 動きの向きの一致の重み。

    Returns:
        (組 (k, 2) の (検出, 物体), 組にならなかった検出, 組にならなかった物体)。
    """
    if len(trks) == 0 or len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.arange(len(trks))
    cxd, cyd = (dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2
    cxp, cyp = (previous_obs[:, 0] + previous_obs[:, 2]) / 2, (previous_obs[:, 1] + previous_obs[:, 3]) / 2
    dx, dy = cxd[None, :] - cxp[:, None], cyd[None, :] - cyp[:, None]  # (m, n) 物体の過去の観測 → 検出
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    cos = np.clip(velocities[:, 1:2] * (dx / norm) + velocities[:, 0:1] * (dy / norm), -1, 1)
    angle = (np.pi / 2 - np.abs(np.arccos(cos))) / np.pi  # (m, n)
    valid = (previous_obs[:, 4] >= 0).astype(float)[:, None]  # 過去の観測がない物体は 0
    angle_cost = (valid * angle * inertia).T * dets[:, 4:5]  # (n, m) 検出の確信度をかける
    iou = iou_xyxy(dets[:, :4], trks)
    a = (iou > iou_threshold).astype(int)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        pairs = np.stack(np.nonzero(a), axis=1)
    else:
        pairs = lap_assignment(-(iou + angle_cost))
    return split_pairs(pairs, iou, iou_threshold, len(dets), len(trks))


class OCSortTracker:
    """OC-SORT の追跡(1 本の動画ごとに作り直す)。ByteTracker と同じ呼び方(update・keep_output)にする。

    Args:
        cfg: 設定。
        fps: 動画のフレームの速さ(OC-SORT は使わない。呼び方をそろえるため)。
    """

    def __init__(self, cfg: OCSortConfig, fps: float = 30.0):
        self.cfg, self.tracks, self.frame_count, self.next_id = cfg, [], 0, 1

    def update(self, boxes: np.ndarray, scores: np.ndarray, extras: list[dict] | None = None) -> list[OCSortTrack]:
        """1 フレーム分を処理する(公式の OCSort.update。use_byte なし)。

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
        keep = np.nonzero(scores > c.det_thresh)[0] if len(scores) else np.empty(0, dtype=int)
        dets = np.array([np.r_[boxes[i][:2], boxes[i][:2] + boxes[i][2:], scores[i]] for i in keep]).reshape(-1, 5)
        dext = [extras[i] for i in keep]
        pred = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        ok = ~np.isnan(pred).any(1)
        self.tracks, pred = [t for t, k in zip(self.tracks, ok) if k], pred[ok]
        vel = np.array([t.velocity if t.velocity is not None else np.zeros(2) for t in self.tracks]).reshape(-1, 2)
        last = np.array([t.last_observation for t in self.tracks]).reshape(-1, 5)
        kobs = np.array([t.k_previous_obs() for t in self.tracks]).reshape(-1, 5)
        # 1 段目: IoU + 動きの向きの一致(OCM)
        pairs, u_det, u_trk = associate_ocm(dets, pred, c.iou_threshold, vel, kobs, c.inertia)
        for d, t in pairs:
            self.tracks[t].update(dets[d], dext[d])
        # 2 段目(OCR): 残った物体の最後に観測した枠と、残った検出の IoU
        if len(u_det) and len(u_trk):
            iou = iou_xyxy(dets[u_det, :4], last[u_trk, :4])
            if iou.max() > c.iou_threshold:
                done_d, done_t = [], []
                for i, j in lap_assignment(-iou):
                    if iou[i, j] < c.iou_threshold:
                        continue
                    self.tracks[u_trk[j]].update(dets[u_det[i]], dext[u_det[i]])
                    done_d.append(u_det[i]); done_t.append(u_trk[j])
                u_det, u_trk = np.setdiff1d(u_det, done_d), np.setdiff1d(u_trk, done_t)
        for t in u_trk:
            self.tracks[t].update(None)
        for d in u_det:
            self.tracks.append(OCSortTrack(dets[d], self.next_id, c.delta_t, dext[d]))
            self.next_id += 1
        out = [t for t in self.tracks if t.time_since_update < 1 and (t.hit_streak >= c.min_hits or self.frame_count <= c.min_hits)]
        self.tracks = [t for t in self.tracks if t.time_since_update <= c.max_age]
        return out

    def keep_output(self, t: OCSortTrack) -> bool:
        """出力に書くか(公式の評価の、面積と縦横比の条件)。

        Args:
            t: 物体。

        Returns:
            書くなら True。
        """
        w, h = t.box[2:]
        return w * h > self.cfg.min_box_area and (self.cfg.max_aspect is None or w / h <= self.cfg.max_aspect)
