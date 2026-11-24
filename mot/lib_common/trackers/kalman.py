"""枠のカルマンフィルタ(DeepSORT・ByteTrack の公式の KalmanFilter と同じ式)。

状態は 8 次元 (cx, cy, a, h, vcx, vcy, va, vh)(中心・縦横比 = 幅 ÷ 高さ・高さと、その速度)。等速の動きを仮定し、観測は
(cx, cy, a, h)。ノイズの大きさは高さ h に比例する(位置の重み 1/20、速度の重み 1/160。縦横比のノイズは小さい定数)。
"""
import numpy as np
import scipy.linalg

STD_POS, STD_VEL = 1.0 / 20, 1.0 / 160


class KalmanFilter:
    """枠の等速のカルマンフィルタ。"""

    def __init__(self):
        self.motion = np.eye(8)
        self.motion[:4, 4:] = np.eye(4)  # 1 フレームで 位置 += 速度
        self.observe = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """最初の観測から状態を作る(速度は 0)。

        Args:
            z: 観測 (cx, cy, a, h)。

        Returns:
            (平均 (8,), 共分散 (8, 8))。
        """
        h = z[3]
        std = [2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h,
               10 * STD_VEL * h, 10 * STD_VEL * h, 1e-5, 10 * STD_VEL * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """1 フレーム先を予測する(まとめて (N, 8)・(N, 8, 8) でもよい)。

        Args:
            mean: 平均 (8,) か (N, 8)。
            cov: 共分散 (8, 8) か (N, 8, 8)。

        Returns:
            (予測の平均, 予測の共分散)(入力と同じ形)。
        """
        single = mean.ndim == 1
        mean, cov = np.atleast_2d(mean), cov.reshape(-1, 8, 8)
        h = mean[:, 3]
        std = np.stack([STD_POS * h, STD_POS * h, np.full_like(h, 1e-2), STD_POS * h,
                        STD_VEL * h, STD_VEL * h, np.full_like(h, 1e-5), STD_VEL * h], 1)  # (N, 8)
        q = np.stack([np.diag(s) for s in np.square(std)])
        mean = mean @ self.motion.T
        cov = self.motion @ cov @ self.motion.T + q
        return (mean[0], cov[0]) if single else (mean, cov)

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """観測で直す。

        Args:
            mean: 平均 (8,)。
            cov: 共分散 (8, 8)。
            z: 観測 (cx, cy, a, h)。

        Returns:
            (直した平均, 直した共分散)。
        """
        h = mean[3]
        r = np.diag(np.square([STD_POS * h, STD_POS * h, 1e-1, STD_POS * h]))  # 観測のノイズ
        proj_mean, proj_cov = self.observe @ mean, self.observe @ cov @ self.observe.T + r
        chol = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(chol, (cov @ self.observe.T).T, check_finite=False).T  # (8, 4)
        return mean + (z - proj_mean) @ gain.T, cov - gain @ proj_cov @ gain.T
