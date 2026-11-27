"""追跡の中の ReID の評価のデータ: 追跡のデータの正解の枠(動画を通した ID つき)から、問い(query)と探す先(gallery)を作る。

ReID のベンチマーク(複数のカメラの間で人を探す)は使わず、追跡のデータの同じ動画の中で、隠れる前と後の同じ物体を見分けられるかを
測る(mot/exp00201 の README の「評価」)。
    - 正解の物体: MOT17・DanceTrack は gt.txt の「評価する」印(7 列目)が 1 で、クラスが歩行者(8 列目が 1)の行(TrackEval の
      MotChallenge2DBox と同じ)。KITTI MOTS は車(クラス 1)と歩行者(クラス 2)のマスク(ID 10000 の無視の領域は除く)を囲む枠
    - 問いと探す先: 動画ごとに、stride フレームおきのフレームの物体の切り出しを使う。問い 1 つに対して、探す先は同じ動画の、
      問いのフレームから gap フレーム以上離れたフレームの切り出し(同じ物体も違う物体も含む)。すぐ隣のフレームは同じ物体の
      ほぼ同じ画像になり簡単すぎるので、gap で除く
形の記号: N = 切り出しの数。
"""
from dataclasses import dataclass

import numpy as np

from mot.lib_common.sets import SETS_DIR, load_sequences


@dataclass
class Instance:
    """正解の物体の 1 フレームの枠。

    Attributes:
        seq: 動画の名前。
        frame: セットの中のフレームの番号(0 始まり)。
        track_id: 動画の中の物体の ID。
        cls: クラスの名前("pedestrian" / "car")。
        box: 枠 (左上の x, 左上の y, 幅, 高さ)(画素)。
        visibility: 見えている割合(MOT17 の 9 列目。分からないデータは 1.0)。
    """
    seq: str
    frame: int
    track_id: int
    cls: str
    box: tuple[float, float, float, float]
    visibility: float


def load_instances(set_name: str) -> list[Instance]:
    """評価のセットの正解の物体を全部読む。

    Args:
        set_name: セットの名前(例 "mot17_val"、"dancetrack_val"、"kitti_mots_val")。

    Returns:
        Instance の並び。
    """
    out = []
    root = SETS_DIR / set_name
    for seq in load_sequences(set_name):
        if set_name.startswith("kitti_mots"):
            from pycocotools import mask as mask_util
            for line in (root / "label_02" / f"{seq.name}.txt").read_text().splitlines():
                f, oid, cls, h, w, rle = line.split(" ", 5)
                if int(oid) == 10000 or int(cls) not in (1, 2):  # 無視の領域
                    continue
                x, y, bw, bh = mask_util.toBbox({"size": [int(h), int(w)], "counts": rle.encode()}).tolist()
                out.append(Instance(seq.name, int(f), int(oid), "car" if int(cls) == 1 else "pedestrian", (x, y, bw, bh), 1.0))
        else:
            for line in (root / seq.name / "gt" / "gt.txt").read_text().splitlines():
                v = line.split(",")
                if int(float(v[6])) != 1 or int(float(v[7])) != 1:  # 評価する歩行者だけ
                    continue
                vis = float(v[8]) if len(v) > 8 else 1.0
                out.append(Instance(seq.name, int(v[0]) - 1, int(v[1]), "pedestrian",
                                    (float(v[2]), float(v[3]), float(v[4]), float(v[5])), vis))
    return out


def absence_lengths(instances: list[Instance]) -> np.ndarray:
    """物体ごとに、正解から消えていた(隠れていた)区間の長さ(フレームの数)を全部集める(gap を決めるための分布)。

    Args:
        instances: load_instances の返り値。

    Returns:
        消えていた区間の長さ (M,)(同じ物体の、続けて写った 2 つのフレームの間の、写っていなかったフレームの数。0 は除く)。
    """
    frames: dict[tuple[str, int], list[int]] = {}
    for ins in instances:
        frames.setdefault((ins.seq, ins.track_id), []).append(ins.frame)
    gaps = []
    for fs in frames.values():
        d = np.diff(np.sort(np.unique(fs))) - 1
        gaps += d[d > 0].tolist()
    return np.asarray(gaps, dtype=np.int64)


def sample_instances(instances: list[Instance], stride: int, min_size: float = 16.0) -> list[Instance]:
    """stride フレームおきのフレームの物体だけを残す(切り出しの数を抑える。枠が小さすぎる物体は除く)。

    Args:
        instances: load_instances の返り値。
        stride: 残すフレームの間隔。
        min_size: 枠の幅と高さの下限(画素。これより小さい切り出しは見た目の特徴が意味を持たない)。

    Returns:
        残した Instance の並び。
    """
    return [i for i in instances if i.frame % stride == 0 and i.box[2] >= min_size and i.box[3] >= min_size]
