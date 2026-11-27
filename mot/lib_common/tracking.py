"""保存した検出(exp00003)に追跡の手法をかけ、評価の形の結果の行にする(mot/ の exp で共通)。

検出の dict に "feat"(見た目の特徴)があれば、追跡の手法に付ける情報(extras)として渡す(mot/exp00201。見た目の特徴を使う設定の
追跡の手法だけが使う)。
"""
import numpy as np
from pycocotools import mask as mask_utils

from mot.lib_common.sets import Sequence

KITTI_CLASS = {"car": 1, "person": 2}  # KITTI MOTS のクラスの番号(1 = 車、2 = 歩行者)


def dataset_of(set_name: str) -> str:
    """評価のセットの名前から、検出を保存したデータセットの名前を返す。

    Args:
        set_name: 例 "mot17_val"。

    Returns:
        "mot17"・"dancetrack"・"kitti_mots" のどれか。
    """
    return next(d for d in ("mot17", "dancetrack", "kitti_mots") if set_name.startswith(d))


def track_boxes(tracker_cls: type, cfg, seq: Sequence, frames: list[list[dict]], label: int) -> list[tuple]:
    """1 本の動画の 1 つのクラスを、指定した追跡の手法で追跡する。

    Args:
        tracker_cls: 追跡の型(update(枠, 確信度, 付ける情報)・keep_output を持つ)。
        cfg: 追跡の設定。
        seq: 動画(セットの範囲)。
        frames: 動画の全部のフレームの検出(exp00003 の形。見た目の特徴を使うときは各検出に "feat")。
        label: 追跡するクラス。

    Returns:
        (フレーム(セットの中の番号、1 始まり), ID, 左, 上, 幅, 高さ, 確信度, マスクの RLE) の並び。
    """
    tracker = tracker_cls(cfg, seq.fps)
    out = []
    for t in range(seq.num_frames):
        dets = [d for d in frames[seq.first_frame - 1 + t] if d["label"] == label]
        boxes = np.array([d["bbox"] for d in dets], float).reshape(-1, 4)
        scores = np.array([d["score"] for d in dets], float)
        extras = [{"rle": d["segmentation"], **({"feat": d["feat"]} if "feat" in d else {})} for d in dets]
        for tr in tracker.update(boxes, scores, extras):
            if tracker.keep_output(tr):
                out.append((t + 1, tr.track_id, *tr.box.tolist(), tr.score, tr.extra["rle"]))
    return out


def kitti_rows(per_class: dict[str, list[tuple]], height: int, width: int) -> list[tuple]:
    """KITTI MOTS の結果の行を作る(フレームの中でマスクが重ならないように、確信度の高い物体から画素を取る)。

    Args:
        per_class: {クラス: track_boxes の返り値}。
        height: 画像の高さ。
        width: 画像の幅。

    Returns:
        evaluate.write_kitti_mots_results に渡す (フレーム(0 始まり), ID, クラス, 縦, 横, RLE) の並び。
    """
    by_frame: dict[int, list] = {}
    for cls, rows in per_class.items():
        offset = 0 if cls == "car" else 100000  # クラスの間で ID が重ならないようにする
        for t, tid, *_, score, rle in rows:
            by_frame.setdefault(t, []).append((score, tid + offset, KITTI_CLASS[cls], rle))
    out = []
    for t in sorted(by_frame):
        taken = np.zeros((height, width), bool)
        for score, tid, c, rle in sorted(by_frame[t], key=lambda r: -r[0]):
            m = mask_utils.decode(rle).astype(bool) & ~taken
            if not m.any():
                continue
            taken |= m
            enc = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            out.append((t - 1, tid, c, height, width, enc["counts"].decode("ascii")))
    return out
