"""追跡の結果の評価(公式の TrackEval)と、結果のファイルの書き出し。

- 枠の追跡(MOT17・DanceTrack): TrackEval の MotChallenge2DBox(公式の MOTChallenge・DanceTrack の評価と同じ設定。
  DO_PREPROC あり = 正解の「歩行者以外」(静止した人・反射など)に重なる予測を誤検出に数えない)。
  結果のファイル: 1 行 = `フレーム, ID, 左, 上, 幅, 高さ, 確信度, -1, -1, -1`(MOTChallenge の形)
- マスクの追跡(KITTI MOTS): TrackEval の KittiMOTS。結果のファイル: 1 行 = `フレーム ID クラス 縦 横 RLE`(KITTI MOTS の形。
  クラス 1 = 車、2 = 歩行者)
指標: HOTA・DetA・AssA・LocA(HOTA)、MOTA・IDSW(CLEAR)、IDF1(Identity)。

TrackEval は numpy 1.24 で消えた np.float・np.int・np.bool をまだ使っているので、import の前にその名前を Python の float・int・bool
として足す(TrackEval のコードは変えない。消えた名前は元々この型の別名だった)。
"""
from pathlib import Path

import numpy as np

for _name, _type in {"float": float, "int": int, "bool": bool}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _type)
import trackeval  # noqa: E402

from mot.lib_common.sets import SETS_DIR  # noqa: E402

KEYS = {"HOTA": ["HOTA", "DetA", "AssA", "LocA"], "CLEAR": ["MOTA", "IDSW"], "Identity": ["IDF1"]}


def write_mot_results(path: Path, tracks: list[tuple[int, int, float, float, float, float, float]]) -> None:
    """枠の追跡の結果を MOTChallenge の形で書く。

    Args:
        path: 書くファイル。
        tracks: (フレーム(セットの中の番号、1 始まり), ID, 左, 上, 幅, 高さ, 確信度) の並び。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{t},{i},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1\n" for t, i, x, y, w, h, s in tracks))


def write_kitti_mots_results(path: Path, tracks: list[tuple[int, int, int, int, int, str]]) -> None:
    """マスクの追跡の結果を KITTI MOTS の形で書く。

    Args:
        path: 書くファイル。
        tracks: (フレーム(0 始まり), ID, クラス(1 = 車、2 = 歩行者), 縦, 横, RLE の文字列) の並び。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{t} {i} {c} {h} {w} {r}\n" for t, i, c, h, w, r in tracks))


def _summary(res: dict, tracker: str, classes: list[str]) -> dict:
    """TrackEval の結果から、クラスごと・全部の動画を合わせた主な指標を取り出す。

    Args:
        res: trackeval.Evaluator.evaluate の返り値の 1 つ目。
        tracker: 追跡の手法の名前。
        classes: クラスの名前の並び。

    Returns:
        {クラス: {指標: 値(%)}}(HOTA 系は IoU の閾値で平均した値)。
    """
    out = {}
    for c in classes:
        comb = res[list(res)[0]][tracker]["COMBINED_SEQ"][c]
        out[c] = {k: float(np.mean(comb[m][k])) * (1 if k == "IDSW" else 100) for m, ks in KEYS.items() for k in ks}
    return out


def evaluate_mot(set_name: str, trackers_dir: Path, tracker: str) -> dict:
    """枠の追跡の結果を評価する。

    Args:
        set_name: 評価のセットの名前(sets.mot_sets の名前)。
        trackers_dir: 追跡の手法ごとのフォルダの親({trackers_dir}/{tracker}/{動画}.txt)。
        tracker: 追跡の手法の名前。

    Returns:
        {"pedestrian": {"HOTA", "DetA", "AssA", "LocA", "MOTA", "IDSW", "IDF1"}}(%。IDSW は回数)。
    """
    gt = SETS_DIR / set_name
    ev_cfg = {**trackeval.Evaluator.get_default_eval_config(), "PRINT_RESULTS": False, "PRINT_CONFIG": False,
              "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "PRINT_ONLY_COMBINED": True,
              "TIME_PROGRESS": False}
    ds_cfg = {**trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(), "GT_FOLDER": str(gt),
              "TRACKERS_FOLDER": str(trackers_dir), "TRACKERS_TO_EVAL": [tracker], "SEQMAP_FILE": str(gt / "seqmap.txt"),
              "SKIP_SPLIT_FOL": True, "TRACKER_SUB_FOLDER": "", "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}), trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    res, _ = trackeval.Evaluator(ev_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    return _summary(res, tracker, ["pedestrian"])


def evaluate_kitti_mots(set_name: str, trackers_dir: Path, tracker: str) -> dict:
    """マスクの追跡の結果を評価する(マスクの IoU で HOTA などを計算する)。

    Args:
        set_name: 評価のセットの名前(sets.kitti_mots_sets の名前)。
        trackers_dir: 追跡の手法ごとのフォルダの親({trackers_dir}/{tracker}/{動画}.txt)。
        tracker: 追跡の手法の名前。

    Returns:
        {"car": {...}, "pedestrian": {...}}(evaluate_mot と同じ指標)。
    """
    gt = SETS_DIR / set_name
    ev_cfg = {**trackeval.Evaluator.get_default_eval_config(), "PRINT_RESULTS": False, "PRINT_CONFIG": False,
              "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "PRINT_ONLY_COMBINED": True,
              "TIME_PROGRESS": False}
    ds_cfg = {**trackeval.datasets.KittiMOTS.get_default_dataset_config(), "GT_FOLDER": str(gt),
              "TRACKERS_FOLDER": str(trackers_dir), "TRACKERS_TO_EVAL": [tracker], "SPLIT_TO_EVAL": "val",
              "TRACKER_SUB_FOLDER": "", "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}), trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    res, _ = trackeval.Evaluator(ev_cfg).evaluate([trackeval.datasets.KittiMOTS(ds_cfg)], metrics)
    return _summary(res, tracker, ["car", "pedestrian"])
