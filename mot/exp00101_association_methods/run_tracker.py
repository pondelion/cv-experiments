"""保存した検出(exp00003)に追跡の手法をかけ、評価のセットで評価する。

流れ:
    1. 評価のセット(mot/lib_common/sets.py)の動画ごとに、保存した検出のうちセットの範囲のフレームを読む
    2. 追跡の手法に 1 フレームずつ渡し、出力の物体(ID つきの枠。KITTI MOTS は照合した検出のマスク)を書く
    3. TrackEval で評価する(mot/lib_common/evaluate.py)。MOT17・DanceTrack は枠、KITTI MOTS はマスクの HOTA など

クラス: MOT17・DanceTrack は person だけ。KITTI MOTS は car と person(pedestrian)を別々の追跡にかける(クラスをまたいで照合しない)。
KITTI MOTS のマスク: 評価(TrackEval の KittiMOTS)は 1 つのフレームの中でマスクが重なることを許さないので、確信度の高い物体から
順に画素を取り、後の物体からは取られた画素を除く(MOTS の論文の TrackR-CNN の評価と同じ考え方)。

追跡の手法(--tracker)と、val の役のセットで選ぶ閾値(--thresh):
    bytetrack: ByteTrack の track_thresh(確信度の高い検出の閾値)
    sort: SORT の det_thresh(使う検出の確信度の下限。0 で公式と同じく全部の検出を使う)
    ocsort: OC-SORT の det_thresh(使う検出の確信度の下限。公式の値は 0.6)
結果は results/{検出のモデル}/{手法}_{セット}.json に、手法と閾値の組ごとに足していく。

使い方(リポジトリ直下で):
    .venv/bin/python -u mot/exp00101_association_methods/run_tracker.py --tracker ocsort \
        --sets mot17_val dancetrack_val kitti_mots_val --thresh 0.2 0.3 0.4 0.5 0.6
"""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import common
import numpy as np
from pycocotools import mask as mask_utils

from mot.lib_common.detections import load_detections
from mot.lib_common.evaluate import evaluate_kitti_mots, evaluate_mot, write_kitti_mots_results, write_mot_results
from mot.lib_common.sets import Sequence, load_sequences
from mot.lib_common.trackers.bytetrack import ByteTrackConfig, ByteTracker
from mot.lib_common.trackers.ocsort import OCSortConfig, OCSortTracker
from mot.lib_common.trackers.sort import SortConfig, SortTracker

DEFAULT_DETECTOR = "gtr_cnxtb_dinov3_bighead640s4_cls8_2k_ep100"  # exp00003 で検出を作った画像のモデル
# 手法の名前 → (設定の型, 追跡の型, val で選ぶ閾値の設定の名前)。どの追跡の型も update(枠, 確信度, 付ける情報)・keep_output を持つ
TRACKERS = {"bytetrack": (ByteTrackConfig, ByteTracker, "track_thresh"),
            "sort": (SortConfig, SortTracker, "det_thresh"),
            "ocsort": (OCSortConfig, OCSortTracker, "det_thresh")}
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
        tracker_cls: 追跡の型(TRACKERS の 2 つ目)。
        cfg: 追跡の設定(TRACKERS の 1 つ目の型)。
        seq: 動画(セットの範囲)。
        frames: 動画の全部のフレームの検出(exp00003 の形)。
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
        for tr in tracker.update(boxes, scores, [{"rle": d["segmentation"]} for d in dets]):
            if tracker.keep_output(tr):
                out.append((t + 1, tr.track_id, *tr.box.tolist(), tr.score, tr.extra["rle"]))
    return out


def kitti_rows(per_class: dict[int, list[tuple]], height: int, width: int) -> list[tuple]:
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


def run(detector: str, set_name: str, tracker: str, cfg, name: str) -> dict:
    """1 つのセットを 1 つの手法・1 つの設定で追跡し、評価する。

    Args:
        detector: 検出を作った画像のモデルの名前(exp00003 の runs/{detector}/)。
        set_name: 評価のセットの名前。
        tracker: 追跡の手法の名前(TRACKERS のキー)。
        cfg: 追跡の設定。
        name: 結果のフォルダの名前(runs/{detector}/{セット}/{name}/)。

    Returns:
        evaluate_mot・evaluate_kitti_mots の返り値。
    """
    ds = dataset_of(set_name)
    out_dir = common.RUNS_DIR / detector / set_name
    for seq in load_sequences(set_name):
        det = load_detections(common.DETECTIONS_DIR / detector / ds / f"{seq.name}.json.gz")
        # 保存したクラスの番号(画像のモデルの番号)は make_detections.LABEL_NAMES(person, car)の順
        label = dict(zip(["person", "car"], det["labels"]))
        frames = det["frames"]
        tracker_cls = TRACKERS[tracker][1]
        if ds == "kitti_mots":
            # 横長の枠を出さない条件(歩行者のデータのため)は pedestrian だけに使う(その条件を持つ手法だけ)
            cls_cfg = lambda c: replace(cfg, max_aspect=cfg.max_aspect if c == "person" else None) if hasattr(cfg, "max_aspect") else cfg
            per_class = {c: track_boxes(tracker_cls, cls_cfg(c), seq, frames, label[c]) for c in ("car", "person")}
            write_kitti_mots_results(out_dir / name / f"{seq.name}.txt", kitti_rows(per_class, seq.height, seq.width))
        else:
            rows = track_boxes(tracker_cls, cfg, seq, frames, label["person"])
            write_mot_results(out_dir / name / f"{seq.name}.txt", [r[:7] for r in rows])
    return (evaluate_kitti_mots if ds == "kitti_mots" else evaluate_mot)(set_name, out_dir, name)


def main() -> None:
    """引数のセットと閾値の組をすべて追跡・評価し、results/{検出のモデル}/{手法}_{セット}.json に保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--tracker", choices=list(TRACKERS), default="bytetrack")
    p.add_argument("--thresh", "--track-thresh", nargs="+", type=float, default=None,
                   help="val で選ぶ閾値の候補(手法ごとの設定の名前は TRACKERS。省略すると公式の値)")
    p.add_argument("--detector", default=DEFAULT_DETECTOR, help="検出を作った画像のモデルの名前(exp00003 の runs/ の下)")
    a = p.parse_args()
    cfg_cls, _, th_name = TRACKERS[a.tracker]
    thresholds = a.thresh if a.thresh is not None else [getattr(cfg_cls, th_name)]
    (common.RESULTS_DIR / a.detector).mkdir(parents=True, exist_ok=True)
    for s in a.sets:
        path = common.RESULTS_DIR / a.detector / f"{a.tracker}_{s}.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        for th in thresholds:
            cfg = cfg_cls(**{th_name: th})
            name = f"{a.tracker}_th{th:g}"
            res = run(a.detector, s, a.tracker, cfg, name)
            results[name] = {"config": asdict(cfg), "detector": a.detector, "metrics": res}
            print(s, name, {c: {k: round(v, 1) for k, v in m.items()} for c, m in res.items()}, flush=True)
            path.write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
