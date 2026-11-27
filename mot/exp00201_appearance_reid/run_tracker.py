"""段階 2: 保存した検出(exp00003)と、その検出ごとの見た目の特徴(make_features.py)で追跡し、評価する。

追跡の手法は exp00101 の最良の ByteTrack に、見た目の特徴の BoT-SORT の使い方(U1。mot/lib_common/trackers/bytetrack.py の
with_reid)を足したもの。見た目の特徴の閾値(appearance_thresh 0.25・proximity_thresh 0.5・特徴の指数移動平均の係数 0.9)は
BoT-SORT の公式の値のまま。val の役のセットで選ぶのは exp00101 と同じ track_thresh だけ(見た目の特徴を足した分だけを比べるため)。
クラス・KITTI MOTS のマスクの扱いは exp00101 の run_tracker.py と同じ(mot/lib_common/tracking.py)。
結果: results/{検出のモデル}/bytetrack_reid_{特徴}_{セット}.json に、閾値ごとに足していく。

使い方(このフォルダで):
    ../../.venv/bin/python -u run_tracker.py --feature dinov3_vitb --sets mot17_val dancetrack_val kitti_mots_val \
        --thresh 0.2 0.3 0.4 0.5 0.6
"""
import argparse
import json
from dataclasses import asdict, replace

import numpy as np

import common
from mot.lib_common.detections import load_detections
from mot.lib_common.evaluate import evaluate_kitti_mots, evaluate_mot, write_kitti_mots_results, write_mot_results
from mot.lib_common.sets import load_sequences
from mot.lib_common.trackers.bytetrack import ByteTrackConfig, ByteTracker
from mot.lib_common.tracking import dataset_of, kitti_rows, track_boxes

DETECTIONS_DIR = common.EXP_DIR.parent / "exp00003_detections" / "runs"


def attach_features(frames: list[list[dict]], first_frame: int, path) -> None:
    """保存した特徴を、検出の dict の "feat" に入れる(その場で書き換える。特徴のない検出(確信度が下限以下)には入れない)。

    Args:
        frames: 動画の全部のフレームの検出(exp00003 の形)。
        first_frame: セットの最初のフレームの、元の動画での番号(1 始まり)。
        path: make_features.py が保存した npz。
    """
    z = np.load(path)
    for f, t, j in zip(z["feat"], z["frame"], z["index"]):
        frames[first_frame - 1 + int(t)][int(j)]["feat"] = f.astype(np.float32)


def run(detector: str, feature: str, set_name: str, cfg: ByteTrackConfig, name: str) -> dict:
    """1 つのセットを 1 つの設定で追跡し、評価する。

    Args:
        detector: 検出を作った画像のモデルの名前。
        feature: 特徴の名前(make_features.py の --feature)。
        set_name: 評価のセットの名前。
        cfg: 追跡の設定(with_reid を入れたもの)。
        name: 結果のフォルダの名前(runs/tracks/{detector}/{セット}/{name}/)。

    Returns:
        evaluate_mot・evaluate_kitti_mots の返り値。
    """
    ds = dataset_of(set_name)
    out_dir = common.RUNS_DIR / "tracks" / detector / set_name
    for seq in load_sequences(set_name):
        det = load_detections(DETECTIONS_DIR / detector / ds / f"{seq.name}.json.gz")
        label = dict(zip(["person", "car"], det["labels"]))
        frames = det["frames"]
        attach_features(frames, seq.first_frame, common.RUNS_DIR / "features" / feature / detector / set_name / f"{seq.name}.npz")
        if ds == "kitti_mots":
            cls_cfg = lambda c: replace(cfg, max_aspect=cfg.max_aspect if c == "person" else None)
            per_class = {c: track_boxes(ByteTracker, cls_cfg(c), seq, frames, label[c]) for c in ("car", "person")}
            write_kitti_mots_results(out_dir / name / f"{seq.name}.txt", kitti_rows(per_class, seq.height, seq.width))
        else:
            rows = track_boxes(ByteTracker, cfg, seq, frames, label["person"])
            write_mot_results(out_dir / name / f"{seq.name}.txt", [r[:7] for r in rows])
    return (evaluate_kitti_mots if ds == "kitti_mots" else evaluate_mot)(set_name, out_dir, name)


def main() -> None:
    """引数のセットと閾値の組をすべて追跡・評価して保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--thresh", nargs="+", type=float, default=[0.6], help="val で選ぶ track_thresh の候補")
    p.add_argument("--detector", default="gtr_cnxtb_dinov3_bighead640s4_frames_cls8_ep100_const")
    a = p.parse_args()
    (common.RESULTS_DIR / a.detector).mkdir(parents=True, exist_ok=True)
    for s in a.sets:
        path = common.RESULTS_DIR / a.detector / f"bytetrack_reid_{a.feature}_{s}.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        for th in a.thresh:
            cfg = ByteTrackConfig(track_thresh=th, with_reid=True)
            name = f"bytetrack_reid_{a.feature}_th{th:g}"
            res = run(a.detector, a.feature, s, cfg, name)
            results[name] = {"config": asdict(cfg), "detector": a.detector, "feature": a.feature, "metrics": res}
            print(s, name, {c: {k: round(v, 1) for k, v in m.items()} for c, m in res.items()}, flush=True)
            path.write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
