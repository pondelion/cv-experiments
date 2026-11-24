"""exp00502: 画像のモデル単体(フレーム独立)の予測に ByteTrack でフレーム間の ID をつなぎ、動画のモデルと同じ評価にかける。

画像のモデル単体はフレームどうしで物体の ID をつながないので、そのままでは動画の AP を測れない。ここでは mot/ の ByteTrack
(`mot/lib_common/trackers/bytetrack.py`)で ID をつなぎ、動画のモデル(VidEoMT・MinVIS)と同じ形の予測にする:
    1. 検出: 動画の全フレームを 1 枚ずつ画像のモデルにかける(mot/ の検出と同じ作り方 `mot/lib_common/detections.py`。確信度 0.05 以上、
       8 クラス、元の大きさのマスク)。runs/{学習の名前}_detections/{セット}/{動画}.json.gz に保存し、2 回目からは読むだけ
    2. 追跡: クラスごとに ByteTrack をかける(クラスをまたいで照合しない)。設定は track_thresh 以外は公式の値。歩行者向けの横長の枠を
       出さない条件は使わない(8 クラスには横長の物体がある)
    3. 予測: 1 つの ID を 1 つの物体の予測にする。マスクは追跡が出力したフレームの照合した検出のマスク(ほかのフレームは写っていない扱い)、
       確信度は出力したフレームの検出の確信度の平均。動画ごとに確信度の上位 100 個(動画のモデルと同じ)
    4. 評価: video_eval の evaluate_vis(動画の AP)と evaluate_frames(画像 AP(動画の予測))
track_thresh は dev のセットで選び(動画の AP が最良の値)、val のセットでは選んだ値で 1 回だけ測る。

使い方(リポジトリ直下で):
    .venv/bin/python -u segmentation/instance_segmentation/exp00502_video_arch_comparison/eval_image_bytetrack.py \
        --run gtr_cnxtb_dinov3_bighead640s4_frames_cls8_ep100_const --max-vram-gb 4
"""
import argparse
import json
from pathlib import Path

import common
import numpy as np
import torch

from lib_common import DATASETS_DIR  # noqa: E402
from mot.lib_common.detections import detect_video, load_detections, load_detector, save_detections  # noqa: E402
from mot.lib_common.trackers.bytetrack import ByteTrackConfig, ByteTracker  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_data import load_video_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_eval import evaluate_frames, evaluate_vis  # noqa: E402

TOP_K = 100  # 動画ごとに残す予測の数(動画のモデルの評価と同じ)


def detections(run: str, set_name: str, max_vram_gb: float) -> dict[int, list[list[dict]]]:
    """セットの全部の動画の検出を作る(保存してあれば読む)。

    Args:
        run: 画像のモデルの学習の名前(runs/{run}/best.pt)。
        set_name: 動画のセットの名前。
        max_vram_gb: GPU のメモリの上限(GB)。

    Returns:
        {動画の id: フレームごとの検出の並び(detect_video の形。"label" は動画のセットのクラスの番号)}。
    """
    gt = load_video_set(set_name)
    out_dir = common.RUNS_DIR / f"{run}_detections" / set_name
    todo = [v for v in gt["videos"] if not (out_dir / f"{v['name']}.json.gz").exists()]
    if todo:
        torch.cuda.set_per_process_memory_fraction(max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        seg, cfg = load_detector(str(common.RUNS_DIR / run / "best.pt"))
        names = [c["name"] for c in json.loads(Path(cfg["train_json"]).read_text())["categories"]]
        for i, v in enumerate(todo):
            frames = detect_video(seg, cfg, [DATASETS_DIR / f for f in v["file_names"]], list(range(len(names))), 0.05, 4, 4, "orig")
            save_detections(out_dir / f"{v['name']}.json.gz", frames, run, 0.05, list(range(len(names))), "orig")
            if i % 20 == 0:
                print(f"detect {set_name}: {i + 1}/{len(todo)}", flush=True)
    # 画像のモデルのクラスの番号 → 動画のセットのクラスの番号(名前で対応させる。並びが違うことがあるため)
    train_json = json.loads((common.RUNS_DIR / run / "config.json").read_text())["train_json"]
    names = [c["name"] for c in json.loads(Path(train_json).read_text())["categories"]]
    vnames = [c["name"] for c in gt["categories"]]
    to_video = {i: vnames.index(n) for i, n in enumerate(names)}
    res = {}
    for v in gt["videos"]:
        frames = load_detections(out_dir / f"{v['name']}.json.gz")["frames"]
        res[v["id"]] = [[{**d, "label": to_video[d["label"]]} for d in f] for f in frames]
    return res


def track_video(frames: list[list[dict]], num_classes: int, cfg: ByteTrackConfig, video_id: int, cat_ids: list[int]) -> list[dict]:
    """1 本の動画の検出に、クラスごとに ByteTrack をかけて、動画の予測(評価の形)にする。

    Args:
        frames: フレームごとの検出の並び。
        num_classes: クラスの数。
        cfg: ByteTrack の設定。
        video_id: 動画の id。
        cat_ids: 0 始まりのクラスの番号 → カテゴリーの id。

    Returns:
        [{"video_id", "category_id", "score", "segmentations": [フレームごとの RLE または None]}](確信度の上位 TOP_K 個)。
    """
    preds = []
    for c in range(num_classes):
        tracker, tracks = ByteTracker(cfg), {}  # tracks: ID → {"segs": [フレームごとの RLE または None], "scores": [...]}
        for t, dets in enumerate(frames):
            dets = [d for d in dets if d["label"] == c]
            boxes = np.array([d["bbox"] for d in dets], float).reshape(-1, 4)
            scores = np.array([d["score"] for d in dets], float)
            for tr in tracker.update(boxes, scores, [{"rle": d["segmentation"]} for d in dets]):
                rec = tracks.setdefault(tr.track_id, {"segs": [None] * len(frames), "scores": []})
                rec["segs"][t] = tr.extra["rle"]
                rec["scores"].append(tr.score)
        for rec in tracks.values():
            preds.append({"video_id": video_id, "category_id": cat_ids[c], "score": float(np.mean(rec["scores"])),
                          "segmentations": rec["segs"]})
    return sorted(preds, key=lambda p: -p["score"])[:TOP_K]


def evaluate(dets: dict, set_name: str, track_thresh: float) -> dict:
    """1 つのセットを 1 つの track_thresh で追跡して評価する。

    Args:
        dets: detections の返り値。
        set_name: 動画のセットの名前。
        track_thresh: ByteTrack の track_thresh。

    Returns:
        evaluate_vis と evaluate_frames の結果をまとめた dict。
    """
    gt = load_video_set(set_name)
    cat_ids = [c["id"] for c in gt["categories"]]
    cfg = ByteTrackConfig(track_thresh=track_thresh, max_aspect=None)
    preds = [p for v in gt["videos"] for p in track_video(dets[v["id"]], len(cat_ids), cfg, v["id"], cat_ids)]
    return {**evaluate_vis(gt, preds), **evaluate_frames(gt, preds)}


def main() -> None:
    """dev で track_thresh を選び、val で 1 回だけ測って results/{学習の名前}_bytetrack.json に保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dev-set", default="vipseg_vis_cls8_dev")
    p.add_argument("--val-set", default="vipseg_vis_cls8_val")
    p.add_argument("--track-thresh", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--max-vram-gb", type=float, default=4.0)
    a = p.parse_args()
    dev, val = detections(a.run, a.dev_set, a.max_vram_gb), detections(a.run, a.val_set, a.max_vram_gb)
    dev_res = {}
    for th in a.track_thresh:
        dev_res[th] = evaluate(dev, a.dev_set, th)
        print(f"dev track_thresh {th:g}: 動画の AP {dev_res[th]['AP'] * 100:.1f} 画像 AP {dev_res[th]['image_AP'] * 100:.1f}", flush=True)
    best = max(dev_res, key=lambda th: dev_res[th]["AP"])
    val_res = evaluate(val, a.val_set, best)
    print(f"val track_thresh {best:g}(dev で選択): 動画の AP {val_res['AP'] * 100:.1f} 画像 AP {val_res['image_AP'] * 100:.1f}", flush=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"run": a.run, "tracker": "ByteTrack", "track_thresh": best, "dev": {str(k): v for k, v in dev_res.items()}, "val": val_res}
    (common.RESULTS_DIR / f"{a.run}_bytetrack.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
