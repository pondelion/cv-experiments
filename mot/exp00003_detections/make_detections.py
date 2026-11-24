"""画像のモデルの検出を、使う全部の動画(MOT17 の train 7 本・DanceTrack の train 40 本と val 25 本・KITTI MOTS の train 21 本)で
作って保存する(mot/lib_common/detections.py)。保存先: runs/{モデルの名前}/{データセット}/{動画}.json.gz。作った動画は飛ばす。

使い方(リポジトリ直下で):
    .venv/bin/python -u mot/exp00003_detections/make_detections.py --run <画像のモデルの runs/{名前}/best.pt> --max-vram-gb 4
    (動作確認: --limit-videos 1 --limit-frames 20)
"""
import argparse
import json
import time
from pathlib import Path

import common
import torch

from mot.lib_common.detections import detect_video, load_detector, save_detections
from mot.lib_common.sets import DANCETRACK_DIR, KITTI_MOTS_DIR, MOT17_DIR

LABEL_NAMES = ["person", "car"]  # 保存するクラス(画像のモデルの学習のデータのクラスの名前で選ぶ。番号の並びはセットで違うことがある)
# マスクの解像度: マスクを評価するのは KITTI MOTS だけなので、ほかは入力の解像度で保存する(detections.detect_video の mask_res)
MASK_RES = {"mot17": "input", "dancetrack": "input", "kitti_mots": "orig"}


def videos() -> list[tuple[str, str, list[Path]]]:
    """検出を作る動画の一覧。

    Returns:
        (データセット, 動画の名前, フレームの順の画像のパス) の並び。
    """
    out = [("mot17", d.name, sorted((d / "img1").glob("*.jpg"))) for d in sorted(MOT17_DIR.glob("*-FRCNN"))]
    for part in ("train1", "train2", "val"):
        out += [("dancetrack", d.name, sorted((d / "img1").glob("*.jpg"))) for d in sorted((DANCETRACK_DIR / part).glob("dancetrack*"))]
    out += [("kitti_mots", d.name, sorted(d.glob("*.png"))) for d in sorted((KITTI_MOTS_DIR / "training" / "image_02").iterdir())]
    return out


def main() -> None:
    """引数のモデルで、まだ作っていない動画の検出を作る。"""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="画像のモデルの学習の重み(runs/{名前}/best.pt)")
    p.add_argument("--min-score", type=float, default=0.05)
    p.add_argument("--max-vram-gb", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument("--limit-frames", type=int, default=None)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    seg, cfg = load_detector(a.run)
    cats = [c["name"] for c in json.loads(Path(cfg["train_json"]).read_text())["categories"]]
    labels = [cats.index(n) for n in LABEL_NAMES]  # 0 始まりの画像のモデルのクラスの番号(LABEL_NAMES の順)
    print("labels", dict(zip(LABEL_NAMES, labels)), flush=True)
    name = Path(a.run).parent.name
    out_root = common.RUNS_DIR / name
    vids = videos()[: a.limit_videos]
    t0, n_done = time.time(), 0
    for ds, vid, paths in vids:
        out = out_root / ds / f"{vid}.json.gz"
        if out.exists() and a.limit_frames is None:
            continue
        paths = paths[: a.limit_frames]
        t = time.time()
        mask_res = MASK_RES[ds]
        frames = detect_video(seg, cfg, paths, labels, a.min_score, a.batch_size, a.workers, mask_res)
        save_detections(out, frames, name, a.min_score, labels, mask_res)
        n_done += len(paths)
        n_det = sum(len(f) for f in frames)
        print(f"{ds} {vid}: {len(paths)} frames, {n_det / max(len(paths), 1):.1f} dets/frame, "
              f"{(time.time() - t) / max(len(paths), 1) * 1000:.0f} ms/frame, total {n_done} frames {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
