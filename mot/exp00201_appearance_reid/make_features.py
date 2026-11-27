"""保存した検出(exp00003)ごとに見た目の特徴を計算して保存する(段階 2・3 の追跡で使う)。

セットの範囲のフレームの、確信度が下限(ByteTrack の確信度の低い検出の下限 0.1)を超える検出の枠の切り出しを、特徴のモデルに通す。
保存: runs/features/{特徴}/{検出のモデル}/{セット}/{動画}.npz(feat (N, C) の float16、frame (N,)(セットの中のフレームの番号、
0 始まり)、index (N,)(そのフレームの検出の並びの中の番号))。
"""
import argparse
import time

import numpy as np
import torch

import common
from eval_reid import build_model
from mot.lib_common.detections import load_detections
from mot.lib_common.reid.features import extract
from mot.lib_common.sets import load_sequences
from mot.lib_common.tracking import dataset_of

DETECTIONS_DIR = common.EXP_DIR.parent / "exp00003_detections" / "runs"
MIN_SCORE = 0.1  # 追跡の手法が使う検出の確信度の下限(ByteTrack の low_thresh)


def main() -> None:
    """指定したセットの全部の動画の検出の特徴を計算して保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True)
    p.add_argument("--detector", default="gtr_cnxtb_dinov3_bighead640s4_frames_cls8_ep100_const")
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-vram-gb", type=float, default=3.0)
    a = p.parse_args()
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model = build_model(a.feature).to(a.device)
    for s in a.sets:
        t0, n = time.time(), 0
        out_dir = common.RUNS_DIR / "features" / a.feature / a.detector / s
        out_dir.mkdir(parents=True, exist_ok=True)
        for seq in load_sequences(s):
            frames = load_detections(DETECTIONS_DIR / a.detector / dataset_of(s) / f"{seq.name}.json.gz")["frames"]
            paths = seq.image_paths()
            items, fr, idx = [], [], []
            for t in range(seq.num_frames):
                for j, d in enumerate(frames[seq.first_frame - 1 + t]):
                    if d["score"] > MIN_SCORE:
                        items.append((paths[t], tuple(d["bbox"])))
                        fr.append(t)
                        idx.append(j)
            feats = extract(model, items, model.crop_size, batch=256, device=a.device)
            np.savez(out_dir / f"{seq.name}.npz", feat=feats.astype(np.float16), frame=np.array(fr, np.int32),
                     index=np.array(idx, np.int32))
            n += len(items)
        print(f"{s}: 検出 {n} 個、{time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
