"""exp00502: 学習済みの重みで予測し、正解と予測を左右に並べた見本(mp4 と、数フレームを並べた JPEG)を作る。

動画は、セットの 8 クラスそれぞれについて、そのクラスを含む動画から 1 本ずつ選ぶ(seed 0。同じ動画が重なれば次の候補)。
保存先: results/samples/{学習の名前}/{セット}/{動画の名前}.mp4・.jpg(lib_common/video_samples.py)。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00502_video_arch_comparison/render_samples.py \
        --run videomt_vitb_dinov3_cls8_ep100_const --weights best.pt
"""
import argparse
import dataclasses
import json
import random

import common

import torch  # noqa: E402

from segmentation.instance_segmentation.lib_common.video_data import VideoEvalDataset, load_video_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_infer import video_predictions  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_samples import render_video_compare  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_train import VideoTrainConfig, build_video_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--set", default="vipseg_vis_cls8_val")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--max-vram-gb", type=float, default=8.0)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    run = common.RUNS_DIR / a.run
    fields = {f.name for f in dataclasses.fields(VideoTrainConfig)}
    cfg = VideoTrainConfig(**{k: v for k, v in json.loads((run / "config.json").read_text()).items() if k in fields})
    cfg.init_weights = None
    gt = load_video_set(a.set)
    names = {c["id"]: c["name"] for c in gt["categories"]}
    model = build_video_model(cfg, len(names)).cuda().eval()
    model.load_state_dict(torch.load(run / a.weights, map_location="cuda"))
    rng, chosen = random.Random(0), []
    for cid in names:  # クラスごとに、そのクラスを含む動画から 1 本
        cand = sorted({x["video_id"] for x in gt["annotations"] if x["category_id"] == cid} - set(chosen))
        if cand:
            chosen.append(rng.choice(cand))
    ds = VideoEvalDataset(a.set, cfg.size)
    index = {v["id"]: i for i, v in enumerate(ds.videos)}
    out_dir = common.RESULTS_DIR / "samples" / a.run / a.set
    for vid in chosen:
        item = ds[index[vid]]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, masks = model.predict_video(item["images"].cuda())
        preds = video_predictions(logits, masks, vid, list(names), cfg.size, item["valid_size"], item["orig_size"], cfg.top_k)
        video = ds.videos[index[vid]]
        render_video_compare(video, [x for x in gt["annotations"] if x["video_id"] == vid], preds, names, out_dir,
                             a.score_thr)
        print("saved", out_dir / video["name"], flush=True)


if __name__ == "__main__":
    main()
