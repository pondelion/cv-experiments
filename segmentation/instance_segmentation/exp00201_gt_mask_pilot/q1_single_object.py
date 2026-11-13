"""Q1: 単一の物体を無地の背景(グリーンバック・白)で生成できるか、背景の色との差で抜き出せるかを確かめる。

学習済みの認識モデルは使わない(① 生成だけ)。抜き出しの規則と、使えるかの判定は lib_common/synthetic_data/chroma_key.py。

- クラスごと・背景の色ごとに N 枚を FLUX.2 [klein] 4B で生成する(768×768、4 step)
- 背景の色との差で物体のマスクを作り、背景の色のそろい方・まとまりの数・端に触れているか・面積を記録する

出力:
    runs/q1/{背景}/{クラス}_{番号}.png(生成した画像。輪郭の色を正しく扱うため PNG)と runs/q1/records.json
    results/q1_single_object.json(まとめ)
    results/samples/q1/{背景}/{クラス}.jpg(見本: クラスごとに 1 枚、抜き出したマスクの輪郭を重ねたもの)

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00201_gt_mask_pilot/q1_single_object.py
"""
import argparse
import json
import random

import common
import numpy as np
import torch

from lib_common.classes import VOC20
from lib_common.latency import Timer, summarize
from lib_common.masks import mask_to_rle
from lib_common.visualize import draw_instances, save_jpeg
from lib_common.synthetic_data.chroma_key import is_usable, key_mask
from lib_common.synthetic_data.flux2_klein import DEFAULT_STEPS, load_pipeline
from lib_common.synthetic_data.prompts import BACKGROUNDS, object_prompt


def summarize_records(records: list[dict]) -> dict:
    """使える割合と、各値の中央値をまとめる。"""
    s = [r["stats"] for r in records]
    return {
        "n": len(records),
        "usable_rate": float(np.mean([r["usable"] for r in records])),
        "bg_spread_median": float(np.median([x["bg_spread"] for x in s])),
        "single_component_rate": float(np.mean([x["n_components"] == 1 for x in s])),
        "touches_border_rate": float(np.mean([x["touches_border"] for x in s])),
        "area_ratio_median": float(np.median([x["area_ratio"] for x in s])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--backgrounds", nargs="+", default=list(BACKGROUNDS))
    args = parser.parse_args()

    pipe = load_pipeline()
    rng = random.Random(0)
    records, t_gen = [], []
    for bg in args.backgrounds:
        out_dir = common.RUNS_DIR / "q1" / bg
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, _ in VOC20:
            for i in range(args.per_class):
                prompt = object_prompt(name, rng, bg)
                seed = i  # 背景の色だけを変えて比べられるよう、同じクラス・番号は同じ seed にする
                with Timer() as t:
                    image = pipe(prompt=prompt, height=common.SIZE, width=common.SIZE, num_inference_steps=DEFAULT_STEPS,
                                 generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                t_gen.append(t.ms)
                arr = np.array(image)
                mask, stats = key_mask(arr)
                file = f"{name.replace(' ', '_')}_{i}.png"
                image.save(out_dir / file)
                records.append({"background": bg, "class": name, "index": i, "file": f"{bg}/{file}", "prompt": prompt,
                                "seed": seed, "stats": stats, "usable": is_usable(stats), "mask": mask_to_rle(mask)})
                if i == 0:
                    label = "usable" if records[-1]["usable"] else "NOT usable"
                    save_jpeg(draw_instances(arr, [mask], [label], alpha=0.25),
                              common.RESULTS_DIR / "samples" / "q1" / bg / f"{name.replace(' ', '_')}.jpg")
            done = [r for r in records if r["background"] == bg and r["class"] == name]
            print(f"{bg:5s} {name:12s} 使える {sum(r['usable'] for r in done)}/{len(done)}  "
                  f"背景のばらつき {np.median([r['stats']['bg_spread'] for r in done]):5.1f}  "
                  f"背景の色 {done[0]['stats']['bg_color']}", flush=True)
    (common.RUNS_DIR / "q1" / "records.json").write_text(json.dumps(records))

    summary = {"by_background": {bg: summarize_records([r for r in records if r["background"] == bg])
                                 for bg in args.backgrounds},
               "by_class": {f"{bg}/{name}": summarize_records([r for r in records if r["background"] == bg and r["class"] == name])
                            for bg in args.backgrounds for name, _ in VOC20},
               "time_ms": {"generate": summarize(t_gen, 2)}}
    common.RESULTS_DIR.mkdir(exist_ok=True)
    (common.RESULTS_DIR / "q1_single_object.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary["by_background"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
