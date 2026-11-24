"""合成データに使う画像生成モデル(FLUX.2 [klein] 4B、fp8 版の本体)の、1 枚の生成にかかる時間を測る。

以降の exp で何枚の合成画像を作れるかの見積もりに使う。ほかの処理と同時に動かさずに測る。

- 解像度ごと(512・768・1024 の正方形)に、warmup の後で複数回生成し、1 枚あたりの時間の中央値・p90 と、GPU メモリの最大の使用量を出す
- step 数は蒸留版の既定の 4
- 見本の画像を results/samples/generation/ に小さい JPEG で保存する

出力: results/generation_speed.json

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00001_eval_and_baselines/measure_generation.py
"""
import argparse
import json

import common
import numpy as np
import torch

from lib_common.latency import Timer, summarize
from lib_common.visualize import save_jpeg
from lib_common.synthetic_data.flux2_klein import DEFAULT_STEPS, load_pipeline

PROMPTS = [
    "a photo of a dog and a person sitting on a couch in a living room",
    "a photo of cars and a bus on a city street",
    "a photo of sheep and cows on a green field",
    "a photo of a cat next to a potted plant and a bottle on a dining table",
]


def main() -> None:
    """画像の大きさごとに 1 枚の生成の時間を測り、結果を保存する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 768, 1024])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=8)
    args = parser.parse_args()

    with Timer() as t_load:
        pipe = load_pipeline()
    result = {"model": "FLUX.2 [klein] 4B (fp8 の本体を bf16 に戻して実行)", "steps": DEFAULT_STEPS,
              "load_s": t_load.ms / 1000, "gpu": torch.cuda.get_device_name(0), "by_size": {}}
    samples_dir = common.RESULTS_DIR / "samples" / "generation"
    for size in args.sizes:
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(args.warmup + args.repeat):
            generator = torch.Generator("cuda").manual_seed(i)
            prompt_id = i % len(PROMPTS)
            with Timer() as t:
                image = pipe(prompt=PROMPTS[prompt_id], height=size, width=size, num_inference_steps=DEFAULT_STEPS,
                             generator=generator).images[0]
            times.append(t.ms)
            if i >= args.warmup and i - args.warmup < len(PROMPTS):  # warmup の後の 1 巡分を、プロンプトの番号の名前で保存する
                save_jpeg(np.array(image), samples_dir / f"{size}_prompt{prompt_id}.jpg")
        result["by_size"][size] = summarize(times, args.warmup) | {
            "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9}
        print(size, json.dumps(result["by_size"][size]), flush=True)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    (common.RESULTS_DIR / "generation_speed.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
