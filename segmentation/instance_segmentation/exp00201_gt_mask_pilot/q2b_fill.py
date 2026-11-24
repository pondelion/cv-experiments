"""Q3b: 背景を一から描く inpainting を、inpainting 専用のモデル FLUX.1 Fill [dev] に替えて比べる(① 生成だけ。認識モデルは使わない)。

q2_compose.py の inpaint_full_*(FLUX.2 [klein] の inpainting)は、物体を条件として直接受け取らない方式で、物体と背景の物理的な整合
(水面・接地・大きさ)が弱かった。FLUX.1 Fill はマスクとマスクした画像を追加の入力として受け取るよう学習されている(lib_common.synthetic_data.flux1_fill)。

- 場面: q2_compose.plan_scenes と同じ(同じ物体・置き方・正解)。保存済みの q2 の正解と画素単位で同じかを確かめてから生成する
- 出発の画像: 物体だけを灰色の無地に置いた画像(q2 の inpaint_full_* と同じ)
- 描き直す範囲: 物体の外側全部 + 物体の輪郭から FILL_EDGE 画素内側まで(輪郭のなじみのため)。最後に物体を貼ったときの透明度で重ねて戻す
- プロンプト(q2 の inpaint_full_* と同じ文面):
    fill_empty:   場所だけ + 何もいないことを強調(BACKGROUND_TEMPLATE)
    fill_noclass: 場所だけ + 全クラスを名前で否定(NO_CLASS_TEMPLATE)
- q2 の結果(runs/q2/・results/samples/q2/)には書き込まない

出力:
    runs/q2b_fill/{fill_empty,fill_noclass}/{番号}.png と {fill_empty,fill_noclass}.json(COCO 形式の正解。q2 と同じ)
    results/q2b_fill.json(まとめ)、results/samples/q2b_fill/{番号}_{版}{,_gt}.jpg

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00201_gt_mask_pilot/q2b_fill.py --n 40
"""
import argparse
import json

import common
import cv2
import numpy as np
import torch
from PIL import Image

from lib_common.classes import VOC20, categories
from lib_common.latency import Timer, summarize
from lib_common.masks import rle_to_mask
from lib_common.visualize import draw_instances, save_jpeg
from q2_compose import coco_dict, plan_scenes
from lib_common.synthetic_data.compose import blend, compose_layers
from lib_common.synthetic_data.flux1_fill import DEFAULT_GUIDANCE, DEFAULT_STEPS, load_fill_pipeline

VARIANTS = {"fill_empty": "background", "fill_noclass": "noclass"}
FILL_EDGE = 4
OUT = common.RUNS_DIR / "q2b_fill"
SAMPLES = common.RESULTS_DIR / "samples" / "q2b_fill"


def check_same_as_q2(k: int, gts: list) -> None:
    """作り直した場面の正解が、保存済みの q2 の正解(paste.json)と画素単位で同じかを確かめる。

    Args:
        k: 場面の番号。
        gts: 作り直した場面の正解 [(クラス名, マスク)]。

    Raises:
        AssertionError: 物体の数かクラスかマスクが q2 と違うとき。
    """
    coco = check_same_as_q2.coco
    anns = [a for a in coco["annotations"] if a["image_id"] == k + 1]
    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert len(anns) == len(gts), f"場面 {k}: 物体の数が q2 と違う"
    for a, (name, m) in zip(anns, gts):
        assert names[a["category_id"]] == name and (rle_to_mask(a["segmentation"]) == m).all(), f"場面 {k}: 正解が q2 と違う"


def main() -> None:
    """q2 と同じ場面を作り直し、背景の作り方を足して保存する(正解が q2 と同じかも確かめる)。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    check_same_as_q2.coco = json.loads((common.RUNS_DIR / "q2" / "paste.json").read_text())
    scenes = plan_scenes(args.n)
    fill = load_fill_pipeline()
    torch.cuda.reset_peak_memory_stats()
    for v in VARIANTS:
        (OUT / v).mkdir(parents=True, exist_ok=True)
    entries = {v: [] for v in VARIANTS}
    times = []
    names = {c["id"]: c["name"] for c in categories(VOC20)}
    for k, s in enumerate(scenes):
        fg, alpha, gts = compose_layers((common.SIZE, common.SIZE), s["layout"])
        check_same_as_q2(k, gts)
        union = np.zeros((common.SIZE, common.SIZE), bool)
        for _, m in gts:
            union |= m
        canvas = Image.fromarray(blend(np.full((common.SIZE, common.SIZE, 3), 128, np.uint8), fg, alpha))
        keep = cv2.erode(union.astype(np.uint8), np.ones((2 * FILL_EDGE + 1,) * 2, np.uint8)).astype(bool)
        mask = Image.fromarray((~keep * 255).astype(np.uint8))  # 白 = 描き直す
        for v, kind in VARIANTS.items():
            with Timer() as t:
                img = np.array(fill(prompt=s[kind], image=canvas, mask_image=mask, height=common.SIZE, width=common.SIZE,
                                    num_inference_steps=DEFAULT_STEPS, guidance_scale=DEFAULT_GUIDANCE,
                                    generator=torch.Generator("cuda").manual_seed(k)).images[0])
            times.append(t.ms)
            img = blend(img, fg, alpha)  # 物体の画素は貼ったときのまま(正解のマスクは変わらない)
            Image.fromarray(img).save(OUT / v / f"{k:04d}.png")
            entries[v].append({"file": f"{k:04d}.png", "height": img.shape[0], "width": img.shape[1], "objects": gts})
            if k < args.samples:
                save_jpeg(img, SAMPLES / f"{k:04d}_{v}.jpg")
                save_jpeg(draw_instances(img, [m for _, m in gts], [n for n, _ in gts], alpha=0.6),
                          SAMPLES / f"{k:04d}_{v}_gt.jpg")
        if k == 0:
            print(f"GPU のメモリの最大の使用量(1 場面目まで): {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        print(f"{k:3d} {s['context']:15s} 物体 {len(gts)} 個  {times[-1] / 1000:.1f}s", flush=True)
    for v in VARIANTS:
        (OUT / f"{v}.json").write_text(json.dumps(coco_dict(entries[v])))
    summary = {"n_scenes": args.n, "steps": DEFAULT_STEPS, "guidance": DEFAULT_GUIDANCE,
               "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9, "time_ms": summarize(times, 2)}
    (common.RESULTS_DIR / "q2b_fill.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
