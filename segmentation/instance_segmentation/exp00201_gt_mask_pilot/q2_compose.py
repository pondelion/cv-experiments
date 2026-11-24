"""Q2・Q3: 抜き出した物体を合成した画像と正解のマスクを作り、背景の作り方 3 通り(paste・inpaint_full_empty・inpaint_full_noclass)で
比べる(① 生成だけ。認識モデルは使わない)。

- 物体: Q1c の llm_v3 で生成した物体(runs/q1c/llm_v3/)を、背景の色に合った規則で抜き出し直し、規則で使えると判定したものを使う
- 場面: 場所(公園・リビングなど)を 1 つ選び、その場所にいそうなクラス(lib_common.synthetic_data.prompts.CONTEXTS)の物体を 1〜4 個置く
  (同じクラスが複数入ってもよい)。置き方・重なり・正解のマスクは lib_common.synthetic_data.compose
- 背景の作り方:
    paste:   場所だけの画像を生成し(BACKGROUND_TEMPLATE)、物体を貼る
    inpaint_full_{empty,noclass}: 物体だけを灰色の無地に置いた画像から、背景を一から inpainting で描く。描き直すマスクを物体の
             輪郭から 16 画素内側まで広げ、最後に物体を貼ったときの透明度で重ねて戻す(物体の画素・正解のマスクは paste と同じ)。
             文面は empty が「場所だけ + 何もいない」(paste の背景と同じ文面)、noclass が「場所だけ + 全クラスを名前で否定」
             (以前の版の inpaint(paste の画像の物体以外を描き直す)は、元の背景が残りやすいのでやめた)
    乱数の種は場面の番号で、3 通りとも同じ値を使う(そのため inpaint_full の背景は paste の背景と大まかな配置が似る)
- 測ること: 1 枚を作る時間(背景の生成・合成・inpainting)

出力:
    runs/q2/{paste,inpaint_full_empty,inpaint_full_noclass}/{番号}.png と {同じ名前}.json(COCO 形式の正解。クラスは VOC20 の ID)
    results/q2_compose.json(まとめ)、results/samples/q2/{番号}_{作り方}{,_gt}.jpg(見本。_gt は正解のマスクを重ねたもの)

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00201_gt_mask_pilot/q2_compose.py --n 40
"""
import argparse
import json
import random

import common
import cv2
import numpy as np
import torch
from PIL import Image

from lib_common.classes import VOC20, categories, name_to_ours
from lib_common.latency import Timer, summarize
from lib_common.masks import mask_bbox, mask_to_rle
from lib_common.visualize import draw_instances, save_jpeg
from lib_common.synthetic_data.chroma_key import is_usable, key_for_background
from lib_common.synthetic_data.compose import blend, compose_layers, crop_object, random_layout

VARIANTS = ["paste", "inpaint_full_empty", "inpaint_full_noclass"]
# 背景を一から inpainting で描くときのプロンプトの違い。前の比較(置いた物体の名前 + 場所 / 場所だけ / 場所だけ + 何もいない)では、
# 置いた物体の名前を入れると背景にも同じクラスの物体が描き足され、余計な物体が入らなかったのは「何もいない」の文面だけだった
#   inpaint_full_empty:   場所だけ + 何もいないことを強調(BACKGROUND_TEMPLATE。背景の生成と同じ文面)
#   inpaint_full_noclass: 場所だけ + 全クラスを名前で否定(NO_CLASS_TEMPLATE)
FULL_PROMPTS = {"inpaint_full_empty": "background", "inpaint_full_noclass": "noclass"}
CELL = 16  # inpainting のマスクが縮められるマス目の一辺の画素数(VAE の 8 倍 × 2 × 2 のまとめ)
from lib_common.synthetic_data.flux2_klein import DEFAULT_STEPS, encode_prompts, inpaint_pipeline, load_pipeline, offload_text_encoder
from lib_common.synthetic_data.prompts import BACKGROUND_TEMPLATE, CONTEXTS, background_for, no_class_prompt


def load_objects() -> dict[str, list]:
    """Q1c の llm_v3 の物体を抜き出し直し、クラスごとの切り抜きの一覧にする。

    Returns:
        {クラス名: [切り抜き]}。
    """
    recs = json.loads((common.RUNS_DIR / "q1c" / "records.json").read_text())
    library: dict[str, list] = {}
    for r in recs:
        if r["strategy"] != "llm_v3":
            continue
        arr = np.array(Image.open(common.RUNS_DIR / "q1c" / r["file"]).convert("RGB"))
        mask, stats = key_for_background(arr, background_for(r["class"]))
        if is_usable(stats):
            library.setdefault(r["class"], []).append(crop_object(r["class"], arr, mask, stats["bg_color"]))
    return library


def plan_scenes(n: int) -> list[dict]:
    """全部の場面(場所・物体・置き方・プロンプト)を先に決める(seed 0。同じ n なら毎回同じ場面になる)。

    q2b_fill.py でも、同じ場面を作り直すのに使う。プロンプトの文章の特徴をまとめて計算してから、文章を読む部分を CPU に退かせて、
    生成中の GPU のメモリを減らすために、生成の前に決めておく。

    Args:
        n: 場面の数。

    Returns:
        場面ごとの {場所・物体・置き方・プロンプト}。
    """
    library = load_objects()
    print("物体の数:", {k: len(v) for k, v in library.items()}, flush=True)
    contexts = sorted({c for cs in CONTEXTS.values() for c in cs})
    rng = random.Random(0)
    scenes = []
    for _ in range(n):
        context = rng.choice(contexts)
        names = [name for name, cs in CONTEXTS.items() if context in cs and name in library]
        objects = [rng.choice(library[name]) for name in [rng.choice(names) for _ in range(rng.randint(1, 4))]]
        layout = random_layout(objects, common.SIZE, rng)
        scenes.append({"context": context, "layout": layout,
                       "background": BACKGROUND_TEMPLATE.format(context=context),
                       "noclass": no_class_prompt(context, [name for name, _ in VOC20])})
    return scenes


def coco_dict(entries: list[dict]) -> dict:
    """合成した画像の一覧を COCO 形式にする。

    Args:
        entries: [{file, height, width, objects: [(クラス名, マスク)]}]。

    Returns:
        COCO 形式の dict。
    """
    to_id = name_to_ours(VOC20)
    images, anns = [], []
    for i, e in enumerate(entries, start=1):
        images.append({"id": i, "file_name": e["file"], "height": e["height"], "width": e["width"]})
        for name, m in e["objects"]:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": to_id[name], "segmentation": mask_to_rle(m),
                         "area": float(m.sum()), "bbox": mask_bbox(m), "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": categories(VOC20)}


def make_samples(n: int) -> None:
    """保存した画像と正解から、見本を作る(先頭の n 場面。作り方ごとに画像と正解を重ねたものの 2 枚を別のファイルにする)。

    {番号}_{作り方}.jpg / {番号}_{作り方}_gt.jpg(作り方は VARIANTS。_gt は正解のマスクを重ねたもの)

    Args:
        n: 見本にする場面の数(先頭から)。
    """
    from lib_common.masks import rle_to_mask
    names = {c["id"]: c["name"] for c in categories(VOC20)}
    for v in VARIANTS:
        coco = json.loads((common.RUNS_DIR / "q2" / f"{v}.json").read_text())
        for im in coco["images"][:n]:
            img = np.array(Image.open(common.RUNS_DIR / "q2" / v / im["file_name"]).convert("RGB"))
            anns = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
            stem = im["file_name"].removesuffix(".png")
            save_jpeg(img, common.RESULTS_DIR / "samples" / "q2" / f"{stem}_{v}.jpg")
            save_jpeg(draw_instances(img, [rle_to_mask(a["segmentation"]) for a in anns],
                                     [names[a["category_id"]] for a in anns], alpha=0.6),
                      common.RESULTS_DIR / "samples" / "q2" / f"{stem}_{v}_gt.jpg")


def main() -> None:
    """場面を決め、背景の作り方ごとに合成した画像と正解を作って保存する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="作る場面の数")
    parser.add_argument("--samples", type=int, default=12, help="見本にする場面の数")
    parser.add_argument("--samples-only", action="store_true", help="生成はせず、保存済みの結果から見本だけを作り直す")
    args = parser.parse_args()
    if args.samples_only:
        make_samples(args.samples)
        return

    scenes = plan_scenes(args.n)
    pipe = load_pipeline()
    kinds = ["background", "noclass"]
    embeds = encode_prompts(pipe, [s[kind] for kind in kinds for s in scenes])
    emb = {kind: embeds[i * args.n: (i + 1) * args.n] for i, kind in enumerate(kinds)}
    bg_embeds = emb["background"]
    offload_text_encoder(pipe)
    inpaint = inpaint_pipeline(pipe)
    torch.cuda.reset_peak_memory_stats()
    out = {v: common.RUNS_DIR / "q2" / v for v in VARIANTS}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    entries = {v: [] for v in VARIANTS}
    times = {"background": [], "compose": [], "inpaint_full": []}
    n_objects = []
    for k, s in enumerate(scenes):
        context = s["context"]
        gen = torch.Generator("cuda").manual_seed(k)
        with Timer() as t:
            bg = np.array(pipe(prompt_embeds=bg_embeds[k], height=common.SIZE, width=common.SIZE,
                               num_inference_steps=DEFAULT_STEPS, generator=gen).images[0])
        times["background"].append(t.ms)
        with Timer() as t:
            fg, alpha, gts = compose_layers(bg.shape[:2], s["layout"])
            pasted = blend(bg, fg, alpha)
        times["compose"].append(t.ms)
        union = np.zeros(pasted.shape[:2], bool)
        for _, m in gts:
            union |= m
        # inpaint_full: 物体だけを灰色の無地に置いた画像から、背景を一から描く。描き直すマスクは物体の輪郭から CELL 画素内側まで
        # 広げる(マスクは 16 画素のマス目に縮めて混ぜられるので、輪郭にかかるマス目に灰色が残らないように)。最後に物体を透明度で重ねて戻す
        keep = cv2.erode(union.astype(np.uint8), np.ones((CELL, CELL), np.uint8)).astype(bool)
        canvas = Image.fromarray(blend(np.full_like(bg, 128), fg, alpha))
        fulls = {}
        for v, kind in FULL_PROMPTS.items():
            with Timer() as t:
                img = np.array(inpaint(prompt_embeds=emb[kind][k], image=canvas,
                                       mask_image=Image.fromarray((~keep * 255).astype(np.uint8)),
                                       height=common.SIZE, width=common.SIZE, strength=1.0, guidance_scale=1.0,
                                       num_inference_steps=DEFAULT_STEPS, generator=torch.Generator("cuda").manual_seed(k)).images[0])
            times["inpaint_full"].append(t.ms)
            fulls[v] = blend(img, fg, alpha)  # 物体の画素は貼ったときのまま(正解のマスクは変わらない)、輪郭の 1 画素は新しい背景になじむ
        if k == 0:
            print(f"GPU のメモリの最大の使用量(1 場面目まで): {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        n_objects.append(len(gts))
        for v, img in [("paste", pasted)] + list(fulls.items()):
            Image.fromarray(img).save(out[v] / f"{k:04d}.png")
            entries[v].append({"file": f"{k:04d}.png", "height": img.shape[0], "width": img.shape[1], "objects": gts})
        print(f"{k:3d} {context:15s} 物体 {len(gts)} 個 {[n for n, _ in gts]}  物体の面積 {union.mean():.2f}", flush=True)
    for v in entries:
        (common.RUNS_DIR / "q2" / f"{v}.json").write_text(json.dumps(coco_dict(entries[v])))
    make_samples(args.samples)
    summary = {"n_scenes": args.n, "objects_per_scene_mean": float(np.mean(n_objects)),
               "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9,
               "time_ms": {k: summarize(v, min(2, len(v) - 1)) for k, v in times.items()}}
    (common.RESULTS_DIR / "q2_compose.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
