"""Q1c: 形・色・種類・ポーズ・向きが多様な単一の物体を生成できるかを、プロンプトの作り方で比べる(認識モデルは使わない)。

プロンプトの作り方(どれも、影を描かせない文面・クラスごとの背景の色(lib_common.synthetic_data.prompts.background_for))
    seed: 同じ文面のまま seed(と向きの言葉)だけを変える
    attribute: クラスごとの属性(種類・色・ポーズ)の一覧から組み合わせる(lib_common.synthetic_data.prompts.ATTRIBUTES)
    llm: 手元の ollama の LLM(qwen3.5:9b)に、クラスごとに多様な説明文を作らせる(lib_common.synthetic_data.llm_prompts.describe)
    llm_v2: llm に、クラスの範囲(含めないもの)と、物の種類に合った変える観点を足したもの(describe_v2)。
            llm では bottle に缶が混じるなど、クラスの外の物が書かれたため
    llm_v3: llm_v2 に加え、向き(動物・人はポーズ・動作も)をこちらで一覧から割り当てて守らせる(describe_v3)。
            llm_v2 では person がほぼ全部「まっすぐ立って正面」になり、形の多様さが落ちたため

多様さの値(抜き出しの規則で使えると判定した物体だけで、クラスごとに計算してクラスで平均。認識モデルは使わない):
    aspect_std:   マスクを囲む枠の縦横の比(log)の標準偏差。大きいほど形の縦横が多様
    area_cv:      マスクの面積の変動係数(標準偏差 ÷ 平均)
    shape_iou:    枠で切り出して 64×64 にそろえたシルエットどうしの IoU の平均。小さいほど形が多様
    color_dist:   物体の画素の平均の色(Lab)どうしの距離の平均。大きいほど色が多様

出力:
    runs/q1c/llm_descriptions.json(LLM の説明文。あれば使い回す)、runs/q1c/{作り方}/{クラス}_{番号}.png、runs/q1c/records.json
    results/q1c_diversity.json(まとめ)
    results/samples/q1c/{クラス}.jpg(見本: 作り方ごとの 1 行に 10 枚、抜いた物体を市松模様の上に並べたもの)

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00201_gt_mask_pilot/q1c_diversity.py
"""
import itertools
import json
import random

import common
import cv2
import numpy as np
import torch

from lib_common.classes import VOC20
from lib_common.masks import mask_to_rle
from lib_common.visualize import JPEG_QUALITY
from q1b_shadow import cutout
from lib_common.synthetic_data.chroma_key import is_usable, key_for_background
from lib_common.synthetic_data.flux2_klein import DEFAULT_STEPS, load_pipeline
from lib_common.synthetic_data.llm_prompts import describe, describe_v2, describe_v3
from lib_common.synthetic_data.prompts import attribute_description, background_for, object_prompt, object_prompt_from_description

N = 10  # クラス × 作り方ごとの枚数
STRATEGIES = ["seed", "attribute", "llm", "llm_v2", "llm_v3"]
LLM_FUNCS = {"llm": describe, "llm_v2": describe_v2, "llm_v3": describe_v3}
TILE = 120


def diversity(arrs: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    """物体の画像とマスクの組から、多様さの値を計算する。"""
    aspects, areas, sils, colors = [], [], [], []
    for a, m in zip(arrs, masks):
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        aspects.append(np.log((x1 - x0) / (y1 - y0)))
        areas.append(m.sum())
        sils.append(cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST).astype(bool))
        colors.append(cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[m].astype(np.float32).mean(axis=0))  # (H, W, 3) → 物体の画素の平均 (3,)
    pairs = list(itertools.combinations(range(len(sils)), 2))
    return {
        "aspect_std": float(np.std(aspects)),
        "area_cv": float(np.std(areas) / np.mean(areas)),
        "shape_iou": float(np.mean([(sils[i] & sils[j]).sum() / (sils[i] | sils[j]).sum() for i, j in pairs])),
        "color_dist": float(np.mean([np.linalg.norm(colors[i] - colors[j]) for i, j in pairs])),
    }


def main() -> None:
    q_dir = common.RUNS_DIR / "q1c"
    q_dir.mkdir(parents=True, exist_ok=True)
    llm_desc = {}
    for key, fn in LLM_FUNCS.items():  # LLM の説明文を先に全部作っておく(生成モデルを読み込む前。あれば使い回す)
        path = q_dir / ("llm_descriptions.json" if key == "llm" else f"{key}_descriptions.json")
        if not path.exists():
            path.write_text(json.dumps({name: fn(name, N) for name, _ in VOC20}, ensure_ascii=False, indent=1))
        llm_desc[key] = json.loads(path.read_text())

    records_path = q_dir / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []  # 生成済みの作り方は使い回す
    todo = [s for s in STRATEGIES if s not in {r["strategy"] for r in records}]
    pipe = load_pipeline() if todo else None
    for strategy in todo:
        (q_dir / strategy).mkdir(exist_ok=True)
        rng = random.Random(0)
        for name, _ in VOC20:
            bg = background_for(name)
            for i in range(N):
                if strategy == "seed":
                    prompt = object_prompt(name, rng, bg, no_shadow=True)
                elif strategy == "attribute":
                    prompt = object_prompt_from_description(attribute_description(name, rng), name, rng)
                else:
                    prompt = object_prompt_from_description(llm_desc[strategy][name][i], name, rng,
                                                            add_view=strategy != "llm_v3")  # v3 は説明文に向きが入っている
                image = pipe(prompt=prompt, height=common.SIZE, width=common.SIZE, num_inference_steps=DEFAULT_STEPS,
                             generator=torch.Generator("cuda").manual_seed(i)).images[0]
                file = f"{strategy}/{name.replace(' ', '_')}_{i}.png"
                image.save(q_dir / file)
                mask, stats = key_for_background(np.array(image), bg)
                records.append({"strategy": strategy, "class": name, "index": i, "file": file, "background": bg,
                                "prompt": prompt, "stats": stats, "usable": is_usable(stats), "mask": mask_to_rle(mask)})
            print(strategy, name, "使える", sum(r["usable"] for r in records[-N:]), "/", N, flush=True)
    if pipe:
        records_path.write_text(json.dumps(records))

    from lib_common.masks import rle_to_mask
    from PIL import Image
    summary = {s: {"usable_rate": float(np.mean([r["usable"] for r in records if r["strategy"] == s])), "per_class": {}}
               for s in STRATEGIES}
    for name, _ in VOC20:
        rows = []
        for s in STRATEGIES:
            rs = [r for r in records if r["strategy"] == s and r["class"] == name]
            arrs = [np.array(Image.open(q_dir / r["file"]).convert("RGB")) for r in rs]
            masks = [rle_to_mask(r["mask"]) for r in rs]
            ok = [k for k, r in enumerate(rs) if r["usable"]]
            if len(ok) >= 3:
                summary[s]["per_class"][name] = diversity([arrs[k] for k in ok], [masks[k] for k in ok])
            tiles = [cv2.resize(cutout(a, m), (TILE, TILE), interpolation=cv2.INTER_AREA) for a, m in zip(arrs, masks)]
            row = np.concatenate(tiles, axis=1)
            cv2.putText(row, s, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(row, s, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(row)
        out = common.RESULTS_DIR / "samples" / "q1c" / f"{name.replace(' ', '_')}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    for s in STRATEGIES:
        pc = summary[s]["per_class"]
        summary[s]["mean"] = {k: float(np.mean([v[k] for v in pc.values()])) for k in ["aspect_std", "area_cv", "shape_iou", "color_dist"]}
    (common.RESULTS_DIR / "q1c_diversity.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({s: {"usable_rate": summary[s]["usable_rate"], **summary[s]["mean"]} for s in STRATEGIES}, indent=1))


if __name__ == "__main__":
    main()
