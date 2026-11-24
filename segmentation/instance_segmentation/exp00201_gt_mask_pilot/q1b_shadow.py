"""Q1b: 物体の影がマスクに入らないようにできるかを、抜き出しの規則 × プロンプトで比べる(認識モデルは使わない)。

- 規則: (1) RGB の距離(lib_common.synthetic_data.chroma_key.key_mask)、(2) 緑の強さ(dominance_key_mask。影は暗くても緑が強いので背景になる)
- プロンプト: (a) 今の文面(Q1 の画像をそのまま使う)、(b) 宙に浮かせ・地面をなくし・照明を平らにする文面(新しく生成する)
- 背景はグリーンバック(白の背景の影は灰色で、灰色の物体と規則で分けられないため)

見本: クラスごとに [元の画像 | 規則 (1) で抜いた物体 | 規則 (2) で抜いた物体] を横に並べる。抜いた物体の外は市松模様にして、
マスクに含まれた範囲がはっきりわかるようにする。

出力:
    runs/q1b/no_shadow/{クラス}_{番号}.png(プロンプト (b) の画像)
    results/q1b_shadow.json(規則・プロンプトごとの、使える割合と面積の中央値)
    results/samples/q1b/{prompt_current,prompt_no_shadow}/{クラス}.jpg と grid_*.jpg(一覧)

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00201_gt_mask_pilot/q1b_shadow.py
"""
import json
import random

import common
import cv2
import numpy as np
import torch
from PIL import Image

from lib_common.classes import VOC20
from lib_common.visualize import JPEG_QUALITY, save_jpeg
from lib_common.synthetic_data.chroma_key import dominance_key_mask, is_usable, key_mask
from lib_common.synthetic_data.flux2_klein import DEFAULT_STEPS, load_pipeline
from lib_common.synthetic_data.prompts import object_prompt

RULES = {"rgb_distance": key_mask, "green_dominance": dominance_key_mask}
PANEL = 240  # 見本の 1 枚の一辺の画素数


def checkerboard(h: int, w: int, cell: int = 16) -> np.ndarray:
    """灰色と濃い紫の市松模様(緑・白の背景とも、物体の色とも見分けやすい色)を作る。

    Args:
        h: 縦の画素数。
        w: 横の画素数。
        cell: 升目の一辺の画素数。

    Returns:
        (h, w, 3) の uint8。
    """
    yy, xx = np.mgrid[:h, :w]
    board = ((yy // cell + xx // cell) % 2).astype(bool)
    out = np.empty((h, w, 3), np.uint8)
    out[board], out[~board] = (110, 110, 110), (90, 40, 110)
    return out


def cutout(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """マスクの中だけ元の画像を残し、外を市松模様にする。

    Args:
        image: (H, W, 3) の uint8。
        mask: (H, W) の bool。

    Returns:
        (H, W, 3) の uint8。
    """
    out = checkerboard(*mask.shape)
    out[mask] = image[mask]
    return out


def panel(image: np.ndarray, masks: dict[str, np.ndarray], title: str) -> np.ndarray:
    """[元の画像 | 規則ごとに抜いた物体] を横に並べ、見出しを付ける。

    Args:
        image: 元の画像 (H, W, 3)。
        masks: {規則の名前: マスク (H, W)}。
        title: 見出し。

    Returns:
        並べた画像。
    """
    tiles = [cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_AREA)]
    tiles += [cv2.resize(cutout(image, m), (PANEL, PANEL), interpolation=cv2.INTER_AREA) for m in masks.values()]
    labels = [title] + list(masks)
    for t, label in zip(tiles, labels):
        cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate(tiles, axis=1)


def main() -> None:
    """規則 × プロンプトの組ごとに物体を抜き出し、並べた画像と面積の比べを保存する。"""
    pipe = load_pipeline()
    rng = random.Random(0)
    out_dir = common.RUNS_DIR / "q1b" / "no_shadow"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in ["prompt_current", "prompt_no_shadow"]:
        stats = {rule: [] for rule in RULES}
        panels = []
        for name, _ in VOC20:
            fname = name.replace(" ", "_")
            for i in range(5):
                if variant == "prompt_current":  # Q1 で生成した画像(同じ seed・今の文面)
                    image = np.array(Image.open(common.RUNS_DIR / "q1" / "green" / f"{fname}_{i}.png").convert("RGB"))
                else:
                    prompt = object_prompt(name, rng, "green", no_shadow=True)
                    pil = pipe(prompt=prompt, height=common.SIZE, width=common.SIZE, num_inference_steps=DEFAULT_STEPS,
                               generator=torch.Generator("cuda").manual_seed(i)).images[0]
                    pil.save(out_dir / f"{fname}_{i}.png")
                    image = np.array(pil)
                masks = {}
                for rule, fn in RULES.items():
                    masks[rule], s = fn(image)
                    stats[rule].append(s | {"usable": is_usable(s)})
                if i == 0:
                    p = panel(image, masks, name)
                    save_jpeg(p, common.RESULTS_DIR / "samples" / "q1b" / variant / f"{fname}.jpg")
                    panels.append(p)
            print(variant, name, {r: round(float(np.median([x["area_ratio"] for x in stats[r][-5:]])), 3) for r in RULES},
                  flush=True)
        grid = np.concatenate([np.concatenate(panels[k:k + 2], axis=1) for k in range(0, len(panels), 2)], axis=0)
        cv2.imwrite(str(common.RESULTS_DIR / "samples" / "q1b" / f"grid_{variant}.jpg"),
                    cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        results[variant] = {rule: {"usable_rate": float(np.mean([x["usable"] for x in v])),
                                   "area_ratio_median": float(np.median([x["area_ratio"] for x in v])),
                                   "single_component_rate": float(np.mean([x["n_components"] == 1 for x in v]))}
                            for rule, v in stats.items()}
    (common.RESULTS_DIR / "q1b_shadow.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
