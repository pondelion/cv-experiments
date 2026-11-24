"""検品の確かめに使う、答えのわかる問題を作る(画像と正解)。

(1) 物体のクラス(t1)
    coco:      COCO dev の、VOC の 20 クラスの物体の切り出し(群衆の領域は除く、枠の短い辺が 64 画素以上。枠を 10% 広げて切り出す)。各クラス 10
    coco_none: COCO val2017 の、20 クラス以外の物体(トラック・ベンチなど)の切り出し(同じ条件)。正解は none。100
    gen:       exp00201 の Q1c の llm_v3 で生成した単一の物体。正解は頼んだクラス。200
(2) 背景に学習するクラスの物体があるか(t2)
    pos:   COCO dev で、20 クラスの物体(群衆の領域は除く、面積 32² 画素以上)が 1 つ以上ある画像。正解は yes。150
    neg:   COCO val2017 で、20 クラスの物体が 1 つもない画像(群衆の領域も含めて)。正解は no。150
    boxed: pos の画像の、20 クラスの物体を全部(小さいもの・群衆の領域も)、枠を 10% + 4 画素広げた灰色(128)の四角で隠したもの。正解は no

出力: runs/checks/{t1,t2}/{種類}/{番号}.jpg と runs/checks/labels.json([{task, kind, file, label}, ...])

使い方: .venv/bin/python segmentation/instance_segmentation/exp00002_decision_model_qc/build_checks.py
"""
import json
import random

import common
import numpy as np
from PIL import Image

from lib_common import DATASETS_DIR
from lib_common.classes import VOC20
from lib_common.datasets import load_set

EXP00201_RUNS = common.EXP_DIR.parent / "exp00201_gt_mask_pilot" / "runs"
OUT = common.RUNS_DIR / "checks"
VOC_NAMES = {n for n, _ in VOC20}


def crop(image: Image.Image, bbox, margin: float = 0.1) -> Image.Image:
    """枠を幅・高さの margin 倍ずつ広げて切り出す(画像の外は切り詰める)。

    Args:
        image: 画像。
        bbox: COCO の枠 (x, y, w, h)。
        margin: 広げる割合。

    Returns:
        切り出した画像。
    """
    x, y, w, h = bbox
    dx, dy = w * margin, h * margin
    return image.crop((max(0, x - dx), max(0, y - dy), min(image.width, x + w + dx), min(image.height, y + h + dy)))


def main() -> None:
    """検品の問題の画像と正解を作り、保存する。"""
    rng = random.Random(0)
    labels = []

    def save(img: Image.Image, task: str, kind: str, k: int, label: str) -> None:
        """問題の画像を保存し、正解を labels に足す。

        Args:
            img: 問題の画像。
            task: 検品の種類(t1・t2)。
            kind: 問題の種類。
            k: 番号。
            label: 正解。
        """
        path = OUT / task / kind / f"{k:04d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(path, quality=95)
        labels.append({"task": task, "kind": kind, "file": str(path.relative_to(OUT)), "label": label})

    dev = load_set("coco_val_dev")
    full = load_set("coco_val2017_80cls")
    dev_names = {c["id"]: c["name"] for c in dev["categories"]}
    full_names = {c["id"]: c["name"] for c in full["categories"]}
    dev_imgs = {im["id"]: im for im in dev["images"]}
    full_imgs = {im["id"]: im for im in full["images"]}
    big = lambda a: not a["iscrowd"] and min(a["bbox"][2], a["bbox"][3]) >= 64  # noqa: E731

    # (1) coco: 20 クラスそれぞれ 10 個
    for name in sorted(VOC_NAMES):
        anns = [a for a in dev["annotations"] if dev_names[a["category_id"]] == name and big(a)]
        for k, a in enumerate(rng.sample(anns, min(10, len(anns)))):
            im = Image.open(DATASETS_DIR / dev_imgs[a["image_id"]]["file_name"])
            save(crop(im, a["bbox"]), "t1", "coco", len(labels), name)
    # (1) coco_none: 20 クラス以外の物体 100 個
    others = [a for a in full["annotations"] if full_names[a["category_id"]] not in VOC_NAMES and big(a)]
    for a in rng.sample(others, 100):
        im = Image.open(DATASETS_DIR / full_imgs[a["image_id"]]["file_name"])
        save(crop(im, a["bbox"]), "t1", "coco_none", len(labels), "none")
    # (1) gen: 生成した単一の物体(グリーンバック・マゼンタのまま)
    for r in json.loads((EXP00201_RUNS / "q1c" / "records.json").read_text()):
        if r["strategy"] == "llm_v3":
            save(Image.open(EXP00201_RUNS / "q1c" / r["file"]), "t1", "gen", len(labels), r["class"])

    # (2) pos / boxed: 20 クラスの物体がある画像と、その物体を灰色の四角で隠したもの
    by_img: dict[int, list] = {}
    for a in dev["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    pos_ids = sorted(i for i, anns in by_img.items() if any(not a["iscrowd"] and a["area"] >= 32 * 32 for a in anns))
    for i in rng.sample(pos_ids, 150):
        im = Image.open(DATASETS_DIR / dev_imgs[i]["file_name"]).convert("RGB")
        save(im, "t2", "pos", len(labels), "yes")
        arr = np.array(im)
        for a in by_img[i]:  # dev の正解は 20 クラスだけなので、全部隠す
            x, y, w, h = a["bbox"]
            dx, dy = w * 0.1 + 4, h * 0.1 + 4
            arr[max(0, int(y - dy)): int(y + h + dy) + 1, max(0, int(x - dx)): int(x + w + dx) + 1] = 128
        save(Image.fromarray(arr), "t2", "boxed", len(labels), "no")
    # (2) neg: 20 クラスの物体が 1 つもない画像
    has_voc = {a["image_id"] for a in full["annotations"] if full_names[a["category_id"]] in VOC_NAMES}
    neg_ids = sorted(i for i in full_imgs if i not in has_voc)
    for i in rng.sample(neg_ids, 150):
        save(Image.open(DATASETS_DIR / full_imgs[i]["file_name"]), "t2", "neg", len(labels), "no")

    (OUT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1))
    counts = {}
    for l in labels:
        counts[(l["task"], l["kind"])] = counts.get((l["task"], l["kind"]), 0) + 1
    print(counts, "画像のない COCO の候補:", len(neg_ids))


if __name__ == "__main__":
    main()
