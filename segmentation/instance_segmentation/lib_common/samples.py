"""結果のページ(GitHub Pages)に載せる、予測の見本の画像を作る。

どの exp でも同じ画像を見本にする(条件どうしを並べて見比べるため)。見本の画像は SAMPLE_COUNTS の枚数を、
正解の物体が 1 つ以上ある画像から seed 0 で選ぶ。

保存先(exp ごと):
    render_compare(学習): results/samples/{学習の名前}/{セット}/{画像の ID}.jpg(最後の評価)、
                          results/samples/{学習の名前}/epoch{エポック}/{セット}/{画像の ID}.jpg(途中の評価)
    render(exp00001):    results/samples/{セット}/{条件}/{画像の ID}.jpg   条件は "gt"(正解)や公開モデルの名前
    results/samples/index.json  見本の一覧([{group(セット), condition(学習の名前など), id, file, ...}, ...])
render_compare は 1 枚に「元の画像 | 正解 | 予測」を、物体の数の見出しを付けて並べる(人が見比べやすいように。学習ではこちらを使う)。
render は正解または予測だけを描く(exp00001 の公開モデルの見本)。
予測は、決めたスコアの閾値(最後の評価では dev で PQ が最大になる値)以上のものだけを描く。
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from lib_common import DATASETS_DIR
from lib_common.datasets import load_set, set_path
from lib_common.masks import rle_to_mask
from lib_common.visualize import draw_instances, save_jpeg, side_by_side

SAMPLE_COUNTS = {"coco_val_dev": 12, "voc_val": 6, "coco_val_dev_cls8": 12, "voc_val_cls8": 6}
SEED = 0


def sample_images(set_name: str) -> list[dict]:
    """見本にする画像を、決まった順で返す(物体のある画像から、決まった乱数で SAMPLE_COUNTS 枚)。

    Args:
        set_name: 評価のセット名(SAMPLE_COUNTS のキー)。

    Returns:
        COCO の images の要素の並び。
    """
    data = load_set(set_name)
    has_obj = {a["image_id"] for a in data["annotations"] if not a["iscrowd"]}
    images = sorted((im for im in data["images"] if im["id"] in has_obj), key=lambda im: im["id"])
    return random.Random(SEED).sample(images, SAMPLE_COUNTS[set_name])


def sample_set(set_name: str) -> str:
    """見本の画像だけのセットを作り(なければ)、その名前を返す(学習の途中に、見本の画像だけを予測するため)。

    Args:
        set_name: 評価のセット名(SAMPLE_COUNTS のキー)。

    Returns:
        作ったセットの名前("{set_name}_samples")。
    """
    sub = f"{set_name}_samples"
    if not set_path(sub).exists():
        data = load_set(set_name)
        images = sample_images(set_name)
        ids = {im["id"] for im in images}
        set_path(sub).write_text(json.dumps({"images": images, "categories": data["categories"],
                                             "annotations": [a for a in data["annotations"] if a["image_id"] in ids]}))
    return sub


def _to_rle(seg, h: int, w: int) -> dict:
    """正解の多角形・RLE を RLE にそろえる。

    Args:
        seg: COCO の注釈の segmentation(多角形・非圧縮の RLE・圧縮の RLE)。
        h, w: 画像の縦・横の画素数。

    Returns:
        圧縮の RLE。
    """
    from pycocotools import mask as mask_utils
    if isinstance(seg, list):  # 多角形
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg["counts"], list):  # 圧縮していない RLE
        return mask_utils.frPyObjects(seg, h, w)
    return seg


def _masks_and_labels(objs: list[dict], im: dict, names: dict[int, str]) -> tuple[list[np.ndarray], list[str]]:
    """1 枚の画像の正解または予測を、描く順(大きい正解・低いスコアの予測から)のマスクとラベルにする。

    Args:
        objs: その画像の正解(score が None)または予測(COCO の結果の形式)の並び。
        im: COCO の images の要素(縦・横の画素数を使う)。
        names: {category_id: クラス名}。

    Returns:
        (物体ごとの (H(縦), W(横)) の bool のマスク, 物体ごとのラベル("dog" または "dog 0.93"))。
    """
    objs = sorted(objs, key=lambda o: -(o.get("area") or 0) if o["score"] is None else o["score"])
    masks = [rle_to_mask(_to_rle(o["segmentation"], im["height"], im["width"])) for o in objs]
    labels = [names[o["category_id"]] + ("" if o["score"] is None else f" {o['score']:.2f}") for o in objs]
    return masks, labels


def render_compare(samples_dir: Path, set_name: str, condition: str, predictions: list[dict],
                   score_threshold: float) -> None:
    """見本の画像ごとに「元の画像 | 正解 | 予測」を、物体の数の見出しを付けて横に並べた 1 枚を保存し、index.json に追記する。

    保存先: samples_dir/{condition}/{set_name}/{画像の ID}.jpg(学習(run)ごとにまとめる。途中の評価は condition が "{名前}/epoch010")

    Args:
        samples_dir: exp の results/samples。
        set_name: 評価のセットの名前(SAMPLE_COUNTS のキー)。
        condition: 条件の名前(例: "m2f_r50_coco10pct" や "m2f_r50_coco10pct/epoch010")。
        predictions: 予測のリスト(COCO の結果の形式。見本の画像以外の予測が入っていてもよい)。
        score_threshold: これ以上のスコアの予測だけを描く。
    """
    data = load_set(set_name)
    names = {c["id"]: c["name"] for c in data["categories"]}
    gts = [dict(a, score=None) for a in data["annotations"] if not a["iscrowd"]]
    entries = []
    for im in sample_images(set_name):
        image = np.array(Image.open(DATASETS_DIR / im["file_name"]).convert("RGB"))
        gt_masks, gt_labels = _masks_and_labels([o for o in gts if o["image_id"] == im["id"]], im, names)
        pr_masks, pr_labels = _masks_and_labels(
            [o for o in predictions if o["image_id"] == im["id"] and o["score"] >= score_threshold], im, names)
        panel = side_by_side([(f"Image {im['id']}", image),
                              (f"GT: {len(gt_masks)} inst", draw_instances(image, gt_masks, gt_labels)),
                              (f"Pred: {len(pr_masks)} inst (score>={score_threshold:.2f})",
                               draw_instances(image, pr_masks, pr_labels))])
        file = f"{condition}/{set_name}/{im['id']}.jpg"  # 学習(run)ごとにまとめる
        save_jpeg(panel, samples_dir / file, max_side=None)  # 各パネルは side_by_side で縮めてある
        entries.append({"group": set_name, "condition": condition, "id": str(im["id"]), "file": file,
                        "n_gt": len(gt_masks), "n_pred": len(pr_masks), "score_threshold": score_threshold})
    _update_index(samples_dir, set_name, condition, entries)


def render_train_inputs(dataset, names: list[str], path: Path, n: int = 8, seed: int = 0) -> None:
    """データ拡張をかけた後の学習の画像に正解を描いて並べた 1 枚を保存する(変換とマスクがずれていないかを学習の前に確かめる)。

    Args:
        dataset: InstanceDataset(学習の変換を渡したもの)。
        names: クラス名の並び(ラベルの番号の順)。
        path: 保存先の JPEG のパス。
        n: 並べる枚数(4 枚ずつ 2 段)。
        seed: 選ぶ画像とデータ拡張の乱数の種。
    """
    from einops import rearrange

    from segmentation.instance_segmentation.lib_common.transforms import MEAN, STD
    rng = random.Random(seed)
    rows = []
    for _ in range(0, n, 4):
        panels = []
        for _ in range(4):
            i = rng.randrange(len(dataset))
            s = dataset[i]
            img = rearrange(s["image"].numpy(), "c h w -> h w c") * np.array(STD) + np.array(MEAN)  # 正規化を戻す: (S, S, 3)
            img = (img.clip(0, 1) * 255).astype(np.uint8)
            masks = [m.numpy().astype(bool) for m in s["masks"]]  # (T, S, S) -> T 個の (S, S)
            labels = [names[int(c)] for c in s["labels"]]
            panels.append((f"{s['image_id']}: {len(masks)} instances", draw_instances(img, masks, labels)))
        rows.append(side_by_side(panels, side=256))
    save_jpeg(np.concatenate(rows, axis=0), path, max_side=None)


def _update_index(samples_dir: Path, set_name: str, condition: str, entries: list[dict]) -> None:
    """index.json の (set_name, condition) の項目を entries に置き換える。

    学習の途中の見本(condition が "{名前}/epoch{エポック}")は index_epochs.json に書く(途中の見本の画像は GitHub に上げず
    ローカルだけに残すので(.gitignore)、GitHub の index.json には最後の見本だけを載せる)。
    原因の切り分け用の学習(condition が "DIAG_" で始まる)の見本も GitHub に上げないので、index_local.json に書く。

    Args:
        samples_dir: exp の results/samples。
        set_name: 評価のセットの名前。
        condition: 条件の名前。
        entries: 新しい項目の並び。
    """
    if "/epoch" in condition:
        index_path = samples_dir / "index_epochs.json"
    elif condition.startswith("DIAG_"):
        index_path = samples_dir / "index_local.json"
    else:
        index_path = samples_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [e for e in index if not (e["group"] == set_name and e["condition"] == condition)] + entries
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))


def render(samples_dir: Path, set_name: str, condition: str, objects: list[dict] | None = None,
           score_threshold: float = 0.0) -> None:
    """見本の画像に正解または予測を描いて保存し、index.json に追記する。

    Args:
        samples_dir: exp の results/samples。
        set_name: 評価のセットの名前。
        condition: 条件の名前("gt" なら正解を描く)。
        objects: 予測のリスト(COCO の結果の形式)。condition が "gt" のときは不要。
        score_threshold: これ以上のスコアの予測だけを描く。
    """
    data = load_set(set_name)
    names = {c["id"]: c["name"] for c in data["categories"]}
    if condition == "gt":
        objects = [dict(a, score=None) for a in data["annotations"] if not a["iscrowd"]]
    entries = []
    for im in sample_images(set_name):
        image = np.array(Image.open(DATASETS_DIR / im["file_name"]).convert("RGB"))
        masks, labels = _masks_and_labels([o for o in objects if o["image_id"] == im["id"] and
                                           (o["score"] is None or o["score"] >= score_threshold)], im, names)
        file = f"{set_name}/{condition}/{im['id']}.jpg"
        save_jpeg(draw_instances(image, masks, labels), samples_dir / file)
        entries.append({"group": set_name, "condition": condition, "id": str(im["id"]), "file": file,
                        "n_objects": len(masks)})
    _update_index(samples_dir, set_name, condition, entries)
