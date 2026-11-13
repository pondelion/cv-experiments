"""評価のデータセットと、基準に使う実データの学習セットを、共通の COCO 形式に変換して保存・読み込みする。

どのセットも、クラスは lib_common/classes.py の VOC20(ID は 1〜20 の通し番号)にそろえ、
datasets/sets/{セット名}.json に書き出す。画像のパス(images の file_name)は datasets/ からの相対パス。

セット:
    coco_val_dev:   COCO val2017 の 5,000 枚を seed 0 で並べ替えた前半 2,500 枚。手法選びに何度も使う
    coco_val_test:  同じく後半 2,500 枚。節目でだけ使う(dev への過適合を防ぐため)
    voc_val:        Pascal VOC 2012 の val(インスタンスのマスクがある 1,449 枚)
    coco_train:     COCO train2017 のうち、20 クラスの物体(群衆の領域を除く)が 1 つ以上ある画像。実データでの基準用
    coco_train_10pct / coco_train_1pct: coco_train から seed 0 で 10% / 1% の画像を選んだもの
    coco_val2017_80cls: COCO val2017 の 5,000 枚全部・80 クラス全部(ID は 1〜80 の通し番号)。公式の値が再現できるかの確認用

8 クラス(classes.CLS8。ID は 1〜8)のセット(比較の実験の基本。1 回の学習を短くして、何十回も比べるため):
    coco_val_dev_cls8 / coco_val_test_cls8 / voc_val_cls8: 上の評価のセットと同じ画像で、正解を 8 クラスだけにしたもの
        (ほかのクラスの物体は正解から外す。画像は全部残す)
    coco_train_cls8:    COCO train2017 のうち、8 クラスの物体(群衆の領域を除く)が 1 つ以上ある画像全部(データの量の比較の元)
    coco_train_cls8_2k: 学習の基本。8 クラスのどれかを含む画像 1,900 枚 + どれも含まない画像 100 枚(誤検出を抑えるため)。
        含む画像は、クラスごとの画像の数がおおよそそろうように選ぶ(build_cls8_sets)

COCO の評価セットは、20 クラス(8 クラスのセットは 8 クラス)の物体が写っていない画像も残す(そこに出した予測は誤検出として数えるため)。

使い方(リポジトリ直下で): .venv/bin/python -m lib_common.datasets
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from lib_common import DATASETS_DIR
from lib_common.classes import (CLS8, VOC20, VOC_CLASS_NAMES, VOC_TO_COCO_NAME, categories, coco80, coco_id_to_ours,
                                name_to_ours)
from lib_common.masks import mask_bbox, mask_to_rle

SETS_DIR = DATASETS_DIR / "sets"
SEED = 0


def set_path(name: str) -> Path:
    """セット名から、COCO 形式の JSON のパスを返す。

    Args:
        name: セット名(例: "coco_val_dev")。

    Returns:
        datasets/sets/{name}.json のパス。
    """
    return SETS_DIR / f"{name}.json"


def load_set(name: str) -> dict:
    """保存したセットの COCO 形式の辞書を読み込む。

    Args:
        name: セット名。

    Returns:
        COCO 形式の辞書(images・annotations・categories)。
    """
    with open(set_path(name)) as f:
        return json.load(f)


def _save(name: str, images: list[dict], annotations: list[dict], class_set: list[tuple[str, int]] = VOC20) -> None:
    """COCO 形式で保存する(annotation の id は 1 から振り直す)。

    Args:
        name: セット名(datasets/sets/{name}.json に保存する)。
        images: COCO の images。
        annotations: COCO の annotations。
        class_set: (COCO のクラス名, COCO の category_id) の並び(categories を作る)。
    """
    for i, ann in enumerate(annotations):
        ann["id"] = i + 1
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(set_path(name), "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories(class_set)}, f)
    n_crowd = sum(a["iscrowd"] for a in annotations)
    print(f"{name}: 画像 {len(images)} 枚、物体 {len(annotations) - n_crowd} 個(群衆の領域 {n_crowd} 個)")


def _convert_coco(split: str, class_set: list[tuple[str, int]] = VOC20) -> tuple[list[dict], list[dict]]:
    """COCO の instances_{split}.json を、class_set のクラスだけ・この実験の ID に変換する。

    Args:
        split: COCO の分け方("val2017" または "train2017")。
        class_set: (COCO のクラス名, COCO の category_id) の並び。

    Returns:
        (images, annotations)。images の file_name は datasets/ からの相対パスにしてある。
    """
    with open(DATASETS_DIR / "coco" / "annotations" / f"instances_{split}.json") as f:
        coco = json.load(f)
    to_ours = coco_id_to_ours(class_set)
    images = [
        {"id": im["id"], "file_name": f"coco/{split}/{im['file_name']}", "height": im["height"], "width": im["width"]}
        for im in coco["images"]
    ]
    annotations = []
    for ann in coco["annotations"]:
        if ann["category_id"] in to_ours:
            ann = dict(ann, category_id=to_ours[ann["category_id"]])
            annotations.append(ann)
    return images, annotations


def _split_by_images(images: list[dict], annotations: list[dict], image_ids: set[int]) -> tuple[list, list]:
    """指定した画像だけを残す。

    Args:
        images: COCO の images。
        annotations: COCO の annotations。
        image_ids: 残す画像の ID。

    Returns:
        (残した images, 残した画像の annotations)。
    """
    return [im for im in images if im["id"] in image_ids], [a for a in annotations if a["image_id"] in image_ids]


def build_coco_val() -> None:
    """COCO val2017 を dev / test の 2,500 枚ずつに分けて保存する。"""
    images, annotations = _convert_coco("val2017")
    ids = sorted(im["id"] for im in images)
    random.Random(SEED).shuffle(ids)
    half = len(ids) // 2
    for name, part in [("coco_val_dev", ids[:half]), ("coco_val_test", ids[half:])]:
        _save(name, *_split_by_images(images, annotations, set(part)))


def build_coco_val_80cls() -> None:
    """COCO val2017 の全部の画像・80 クラスを保存する(公式の値が再現できるかの確認用)。"""
    class_set = coco80()
    _save("coco_val2017_80cls", *_convert_coco("val2017", class_set), class_set)


def build_coco_train() -> None:
    """COCO train2017 から、20 クラスの物体がある画像だけを残して保存する(全部 / 10% / 1%)。"""
    images, annotations = _convert_coco("train2017")
    has_object = {a["image_id"] for a in annotations if not a["iscrowd"]}
    ids = sorted(has_object)
    _save("coco_train", *_split_by_images(images, annotations, set(ids)))
    random.Random(SEED).shuffle(ids)
    for name, ratio in [("coco_train_10pct", 0.1), ("coco_train_1pct", 0.01)]:
        _save(name, *_split_by_images(images, annotations, set(ids[: round(len(ids) * ratio)])))


def _restrict_classes(src: str, dst: str, class_set: list[tuple[str, int]]) -> None:
    """保存したセットの正解を class_set のクラスだけにし、ID を振り直して別の名前で保存する(画像は全部残す)。

    Args:
        src: 元のセット名(VOC20 の ID のもの)。
        dst: 保存するセット名。
        class_set: 残すクラスの (COCO のクラス名, COCO の category_id) の並び。
    """
    data = load_set(src)
    src_names = {c["id"]: c["name"] for c in data["categories"]}
    to_new = name_to_ours(class_set)
    anns = [dict(a, category_id=to_new[src_names[a["category_id"]]]) for a in data["annotations"]
            if src_names[a["category_id"]] in to_new]
    _save(dst, data["images"], anns, class_set)


def select_balanced(images: list[dict], annotations: list[dict], n_positive: int, n_negative: int,
                    seed: int = SEED) -> set[int]:
    """クラスごとの画像の数がおおよそそろうように画像を選び、どのクラスも含まない画像を足す。

    含む画像の選び方: 選んだ画像の中で「そのクラスを含む画像の数」がいちばん少ないクラスを選び、そのクラスを含む
    まだ選んでいない画像を、乱数で並べた順に 1 枚足す。を n_positive 枚になるまで繰り返す(1 枚にほかのクラスも
    写っていれば、そのクラスの数も増えるので、person のように一緒に写りやすいクラスは多めになる)。
    どのクラスも含まない画像: annotations(群衆の領域も含む)が 1 つもない画像から、乱数で n_negative 枚。

    Args:
        images: COCO の images(候補の全部)。
        annotations: images の画像の annotations(残すクラスだけ。群衆の領域も含む)。
        n_positive: クラスを含む画像の数。
        n_negative: どのクラスも含まない画像の数。
        seed: 乱数の種。

    Returns:
        選んだ画像の ID。
    """
    rng = random.Random(seed)
    classes_of: dict[int, set[int]] = {}
    for a in annotations:
        if not a["iscrowd"]:
            classes_of.setdefault(a["image_id"], set()).add(a["category_id"])
    has_any = {a["image_id"] for a in annotations}
    candidates: dict[int, list[int]] = {}
    for iid in sorted(classes_of):
        for c in classes_of[iid]:
            candidates.setdefault(c, []).append(iid)
    for c in candidates:
        rng.shuffle(candidates[c])
    count = {c: 0 for c in candidates}
    chosen: set[int] = set()
    while len(chosen) < n_positive:
        c = min(count, key=lambda k: (count[k], k))  # 画像の数がいちばん少ないクラス(同じなら ID の小さい方)
        while candidates[c] and candidates[c][-1] in chosen:
            candidates[c].pop()
        if not candidates[c]:  # そのクラスの画像を使い切った(この規模では起きない)
            count[c] = float("inf")
            continue
        iid = candidates[c].pop()
        chosen.add(iid)
        for k in classes_of[iid]:
            count[k] += 1
    negatives = sorted(im["id"] for im in images if im["id"] not in has_any)
    rng.shuffle(negatives)
    return chosen | set(negatives[:n_negative])


def build_cls8_sets() -> None:
    """8 クラス(CLS8)のセットを作る(モジュールの説明を参照)。"""
    for src in ["coco_val_dev", "coco_val_test", "voc_val"]:
        _restrict_classes(src, f"{src}_cls8", CLS8)
    images, annotations = _convert_coco("train2017", CLS8)
    has_object = {a["image_id"] for a in annotations if not a["iscrowd"]}
    _save("coco_train_cls8", *_split_by_images(images, annotations, has_object), CLS8)
    chosen = select_balanced(images, annotations, n_positive=1900, n_negative=100)
    _save("coco_train_cls8_2k", *_split_by_images(images, annotations, chosen), CLS8)


def build_voc_val() -> None:
    """Pascal VOC 2012 val のインスタンスのマスクを COCO 形式に変換して保存する。

    SegmentationObject の PNG は、画素値が物体の番号(0 = 背景、1〜N = 物体、255 = 境界・判定しない領域)。
    物体のクラスは、同じ画素の SegmentationClass の値(1〜20 = VOC のクラス、255 = 判定しない領域)の多数決で決める。
    255 の領域(物体の輪郭に沿った数画素の帯)は COCO 形式では表せないので、どの物体のマスクにも含めない。
    """
    voc_dir = DATASETS_DIR / "voc" / "VOCdevkit" / "VOC2012"
    names = (voc_dir / "ImageSets" / "Segmentation" / "val.txt").read_text().split()
    to_ours = name_to_ours(VOC20)
    images, annotations = [], []
    for image_id, name in enumerate(names, start=1):
        obj = np.array(Image.open(voc_dir / "SegmentationObject" / f"{name}.png"))  # (H, W) 物体の番号
        cls = np.array(Image.open(voc_dir / "SegmentationClass" / f"{name}.png"))  # (H, W) VOC のクラスの番号
        images.append({"id": image_id, "file_name": f"voc/VOCdevkit/VOC2012/JPEGImages/{name}.jpg",
                       "height": int(obj.shape[0]), "width": int(obj.shape[1])})
        for k in np.unique(obj):
            if k in (0, 255):
                continue
            mask = obj == k
            values = cls[mask]
            values = values[(values != 0) & (values != 255)]
            voc_name = VOC_CLASS_NAMES[np.bincount(values).argmax() - 1]
            annotations.append({
                "image_id": image_id,
                "category_id": to_ours[VOC_TO_COCO_NAME.get(voc_name, voc_name)],
                "segmentation": mask_to_rle(mask),
                "area": float(mask.sum()),
                "bbox": mask_bbox(mask),
                "iscrowd": 0,
            })
    _save("voc_val", images, annotations)


if __name__ == "__main__":
    import sys
    builders = {"coco_val": build_coco_val, "voc_val": build_voc_val, "coco_train": build_coco_train,
                "coco_val_80cls": build_coco_val_80cls, "cls8": build_cls8_sets}
    for name in sys.argv[1:] or builders:  # 引数でどれを作るかを選べる(省略すると全部)
        builders[name]()
