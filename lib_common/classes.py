"""実験で使うクラスの集合と、各データセットのクラス名との対応。

クラスの名前は COCO の名前にそろえる。どのデータセットで評価するときも、ここで決めた順番の ID
(1 始まりの通し番号)に変換してから評価する。
"""

# Pascal VOC の 20 クラス(COCO の 80 クラスに全部含まれる)。値は COCO の名前と、COCO の category_id。
# ここでの並び順で ID を 1, 2, ... と振る(この順番は VOC の公式のクラスの順)。
VOC20 = [
    ("airplane", 5),
    ("bicycle", 2),
    ("bird", 16),
    ("boat", 9),
    ("bottle", 44),
    ("bus", 6),
    ("car", 3),
    ("cat", 17),
    ("chair", 62),
    ("cow", 21),
    ("dining table", 67),
    ("dog", 18),
    ("horse", 19),
    ("motorcycle", 4),
    ("person", 1),
    ("potted plant", 64),
    ("sheep", 20),
    ("couch", 63),
    ("train", 7),
    ("tv", 72),
]

# 比較の実験の基本に使う 8 クラス(VOC20 の一部。値は COCO の名前と category_id)。ここでの並び順で ID を 1, 2, ... と振る。
# 選び方: 小さい物体が多いクラス(car・bottle)と大きい物体が多いクラス(cat・tv)、見分けやすいクラス(cat・tv)と
# 見分けにくいクラス(chair・bicycle)を混ぜ、紛らわしい cat と dog は両方入れる。屋外の道路(person・car・bicycle・dog)と
# 屋内(person・cat・dog・chair・bottle・tv)で一緒に写りやすく、合成データで複数のクラスの場面を作りやすい
CLS8 = [
    ("person", 1),
    ("bicycle", 2),
    ("car", 3),
    ("cat", 17),
    ("dog", 18),
    ("bottle", 44),
    ("chair", 62),
    ("tv", 72),
]

def coco80() -> list[tuple[str, int]]:
    """COCO の 80 クラス全部(公式の値との比較用)。並びは COCO の category_id の順。

    COCO の annotations から読む(datasets/coco/annotations/instances_val2017.json が必要)。

    Returns:
        (COCO のクラス名, COCO の category_id) の並び(80 個)。
    """
    import json

    from lib_common import DATASETS_DIR
    with open(DATASETS_DIR / "coco" / "annotations" / "instances_val2017.json") as f:
        cats = json.load(f)["categories"]
    return [(c["name"], c["id"]) for c in sorted(cats, key=lambda c: c["id"])]


# VOC のクラス名 → COCO のクラス名(名前が違うものだけ)
VOC_TO_COCO_NAME = {
    "aeroplane": "airplane",
    "diningtable": "dining table",
    "motorbike": "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
}

# VOC の SegmentationClass の画素値(1〜20)に対応するクラス名(VOC の公式の順)
VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def categories(class_set: list[tuple[str, int]] = VOC20) -> list[dict]:
    """COCO 形式の categories(id は 1 始まりの通し番号)を作る。

    Args:
        class_set: (COCO のクラス名, COCO の category_id) の並び。

    Returns:
        [{"id": 1, "name": "airplane"}, ...] の形のリスト。
    """
    return [{"id": i + 1, "name": name} for i, (name, _) in enumerate(class_set)]


def coco_id_to_ours(class_set: list[tuple[str, int]] = VOC20) -> dict[int, int]:
    """COCO の category_id → この実験の ID への対応(集合に入らない COCO のクラスは含まない)。

    Args:
        class_set: (COCO のクラス名, COCO の category_id) の並び。

    Returns:
        {COCO の category_id: この実験の ID(1 始まりの通し番号)}。
    """
    return {coco_id: i + 1 for i, (_, coco_id) in enumerate(class_set)}


def name_to_ours(class_set: list[tuple[str, int]] = VOC20) -> dict[str, int]:
    """COCO のクラス名 → この実験の ID への対応。

    Args:
        class_set: (COCO のクラス名, COCO の category_id) の並び。

    Returns:
        {クラス名: この実験の ID(1 始まりの通し番号)}。
    """
    return {name: i + 1 for i, (name, _) in enumerate(class_set)}
