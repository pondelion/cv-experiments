"""exp00501: 画像の段階と同じ 8 クラスの VIPSeg のセットを作る(make_sets.py の vipseg_vis_* から作る。フレームの変換はし直さない)。

画像の段階の 8 クラス(COCO と VOC で共通: person・bicycle・car・cat・dog・bottle・chair・tv)に当たる VIPSeg の物体のクラス
(person・bicycle・car・cat・dog・bottle_or_cup・chair_or_seat・screen_or_television)の物体だけを残し、8 クラスのどれも写って
いない動画は除く。ほかのクラスの物体は背景として扱う(画像の段階の `coco_*_cls8` と同じ)。

    vipseg_vis_cls8_train: 公式の train から dev を除いた残り
    vipseg_vis_cls8_dev:   公式の train から選んだ 200 本(学習の途中の評価・最良の重みの選択)。動画の AP はクラスごとの AP の平均
                           なので、本数の少ないクラスの揺れが全体に効く。そのため、本数の少ないクラスから順に、各クラスが
                           MIN_PER_CLASS 本以上入るように選び(seed 0)、残りをランダムに足す
    vipseg_vis_cls8_val:   公式の val(節目の評価)
    vipseg_frames_cls8_{train,dev,val}: 上の 3 つの動画の全部のフレームを、1 枚ずつの画像にした COCO 形式のセット
                           (画像のインスタンスセグメンテーションのモデル(MinVIS・DVIS の土台)の学習・評価用)

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00501_video_eval_and_baselines/make_sets_cls8.py
"""
import json
import random

import common

from lib_common import DATASETS_DIR  # noqa: E402

CLASSES = ["person", "bicycle", "car", "cat", "dog", "bottle_or_cup", "chair_or_seat", "screen_or_television"]
DEV_VIDEOS = 200
MIN_PER_CLASS = 5


def load(name: str) -> dict:
    return json.loads((DATASETS_DIR / "sets" / f"{name}.json").read_text())


def keep_cls8(d: dict, video_ids: set | None = None) -> dict:
    """8 クラスの物体だけを残し、8 クラスのどれも写っていない動画を除く(video_ids があれば、その動画だけにする)。"""
    cats = [c for c in d["categories"] if c["name"] in CLASSES]
    ids = {c["id"] for c in cats}
    anns = [a for a in d["annotations"] if a["category_id"] in ids]
    has = {a["video_id"] for a in anns} if video_ids is None else video_ids & {a["video_id"] for a in anns}
    return {"videos": [v for v in d["videos"] if v["id"] in has], "annotations": [a for a in anns if a["video_id"] in has],
            "categories": cats}


def merge(a: dict, b: dict) -> dict:
    """2 つのセットを合わせる(どちらも id が 1 から振られているので、b の動画・物体の id を a の最大値だけずらす)。"""
    vo, ao = max(v["id"] for v in a["videos"]), max(x["id"] for x in a["annotations"])
    return {"videos": a["videos"] + [v | {"id": v["id"] + vo} for v in b["videos"]],
            "annotations": a["annotations"] + [x | {"id": x["id"] + ao, "video_id": x["video_id"] + vo} for x in b["annotations"]],
            "categories": a["categories"]}


def to_frames(d: dict) -> dict:
    """動画のセットを、フレームを 1 枚の画像とした COCO 形式の画像のセットにする(画像のインスタンスセグメンテーションのモデルの
    学習・評価用。MinVIS・DVIS の画像のモデル)。画像の id = 動画の id × 1000 + フレームの番号。
    画像の学習のコードはクラスの id が 1〜K の連番であることを前提にする(ラベル = id − 1)ので、クラスの id を CLASSES の順に
    1〜8 へ振り直す(動画のセットとの対応はクラス名で取る)。"""
    new_id = {c["id"]: CLASSES.index(c["name"]) + 1 for c in d["categories"]}
    images, anns = [], []
    for v in d["videos"]:
        for t, f in enumerate(v["file_names"]):
            images.append({"id": v["id"] * 1000 + t, "file_name": f, "height": v["height"], "width": v["width"]})
    for a in d["annotations"]:
        for t, rle in enumerate(a["segmentations"]):
            if rle is None:
                continue
            anns.append({"id": len(anns) + 1, "image_id": a["video_id"] * 1000 + t, "category_id": new_id[a["category_id"]],
                         "segmentation": rle, "area": a["areas"][t], "bbox": a["bboxes"][t], "iscrowd": 0})
    cats = sorted(({"id": new_id[c["id"]], "name": c["name"]} for c in d["categories"]), key=lambda c: c["id"])
    return {"images": images, "annotations": anns, "categories": cats}


def main() -> None:
    pool = keep_cls8(merge(load("vipseg_vis_train"), load("vipseg_vis_dev")))  # 公式の train(元の train + 元の dev)
    name_of = {c["id"]: c["name"] for c in pool["categories"]}
    videos_of = {n: set() for n in CLASSES}
    for a in pool["annotations"]:
        videos_of[name_of[a["category_id"]]].add(a["video_id"])
    rng, dev = random.Random(0), set()
    for n in sorted(CLASSES, key=lambda n: len(videos_of[n])):  # 本数の少ないクラスから
        cand = sorted(videos_of[n] - dev)
        need = max(0, MIN_PER_CLASS - len(videos_of[n] & dev))
        dev |= set(rng.sample(cand, need))
    rest = sorted({v["id"] for v in pool["videos"]} - dev)
    dev |= set(rng.sample(rest, DEV_VIDEOS - len(dev)))
    all_ids = {v["id"] for v in pool["videos"]}
    sets = {"vipseg_vis_cls8_train": keep_cls8(pool, all_ids - dev), "vipseg_vis_cls8_dev": keep_cls8(pool, dev),
            "vipseg_vis_cls8_val": keep_cls8(load("vipseg_vis_val"))}
    for name, d in sets.items():
        (DATASETS_DIR / "sets" / f"{name}.json").write_text(json.dumps(d))
        frames = to_frames(d)  # 同じ動画のフレームの画像のセット(vipseg_frames_cls8_*)
        fname = name.replace("vipseg_vis_cls8", "vipseg_frames_cls8")
        (DATASETS_DIR / "sets" / f"{fname}.json").write_text(json.dumps(frames))
        print(f"{fname}: 画像 {len(frames['images'])} 枚・物体 {len(frames['annotations'])} 個", flush=True)
        per = {n: len({a["video_id"] for a in d["annotations"] if name_of[a["category_id"]] == n}) for n in CLASSES}
        print(f"{name}: 動画 {len(d['videos'])} 本・フレーム {sum(v['length'] for v in d['videos'])} 枚・物体 {len(d['annotations'])} 個 |",
              " ".join(f"{n} {k}" for n, k in per.items()), flush=True)


if __name__ == "__main__":
    main()
