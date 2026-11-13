"""インスタンスセグメンテーションのデータの読み込み。

COCO 形式の JSON(lib_common/datasets.py のセット、または合成データから作った JSON)を読み、1 枚ごとに画像と物体ごとのマスクを
作って、外から渡した変換(transforms.py の albumentations の変換。データ拡張を含む)をかける。群衆の領域(iscrowd=1)は使わない。

形の記号: S = 変換の後の画像の一辺の画素数、T = 物体の数、h・w = 元の画像の縦・横の画素数
"""
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from lib_common import DATASETS_DIR


def ann_mask(ann: dict, h: int, w: int) -> np.ndarray:
    """COCO の注釈の 1 物体のマスクを、0/1 の配列にする(多角形・非圧縮の RLE・圧縮の RLE のどれにも対応)。

    Args:
        ann: COCO の注釈(segmentation を持つ)。
        h, w: 画像の縦・横の画素数。

    Returns:
        (h, w) の uint8 の 0/1。
    """
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


class InstanceDataset(Dataset):
    """COCO 形式の JSON から、1 枚ずつ画像と正解を作る。

    Args:
        coco_json: COCO 形式の JSON のパス。
        transforms: 画像とマスクにかける albumentations の変換(transforms.train_transforms / eval_transforms)。
        image_root: 画像のパス(images の file_name)の基準のフォルダ。
    """

    def __init__(self, coco_json: Path, transforms: A.Compose, image_root: Path = DATASETS_DIR):
        coco = json.loads(Path(coco_json).read_text())
        self.images = coco["images"]
        self.anns: dict[int, list] = {}
        for a in coco["annotations"]:
            if not a.get("iscrowd", 0):
                self.anns.setdefault(a["image_id"], []).append(a)
        self.transforms, self.image_root = transforms, Path(image_root)

    def __len__(self) -> int:
        """データの数を返す。

        Returns:
            画像の数。
        """
        return len(self.images)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """i 番目の画像と正解を、変換の前の形で読む。

        Args:
            i: 画像の番号。

        Returns:
            (画像 (h, w, 3) の uint8 の RGB, マスク (T, h, w) の uint8 の 0/1, ラベル (T,) の int64(クラスの ID − 1))。
        """
        info = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(self.image_root / info["file_name"])), cv2.COLOR_BGR2RGB)  # (h, w, 3)
        h, w = img.shape[:2]
        anns = self.anns.get(info["id"], [])
        masks = np.stack([ann_mask(a, h, w) for a in anns]) if anns else np.zeros((0, h, w), np.uint8)
        return img, masks, np.array([a["category_id"] - 1 for a in anns], dtype=np.int64)

    def __getitem__(self, i: int) -> dict:
        """i 番目の画像のデータを作る。

        Args:
            i: 画像の番号。

        Returns:
            {"image": (3, S, S) の float32(変換で正規化済み), "masks": (T, S, S) の uint8 の 0/1,
             "labels": (T,) の int64(クラスの ID − 1), "image_id": COCO の画像の ID, "orig_size": 元の (h, w)}。
            変換(切り抜き)で見えなくなった物体は除く。
        """
        info = self.images[i]
        img, raw_masks, raw_labels = self.load(i)
        h, w = img.shape[:2]
        labels = torch.as_tensor(raw_labels, dtype=torch.int64)  # (T,)
        if len(raw_labels):
            out = self.transforms(image=img, masks=raw_masks)  # masks: (T, h, w)
            masks = out["masks"].to(torch.uint8)  # (T, S, S)
            keep = masks.amax(dim=(1, 2)) > 0  # (T, S, S) -> (T,) 切り抜きで消えた物体を除く
            masks, labels = masks[keep], labels[keep]
        else:  # albumentations は 0 枚のマスクを受け付けないので、画像だけを変換する
            out = self.transforms(image=img)
            masks = torch.zeros((0, *out["image"].shape[1:]), dtype=torch.uint8)  # (0, S, S)
        return {"image": out["image"], "masks": masks, "labels": labels, "image_id": info["id"], "orig_size": (h, w)}


class MosaicInstanceDataset(InstanceDataset):
    """4 枚の画像をつなげる Mosaic のデータ拡張をかけるデータ(YOLO 系の学習。transforms.yolo_train_transforms)。

    albumentations の Mosaic は物体ごとのマスクの並び(masks)を受け付けないので、物体ごとのマスクを 1 枚の「物体の番号の地図」
    (mask。0 = 背景、1 以上 = 物体の番号。4 枚の画像で番号が重ならないようにずらす)にして渡し、変換の後に番号ごとのマスクに戻す。
    物体どうしが重なる画素は、後の物体の番号になる(COCO では重なりは少ない)。

    Args:
        coco_json: COCO 形式の JSON のパス。
        transforms: A.Mosaic を含む変換(image・mask・mosaic_metadata を受け取るもの)。
        image_root: 画像のパスの基準のフォルダ。
    """

    def __getitem__(self, i: int) -> dict:
        """i 番目の画像と、ランダムに選んだ 3 枚を Mosaic でつなげたデータを作る。

        Args:
            i: 画像の番号。

        Returns:
            InstanceDataset.__getitem__ と同じ形(orig_size は i 番目の画像の大きさ)。
        """
        picks = [i] + [random.randrange(len(self)) for _ in range(3)]  # random は DataLoader の worker ごとに種が変わる
        samples, all_labels = [], []
        for j in picks:
            img, masks, labels = self.load(j)
            id_map = np.zeros(img.shape[:2], np.int32)  # (h, w) 物体の番号の地図
            for t, m in enumerate(masks):
                id_map[m > 0] = len(all_labels) + t + 1
            all_labels += labels.tolist()
            samples.append({"image": img, "mask": id_map})
        out = self.transforms(**samples[0], mosaic_metadata=samples[1:])
        id_map = torch.as_tensor(out["mask"])  # (S, S)
        ids = [k for k in torch.unique(id_map).tolist() if k > 0]  # 変換の後に残った物体の番号
        masks = (id_map[None] == torch.tensor(ids).view(-1, 1, 1)).to(torch.uint8) if ids else \
            torch.zeros((0, *id_map.shape), dtype=torch.uint8)  # (S, S) -> (T, S, S)
        labels = torch.tensor([all_labels[k - 1] for k in ids], dtype=torch.int64)  # (T,)
        info = self.images[i]
        return {"image": out["image"], "masks": masks, "labels": labels, "image_id": info["id"],
                "orig_size": samples[0]["image"].shape[:2]}


def collate(batch: list[dict]) -> dict:
    """画像はまとめ、物体の数が違うマスク・ラベルは画像ごとのリストのままにする。

    Args:
        batch: InstanceDataset の返り値の並び(長さ B(画像の数))。

    Returns:
        {"image": (B, 3, S, S), "targets": 画像ごとの {"masks": (T_i, S, S), "labels": (T_i,)}(T_i = その画像の物体の数),
         "meta": 画像ごとの {"image_id", "orig_size"}}。
    """
    return {"image": torch.stack([b["image"] for b in batch]),
            "targets": [{"masks": b["masks"], "labels": b["labels"]} for b in batch],
            "meta": [{k: b[k] for k in ("image_id", "orig_size")} for b in batch]}
