"""動画インスタンスセグメンテーションのデータ: データセットの変換(共通の形式へ)と、学習用の区間の読み込み。

共通の形式(YouTube-VIS と同じ JSON。`datasets/sets/{セット名}.json`):
    {"videos": [{"id", "name", "width", "height", "length", "file_names": [datasets/ からの相対のパス, ...]}],
     "annotations": [{"id", "video_id", "category_id", "segmentations": [フレームごとの RLE または None], "areas": [...],
                      "bboxes": [...], "iscrowd": 0}],
     "categories": [{"id", "name", "isthing"}]}
1 つの annotation が 1 つの物体(動画を通した ID)で、segmentations は動画のフレームの数だけ並ぶ(写っていないフレームは None)。
クラスは名前の文字列も持つので、オープンボキャブラリーの評価にも使える。

VIPSeg(動画のパノプティック)からの変換:
    - 公式の前処理(配布の change2_720p.py)と同じく、フレームを高さ 720 に縮める(画像は bilinear、マスクは nearest)。
      保存先も公式と同じ `VIPSeg_720P/images/{動画}/`・`VIPSeg_720P/panomasks/{動画}/`
    - マスクの値: 物体のクラスは「クラスの値 × 100 + 物体の番号」、背景のクラスはクラスの値そのもの(どちらもクラスの値は 1 始まり、
      0 は VOID)。カテゴリーの JSON の id は 0 始まりなので、クラスの値 − 1 が id
    - 物体のクラス(isthing = 1、58 クラス)だけを annotation にする。背景のクラスの値の画素と、物体のクラス以外のクラスに物体の番号が
      付いた値(val に 1 個ある ceiling)は除く
"""
import json
import random
from multiprocessing import Pool
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from lib_common import DATASETS_DIR

VIPSEG_HEIGHT = 720  # 公式の前処理の高さ
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # 画像の段階と同じ ImageNet の正規化


def _vipseg_frame(args: tuple) -> tuple[str, int, int, dict]:
    """1 フレームを 720p に縮めて保存し、物体ごとのマスクの RLE を返す。

    Args:
        args: (VIPSeg の根のフォルダ, 動画の名前, フレームの名前(拡張子なし), 物体のクラスの値の集合)。

    Returns:
        (動画の名前, 縮めた幅, 縮めた高さ, {マスクの値: RLE})。
    """
    root, video, frame, thing_values = args
    root = Path(root)
    out_img = root / "VIPSeg_720P" / "images" / video / f"{frame}.jpg"
    out_mask = root / "VIPSeg_720P" / "panomasks" / video / f"{frame}.png"
    if out_img.exists() and out_mask.exists():  # 途中で止まったときのやり直し用
        mask = np.array(Image.open(out_mask))
        w, h = mask.shape[1], mask.shape[0]
    else:
        img = Image.open(root / "imgs" / video / f"{frame}.jpg")
        w0, h0 = img.size
        w, h = int(VIPSEG_HEIGHT * w0 / h0), VIPSEG_HEIGHT
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        img.resize((w, h), Image.BILINEAR).save(out_img)
        m = Image.open(root / "panomasks" / video / f"{frame}.png").resize((w, h), Image.NEAREST)
        m.save(out_mask)
        mask = np.array(m)
    rles = {}
    for value in np.unique(mask):
        if value > 124 and value // 100 in thing_values:  # 物体のクラスの物体だけ
            rle = mask_util.encode(np.asfortranarray(mask == value))
            rle["counts"] = rle["counts"].decode("ascii")
            rles[int(value)] = rle
    return video, w, h, rles


def convert_vipseg(root: Path, split_videos: dict[str, list[str]], workers: int = 4) -> dict[str, dict]:
    """VIPSeg を 720p に縮め、物体のクラスだけの共通の形式の JSON の中身を作る。

    Args:
        root: VIPSeg の根のフォルダ(imgs/・panomasks/・VIPSeg_720P/panoVIPSeg_categories.json がある)。
        split_videos: {セット名: [動画の名前, ...]}。
        workers: 並べて動かすプロセスの数。

    Returns:
        {セット名: 共通の形式の dict}。file_names は datasets/ からの相対のパス。
    """
    cats = json.loads((root / "VIPSeg_720P" / "panoVIPSeg_categories.json").read_text())
    things = [c for c in cats if c["isthing"]]
    thing_values = {c["id"] + 1 for c in things}  # マスクのクラスの値は id + 1
    categories = [{"id": c["id"] + 1, "name": c["name"], "isthing": 1} for c in things]  # 共通の形式の id = マスクのクラスの値
    rel = root.relative_to(root.parents[1])  # datasets/ からの相対(vipseg/VIPSeg)
    out = {}
    for split, videos in split_videos.items():
        jobs = [(str(root), v, f.stem, thing_values) for v in videos for f in sorted((root / "imgs" / v).glob("*.jpg"))]
        frames = {v: [] for v in videos}
        with Pool(workers) as pool:
            for (video, w, h, rles), job in zip(pool.imap(_vipseg_frame, jobs, chunksize=16), jobs):
                frames[video].append((job[2], w, h, rles))
        vids, anns = [], []
        for vid, video in enumerate(videos, start=1):
            fr = frames[video]
            vids.append({"id": vid, "name": video, "width": fr[0][1], "height": fr[0][2], "length": len(fr),
                         "file_names": [f"{rel}/VIPSeg_720P/images/{video}/{f[0]}.jpg" for f in fr]})
            for value in sorted({v for f in fr for v in f[3]}):
                segs = [f[3].get(value) for f in fr]
                areas = [int(mask_util.area(s)) if s else None for s in segs]
                boxes = [[float(x) for x in mask_util.toBbox(s)] if s else None for s in segs]
                anns.append({"id": len(anns) + 1, "video_id": vid, "category_id": value // 100, "segmentations": segs,
                             "areas": areas, "bboxes": boxes, "iscrowd": 0})
        out[split] = {"videos": vids, "annotations": anns, "categories": categories}
    return out


def load_video_set(name: str) -> dict:
    """共通の形式のセット(`datasets/sets/{name}.json`)を読む。"""
    return json.loads((DATASETS_DIR / "sets" / f"{name}.json").read_text())


def video_train_transforms(size: int = 512, scale_range: tuple[float, float] = (0.1, 2.0)) -> A.Compose:
    """区間の全フレームに同じ変換をかける学習の変換(画像の段階の LSJ + 左右の反転と同じ部品)。

    `images=(T, H, W, 3)`・`masks=(T, H, W)`(物体の番号の画像)で呼ぶと、全フレームに同じ拡大縮小・切り抜き・反転をかける。
    正規化と tensor への変換は VideoClipDataset で行う。

    Args:
        size: 入力の一辺 S。
        scale_range: 長い辺を S の何倍にするかの範囲。

    Returns:
        A.Compose。
    """
    sizes = list(range(max(1, round(size * scale_range[0])), round(size * scale_range[1]) + 1))
    return A.Compose([
        A.LongestMaxSize(max_size=sizes, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=size, min_width=size, position="top_left", border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(height=size, width=size),
        A.HorizontalFlip(p=0.5),
    ])


def video_eval_transforms(size: int = 512) -> A.Compose:
    """評価の変換(長い辺を S にそろえて、右と下を埋める。画像の段階の eval_transforms と同じ)。"""
    return A.Compose([
        A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=size, min_width=size, position="top_left", border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    ])


def _normalize(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) の uint8 → (T, 3, H, W) の float32(ImageNet の平均・標準偏差で正規化)。"""
    x = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255)  # (T, H, W, 3)
    x = (x - torch.tensor(MEAN)) / torch.tensor(STD)
    return x.permute(0, 3, 1, 2).contiguous()  # (T, H, W, 3) -> (T, 3, H, W)


def sample_clip(length: int, clip_len: int, frame_range: int, rng: random.Random) -> list[int]:
    """動画から学習の区間のフレームの番号を選ぶ(VidEoMT の公式の dataset_mapper の select_frames と同じ規則)。

    - frame_range × 2 + 1 == clip_len(公式の VidEoMT の設定 5 枚・2)、または frame_range < 0(SAM 2 の 8 枚): 連続した clip_len 枚
      (始まりは無作為)
    - それ以外: 基準のフレームを 1 つ無作為に選び、その前後 frame_range 枚以内(基準を除く)から clip_len − 1 枚を
      重複を許して選び、基準と合わせて時刻の順に並べる
    動画が clip_len より短いときは、全部のフレームに、無作為に選んだフレームを重ねて足す。

    Args:
        length: 動画のフレームの数。
        clip_len: 区間のフレームの数 T。
        frame_range: 基準のフレームから離れてよいフレームの数(公式の SAMPLING_FRAME_RANGE)。
        rng: 乱数。

    Returns:
        フレームの番号の並び(長さ T、時刻の順)。
    """
    if length <= clip_len:
        return sorted(rng.choices(range(length), k=clip_len - length) + list(range(length)))
    if frame_range * 2 + 1 == clip_len or frame_range < 0:  # frame_range < 0: 連続した clip_len 枚(SAM 2 の RandomUniformSampler)
        start = rng.randrange(length - clip_len + 1)
        return list(range(start, start + clip_len))
    ref = rng.randrange(length)
    near = [i for i in range(max(0, ref - frame_range), min(length, ref + frame_range + 1)) if i != ref]
    return sorted(rng.choices(near, k=clip_len - 1) + [ref])


class VideoClipDataset(Dataset):
    """学習の区間を返す(1 つの要素 = 1 本の動画から選んだ T 枚のフレーム。公式の VidEoMT の設定では連続した T 枚)。

    物体のマスクは、フレームごとに「物体の番号の画像」(背景 0、物体 k は k)にまとめて画像と同じ変換をかけ、変換の後に物体ごとの
    マスクに戻す(同じフレームで物体どうしが重なる正解は、後の物体が上書きする。VIPSeg はパノプティックなので重ならない)。
    区間のどのフレームにも写っていない物体は除く。

    Args:
        set_name: 共通の形式のセットの名前。
        clip_len: 区間のフレームの数 T(VidEoMT の公式は 5)。
        frame_range: sample_clip の frame_range(公式は 2)。
        size: 入力の一辺 S。
        seed: 乱数の種(要素ごとに seed と番号から作るので、worker の数によらず同じ区間になる)。
    """

    def __init__(self, set_name: str, clip_len: int = 5, frame_range: int = 2, size: int = 512, seed: int = 0):
        d = load_video_set(set_name)
        self.videos = d["videos"]
        self.cat_ids = [c["id"] for c in d["categories"]]
        self.label_of = {c: i for i, c in enumerate(self.cat_ids)}  # カテゴリーの id → 0 始まりのクラスの番号
        self.anns = {v["id"]: [] for v in self.videos}
        for a in d["annotations"]:
            self.anns[a["video_id"]].append(a)
        self.clip_len, self.frame_range, self.size, self.seed = clip_len, frame_range, size, seed
        self.transform = video_train_transforms(size)
        self.epoch = 0  # set_epoch で変えると、同じ動画から別の区間を選ぶ

    def set_epoch(self, epoch: int) -> None:
        """エポックごとに区間の選び方を変える。"""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> dict:
        """Returns: {"images": (T, 3, S, S), "labels": (N,), "masks": (N, T, S, S) の bool, "video_id"}。"""
        rng = random.Random((self.seed, self.epoch, i).__hash__())
        v = self.videos[i]
        if getattr(self, "deva_max_jump", None):  # DEVA の学習: 公式の VOSDataset のフレームの選び方(deva/training.py)
            from segmentation.instance_segmentation.lib_common.models.video.deva.training import sample_frames
            frames = sample_frames(v["length"], self.deva_max_jump, np.random.default_rng(abs(hash((self.seed, self.epoch, i)))),
                                   self.clip_len)
        else:
            frames = sample_clip(v["length"], self.clip_len, self.frame_range, rng)
        imgs = np.stack([np.array(Image.open(DATASETS_DIR / v["file_names"][f]).convert("RGB")) for f in frames])  # (T, H, W, 3)
        anns = self.anns[v["id"]]
        ids = np.zeros((len(frames), v["height"], v["width"]), np.uint16)  # (T, H, W) の物体の番号の画像
        for k, a in enumerate(anns, start=1):
            for t, f in enumerate(frames):
                s = a["segmentations"][f]
                if s:
                    ids[t][mask_util.decode(s).astype(bool)] = k
        out = self.transform(images=imgs, masks=ids)
        imgs, ids = np.asarray(out["images"]), np.asarray(out["masks"])  # (T, S, S, 3)・(T, S, S)
        masks, labels = [], []
        for k, a in enumerate(anns, start=1):
            m = ids == k  # (T, S, S)
            if m.any():
                masks.append(m)
                labels.append(self.label_of[a["category_id"]])
        masks = torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, len(frames), self.size, self.size), dtype=torch.bool)
        return {"images": _normalize(imgs), "labels": torch.tensor(labels, dtype=torch.long), "masks": masks, "video_id": v["id"]}


def collate_clips(batch: list[dict]) -> dict:
    """区間をまとめる: images は (B, T, 3, S, S) に重ね、正解は区間ごとのリストのまま。"""
    return {"images": torch.stack([b["images"] for b in batch]),
            "targets": [{"labels": b["labels"], "masks": b["masks"]} for b in batch],
            "video_ids": [b["video_id"] for b in batch]}


class VideoEvalDataset(Dataset):
    """評価の動画を返す(1 つの要素 = 1 本の動画の全フレーム)。

    Args:
        set_name: 共通の形式のセットの名前。
        size: 入力の一辺 S。
    """

    def __init__(self, set_name: str, size: int = 512):
        d = load_video_set(set_name)
        self.videos, self.size, self.transform = d["videos"], size, video_eval_transforms(size)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> dict:
        """Returns: {"images": (T, 3, S, S), "video_id", "orig_size": (H, W), "valid_size": (縮めた後の画像の部分の h, w)}。"""
        v = self.videos[i]
        imgs = np.stack([np.array(Image.open(DATASETS_DIR / f).convert("RGB")) for f in v["file_names"]])  # (T, H, W, 3)
        out = np.asarray(self.transform(images=imgs)["images"])  # (T, S, S, 3)
        scale = self.size / max(v["height"], v["width"])
        return {"images": _normalize(out), "video_id": v["id"], "orig_size": (v["height"], v["width"]),
                "valid_size": (round(v["height"] * scale), round(v["width"] * scale))}
