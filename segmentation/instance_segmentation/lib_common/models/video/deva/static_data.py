"""DEVA の伝搬のネットワークの静止画の段階(公式の stage 0)の学習データ。exp00502。

参考: 公式のコード(hkchengrex/Tracking-Anything-with-DEVA の `deva/dataset/static_dataset.py` の StaticTransformDataset、
`deva/dataset/tps.py`、`deva/utils/configuration.py` の s0 の既定)。コードは写さず、規則を読み取って書いた。

公式の stage 0: 1 枚の静止画と 1 個の前景のマスク(顕著物体などのデータセット)から、無作為な変換で 3 フレームの疑似の動画を作り、
動画の段階と同じ forward・損失で学習する(物体は 1 個。参照のフレームは 2 枚。lr 2e-5、80,000 step、batch 16)。
ここでの違い(私の判断): 公式の静止画のデータセット(FSS・DUTS・ECSSD・BIG・HRSOD)は使わず、この exp の学習のセット(VIPSeg)の
フレームから無作為に 1 枚、そこに写る物体から無作為に 1 個を選び、その物体のマスクを前景にする。入力の一辺は公式の 384 ではなく
exp の size。

1 サンプルの作り方(公式と同じ):
    各フレームで、まず全フレーム共通の変換(同じ乱数の種: 拡大縮小 0.5〜2 の affine・左右反転、色の変化 0.1/0.05/0.05/0.05・
    確率 0.05 の白黒)、次にフレームごとの変換(別の種: 回転 ±20°・拡大縮小 0.5〜2・shear 10 の affine、短い辺を S に縮めて
    S × S を切り出す、色の変化 0.1/0.05/0.05)、確率 0.33 で TPS の歪み(12 点、ずれの標準偏差 0.02)。画像の空いた所は公式の
    平均色 (124, 116, 104)、マスクは 0 で埋める。

形の記号: T = フレームの数(3)、S = 入力の一辺。
"""
import random

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from scipy.interpolate import RBFInterpolator
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from lib_common import DATASETS_DIR
from segmentation.instance_segmentation.lib_common.video_data import _normalize, load_video_set

S0_NUM_FRAMES, S0_NUM_REF_FRAMES, S0_MAX_NUM_OBJ = 3, 2, 1  # 公式の s0 の既定
IM_MEAN = (124, 116, 104)  # 公式の埋める色
TPS_PROB, TPS_SCALE, TPS_POINTS = 0.33, 0.02, 12


def random_tps_warp(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """公式の random_tps_warp: 無作為な 12 点を正規化の座標(0〜1)で選び、正規分布のずれを足した点へ写す thin-plate spline で歪める。

    公式(thinplate パッケージ)と同じく、出力の各画素の座標から元の画像の座標へのずれを、ずらした後の点で 0 次 + 1 次の多項式 +
    r² log r の基底(平滑化なし)で補間し、cv2.remap で画像は bilinear、マスクは nearest で読み出す。補間は scipy の
    RBFInterpolator(kernel="thin_plate_spline"、degree=1)を使う(同じ補間の式)。
    Args: img (S, S, 3) の uint8、mask (S, S) の uint8。
    """
    h, w = mask.shape
    src = np.stack([rng.choice(w, TPS_POINTS, replace=False) / w, rng.choice(h, TPS_POINTS, replace=False) / h], 1)  # (12, 2) の (x, y)
    dst = src + rng.normal(scale=TPS_SCALE, size=src.shape)
    shift = RBFInterpolator(dst, src - dst, kernel="thin_plate_spline", degree=1)  # ずらした後の点 → 元の点へのずれ
    gy, gx = np.mgrid[0:h, 0:w]
    grid = np.stack([gx.ravel() / w, gy.ravel() / h], 1)  # (S*S, 2) の出力の画素の正規化の座標
    srcpos = grid + shift(grid)
    mapx = (srcpos[:, 0] * w).reshape(h, w).astype(np.float32)
    mapy = (srcpos[:, 1] * h).reshape(h, w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR), cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)


def _reseed(seed: int) -> None:
    """公式の reseed: torchvision の変換が使う乱数(random と torch)を同じ種にして、画像とマスクに同じ変換をかける。"""
    random.seed(seed)
    torch.manual_seed(seed)


class StaticPseudoVideoDataset(Dataset):
    """静止画 1 枚 + 物体 1 個から 3 フレームの疑似の動画を作る(公式の StaticTransformDataset。物体は 1 個)。

    返す形は VideoClipDataset と同じ({"images": (T, 3, S, S), "labels": (1,), "masks": (1, T, S, S) の bool, "video_id"})。
    1 エポック = 動画の数のサンプル(動画を 1 本ずつ順に、その中のフレームと物体を無作為に選ぶ)。

    Args:
        set_name: 共通の形式のセットの名前(学習のセット)。
        size: 入力の一辺 S。
        num_frames: フレームの数(公式の s0 は 3)。
        seed: 乱数の種(要素ごとに seed・エポック・番号から作る)。
    """

    def __init__(self, set_name: str, size: int = 512, num_frames: int = S0_NUM_FRAMES, seed: int = 0):
        d = load_video_set(set_name)
        self.videos = d["videos"]
        self.label_of = {c["id"]: i for i, c in enumerate(d["categories"])}
        self.anns = {v["id"]: [] for v in self.videos}
        for a in d["annotations"]:
            self.anns[a["video_id"]].append(a)
        self.size, self.num_frames, self.seed, self.epoch = size, num_frames, seed, 0
        s = size
        self.pair_im_lone = transforms.ColorJitter(0.1, 0.05, 0.05, 0)
        self.pair_im_dual = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.5, 2.0), shear=10, interpolation=InterpolationMode.BICUBIC, fill=IM_MEAN),
            transforms.Resize(s, InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomCrop((s, s), pad_if_needed=True, fill=IM_MEAN)])
        self.pair_gt_dual = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.5, 2.0), shear=10, interpolation=InterpolationMode.BICUBIC, fill=0),
            transforms.Resize(s, InterpolationMode.NEAREST),
            transforms.RandomCrop((s, s), pad_if_needed=True, fill=0)])
        self.all_im_lone = transforms.Compose([transforms.ColorJitter(0.1, 0.05, 0.05, 0.05), transforms.RandomGrayscale(0.05)])
        self.all_im_dual = transforms.Compose([transforms.RandomAffine(degrees=0, scale=(0.5, 2.0), fill=IM_MEAN),
                                               transforms.RandomHorizontalFlip()])
        self.all_gt_dual = transforms.Compose([transforms.RandomAffine(degrees=0, scale=(0.5, 2.0), fill=0),
                                               transforms.RandomHorizontalFlip()])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.videos)

    def _pick(self, v: dict, rng: np.random.Generator) -> tuple[int, dict] | None:
        """物体が写っているフレームを無作為に 1 枚、そこに写る物体を無作為に 1 個選ぶ。"""
        anns = self.anns[v["id"]]
        frames = [f for f in range(v["length"]) if any(a["segmentations"][f] for a in anns)]
        if not frames:
            return None
        f = int(rng.choice(frames))
        return f, anns[int(rng.choice([k for k, a in enumerate(anns) if a["segmentations"][f]]))]

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng(abs(hash((self.seed, self.epoch, i))))
        v = self.videos[i]
        picked = self._pick(v, rng)
        if picked is None:  # 物体のない動画(学習のループは物体のない区間を使わない)
            return {"images": torch.zeros((self.num_frames, 3, self.size, self.size)), "labels": torch.zeros(0, dtype=torch.long),
                    "masks": torch.zeros((0, self.num_frames, self.size, self.size), dtype=torch.bool), "video_id": v["id"]}
        f, a = picked
        im = Image.open(DATASETS_DIR / v["file_names"][f]).convert("RGB")
        gt = Image.fromarray(mask_util.decode(a["segmentations"][f]).astype(np.uint8) * 255)
        seq_seed = int(rng.integers(2147483647))
        imgs, masks = [], []
        for _ in range(self.num_frames):
            _reseed(seq_seed)
            this_im = self.all_im_lone(self.all_im_dual(im))
            _reseed(seq_seed)
            this_gt = self.all_gt_dual(gt)
            pair_seed = int(rng.integers(2147483647))
            _reseed(pair_seed)
            this_im = self.pair_im_lone(self.pair_im_dual(this_im))
            _reseed(pair_seed)
            this_gt = self.pair_gt_dual(this_gt)
            this_im, this_gt = np.asarray(this_im), np.asarray(this_gt)
            if rng.random() < TPS_PROB:
                this_im, this_gt = random_tps_warp(this_im, this_gt, rng)
            imgs.append(this_im)
            masks.append(this_gt > 127)  # 公式は ToTensor(0〜1)の後 0.5 で切る
        masks = torch.from_numpy(np.stack(masks))[None]  # (1, T, S, S)
        return {"images": _normalize(np.stack(imgs)), "labels": torch.tensor([self.label_of[a["category_id"]]]),
                "masks": masks, "video_id": v["id"]}
