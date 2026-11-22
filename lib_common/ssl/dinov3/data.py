"""DINOv3 の自己教師ありの学習のデータ: 複数の切り抜きのデータ拡張、iBOT のパッチを隠す位置、まとめる処理。

公式(facebookresearch/dinov3 の `dinov3/data/augmentations.py`・`masking.py`・`collate.py`)と同じ処理を、自分で書いたもの
(コードは写していない)。

データ拡張(1 枚の画像から):
    大きい切り抜き(global)2 枚: RandomResizedCrop(scale 0.32〜1.0、224 画素)+ 左右の反転 + 色の揺らぎ
        (確率 0.8 で明るさ 0.4・コントラスト 0.4・彩度 0.2・色相 0.1、確率 0.2 で白黒)+ ぼかし
        (1 枚目は確率 1.0、2 枚目は確率 0.1)+ 2 枚目だけ確率 0.2 で solarize
    小さい切り抜き(local)n 枚(公式は 8): RandomResizedCrop(scale 0.05〜0.32、96 画素)+ 反転 + 色の揺らぎ + 確率 0.5 でぼかし
    任意: Gram の先生への切り抜き(大きい切り抜きと同じ領域を、別の大きさに)、先生への切り抜きを色の揺らぎなしにする、
          小さい切り抜きを大きい切り抜きの中から取る(パッチの格子に合わせて)
iBOT のパッチを隠す位置: まとめ(batch)の大きい切り抜きの半分(mask_probability)に、隠す割合を 0.1〜0.5 の範囲で等間隔に
    割り当て、縦横比 0.3〜3.3 の長方形のかたまりで隠す(足りない分はランダムな位置で埋める)。残りの半分は隠さない。
"""
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import v2

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass
class CropConfig:
    """データ拡張の設定(既定は公式の ImageNet-1K の設定 vitl_im1k_lin834.yaml)。"""
    global_crops_scale: tuple[float, float] = (0.32, 1.0)
    local_crops_scale: tuple[float, float] = (0.05, 0.32)
    local_crops_number: int = 8
    global_crops_size: int = 224
    local_crops_size: int = 96
    gram_teacher_crops_size: int | None = None  # Gram の先生への切り抜きの大きさ(None なら作らない)
    gram_teacher_no_distortions: bool = False  # Gram の先生への切り抜きに色の揺らぎ・ぼかしをかけない
    teacher_no_color_jitter: bool = False  # 先生への大きい切り抜きに色の揺らぎ・ぼかしをかけない
    local_crops_subset_of_global_crops: bool = False  # 小さい切り抜きを大きい切り抜きの中から取る
    share_color_jitter: bool = False  # 色の揺らぎを切り抜きの前に 1 回だけかける(全部の切り抜きで同じ)
    horizontal_flips: bool = True
    patch_size: int = 16
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def _blur(p: float) -> v2.Transform:
    """確率 p のガウスのぼかし(公式と同じ半径 0.1〜2.0)。"""
    return v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=p)


class MultiCropAugmentation:
    """1 枚の画像(PIL か (3, H, W) の uint8 の tensor)から、学習に使う切り抜きの組を作る。

    Args:
        cfg: CropConfig。
    """

    def __init__(self, cfg: CropConfig):
        self.cfg = cfg
        flip = v2.RandomHorizontalFlip(p=0.5 if cfg.horizontal_flips else 0.0)
        bicubic = v2.InterpolationMode.BICUBIC
        # 大きい切り抜きと Gram の先生への切り抜きは、大きい方の大きさで同じ領域を切り抜いてから、それぞれの大きさにする
        big = max(cfg.global_crops_size, cfg.gram_teacher_crops_size or 0)
        self.geo_global = v2.Compose([v2.RandomResizedCrop(big, scale=cfg.global_crops_scale, interpolation=bicubic), flip])
        self.geo_local = v2.Compose([v2.RandomResizedCrop(cfg.local_crops_size, scale=cfg.local_crops_scale,
                                                          interpolation=bicubic), flip])
        self.to_global = v2.Resize(cfg.global_crops_size, interpolation=bicubic)
        self.to_gram = v2.Resize(cfg.gram_teacher_crops_size, interpolation=bicubic) if cfg.gram_teacher_crops_size else None
        self.color = v2.Compose([v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), v2.RandomGrayscale(p=0.2)])
        self.normalize = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(cfg.mean, cfg.std)])
        jitter = [] if cfg.share_color_jitter else [self.color]  # 共有するときは切り抜きの前に 1 回だけ
        self.distort = [v2.Compose(jitter + [_blur(1.0)]),  # 大きい切り抜きの 1 枚目
                        v2.Compose(jitter + [_blur(0.1), v2.RandomSolarize(threshold=128, p=0.2)])]  # 2 枚目
        self.distort_local = v2.Compose(jitter + [_blur(0.5)])

    def __call__(self, image) -> dict:
        """Returns: {"global_crops": [(3, G, G)] × 2, "global_crops_teacher": 同じ, "local_crops": [(3, L, L)] × n,
        "gram_teacher_crops": [(3, R, R)] × 2(作るときだけ)}。"""
        c = self.cfg
        image = v2.functional.to_image(image)  # (3, H, W) の uint8
        if c.share_color_jitter:
            image = self.color(image)
        out = {"global_crops": [], "global_crops_teacher": [], "gram_teacher_crops": []}
        bases = []
        for k in range(2):
            base = self.geo_global(image)  # 大きい方の大きさで切り抜き・反転
            bases.append(base)
            if self.to_gram is not None and c.gram_teacher_no_distortions:
                # 揺らぎをかけない Gram の切り抜きを先に作り、大きい切り抜きは縮めてから揺らぎをかける(ぼかしの効き方を大きさで変えない)
                out["gram_teacher_crops"].append(self.normalize(self.to_gram(base)))
                distorted = self.distort[k](self.to_global(base))
                out["global_crops"].append(self.normalize(distorted))
            else:
                distorted = self.distort[k](base)  # 揺らぎを共有するときは、大きい方の大きさのまま揺らぎをかけてから縮める
                out["global_crops"].append(self.normalize(self.to_global(distorted)))
                if self.to_gram is not None:
                    out["gram_teacher_crops"].append(self.normalize(self.to_gram(distorted)))
            teacher = self.to_global(base) if c.teacher_no_color_jitter else None
            out["global_crops_teacher"].append(self.normalize(teacher) if teacher is not None else out["global_crops"][-1])
        if c.local_crops_subset_of_global_crops:  # 大きい切り抜きの中の、パッチの格子に合った位置から取る
            out["local_crops"] = []
            for k in range(c.local_crops_number):
                img = self.normalize(self.distort_local(self.to_global(bases[k * 2 // c.local_crops_number])))
                rx, ry = (np.random.randint(0, (c.global_crops_size - c.local_crops_size) // c.patch_size, 2) * c.patch_size)
                out["local_crops"].append(img[:, rx: rx + c.local_crops_size, ry: ry + c.local_crops_size])
        else:
            out["local_crops"] = [self.normalize(self.distort_local(self.geo_local(image))) for _ in range(c.local_crops_number)]
        if not out["gram_teacher_crops"]:
            del out["gram_teacher_crops"]
        return out


class BlockMaskGenerator:
    """パッチの格子 (h, w) に、長方形のかたまりで n 個のパッチを隠す(BEiT・iBOT・DINOv3 と同じ考え方)。

    Args:
        grid: (h, w)。
        min_patches: 1 つのかたまりの最小のパッチの数(公式は 4)。
        min_aspect: かたまりの縦横比の最小(公式は 0.3。最大は 1 / 0.3)。
        max_patches: 1 つのかたまりの最大のパッチの数(公式は全体の 50%)。
    """

    def __init__(self, grid: tuple[int, int], min_patches: int = 4, min_aspect: float = 0.3, max_patches: int | None = None):
        self.h, self.w = grid
        self.min_patches = min_patches
        self.max_patches = max_patches or int(0.5 * self.h * self.w)
        self.log_aspect = (math.log(min_aspect), math.log(1 / min_aspect))

    def _add_block(self, mask: np.ndarray, limit: int) -> int:
        """かたまりを 1 つ足す(最大 10 回試す)。Returns: 新しく隠したパッチの数。"""
        for _ in range(10):
            area = random.uniform(self.min_patches, limit)
            aspect = math.exp(random.uniform(*self.log_aspect))
            bh, bw = int(round(math.sqrt(area * aspect))), int(round(math.sqrt(area / aspect)))
            if bw < self.w and bh < self.h:
                top, left = random.randint(0, self.h - bh), random.randint(0, self.w - bw)
                block = mask[top: top + bh, left: left + bw]
                new = bh * bw - int(block.sum())
                if 0 < new <= limit:
                    block[:] = True
                    return new
        return 0

    def __call__(self, n: int) -> np.ndarray:
        """Returns: (h, w) の bool。n 個のパッチを隠す(かたまりで足りない分はランダムな位置)。"""
        mask = np.zeros((self.h, self.w), dtype=bool)
        count = 0
        while count < n:
            new = self._add_block(mask, min(n - count, self.max_patches))
            if new == 0:
                break
            count += new
        flat = mask.reshape(-1)
        rest = np.flatnonzero(~flat)
        flat[np.random.choice(rest, size=n - int(flat.sum()), replace=False)] = True
        return mask


class DINOCollate:
    """切り抜きの組をまとめ、iBOT のパッチを隠す位置を作る。

    Args:
        mask_generator: BlockMaskGenerator(大きい切り抜きのパッチの格子)。
        mask_ratio: 隠す割合の範囲(公式は 0.1〜0.5)。
        mask_probability: 大きい切り抜きのうち隠すものの割合(公式は 0.5)。
        random_circular_shift: 隠す位置を格子の上でランダムに巡回させる(公式の mask_random_circular_shift)。
    """

    def __init__(self, mask_generator: BlockMaskGenerator, mask_ratio: tuple[float, float] = (0.1, 0.5),
                 mask_probability: float = 0.5, random_circular_shift: bool = False):
        self.gen, self.ratio, self.prob, self.shift = mask_generator, mask_ratio, mask_probability, random_circular_shift

    def __call__(self, samples: list[dict]) -> dict:
        """Returns: {"global_crops": (2·B, 3, G, G)(1 枚目の B 枚、2 枚目の B 枚の順), "global_crops_teacher": 同じ形,
        "local_crops": (n·B, 3, L, L), "gram_teacher_crops": (2·B, 3, R, R)(あれば), "masks": (2·B, N) の bool,
        "mask_indices": 隠したパッチの (2·B·N の中の)番号, "masks_weight": 隠したパッチごとの重み(その切り抜きの隠した数の逆数)}。"""
        stack = lambda key, n: torch.stack([s[key][i] for i in range(n) for s in samples])  # 切り抜きの番号ごとに B 枚ずつ並べる
        out = {"global_crops": stack("global_crops", 2), "global_crops_teacher": stack("global_crops_teacher", 2),
               "local_crops": stack("local_crops", len(samples[0]["local_crops"]))}
        if "gram_teacher_crops" in samples[0]:
            out["gram_teacher_crops"] = stack("gram_teacher_crops", 2)
        b = len(out["global_crops"])
        n_patches = self.gen.h * self.gen.w
        n_masked = int(b * self.prob)
        ratios = torch.linspace(*self.ratio, n_masked + 1)[1:]  # 隠す切り抜きごとの割合(等間隔)
        masks = [torch.from_numpy(self.gen(int(n_patches * r))) for r in ratios] + \
                [torch.zeros(self.gen.h, self.gen.w, dtype=torch.bool) for _ in range(b - n_masked)]
        if self.shift:
            masks = [torch.roll(m, (random.randrange(m.shape[0]), random.randrange(m.shape[1])), (0, 1)) for m in masks]
        random.shuffle(masks)
        masks = rearrange(torch.stack(masks), "b h w -> b (h w)")  # (2·B, N)
        out["masks"] = masks
        out["mask_indices"] = masks.flatten().nonzero().flatten()
        out["masks_weight"] = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
        return out
