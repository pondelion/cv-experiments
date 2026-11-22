"""DINOv3 の自己教師ありの学習(1 枚の GPU)。学生・先生・Gram の先生、学習率などの変化、パラメータの組分け、学習のループ。

公式(facebookresearch/dinov3 の `dinov3/train/ssl_meta_arch.py`・`train.py`・`param_groups.py`・`cosine_lr_scheduler.py`)と
同じ学習を、自分で書いたもの(コードは写していない)。既定の値は公式の ImageNet-1K の設定(`vitl_im1k_lin834.yaml`)。

1 回の optimizer step:
    1. 先生(学生の EMA)に大きい切り抜き 2 枚を入れ、[CLS] を DINO の head、隠すパッチの位置の特徴を iBOT の head に通し、
       それぞれ Sinkhorn-Knopp で確率にする(勾配なし)
    2. 学生に、隠した大きい切り抜き 2 枚と、小さい切り抜き n 枚を入れる
    3. 損失 = DINO(小さい切り抜き × 先生の大きい切り抜き)× 重み + DINO(大きい切り抜きどうし、同じ切り抜きを除く)× 重み
             + KoLeo(学生の大きい切り抜きの [CLS])× 0.1 × 2 + iBOT(隠したパッチ)+ Gram(使うとき)
       DINO の 2 つの重みは、組の数の割合(大きい × 大きい と 大きい × 小さい)
    4. 勾配の大きさの上限(公式は 3.0)をかけて更新し、先生を EMA で更新する。Gram の先生は決めた step で更新する
学習率などの変化(step ごと): learning rate(線形の warmup → コサインで min_lr へ)、weight decay(コサインで 0.04 → 0.4)、
    先生の EMA の係数(コサインで 0.992 → 1)、先生の温度(線形の warmup で 0.04 → 0.07、その後一定)。最初の
    freeze_last_layer_epochs の間は head の最後の層の学習率を 0 にする
パラメータの組分け(学生の backbone・DINO の head・iBOT の head ごと): 層ごとの学習率の減衰(i 番目の層は
    layerwise_decay^(層の数 + 1 − (i + 1))、埋め込み・先頭のトークン・mask token は 0 番目の層)、パッチの埋め込みは学習率 ×
    patch_embed_lr_mult、bias・正規化の層・layer scale は weight decay 0
"""
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lib_common.ssl.dinov3.data import BlockMaskGenerator, CropConfig, DINOCollate, MultiCropAugmentation
from lib_common.ssl.dinov3.losses import GramLoss, dino_loss, ibot_loss, koleo_loss, sinkhorn_knopp
from lib_common.ssl.dinov3.modules import DINOHead, MaskableViT


@dataclass
class DINOv3Config:
    """学習の設定(既定は公式の vitl_im1k_lin834.yaml。大きさ・まとめの数は手元に合わせて変える)。"""
    name: str
    out_dir: str
    backbone: str = "vit_small_patch16_224"  # timm の ViT の名前
    pretrained: bool = False  # timm の学習済みの重みから始める(追加の事前学習)
    init_weights: str | None = None  # 学生・先生の backbone の初期値の重み(timm の state_dict。pretrained より優先)
    drop_path_rate: float = 0.3
    crops: CropConfig = field(default_factory=CropConfig)
    # DINO・iBOT の head
    head_n_prototypes: int = 65536
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    dino_loss_weight: float = 1.0
    dino_global_ignore_diagonal: bool = True
    koleo_loss_weight: float = 0.1
    ibot_loss_weight: float = 1.0
    mask_probability: float = 0.5
    mask_ratio: tuple[float, float] = (0.1, 0.5)
    mask_random_circular_shift: bool = False
    reweight_dino_local_loss: dict | None = None  # 小さい切り抜きの DINO の損失の重みの変化 {start, peak, end, warmup_epochs}
    # Gram anchoring
    gram_loss_weight: float = 0.0  # 0 なら使わない
    gram_ema_teacher: bool = False  # Gram の先生に、EMA の先生そのものを使う
    gram_ckpt: str | None = None  # Gram の先生の初期値の重み(backbone の state_dict)
    gram_it_load_ema_teacher: int = -1  # この step で EMA の先生を Gram の先生に写す
    gram_update_frequency: int = 0  # 0 より大きいと、gram_it_first_update の後この間隔で Gram の先生を先生から更新する
    gram_it_first_update: int = 0
    gram_max_updates: int | None = None
    gram_img_level: bool = False
    gram_tokens_used: str = "all"  # "all" / "masked" / "unmasked"
    gram_normalize: bool = True
    gram_remove_neg: bool = False
    gram_remove_only_teacher_neg: bool = False
    gram_resize_method: str = "bicubic"  # Gram の先生のパッチの特徴を学生の格子に合わせる拡大縮小
    gram_resize_antialias: bool = False
    # 先生
    distill_teacher: str | None = None  # 学生と別の、固定した先生の backbone(timm の名前)。None なら学生の EMA(自己蒸留)
    distill_teacher_weights: str | None = None  # 固定した先生の重み(timm の名前で pretrained なら None)
    momentum_teacher: float = 0.992
    final_momentum_teacher: float = 1.0
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30
    # 最適化
    epochs: int = 100
    epoch_length: int = 1250  # 公式の OFFICIAL_EPOCH_LENGTH(1 エポックの step の数)
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_scaling: str | None = "sqrt_wrt_1024"  # lr を batch size で直す("sqrt_wrt_1024": × 4·√(B/1024)、"linear_wrt_256")
    warmup_epochs: int = 10
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    layerwise_decay: float = 0.9
    patch_embed_lr_mult: float = 0.2
    dino_head_wd_multiplier: float = 1.0
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    workers: int = 8
    seed: int = 0
    max_vram_gb: float | None = None
    max_steps: int | None = None  # 動作確認用
    save_every: int = 5000


def cosine_schedule(base: float, final: float, total: int, warmup: int = 0, start_warmup: float = 0.0) -> np.ndarray:
    """線形の warmup(start_warmup → base)の後、コサインで final へ下げる(公式の CosineScheduler)。Returns: (total,)。"""
    w = np.linspace(start_warmup, base, warmup)
    it = np.arange(total - warmup)
    return np.concatenate([w, final + 0.5 * (base - final) * (1 + np.cos(np.pi * it / max(len(it), 1)))])


def warmup_cosine(start: float, peak: float, end: float, warmup: int, total: int) -> np.ndarray:
    """start → peak の線形の warmup の後、コサインで end へ(公式の linear_warmup_cosine_decay。損失の重みの変化に使う)。"""
    w = np.linspace(start, peak, warmup, endpoint=False)
    it = np.arange(total - warmup)
    return np.concatenate([w, end + 0.5 * (peak - end) * (1 + np.cos(np.pi * it / max(len(it), 1)))])


def param_groups(model: nn.Module, cfg: DINOv3Config, is_backbone: bool) -> list[dict]:
    """1 つの部品(backbone か head)のパラメータを、学習率・weight decay の倍率ごとの組に分ける(公式の param_groups.py)。"""
    n_blocks = len(model.vit.blocks) if is_backbone else 0
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr_mult = 1.0
        if is_backbone:
            layer = n_blocks + 1  # 最後の LayerNorm など(倍率 1)
            if any(k in name for k in ("pos_embed", "patch_embed", "mask_token", "cls_token", "reg_token")):
                layer = 0
            elif ".blocks." in f".{name}":
                layer = int(name.split("blocks.")[1].split(".")[0]) + 1
            lr_mult = cfg.layerwise_decay ** (n_blocks + 1 - layer)
            if "patch_embed" in name:
                lr_mult *= cfg.patch_embed_lr_mult
        wd_mult = cfg.dino_head_wd_multiplier if not is_backbone else 1.0
        if name.endswith("bias") or "norm" in name or "gamma" in name:
            wd_mult = 0.0  # bias・正規化の層・layer scale(timm の名前は ls.gamma)は weight decay なし
        last = "last_layer" in name
        key = (lr_mult, wd_mult, last)
        groups.setdefault(key, {"params": [], "lr_mult": lr_mult, "wd_mult": wd_mult, "is_last_layer": last})["params"].append(p)
    return list(groups.values())


class DINOv3SSL(nn.Module):
    """学生・先生・Gram の先生と、損失の計算。

    Args:
        cfg: DINOv3Config。
    """

    def __init__(self, cfg: DINOv3Config):
        super().__init__()
        self.cfg = cfg
        bb = MaskableViT(cfg.backbone, cfg.pretrained, cfg.drop_path_rate)
        if cfg.init_weights:
            bb.vit.load_state_dict(torch.load(cfg.init_weights, map_location="cpu"), strict=False)
        head = lambda: DINOHead(bb.embed_dim, cfg.head_n_prototypes, cfg.head_hidden_dim, cfg.head_bottleneck_dim, cfg.head_nlayers)
        self.student = nn.ModuleDict({"backbone": bb, "dino_head": head(), "ibot_head": head()})
        if cfg.distill_teacher:  # 学生と別の、固定した先生(公式の distillation)
            tb = MaskableViT(cfg.distill_teacher, cfg.distill_teacher_weights is None, 0.0)
            if cfg.distill_teacher_weights:
                tb.vit.load_state_dict(torch.load(cfg.distill_teacher_weights, map_location="cpu"), strict=False)
            th = lambda: DINOHead(tb.embed_dim, cfg.head_n_prototypes, cfg.head_hidden_dim, cfg.head_bottleneck_dim, cfg.head_nlayers)
            self.teacher = nn.ModuleDict({"backbone": tb, "dino_head": th(), "ibot_head": th()})
            self.ema = copy.deepcopy(self.student).requires_grad_(False)  # 先生は固定し、学生の EMA を別に持つ(公式と同じ)
        else:
            self.teacher = copy.deepcopy(self.student)  # 学生の EMA(最初は学生と同じ)
            self.ema = self.teacher
        self.teacher.requires_grad_(False)
        self.gram_teacher = None
        if cfg.gram_loss_weight > 0 and not cfg.gram_ema_teacher:
            self.gram_teacher = copy.deepcopy(self.student["backbone"]).requires_grad_(False)
            self.gram_ready = cfg.gram_ckpt is not None
            if cfg.gram_ckpt:
                self.gram_teacher.vit.load_state_dict(torch.load(cfg.gram_ckpt, map_location="cpu"), strict=False)
        self.gram_loss = GramLoss(cfg.gram_normalize, cfg.gram_img_level, cfg.gram_remove_neg, cfg.gram_remove_only_teacher_neg)
        self.gram_updates = 0

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()  # 先生は常に推論の形(drop path なし)
        if self.gram_teacher is not None:
            self.gram_teacher.eval()
        return self

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        """学生の EMA ← momentum × EMA + (1 − momentum) × 学生(自己蒸留では EMA が先生そのもの)。"""
        s, t = list(self.student.parameters()), list(self.ema.parameters())
        torch._foreach_mul_(t, momentum)
        torch._foreach_add_(t, s, alpha=1 - momentum)

    @torch.no_grad()
    def load_gram_teacher(self) -> None:
        """Gram の先生 ← 学生の EMA の backbone(公式の gram_load_ema_teacher・update_gram(m=0))。"""
        self.gram_teacher.load_state_dict(self.ema["backbone"].state_dict())
        self.gram_ready = True

    def gram_before_step(self, step: int) -> None:
        """optimizer step の前: gram_it_load_ema_teacher の step で、EMA の先生を Gram の先生に写す(公式と同じ時期)。"""
        if self.gram_teacher is not None and step == self.cfg.gram_it_load_ema_teacher:
            self.load_gram_teacher()

    def gram_after_step(self, step: int) -> None:
        """optimizer step と EMA の更新の後: gram_it_first_update から gram_update_frequency ごとに、Gram の先生を写し直す
        (公式は (step + 1) で判定。最大 gram_max_updates 回)。"""
        c = self.cfg
        if (self.gram_teacher is not None and c.gram_update_frequency > 0 and step + 1 >= c.gram_it_first_update
                and (step + 1) % c.gram_update_frequency == 0
                and (c.gram_max_updates is None or self.gram_updates < c.gram_max_updates)):
            self.load_gram_teacher()
            self.gram_updates += 1

    def forward(self, batch: dict, teacher_temp: float, local_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
        """1 回の損失を計算する。

        Args:
            batch: DINOCollate の出力(GPU の上)。
            teacher_temp: 先生の温度。
            local_weight: 小さい切り抜きの DINO の損失の重み(reweight_dino_local_loss の変化)。

        Returns:
            (損失の合計, 内訳 {名前: float})。
        """
        c = self.cfg
        g, loc, masks = batch["global_crops"], batch["local_crops"], batch["masks"]  # (2B, 3, G, G), (nB, 3, L, L), (2B, N)
        b = g.shape[0] // 2
        n_local = loc.shape[0] // b
        idx = batch["mask_indices"]
        # 1. 先生(勾配なし)
        with torch.no_grad():  # 公式は先生にも学生と同じ大きい切り抜きを入れる(teacher_no_color_jitter のときだけ揺らぎなしの切り抜き)
            t_out = self.teacher["backbone"](batch["global_crops_teacher"] if c.crops.teacher_no_color_jitter else g)
            t_cls = sinkhorn_knopp(self.teacher["dino_head"](t_out["cls"]), teacher_temp)  # (2B, K)
            t_patch = sinkhorn_knopp(self.teacher["ibot_head"](t_out["patch"].flatten(0, 1)[idx]), teacher_temp)  # (M, K)
        # 2. 学生(大きい切り抜きはパッチを隠す)
        s_g = self.student["backbone"](g, None if c.distill_teacher else masks)  # 蒸留のときは隠さない(公式と同じ)
        s_l = self.student["backbone"](loc)
        s_cls = self.student["dino_head"](torch.cat([s_g["cls"], s_l["cls"]]))  # (2B + nB, K)
        s_cls_g, s_cls_l = s_cls[: 2 * b], s_cls[2 * b:]
        s_patch = self.student["ibot_head"](s_g["patch"].flatten(0, 1)[idx])  # (M, K)
        # 3. 損失
        t_cls = rearrange(t_cls, "(n b) k -> n b k", n=2)
        global_terms = 2 * (2 - 1) if c.dino_global_ignore_diagonal else 4
        local_terms = 2 * n_local
        parts = {
            "dino_local": dino_loss(rearrange(s_cls_l, "(n b) k -> n b k", n=n_local), t_cls),
            "dino_global": dino_loss(rearrange(s_cls_g, "(n b) k -> n b k", n=2), t_cls,
                                     ignore_diagonal=c.dino_global_ignore_diagonal),
            "koleo": sum(koleo_loss(x) for x in rearrange(s_g["cls"], "(n b) c -> n b c", n=2)) / 2,
            "ibot": ibot_loss(s_patch, t_patch, batch["masks_weight"], 2 * b),
        }
        total = (c.dino_loss_weight * local_terms / (global_terms + local_terms) * local_weight * parts["dino_local"]
                 + c.dino_loss_weight * global_terms / (global_terms + local_terms) * parts["dino_global"]
                 + c.koleo_loss_weight * 2 * parts["koleo"] + c.ibot_loss_weight * parts["ibot"])
        if c.gram_loss_weight > 0:
            parts["gram"] = self._gram(batch, s_g["patch"], t_out["patch"], masks, g.shape[-1])
            total = total + c.gram_loss_weight * parts["gram"]
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def _gram(self, batch: dict, s_patch: torch.Tensor, t_patch: torch.Tensor, masks: torch.Tensor, size: int) -> torch.Tensor:
        """Gram の損失(学生のパッチの特徴と、Gram の先生のパッチの特徴の似かたの表の二乗誤差)。"""
        c = self.cfg
        if c.gram_ema_teacher:
            target = t_patch
        else:
            if not self.gram_ready:
                return s_patch.sum() * 0  # Gram の先生の準備の前(gram_it_load_ema_teacher まで)は 0
            with torch.no_grad():
                target = self.gram_teacher(batch.get("gram_teacher_crops", batch["global_crops_teacher"]))["patch"]
            if target.shape[1] != s_patch.shape[1]:  # 先生の切り抜きが大きいときは、学生のパッチの格子に縮める
                n_t, n_s = int(target.shape[1] ** 0.5), size // c.crops.patch_size
                grid = rearrange(target, "b (h w) c -> b c h w", h=n_t)
                grid = F.interpolate(grid.float(), size=(n_s, n_s), mode=c.gram_resize_method, align_corners=False,
                                     antialias=c.gram_resize_antialias)
                target = rearrange(grid, "b c h w -> b (h w) c")
        if c.gram_tokens_used == "masked":
            return self.gram_loss(s_patch[masks], target[masks])
        if c.gram_tokens_used == "unmasked":
            return self.gram_loss(s_patch[~masks], target[~masks])
        return self.gram_loss(s_patch, target)


class ImageListDataset(Dataset):
    """画像のパスの並びから、DINOv3 の切り抜きの組を返す(ラベルなし)。

    Args:
        paths: 画像のパスの並び。
        crops: CropConfig。
    """

    def __init__(self, paths: list[str], crops: CropConfig):
        self.paths, self.aug = list(paths), MultiCropAugmentation(crops)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        return self.aug(Image.open(self.paths[i]).convert("RGB"))


def train_dinov3(cfg: DINOv3Config, dataset: Dataset) -> Path:
    """学習して、学生の EMA の backbone の重み(timm の ViT の state_dict。自己蒸留では先生)を
    {out_dir}/{name}/teacher_backbone.pt に保存する。

    Args:
        cfg: DINOv3Config。
        dataset: 切り抜きの組を返す Dataset(ImageListDataset など)。

    Returns:
        保存した先生の backbone の重みのパス。
    """
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if cfg.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(cfg.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1, ensure_ascii=False, default=str))
    model = DINOv3SSL(cfg).cuda().train()
    groups = [g for k, m in model.student.items() for g in param_groups(m, cfg, k == "backbone")]
    opt = torch.optim.AdamW(groups, betas=cfg.adamw_betas, fused=True)
    total = cfg.epochs * cfg.epoch_length
    lr = cfg.lr
    if cfg.lr_scaling == "sqrt_wrt_1024":
        lr *= 4 * math.sqrt(cfg.batch_size / 1024)
    elif cfg.lr_scaling == "linear_wrt_256":
        lr *= cfg.batch_size / 256
    lr_s = cosine_schedule(lr, cfg.min_lr, total, cfg.warmup_epochs * cfg.epoch_length)
    last_s = lr_s.copy()
    last_s[: cfg.freeze_last_layer_epochs * cfg.epoch_length] = 0  # 最初は head の最後の層を固定
    wd_s = cosine_schedule(cfg.weight_decay, cfg.weight_decay_end, total)
    mom_s = cosine_schedule(cfg.momentum_teacher, cfg.final_momentum_teacher, total)
    temp_s = cosine_schedule(cfg.teacher_temp, cfg.teacher_temp, total, cfg.warmup_teacher_temp_epochs * cfg.epoch_length,
                             cfg.warmup_teacher_temp)
    r = cfg.reweight_dino_local_loss
    local_s = warmup_cosine(r["start"], r["peak"], r["end"], r["warmup_epochs"] * cfg.epoch_length, total) if r else None
    grid = cfg.crops.global_crops_size // cfg.crops.patch_size
    collate = DINOCollate(BlockMaskGenerator((grid, grid)), cfg.mask_ratio, cfg.mask_probability, cfg.mask_random_circular_shift)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.workers,
                        collate_fn=collate, persistent_workers=cfg.workers > 0, pin_memory=True)
    log = open(out / "train_log.jsonl", "a")
    step, t0 = 0, time.time()
    end = min(total, cfg.max_steps or total)
    while step < end:
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:  # この step の学習率と weight decay
                g["lr"] = (last_s if g["is_last_layer"] else lr_s)[step] * g["lr_mult"]
                g["weight_decay"] = wd_s[step] * g["wd_mult"]
            model.gram_before_step(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model(batch, float(temp_s[step]), float(local_s[step]) if local_s is not None else 1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for m in model.student.values():  # 公式と同じく、部品ごとに勾配の大きさの上限をかける
                torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.clip_grad)
            opt.step()
            model.update_ema(float(mom_s[step]))
            model.gram_after_step(step)
            step += 1
            if step % 20 == 0 or step == 1:
                rec = {"step": step, "loss": float(loss.detach()), **parts, "lr": float(lr_s[step - 1]),
                       "teacher_temp": float(temp_s[step - 1]), "elapsed_s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
            if step % cfg.save_every == 0 or step == end:  # 使う重みは学生の EMA(自己蒸留では先生そのもの)
                torch.save(model.ema["backbone"].vit.state_dict(), out / "teacher_backbone.pt")
                torch.save(model.student["backbone"].vit.state_dict(), out / "student_backbone.pt")
            if step >= end:
                break
    return out / "teacher_backbone.pt"
