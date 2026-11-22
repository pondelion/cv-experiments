"""DoRA(Discover and tRAck。Venkataramanan ら ICLR 2024、"Is ImageNet worth 1 video?")の自己教師ありの学習(1 枚の GPU)。

論文と公式のコード(shashankvkt/DoRA_ICLR24 の `main.py`・`utils.py` の MOT・`sinkhorn_knopp.py`)を参考に、自分で書いたもの
(コードは写していない)。部品は DINOv3 の自作の実装(`lib_common/ssl/dinov3/`)の timm の ViT の包み・DINO の head・学習率などの
変化・パラメータの組分けを使い回す。

1 回の optimizer step(1 つの要素 = 動画から取った T 枚(公式は 8)のフレームの区間):
    1. データ拡張: global crop は、区間の全フレームで同じ位置の random resized crop(300 画素、scale 0.2〜1)→ 同じ位置の
       random resized crop(224 画素、scale 0.4〜1)と左右反転。color jitter・blur(確率 0.1)はフレームごと。local crop n 枚
       (公式は 6。112 画素)は、フレームごとに別の位置
    2. 物体を見つける(先生、勾配なし): ViT の self-attention は multi-head attention で、head(ViT-S/16 は 6 個)ごとに独立に
       attention の重みを計算する。最後の block の [CLS] トークンから各パッチへの attention map は、head ごとに別の物体・部分に
       集まる(DINO の論文)ので、最初のフレームで head を k 個(公式は 3)ランダムに選び、1 head を物体 1 個の候補にする。
       選んだ head の attention map の重みで各パッチの query のベクトルを足し、物体の特徴(prototype)にする。L2 正規化した
       パッチの特徴と prototype の内積を Sinkhorn-Knopp(温度 0.05、30 回)で各パッチの物体への割り当ての重みにし、その重みで
       パッチの特徴を平均して prototype を作り直す
    3. 物体を追う(先生、勾配なし): フレームごとに、prototype と各パッチの key の内積(× 次元^−0.5)の softmax をパッチの上で
       取り、L2 正規化した値が平均より大きいパッチを、その物体のマスクにする
    4. 学生: 物体ごとに global crop のマスクの外の画素を 0 にしたフレーム(マスクしたフレーム)と、local crop を入れる
    5. 損失(DINO(v1)の損失。中心化 + 温度付きの softmax の先生): フレームごとに、先生の global crop(マスクなし)の出力に
       対する、学生の「マスクしたフレーム」の交差エントロピーと「local crop」の交差エントロピーの平均。フレーム・物体で平均する
    6. 勾配の大きさの上限、最初のエポックは head の最後の層を固定、先生を EMA で更新、先生の出力の中心を移動平均で更新

公式のコードとの違い(どちらも、公式のコードの計算が意図と違うと見られる箇所を、意図の計算にした):
    - local crop の損失: 公式は画像の DINO の「同じ crop どうしの組を除く」処理が残っていて、local crop の 1 枚目を常に除いて
      いる。ここでは全部の local crop を使う
    - 物体での平均: 公式はフレームの損失の和を物体ごとに 0 に戻していないので、後の物体ほど重く足される。ここでは物体ごとに
      フレームで平均し、物体で平均する
    - 先生の ViT が RoPE を使う(timm の Eva。DINOv3 の ViT)ときは、RoPE をかける前の q・k を使う(公式の ViT は RoPE を使わない)

形の記号: B = 区間の数、T = フレームの数、K = 物体の数、n = local cropの数、N = パッチの数、C = ViT の次元、P = prototype の数。
"""
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path

# 動画の復号は 1 スレッドにする(並列は DataLoader の worker の数で取る)。ffmpeg の既定は CPU のコアの数だけスレッドを使うので、
# worker が多いと 1 本あたり 50 を超えるスレッドで CPU を取り合い、同じ GPU で動くほかの学習のデータの読み込みまで遅くなる
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # 壊れた動画の ffmpeg の警告(moov atom not found など)を出さない
import cv2  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from lib_common.ssl.dinov3.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from lib_common.ssl.dinov3.modules import DINOHead, MaskableViT  # noqa: E402
from lib_common.ssl.dinov3.train import cosine_schedule, param_groups  # noqa: E402

cv2.setNumThreads(1)  # cv2 の色の変換なども 1 スレッドに


@dataclass
class DoRAConfig:
    """学習の設定(既定は公式の pretrain.slurm.sh と main.py の既定の値。大きさは手元に合わせて変える)。"""
    name: str
    out_dir: str
    backbone: str = "vit_small_patch16_224"
    pretrained: bool = False
    init_weights: str | None = None
    drop_path_rate: float = 0.1
    num_frames: int = 8
    frame_step_sec: float = 1.0  # 区間のフレームの間隔(秒。公式は 60fps の動画で 60 フレームおき = 1 秒)
    local_crops_number: int = 6
    global_crop_size: int = 224
    local_crop_size: int = 112
    num_objects: int = 3  # 物体の数 K(先生の最後の block の self-attention の head から選ぶ数。1 head = 物体 1 個の候補)
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 30
    out_dim: int = 65536
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    student_temp: float = 0.1
    center_momentum: float = 0.9
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.04  # DINO の既定(公式の引数の既定も 0.04)
    warmup_teacher_temp_epochs: int = 30
    momentum_teacher: float = 0.996
    epochs: int = 100
    batch_size: int = 6  # micro-batch size(区間の本数。公式は 1 GPU あたり 6 本 × 8 GPU)
    grad_accum: int = 1  # gradient accumulation steps(1 回の optimizer step = batch_size × grad_accum 本。公式の 48 本に合わせられる)
    lr: float = 5e-4  # 1 回の optimizer step の区間 256 本あたり(DINO と同じく本数に比例させる)
    min_lr: float = 1e-5
    warmup_epochs: int = 10
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    workers: int = 8
    seed: int = 0
    max_vram_gb: float | None = None
    max_steps: int | None = None
    log_every: int = 200  # ログを書く間隔(optimizer step。48 本ずつで約 17 分、1 エポックに約 4 行)
    snapshot_every_epochs: int = 5  # 先生の backbone の重みを残す間隔(エポック。1 個 約 90MB(ViT-S))。last.pt は毎エポック上書き


def read_clip(path: str, num_frames: int, step_sec: float, rng: random.Random) -> np.ndarray:
    """動画から、step_sec 秒おきの num_frames 枚を、ランダムな開始の位置から読む(足りないときは最後のフレームを重ねる)。

    開始の位置へ 1 回だけ飛び(seek)、そこから順に復号して、使うフレームだけを画像にする(grab は復号だけで画像にしない)。
    フレームごとに飛ぶと、そのたびに手前のキーフレームから復号し直すので、CPU の時間が約 3 倍かかる(取り出すフレームは同じ)。

    Returns:
        (T, H, W, 3) の uint8(RGB)。
    """
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])  # 復号は 1 スレッド(並列は worker で取る)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * step_sec))
    span = step * (num_frames - 1)
    start = rng.randint(0, max(0, total - 1 - span))
    targets = [min(start + i * step, max(total - 1, 0)) for i in range(num_frames)]  # 短い動画は最後のフレームを重ねる
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    got, last = {}, None
    for j in range(start, targets[-1] + 1):
        if j not in targets:
            cap.grab()
            continue
        ok, f = cap.read()
        if ok:
            last = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if last is None:
            raise ValueError(f"読めない動画: {path}")
        got[j] = last  # 読めなかったときは直前に読めたフレーム
    cap.release()
    return np.stack([got[j] for j in targets])


class VideoClipAugmentation:
    """DoRA のデータ拡張(公式の DataAugmentationDINO の動画の版)。

    Args:
        cfg: DoRAConfig。
    """

    def __init__(self, cfg: DoRAConfig):
        bicubic = v2.InterpolationMode.BICUBIC
        self.init_crop = v2.RandomResizedCrop(300, scale=(0.2, 1.0), interpolation=bicubic)  # 区間で同じ位置
        self.global_crop = v2.Compose([v2.RandomResizedCrop(cfg.global_crop_size, scale=(0.4, 1.0), interpolation=bicubic),
                                       v2.RandomHorizontalFlip(p=0.5)])  # 区間で同じ位置・反転
        color = v2.Compose([v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), v2.RandomGrayscale(p=0.2)])
        blur = lambda p: v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=p)
        norm = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.global_frame = v2.Compose([color, blur(0.1), norm])  # フレームごと
        self.local_frame = v2.Compose([v2.RandomResizedCrop(cfg.local_crop_size, scale=(0.05, 0.4), interpolation=bicubic),
                                       color, blur(0.5), norm])  # フレームごとに別の位置
        self.n_local = cfg.local_crops_number

    def __call__(self, clip: np.ndarray) -> dict:
        """Args: clip: (T, H, W, 3) の uint8。Returns: {"global": (T, 3, G, G), "local": (n, T, 3, L, L)}。"""
        video = self.init_crop(torch.from_numpy(rearrange(clip, "t h w c -> t c h w")))  # (T, 3, 300, 300) 区間で同じ位置
        g = self.global_crop(video)  # (T, 3, G, G) 区間で同じ位置
        glob = torch.stack([self.global_frame(f) for f in g])
        local = torch.stack([torch.stack([self.local_frame(f) for f in video]) for _ in range(self.n_local)])
        return {"global": glob, "local": local}


class VideoClipDataset(Dataset):
    """動画のファイルの並びから、DoRA の学習の区間を返す。

    Args:
        paths: 動画のファイルの並び。
        cfg: DoRAConfig。
    """

    def __init__(self, paths: list[str], cfg: DoRAConfig):
        self.paths, self.cfg, self.aug = list(paths), cfg, VideoClipAugmentation(cfg)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        rng = random.Random()
        for _ in range(5):  # 読めない動画は別の動画で代える
            try:
                return self.aug(read_clip(self.paths[i], self.cfg.num_frames, self.cfg.frame_step_sec, rng))
            except Exception:
                i = rng.randrange(len(self.paths))
        raise RuntimeError("動画が続けて読めない")


@torch.no_grad()
def sinkhorn_assign(logits: torch.Tensor, eps: float, n_iter: int) -> torch.Tensor:
    """パッチと物体の内積を、Sinkhorn-Knopp でパッチから物体への割り当てにする(画像ごと。公式の SinkhornKnopp)。

    Args:
        logits: (B, N(パッチ), K(物体))。

    Returns:
        (B, K, N)。各物体の行は、パッチの上の割り当ての重み。
    """
    q = logits.float() / eps
    q = torch.exp(q - q.amax(dim=(1, 2), keepdim=True)).transpose(1, 2)  # (B, K, N)
    k, n = q.shape[1], q.shape[2]
    q /= q.sum(dim=(1, 2), keepdim=True)
    for _ in range(n_iter):
        q /= q.sum(dim=2, keepdim=True) * k  # 各物体の総量を 1 / K に
        q /= q.sum(dim=1, keepdim=True) * n  # 各パッチの総量を 1 / N に
    return q * n


@torch.no_grad()
def discover_and_track(teacher_out: dict, frames: torch.Tensor, b: int, cfg: DoRAConfig, patch: int) -> torch.Tensor:
    """物体を見つけて追い、物体ごとにマスクしたフレームを作る(公式の utils.MOT)。

    Args:
        teacher_out: 先生の backbone の出力(return_last_qk)。区間の全フレーム (B·T, ...)。
        frames: 先生に入れた global crop (B·T, 3, G, G)。
        b: 区間の数 B。
        cfg: DoRAConfig。
        patch: パッチの一辺の画素の数。

    Returns:
        物体ごとにマスクしたフレーム (K, B·T, 3, G, G)。
    """
    t = frames.shape[0] // b
    grid = frames.shape[-1] // patch
    first = lambda x: rearrange(x, "(b t) ... -> b t ...", t=t)[:, 0]  # 最初のフレーム
    attn = first(teacher_out["cls_attn"])  # (B, 頭, N)
    heads = random.sample(range(attn.shape[1]), cfg.num_objects)
    proto = torch.einsum("bkn,bnc->bkc", attn[:, heads].float(), first(teacher_out["q"]).float())  # (B, K, C) prototype
    feats = F.normalize(first(teacher_out["patch"]).float(), dim=-1)  # (B, N, C)
    assign = sinkhorn_assign(feats @ proto.transpose(1, 2), cfg.sinkhorn_eps, cfg.sinkhorn_iters)  # (B, K, N)
    proto = assign @ feats  # (B, K, C) 割り当ての重みで作り直した prototype
    keys = rearrange(teacher_out["k"].float(), "(b t) n c -> b t n c", t=t)  # (B, T, N, C)
    sim = torch.einsum("bkc,btnc->bktn", proto, keys) * proto.shape[-1] ** -0.5  # (B, K, T, N)
    w = F.normalize(sim.softmax(-1), dim=-1)
    m = (w > w.mean(dim=-1, keepdim=True)).float()  # (B, K, T, N) 平均より大きいパッチ
    m = rearrange(m, "b k t (h w) -> (k b t) 1 h w", h=grid)
    m = F.interpolate(m, scale_factor=patch, mode="nearest")  # (K·B·T, 1, G, G)
    return rearrange(repeat(frames, "m c h w -> k m c h w", k=cfg.num_objects), "k m c h w -> (k m) c h w").mul(m).unflatten(
        0, (cfg.num_objects, frames.shape[0]))


class DoRA(nn.Module):
    """学生・先生(学生の EMA)と、DoRA の損失。

    Args:
        cfg: DoRAConfig。
    """

    def __init__(self, cfg: DoRAConfig):
        super().__init__()
        self.cfg = cfg
        bb = MaskableViT(cfg.backbone, cfg.pretrained, cfg.drop_path_rate)
        if cfg.init_weights:
            bb.vit.load_state_dict(torch.load(cfg.init_weights, map_location="cpu"), strict=False)
        self.student = nn.ModuleDict({"backbone": bb, "dino_head": DINOHead(bb.embed_dim, cfg.out_dim, cfg.head_hidden_dim,
                                                                          cfg.head_bottleneck_dim, cfg.head_nlayers)})
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, cfg.out_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        """先生 ← momentum × 先生 + (1 − momentum) × 学生。"""
        s, t = list(self.student.parameters()), list(self.teacher.parameters())
        torch._foreach_mul_(t, momentum)
        torch._foreach_add_(t, s, alpha=1 - momentum)

    def forward(self, batch: dict, teacher_temp: float) -> tuple[torch.Tensor, dict]:
        """Args: batch: {"global": (B, T, 3, G, G), "local": (B, n, T, 3, L, L)}。Returns: (損失, 内訳)。"""
        c = self.cfg
        b, t = batch["global"].shape[:2]
        frames = rearrange(batch["global"], "b t c h w -> (b t) c h w")
        with torch.no_grad():
            t_out = self.teacher["backbone"](frames, return_last_qk=True)
            t_logits = self.teacher["dino_head"](t_out["cls"])  # (B·T, P)
            t_prob = F.softmax((t_logits.float() - self.center) / teacher_temp, dim=-1)
            masked = discover_and_track(t_out, frames, b, c, self.student["backbone"].vit.patch_embed.patch_size[0])
        s_obj = self.student["dino_head"](self.student["backbone"](masked.flatten(0, 1))["cls"])  # (K·B·T, P)
        local = rearrange(batch["local"], "b n t c h w -> (n b t) c h w")
        s_loc = self.student["dino_head"](self.student["backbone"](local)["cls"])  # (n·B·T, P)
        ce = lambda s: -(t_prob * F.log_softmax(s.float() / c.student_temp, dim=-1)).sum(-1).mean()  # フレーム・区間で平均
        obj_loss = torch.stack([ce(x) for x in s_obj.chunk(c.num_objects)]).mean()  # 物体ごとに平均して、物体で平均
        loc_loss = torch.stack([ce(x) for x in s_loc.chunk(c.local_crops_number)]).mean()
        loss = 0.5 * (obj_loss + loc_loss)  # 公式と同じく、マスクしたフレームと local crop を半分ずつ
        batch_center = t_logits.float().mean(0, keepdim=True).detach()  # (1, P) 中心の更新は optimizer step ごとに(update_center)
        return loss, {"object": float(obj_loss.detach()), "local": float(loc_loss.detach())}, batch_center

    @torch.no_grad()
    def update_center(self, batch_center: torch.Tensor) -> None:
        """先生の出力の中心の移動平均を更新する(1 回の optimizer step に 1 回。公式は全 GPU の batch の平均で 1 回)。

        Args:
            batch_center: その optimizer step の全部の micro-batch の先生の出力の平均 (1, P)。
        """
        c = self.cfg.center_momentum
        self.center.mul_(c).add_(batch_center, alpha=1 - c)


def train_dora(cfg: DoRAConfig, dataset: Dataset) -> Path:
    """学習して、先生の backbone の重み(timm の ViT の state_dict)を {out_dir}/{name}/ に保存する。

    保存するもの: 1 エポックごとに、途中から再開する状態 `last.pt`(上書き。学生・先生・head・中心・optimizer・step)と、最新の
    `teacher_backbone.pt`・`student_backbone.pt`(上書き)。snapshot_every_epochs エポックごとに `teacher_backbone_ep{N}.pt` を残す。
    `last.pt` があれば、そこから続ける(途中で止めて、同じ設定で動かし直せば続きから学習する)。
    """
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    if cfg.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(cfg.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1, ensure_ascii=False))
    model = DoRA(cfg).cuda().train()
    pg_cfg = type("PG", (), {"layerwise_decay": 1.0, "patch_embed_lr_mult": 1.0, "dino_head_wd_multiplier": 1.0})()
    groups = [g for k, m in model.student.items() for g in param_groups(m, pg_cfg, k == "backbone")]  # DINO と同じ: 層ごとの減衰なし
    opt = torch.optim.AdamW(groups, fused=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.workers,
                        persistent_workers=cfg.workers > 0, pin_memory=True)
    accum = cfg.grad_accum  # gradient accumulation steps(1 回の optimizer step = batch_size × accum 本の区間)
    epoch_len = len(loader) // accum  # 1 エポックの optimizer step の数
    total = cfg.epochs * epoch_len
    lr = cfg.lr * cfg.batch_size * accum / 256  # DINO と同じ、1 回の optimizer step の区間の数に比例させる
    lr_s = cosine_schedule(lr, cfg.min_lr, total, cfg.warmup_epochs * epoch_len)
    last_s = lr_s.copy()
    last_s[: cfg.freeze_last_layer_epochs * epoch_len] = 0
    wd_s = cosine_schedule(cfg.weight_decay, cfg.weight_decay_end, total)
    mom_s = cosine_schedule(cfg.momentum_teacher, 1.0, total)
    temp_ep = np.concatenate([np.linspace(cfg.warmup_teacher_temp, cfg.teacher_temp, cfg.warmup_teacher_temp_epochs),
                              np.ones(max(cfg.epochs - cfg.warmup_teacher_temp_epochs, 0)) * cfg.teacher_temp])  # エポックごと(公式)
    step, start_epoch = 0, 0
    if (out / "last.pt").exists():  # 途中から続ける
        state = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        step, start_epoch = state["step"], state["epoch"]
        print(f"last.pt から再開: エポック {start_epoch}、step {step}", flush=True)
    log = open(out / "train_log.jsonl", "a")
    t0, end = time.time(), min(total, cfg.max_steps or total)
    for epoch in range(start_epoch, cfg.epochs):
        sums, centers = {}, []
        for i, batch in enumerate(loader):
            if i >= epoch_len * accum:  # 最後の半端な micro-batch は使わない
                break
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            if i % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = (last_s if g["is_last_layer"] else lr_s)[step] * g["lr_mult"]
                    g["weight_decay"] = wd_s[step] * g["wd_mult"]
                opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts, center = model(batch, float(temp_ep[min(epoch, len(temp_ep) - 1)]))
            (loss / accum).backward()  # accum 回分の勾配をためる(micro-batch は同じ大きさなので、全体の平均の勾配になる)
            centers.append(center)
            for k, v in {"loss": float(loss.detach()), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v / accum
            if (i + 1) % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg.clip_grad)
            opt.step()
            model.update_ema(float(mom_s[step]))
            model.update_center(torch.cat(centers).mean(0, keepdim=True))  # その optimizer step の全部の micro-batch の平均
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                rec = {"step": step, "epoch": epoch, **sums, "lr": float(lr_s[step - 1]),
                       "time": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
            sums, centers = {}, []
            if step >= end:
                break
        torch.save(model.teacher["backbone"].vit.state_dict(), out / "teacher_backbone.pt")
        torch.save(model.student["backbone"].vit.state_dict(), out / "student_backbone.pt")
        if (epoch + 1) % cfg.snapshot_every_epochs == 0:
            torch.save(model.teacher["backbone"].vit.state_dict(), out / f"teacher_backbone_ep{epoch + 1}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "epoch": epoch + 1},
                   out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")  # 書きかけで止まっても前の last.pt が残るように
        if step >= end:
            break
    return out / "teacher_backbone.pt"
