"""V-JEPA 2.1(Meta、2026)の自己教師ありの学習(1 枚の GPU)を、timm の ViT の上に自分で書いたもの(コードは写していない)。

参考にした公式のコード: facebookresearch/vjepa2 の `app/vjepa_2_1/`(train.py・models/vision_transformer.py・models/predictor.py・
models/utils/masks_dist.py)と `src/masks/multiseq_multiblock3d.py`、設定は `configs/train_2_1/vitb16/pretrain-256px-16f.yaml`。

1 回の optimizer step(動画の区間のまとめと、画像のまとめを 1 つずつ):
    1. データ: 動画は 4fps で 16 枚(既定)。区間で同じ位置の切り抜き(scale 0.3〜1、縦横比 0.75〜1.35、256 画素)と左右反転。色は
       変えない。画像も同じ切り抜き
    2. 隠す場所: パッチの格子(動画は 2 フレームずつまとめた時間 × 縦 × 横)の上の、時間の全部にわたる直方体のまとまりを
       いくつか重ねる。動画は「面積 15% のまとまり 8 個」と「70% のまとまり 2 個」の 2 通り、画像は「15% × 10 個」の 1 通り。
       隠した場所が予測の対象、残りが学生に見せる場所
    3. 先生(学生の EMA、勾配なし): 全部のパッチを入れ、4 つの層(12 層なら 3・6・9・12 層目)の出力を、層ごとの LayerNorm の後、
       さらに重みなしの LayerNorm をかけてつなげたもの (B, N, 4C) を目標にする
    4. 学生: 見せる場所のパッチだけを入れ、同じ 4 つの層の出力をつなげたもの (B, N_見せる, 4C) を予測器に渡す
    5. 予測器(12 層の小さい ViT): 学生の出力 + 隠した場所の mask token(位置つき)を入れ、隠した場所と見せた場所の両方の
       先生の目標 (4C) を予測する
    6. 損失: 隠した場所の予測の L1 + λ × 見せた場所の予測の L1(パッチごとに、最も近い隠した場所までの格子の距離の平方根の
       逆数で重み付け)。λ は 1.5 万〜3 万 step で 0 から 0.5 に上げる(公式の既定)。隠し方の通りで平均し、動画と画像の損失を
       平均する(公式は GPU の半分ずつで動画と画像を受け持ち、勾配を GPU で平均するので、同じ重みになる)
    7. learning rate は warmup の後は一定(公式の設定は最後の値も同じ)、先生の EMA の係数は一定の 0.99925、勾配の大きさの上限なし

公式との違い:
    - 位置の情報: 公式は 3 次元の RoPE。ここでは timm の ViT の位置の埋め込み(学習する絶対位置か、Eva の 2 次元の RoPE)を
      フレームごとに使い、時間の位置は学習する埋め込みを足す(後の VidEoMT はフレームごとに同じ ViT を使うので、ViT の
      位置の扱いを変えないため)。予測器は位置を 3 次元の固定の sin・cos の埋め込みで入れる(公式は RoPE)
    - 動画のパッチの埋め込み: 公式は 2 フレーム × 16 × 16 の 3 次元の畳み込みを一から学習し、画像は別の 2 次元の埋め込みを使う。
      ここでも別にするが、動画の 3 次元の畳み込みは、ViT の 2 次元の埋め込みの重みを時間の 2 つに半分ずつ分けた値で始める
      (学習済みの ViT から始めるときに、最初から同じ特徴が出るように)
    - 予測器の mask token: 公式は 2 つ作るが、使われるのは 1 つ目だけ(番号が区間の長さの番号で、長さが 1 通りのため)。
      ここでは 1 つだけ作る
    - 先頭のトークン: timm の ViT の [CLS] などはそのまま通す(公式の ViT は持たない)。損失はパッチだけで取る

形の記号: B = まとめの数、T = フレームの数、t = 時間の格子の数(T / 2)、h・w = 縦・横のパッチの数、N = パッチの数(t·h·w)、
C = ViT の次元、D = 予測器の次元、L = 目標にする層の数(4)。
"""
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block
from torch import nn
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from lib_common.ssl.dinov3.data import IMAGENET_MEAN, IMAGENET_STD
from lib_common.ssl.dinov3.train import cosine_schedule
from lib_common.ssl.dora import read_clip


@dataclass
class MaskSpec:
    """隠し方の 1 通り: 面積の割合 spatial_scale のまとまりを num_blocks 個重ねる(時間は全部)。"""
    num_blocks: int
    spatial_scale: float
    aspect_ratio: tuple[float, float] = (0.75, 1.5)


@dataclass
class VJEPAConfig:
    """学習の設定(既定は公式の ViT-B の pretrain-256px-16f.yaml。step の数は手元に合わせて変える)。"""
    name: str
    out_dir: str
    backbone: str = "vit_small_patch16_224"
    pretrained: bool = False
    init_weights: str | None = None
    drop_path_rate: float = 0.0
    num_frames: int = 16
    tubelet_size: int = 2
    fps: float = 4.0
    crop_size: int = 256
    crop_scale: tuple[float, float] = (0.3, 1.0)
    crop_ratio: tuple[float, float] = (0.75, 1.35)
    video_masks: list[MaskSpec] = field(default_factory=lambda: [MaskSpec(8, 0.15), MaskSpec(2, 0.7)])
    image_masks: list[MaskSpec] = field(default_factory=lambda: [MaskSpec(10, 0.15)])
    use_images: bool = True  # 画像のまとめも学習する(公式は GPU の半分)
    batch_size: int = 8  # 動画の区間の数(公式は GPU 1 枚あたり 48)
    image_batch_size: int = 24  # 画像の数(公式は 144。動画の 3 倍)
    pred_depth: int = 12
    pred_dim: int = 384
    pred_heads: int = 12
    target_layers: tuple[int, ...] | None = None  # None なら深さを 4 等分した層(12 層なら 2, 5, 8, 11 の番号)
    lambda_video: float = 0.5
    lambda_image: float = 0.5
    lambda_start_step: int = 15000
    lambda_end_step: int = 30000
    weight_distance: bool = True
    ema: float = 0.99925
    total_steps: int = 300000  # 公式: 1000 エポック × 300 step
    warmup_steps: int = 12000  # 公式: 40 エポック × 300 step
    start_lr: float = 1e-4
    lr: float = 6e-4
    final_lr: float = 6e-4
    lr_horizon_scale: float = 1.25  # コサインの長さ = 1.25 × 全 step(公式の ipe_scale)
    weight_decay: float = 0.04
    final_weight_decay: float = 0.04
    workers: int = 8
    seed: int = 0
    max_vram_gb: float | None = None
    save_every_steps: int = 5000
    log_every: int = 200  # ログを書く間隔(optimizer step)


# ---------------------------------------------------------------- データ


class ClipCrop:
    """区間(か画像)のすべてのフレームを、同じ位置で切り抜き・反転して正規化する(公式の VideoTransform、auto augment なし)。

    Args:
        cfg: VJEPAConfig(切り抜きの大きさ・面積と縦横比の範囲を使う)。
    """

    def __init__(self, cfg: VJEPAConfig):
        self.t = v2.Compose([v2.RandomResizedCrop(cfg.crop_size, scale=cfg.crop_scale, ratio=cfg.crop_ratio, antialias=True),
                             v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """全部のフレームに同じ変換をかける。

        Args:
            frames: (T, H, W, 3) の uint8。

        Returns:
            (3, T, S, S) の正規化した float。
        """
        return rearrange(self.t(torch.from_numpy(rearrange(frames, "t h w c -> t c h w"))), "t c h w -> c t h w")


class VideoClips(Dataset):
    """動画のファイルから、fps おきの num_frames 枚の区間を返す(n_frames=1 なら 1 枚の画像として返す)。

    Args:
        paths: 動画のファイルの並び。
        cfg: VJEPAConfig。
        as_image: True なら、ランダムな 1 フレームを画像 (3, S, S) として返す。
    """

    def __init__(self, paths: list[str], cfg: VJEPAConfig, as_image: bool = False):
        self.paths, self.cfg, self.as_image, self.crop = list(paths), cfg, as_image, ClipCrop(cfg)

    def __len__(self) -> int:
        """動画の数を返す。

        Returns:
            動画の数。
        """
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        """i 番目の動画から区間(か 1 枚)を読み、変換をかける(読めない動画は、別の無作為な動画で 5 回まで代える)。

        Args:
            i: 動画の番号。

        Returns:
            区間 (3, T, S, S)、または画像 (3, S, S)。

        Raises:
            RuntimeError: 5 回続けて読めなかったとき。
        """
        rng = random.Random()
        n = 1 if self.as_image else self.cfg.num_frames
        for _ in range(5):  # 読めない動画は別の動画で代える
            try:
                x = self.crop(read_clip(self.paths[i], n, 1.0 / self.cfg.fps, rng))
                return x[:, 0] if self.as_image else x
            except Exception:
                i = rng.randrange(len(self.paths))
        raise RuntimeError("動画が続けて読めない")


def sample_masks(spec: MaskSpec, grid: tuple[int, int, int], batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """隠す場所を作る(公式の _MaskGenerator。まとまりの大きさはまとめで 1 つ、位置は要素ごと)。

    Args:
        spec: 隠し方。
        grid: パッチの格子 (t, h, w)。
        batch: まとめの数 B。
        gen: 乱数の生成器。

    Returns:
        (見せる場所の番号 (B, N_見せる), 隠す場所の番号 (B, N_隠す))。番号は t·h·w の格子を平らにした順。要素ごとに数が違うので、
        公式と同じく、まとめの中で最も少ない数に切りそろえる。
    """
    t, h, w = grid
    keep = int(h * w * spec.spatial_scale)
    ar = spec.aspect_ratio[0] + torch.rand(1, generator=gen).item() * (spec.aspect_ratio[1] - spec.aspect_ratio[0])
    bh, bw = min(round(math.sqrt(keep * ar)), h), min(round(math.sqrt(keep / ar)), w)
    enc, pred = [], []
    while len(enc) < batch:
        m = torch.ones(t, h, w, dtype=torch.bool)  # True = 見せる
        for _ in range(spec.num_blocks):
            top, left = random.randint(0, h - bh), random.randint(0, w - bw)
            m[:, top: top + bh, left: left + bw] = False
        m = m.flatten()
        if m.any():
            enc.append(m.nonzero().squeeze(1)); pred.append((~m).nonzero().squeeze(1))
    ne, np_ = min(len(e) for e in enc), min(len(p) for p in pred)
    return torch.stack([e[:ne] for e in enc]), torch.stack([p[:np_] for p in pred])


def context_weights(enc: torch.Tensor, pred: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """見せた場所のパッチごとの重み = 1 / sqrt(最も近い隠した場所までの格子の距離)(公式の compute_mask_distance)。

    Args:
        enc: 見せた場所の番号 (B, Ne)。
        pred: 隠した場所の番号 (B, Np)。
        grid: パッチの格子 (t, h, w)。

    Returns:
        (B, Ne)。
    """
    _, h, w = grid
    pos = lambda i: torch.stack([i // (h * w), i % (h * w) // w, i % w], dim=-1).float()  # (B, n, 3) の (時間, 縦, 横)
    d = torch.cdist(pos(enc), pos(pred)).amin(dim=-1)  # (B, Ne)
    return 1.0 / d.sqrt()


# ---------------------------------------------------------------- モデル


class VJEPAEncoder(nn.Module):
    """timm の ViT を、画像 (B, 3, H, W) と動画 (B, 3, T, H, W) の両方を入れ、見せるパッチだけを通せる形で包む。

    Args:
        cfg: VJEPAConfig。
    """

    def __init__(self, cfg: VJEPAConfig):
        super().__init__()
        self.vit = v = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, dynamic_img_size=True,
                                         drop_path_rate=cfg.drop_path_rate)
        if cfg.init_weights:
            v.load_state_dict(torch.load(cfg.init_weights, map_location="cpu"), strict=False)
        c, p = v.embed_dim, v.patch_embed.patch_size
        self.tubelet = cfg.tubelet_size
        self.patch_embed_video = nn.Conv3d(3, c, (cfg.tubelet_size, *p), stride=(cfg.tubelet_size, *p))
        with torch.no_grad():  # 2 次元の埋め込みを時間に半分ずつ分ける(同じフレームが続けば、2 次元と同じ値)
            self.patch_embed_video.weight.copy_(repeat(v.patch_embed.proj.weight, "c i a b -> c i t a b", t=cfg.tubelet_size)
                                                / cfg.tubelet_size)
            self.patch_embed_video.bias.copy_(v.patch_embed.proj.bias)
        self.time_embed = nn.Parameter(trunc_normal_(torch.zeros(cfg.num_frames // cfg.tubelet_size, c), std=0.02))
        self.mod_embed = nn.ParameterDict({k: nn.Parameter(torch.randn(c) * 1e-6) for k in ("image", "video")})
        depth = len(v.blocks)
        self.target_layers = list(cfg.target_layers or [depth // 4 * (i + 1) - 1 for i in range(4)])
        self.norms = nn.ModuleList([copy.deepcopy(v.norm) for _ in self.target_layers])  # 層ごとの LayerNorm(公式の norms_block)
        self.has_rope = getattr(v, "rope", None) is not None

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """見せるパッチだけを ViT に通し、目標の層の出力をつなげる。

        Args:
            x: 画像 (B, 3, H, W) か動画 (B, 3, T, H, W)。
            keep: 通すパッチの番号 (B, N_見せる)(None なら全部)。

        Returns:
            目標の層の出力をつなげたもの (B, N_見せる, L·C)(パッチだけ)。
        """
        v = self.vit
        if x.dim() == 4:
            tok = v.patch_embed(x)  # (B, h, w, C)
            tok = tok.unsqueeze(1)  # (B, 1, h, w, C)
            mod = "image"
        else:
            tok = rearrange(self.patch_embed_video(x), "b c t h w -> b t h w c")
            mod = "video"
        b, t, h, w, c = tok.shape
        npt = 0 if v.no_embed_class else v.num_prefix_tokens
        prefix_pe = None
        if v.pos_embed is not None:  # 学習する絶対位置: フレームごとに同じ 2 次元の位置
            pe = resample_abs_pos_embed(v.pos_embed, new_size=(h, w), old_size=v.patch_embed.grid_size, num_prefix_tokens=npt)
            prefix_pe = pe[:, :npt] if npt else None
            tok = tok + pe[:, npt:].reshape(1, 1, h, w, c)
        if mod == "video":
            tok = tok + self.time_embed[:t].view(1, t, 1, 1, c)
        tok = rearrange(tok + self.mod_embed[mod], "b t h w c -> b (t h w) c")
        rope = None
        if self.has_rope:  # Eva の 2 次元の RoPE を、フレームごとに同じにする
            rope = repeat(v.rope.get_embed(shape=(h, w)), "n d -> (t n) d", t=t)
        if keep is not None:
            tok = torch.gather(tok, 1, repeat(keep, "b n -> b n c", c=c))
            if rope is not None:
                rope = rope[keep].unsqueeze(1)  # (B, 1, N_見せる, d)
        prefix = [z.expand(b, -1, -1) for z in (v.cls_token, v.reg_token) if z is not None]
        if prefix:
            pre = torch.cat(prefix, dim=1)
            tok = torch.cat([pre + prefix_pe if prefix_pe is not None else pre, tok], dim=1)
        tok = v.norm_pre(tok)
        outs, k = [], v.num_prefix_tokens
        for i, blk in enumerate(v.blocks):
            tok = blk(tok, rope=rope) if self.has_rope else blk(tok)
            if i in self.target_layers:
                outs.append(self.norms[self.target_layers.index(i)](tok[:, k:]))
        return torch.cat(outs, dim=-1)

    def export_vit(self) -> dict:
        """後の学習で使う timm の ViT の重みを出す。

        Returns:
            timm の ViT の state_dict(最後の LayerNorm は、最後の目標の層の LayerNorm に置き換える。公式と同じ)。
        """
        sd = self.vit.state_dict()
        for k, val in self.norms[-1].state_dict().items():
            sd[f"norm.{k}"] = val
        return sd


def sincos_3d(grid: tuple[int, int, int], dim: int, device) -> torch.Tensor:
    """3 次元の固定の sin・cos の位置の埋め込み(時間に dim/4、縦・横に 3·dim/8 ずつ)。

    Args:
        grid: パッチの格子 (t, h, w)。
        dim: 埋め込みの次元。
        device: 作る device。

    Returns:
        (t·h·w, dim)。
    """
    def one(n: int, d: int) -> torch.Tensor:
        """1 つの軸の sin・cos の埋め込み。

        Args:
            n: 軸の長さ。
            d: 次元。

        Returns:
            (n, d)。
        """
        omega = 1.0 / 10000 ** (torch.arange(d // 2, device=device, dtype=torch.float32) / (d // 2))
        a = torch.arange(n, device=device, dtype=torch.float32)[:, None] * omega
        return torch.cat([a.sin(), a.cos()], dim=1)  # (n, d)
    t, h, w = grid
    dt = dim // 4 // 2 * 2
    ds = (dim - dt) // 2 // 2 * 2
    dt = dim - 2 * ds
    et, eh, ew = one(t, dt), one(h, ds), one(w, ds)
    return torch.cat([repeat(et, "t d -> (t h w) d", h=h, w=w), repeat(eh, "h d -> (t h w) d", t=t, w=w),
                      repeat(ew, "w d -> (t h w) d", t=t, h=h)], dim=1)


class VJEPAPredictor(nn.Module):
    """予測器: 学生の出力と mask token から、隠した場所と見せた場所の両方の先生の目標を予測する(公式の VisionTransformerPredictor)。

    Args:
        in_dim: 学生の出力の次元 L·C。
        embed_dim: ViT の次元 C。
        cfg: VJEPAConfig。
    """

    def __init__(self, in_dim: int, embed_dim: int, cfg: VJEPAConfig):
        super().__init__()
        d = cfg.pred_dim
        self.embed = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, d))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))  # 公式と同じく 0 で始める
        self.mod_embed = nn.ParameterDict({k: nn.Parameter(torch.randn(d) * 1e-6) for k in ("image", "video")})
        self.blocks = nn.ModuleList([Block(d, cfg.pred_heads, mlp_ratio=4.0, qkv_bias=True,
                                           norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(cfg.pred_depth)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.proj_pred = nn.Linear(d, in_dim)
        self.proj_context = nn.Linear(d, in_dim)
        self.apply(self._init)
        for i, blk in enumerate(self.blocks):  # 公式と同じく、深い層ほど残差の枝の出力を小さく始める
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

    @staticmethod
    def _init(m: nn.Module) -> None:
        """公式と同じ初期化(Linear は切断した正規分布 0.02・bias 0、LayerNorm は 1・0)。

        Args:
            m: 初期化する層。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, enc: torch.Tensor, pred: torch.Tensor, grid: tuple[int, int, int], mod: str):
        """隠した場所と見せた場所の先生の目標を予測する。

        Args:
            z: 学生の出力 (B, Ne, L·C)。
            enc: 見せた場所の番号 (B, Ne)。
            pred: 隠した場所の番号 (B, Np)。
            grid: パッチの格子 (t, h, w)。
            mod: "image" か "video"(種類の埋め込みを選ぶ)。

        Returns:
            (隠した場所の予測 (B, Np, L·C), 見せた場所の予測 (B, Ne, L·C))。
        """
        pos = sincos_3d(grid, self.mask_token.shape[-1], z.device).to(z.dtype)  # (N, D)
        x = self.embed(z) + pos[enc]
        m = self.mask_token + pos[pred]
        ne = x.shape[1]
        x = torch.cat([x, m], dim=1) + self.mod_embed[mod]  # 注意は順番によらないので、公式の位置の順への並べ替えは省く
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj_pred(x[:, ne:]), self.proj_context(x[:, :ne])


class VJEPA(nn.Module):
    """学生・先生(学生の EMA)・予測器と、V-JEPA 2.1 の損失。

    Args:
        cfg: VJEPAConfig。
    """

    def __init__(self, cfg: VJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VJEPAEncoder(cfg)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        c = self.encoder.vit.embed_dim
        self.n_levels = len(self.encoder.target_layers)
        self.predictor = VJEPAPredictor(c * self.n_levels, c, cfg)

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        """先生を学生の EMA で更新する(先生 ← momentum × 先生 + (1 − momentum) × 学生)。

        Args:
            momentum: EMA の係数。
        """
        s, t = list(self.encoder.parameters()), list(self.target.parameters())
        torch._foreach_mul_(t, momentum)
        torch._foreach_add_(t, s, alpha=1 - momentum)

    def branch_loss(self, x: torch.Tensor, specs: list[MaskSpec], lam: float) -> tuple[torch.Tensor, dict]:
        """画像か動画の 1 つのまとめの損失(隠し方の通りで平均)。

        Args:
            x: 画像 (B, 3, S, S) か動画 (B, 3, T, S, S)。
            specs: 隠し方の並び。
            lam: 見せた場所の予測の損失の重み。

        Returns:
            (損失, 内訳)。
        """
        c = self.cfg
        p = self.encoder.vit.patch_embed.patch_size[0]
        grid = (1 if x.dim() == 4 else x.shape[2] // c.tubelet_size, x.shape[-2] // p, x.shape[-1] // p)
        mod = "image" if x.dim() == 4 else "video"
        with torch.no_grad():
            h = self.target(x)  # (B, N, L·C)
            h = torch.cat([F.layer_norm(u, u.shape[-1:]) for u in h.float().chunk(self.n_levels, dim=-1)], dim=-1)
        gen = torch.Generator().manual_seed(random.getrandbits(31))
        losses, pred_l, ctx_l = [], 0.0, 0.0
        for spec in specs:
            enc, pred = (m.to(x.device) for m in sample_masks(spec, grid, x.shape[0], gen))
            z = self.encoder(x, enc)
            zp, zc = self.predictor(z, enc, pred, grid, mod)
            take = lambda idx: torch.gather(h, 1, repeat(idx, "b n -> b n d", d=h.shape[-1]))
            lp = (zp.float() - take(pred)).abs().mean()
            loss = lp
            if lam > 0:
                diff = (zc.float() - take(enc)).abs()  # (B, Ne, L·C)
                if c.weight_distance:
                    diff = diff * context_weights(enc, pred, grid).unsqueeze(-1)
                lc = diff.mean()
                loss = loss + lam * lc
                ctx_l += float(lc.detach()) / len(specs)
            losses.append(loss)
            pred_l += float(lp.detach()) / len(specs)
        return torch.stack(losses).mean(), {f"{mod}_pred": pred_l, f"{mod}_context": ctx_l}

    def forward(self, video: torch.Tensor, image: torch.Tensor | None, step: int) -> tuple[torch.Tensor, dict]:
        """1 step の損失(動画と画像の損失の平均。見せた場所の損失の重みは step に合わせて 0 から上げる)。

        Args:
            video: 動画 (B, 3, T, S, S)。
            image: 画像 (B, 3, S, S) か None。
            step: 今の step。

        Returns:
            (損失, 内訳)。
        """
        c = self.cfg
        ramp = min(max((step - c.lambda_start_step) / max(c.lambda_end_step - c.lambda_start_step, 1), 0.0), 1.0)
        loss, parts = self.branch_loss(video, c.video_masks, c.lambda_video * ramp)
        if image is not None:
            li, pi = self.branch_loss(image, c.image_masks, c.lambda_image * ramp)
            loss = 0.5 * (loss + li)
            parts |= pi
        return loss, parts


# ---------------------------------------------------------------- 学習


def _loop(loader: DataLoader):
    """DataLoader をくり返し回す。

    Args:
        loader: DataLoader。

    Yields:
        loader の要素(終わりなく)。
    """
    while True:
        yield from loader


def train_vjepa(cfg: VJEPAConfig, video_set: Dataset, image_set: Dataset | None = None, max_steps: int | None = None) -> Path:
    """学習して、先生の ViT の重み(timm の state_dict)を {out_dir}/{name}/target_backbone.pt に保存する。

    Args:
        cfg: VJEPAConfig。
        video_set: 動画の区間 (3, T, S, S) を返す Dataset。
        image_set: 画像 (3, S, S) を返す Dataset(cfg.use_images のとき)。
        max_steps: 途中で止める step(確かめるとき。学習率などの変化は total_steps で作る)。

    Returns:
        保存した先生の重みのパス。
    """
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    if cfg.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(cfg.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1, ensure_ascii=False))
    model = VJEPA(cfg).cuda().train()
    model.target.eval()
    named = [(n, p) for n, p in list(model.encoder.named_parameters()) + list(model.predictor.named_parameters()) if p.requires_grad]
    no_wd = lambda n, p: "bias" in n or p.dim() == 1  # 公式と同じく、bias と 1 次元のパラメータは weight decay なし
    opt = torch.optim.AdamW([{"params": [p for n, p in named if not no_wd(n, p)], "wd_mult": 1.0},
                             {"params": [p for n, p in named if no_wd(n, p)], "wd_mult": 0.0}], fused=True)
    horizon = int(cfg.lr_horizon_scale * cfg.total_steps)
    lr_s = cosine_schedule(cfg.lr, cfg.final_lr, horizon, cfg.warmup_steps, cfg.start_lr)
    wd_s = cosine_schedule(cfg.weight_decay, cfg.final_weight_decay, horizon)
    mk = lambda ds, bs: _loop(DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=cfg.workers,
                                         persistent_workers=cfg.workers > 0, pin_memory=True))
    videos = mk(video_set, cfg.batch_size)
    images = mk(image_set, cfg.image_batch_size) if cfg.use_images and image_set is not None else None
    log = open(out / "train_log.jsonl", "a")
    end, t0 = min(cfg.total_steps, max_steps or cfg.total_steps), time.time()
    for step in range(end):
        for g in opt.param_groups:
            g["lr"], g["weight_decay"] = lr_s[step], wd_s[step] * g["wd_mult"]
        video = next(videos).cuda(non_blocking=True)
        image = next(images).cuda(non_blocking=True) if images is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = model(video, image, step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_ema(cfg.ema)
        if step % cfg.log_every == 0 or step + 1 == end:
            rec = {"step": step + 1, "loss": float(loss.detach()), **parts, "lr": float(lr_s[step]),
                   "time": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
        if (step + 1) % cfg.save_every_steps == 0 or step + 1 == end:
            torch.save(model.target.export_vit(), out / "target_backbone.pt")
            torch.save(model.encoder.export_vit(), out / "student_backbone.pt")
    return out / "target_backbone.pt"
