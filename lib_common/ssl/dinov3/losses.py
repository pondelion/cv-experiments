"""DINOv3 の自己教師ありの学習の損失(DINO・iBOT・KoLeo・Gram)。

公式(facebookresearch/dinov3 の `dinov3/loss/`)と同じ計算を、自分で書いたもの(コードは写していない)。公式は DINO と iBOT が
それぞれ Sinkhorn-Knopp を持っていたので 1 つの関数にまとめ、1 枚の GPU で使うので分散の処理は入れない。公式の学習
(SSLMetaArch)は先生の中心化に Sinkhorn-Knopp だけを使う(softmax と中心の値の移動平均の版は使わない)ので、その版も入れない。

    DINO の損失: 先生の [CLS] の head の出力を Sinkhorn-Knopp で「prototype に均等に割り当てた確率」にし、学生の [CLS] の head の
        出力(温度 0.1 の softmax)との交差エントロピー。学生の切り抜き × 先生の切り抜きの全部の組(同じ切り抜きどうしは除ける)
    iBOT の損失: 隠したパッチだけについて、先生(隠さない入力)のパッチの head の出力を Sinkhorn-Knopp で確率にし、学生(隠した入力)
        のパッチの head の出力との交差エントロピー。切り抜きごとに、隠したパッチの数で平均する
    KoLeo の正則化: 学生の [CLS] の特徴(L2 正規化)の、まとめの中の最も近い別の特徴までの距離の対数の平均に、負の符号を付けたもの
        (特徴が一か所に固まらないようにする)
    Gram の損失(Gram anchoring): 学生と Gram の先生のパッチの特徴の、パッチどうしの似かたの表(L2 正規化した特徴の内積)の二乗誤差
"""
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, temp: float, n_iter: int = 3) -> torch.Tensor:
    """先生の出力を、各サンプルの確率の和が 1・各 prototype への割り当ての総量が均等になるように正規化する(SwAV・DINOv2 と同じ)。

    Args:
        logits: (M(サンプルの数), K(prototype の数))。
        temp: 先生の温度。
        n_iter: 行と列の正規化を交互にする回数(公式は 3)。

    Returns:
        (M, K)。各行の和が 1。
    """
    q = torch.exp(logits.float() / temp).t()  # (K, M)
    k, m = q.shape
    q /= q.sum()
    for _ in range(n_iter):
        q /= q.sum(dim=1, keepdim=True) * k  # 各 prototype の総量を 1 / K に
        q /= q.sum(dim=0, keepdim=True) * m  # 各サンプルの総量を 1 / M に
    return (q * m).t()


def dino_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, student_temp: float = 0.1,
              ignore_diagonal: bool = False) -> torch.Tensor:
    """DINO の損失。

    Args:
        student_logits: (S(学生の切り抜きの数), B, K)。
        teacher_probs: (T(先生の切り抜きの数), B, K)(sinkhorn_knopp の出力)。
        student_temp: 学生の温度(公式は 0.1)。
        ignore_diagonal: 同じ番号の切り抜きどうし(s = t)の組を除く(大きい切り抜きどうしで使う)。

    Returns:
        組ごとの交差エントロピーの平均。
    """
    s, b, _ = student_logits.shape
    t = teacher_probs.shape[0]
    logp = F.log_softmax(student_logits.float() / student_temp, dim=-1)
    if not ignore_diagonal:
        return -torch.einsum("sbk,tbk->", logp, teacher_probs) / (b * s * t)
    pair = -torch.einsum("sbk,tbk->st", logp, teacher_probs)  # (S, T) 組ごとの和
    n_diag = min(s, t)
    pair = torch.diagonal_scatter(pair, pair.new_zeros(n_diag))  # 同じ番号の組(対角)を 0 に
    return pair.sum() / (b * s * t - b * n_diag)


def ibot_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, masks_weight: torch.Tensor, n_crops: int,
              student_temp: float = 0.1) -> torch.Tensor:
    """iBOT の損失(隠したパッチだけ)。

    Args:
        student_logits: 隠したパッチの学生の head の出力 (M(隠したパッチの数), K)。
        teacher_probs: 同じパッチの先生の確率 (M, K)。
        masks_weight: パッチごとの重み (M,)(その切り抜きの隠したパッチの数の逆数。切り抜きごとの平均にする)。
        n_crops: 大きい切り抜きの数(2·B。隠していない切り抜きも数える。公式と同じ)。
        student_temp: 学生の温度。

    Returns:
        損失。
    """
    ce = torch.sum(teacher_probs * F.log_softmax(student_logits.float() / student_temp, dim=-1), dim=-1)  # (M,)
    return -(ce * masks_weight).sum() / n_crops


def koleo_loss(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo の正則化(Sablayrolles ら 2018)。

    Args:
        x: (B, C) の特徴(内側で L2 正規化する)。

    Returns:
        −mean(log(最も近い別の特徴までの距離))。
    """
    with torch.autocast("cuda", enabled=False):
        x = F.normalize(x.float(), dim=-1, eps=eps)
        with torch.no_grad():
            dots = x @ x.t()
            dots.fill_diagonal_(-1)  # 自分自身を除く
            nn_idx = dots.argmax(dim=1)
        dist = F.pairwise_distance(x, x[nn_idx], eps=eps)
        return -torch.log(dist + eps).mean()


class GramLoss(nn.Module):
    """Gram の損失(DINOv3 の Gram anchoring)。

    Args:
        normalize: 特徴を L2 正規化してから内積を取る(公式の既定は True)。
        img_level: 似かたの表を画像ごとに作る(False なら、まとめの全部のパッチで 1 つの表。公式の既定は False、
            ViT-7B の Gram anchoring の設定は True)。
        remove_neg: 学生・先生の両方の負の似かたを 0 にする。
        remove_only_teacher_neg: 先生の負の似かただけを 0 にする(公式のコードは先生を先に 0 にしてから「両方が負」の学生を
            0 にするので、実際には学生は変わらない。その動きに合わせる)。
    """

    def __init__(self, normalize: bool = True, img_level: bool = False, remove_neg: bool = False,
                 remove_only_teacher_neg: bool = False):
        super().__init__()
        assert not (remove_neg and remove_only_teacher_neg)
        self.normalize, self.img_level = normalize, img_level
        self.remove_neg, self.remove_only_teacher_neg = remove_neg, remove_only_teacher_neg

    def _gram(self, f: torch.Tensor) -> torch.Tensor:
        """(B, N, C) か (M, C) の特徴 → 似かたの表 (B, N, N) か (M, M)。"""
        f = f.float()
        if self.normalize:
            f = F.normalize(f, dim=-1)
        if not self.img_level and f.dim() == 3:
            f = f.flatten(0, 1)  # (B, N, C) -> (B·N, C)
        return f @ f.transpose(-1, -2)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Args: student・teacher: (B, N, C)(img_level)か (B, N, C)・(M, C)(まとめ全体)。Returns: 二乗誤差の平均。"""
        s, t = self._gram(student), self._gram(teacher)
        if self.remove_neg:
            s, t = s.clamp(min=0), t.clamp(min=0)
        elif self.remove_only_teacher_neg:
            t = t.clamp(min=0)
        return F.mse_loss(s, t)
