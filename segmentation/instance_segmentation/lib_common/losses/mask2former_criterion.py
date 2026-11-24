"""Mask2Former の損失(正解との割り当て + クラスとマスクの損失)。論文と同じ構成。

1. 割り当て(ハンガリアン法): 画像ごとに、クエリ × 正解の物体の費用
       費用 = W_CLASS × (−その正解のクラスの確率) + W_MASK × マスクの BCE + W_DICE × マスクの Dice の損失
   を計算し、費用の合計が最小になる 1 対 1 の組み合わせを選ぶ。マスクの費用は、全画素ではなく、ランダムに選んだ N_POINTS 個の点で計算する
2. クラスの損失: 割り当てられたクエリは正解のクラス、それ以外は「物体なし」を正解とする交差エントロピー(物体なしの重みは EOS_COEF)
3. マスクの損失: 割り当てられた組だけ。点を N_POINTS × OVERSAMPLE 個ランダムに選び、そのうち予測が不確か(値が 0 に近い)な点を
   IMPORTANCE の割合で選び、残りはランダムに選んだ点で、BCE と Dice の損失を計算する(論文の「点を選んで計算する損失」)
4. 補助の損失: 最初のクエリの予測と、decoder の全層の予測に、同じ損失を掛けて足す

形の記号(このファイルで共通):
    B = 画像の数、Q = クエリの数、K = クラスの数、T = 正解の物体の数(全部の画像の分をつなげた数)、
    M = マスクの枚数(関数ごとに、何のマスクかを書く)、P = 点の数、h・w = 予測のマスクの縦・横、H・W = 正解のマスクの縦・横
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
from torch import nn

W_CLASS, W_MASK, W_DICE = 2.0, 5.0, 5.0
EOS_COEF = 0.1
N_POINTS = 112 * 112  # 12544(論文と同じ)
OVERSAMPLE, IMPORTANCE = 3.0, 0.75


def point_sample(masks: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """各マスクの、指定した点の値を双線形の補間で取る。

    Args:
        masks: (M(マスクの枚数), H(縦), W(横))。
        coords: 点の位置 (M, P(点の数), 2(x, y))。0〜1 の割合(0 = 左・上の端、1 = 右・下の端)。

    Returns:
        各点の値 (M, P)。
    """
    grid = rearrange(2 * coords - 1, "m p xy -> m p 1 xy")  # grid_sample は −1〜1 の座標を (M, 出力の縦, 出力の横, 2) で求める
    out = F.grid_sample(rearrange(masks, "m h w -> m 1 h w").float(), grid, align_corners=False)  # -> (M, 1, P, 1)
    return rearrange(out, "m 1 p 1 -> m p")


def uncertain_points(logits: torch.Tensor) -> torch.Tensor:
    """不確かな点を多めに選んだ点の座標を返す(論文と同じ選び方)。

    Args:
        logits: 予測のマスクの値 (M(マスクの枚数), h(縦), w(横))。

    Returns:
        点の位置 (M, N_POINTS, 2(x, y))。先頭 N_POINTS × IMPORTANCE 個が不確かな点、残りがランダムな点。
    """
    n = logits.shape[0]
    n_over = int(N_POINTS * OVERSAMPLE)
    coords = torch.rand(n, n_over, 2, device=logits.device)  # (M, P_over, 2) 候補の点
    uncertainty = -point_sample(logits, coords).abs()  # (M, P_over) 値が 0 に近いほど不確か
    n_imp = int(N_POINTS * IMPORTANCE)
    idx = uncertainty.topk(n_imp, dim=1).indices  # (M, P_imp)
    imp = torch.gather(coords, 1, repeat(idx, "m p -> m p xy", xy=2))  # (M, P_over, 2) -> (M, P_imp, 2)
    rand = torch.rand(n, N_POINTS - n_imp, 2, device=logits.device)  # (M, N_POINTS − P_imp, 2)
    return torch.cat([imp, rand], dim=1)  # (M, N_POINTS, 2)


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """組ごとの Dice の損失(1 − 予測と正解の重なりの割合)。

    Args:
        pred: 予測の値(sigmoid の前)(M(組の数), P(点の数))。
        target: 正解の 0/1 (M, P)。

    Returns:
        組ごとの損失 (M,)。
    """
    p = pred.sigmoid()
    return 1 - (2 * (p * target).sum(-1) + 1) / (p.sum(-1) + target.sum(-1) + 1)


@torch.no_grad()
def match_cost(logits: torch.Tensor, masks: torch.Tensor, tgt_labels: torch.Tensor, tgt_masks: torch.Tensor,
               tgt_image: torch.Tensor, activation: str = "softmax"):
    """割り当ての費用(画像ごとのクエリ × 全部の正解の物体)。

    画像ごとに点の位置を 1 組選び、その画像のクエリと正解の物体を同じ点で比べる(公式と同じ)。速さのため、全部の画像の費用を
    まとめて (B, Q, T) で計算し、CPU へは 1 回だけ移す(ほかの画像の正解との費用も計算されるが、使う側はその部分を使わない)。

    Args:
        logits: クラスの値 (B(画像の数), Q(クエリの数), K+1(クラスの数 + 物体なし))。
        masks: 予測のマスクの値 (B, Q, h, w)。
        tgt_labels: 全部の画像の正解の物体のクラスをつなげたもの (T(正解の物体の数),)。
        tgt_masks: 同じ順の正解のマスク (T, H, W) の float。
        tgt_image: 正解の物体がどの画像のものか (T,)。
        activation: クラスの値の付け方。"softmax"(既定)か "sigmoid"(クラスの値が (B, Q, K) の sigmoid の方式。GTR 系の画像の
            モデルを segmenter に使う DVIS の学習の前半の割り当て。クラスの費用を −sigmoid の確率にする。softmax の方式の
            −softmax の確率と同じ形。私の判断)。

    Returns:
        費用 (B, Q, T) の numpy の配列。
    """
    b, q = logits.shape[:2]
    coords = torch.rand(b, N_POINTS, 2, device=masks.device)  # (B, P, 2) 画像ごとの点の位置
    pm = point_sample(rearrange(masks, "b q h w -> (b q) h w"),
                      repeat(coords, "b p xy -> (b q) p xy", q=q))  # (B × Q, P) 各クエリを自分の画像の点で
    pm = rearrange(pm, "(b q) p -> b q p", b=b)  # (B × Q, P) -> (B, Q, P)
    tm = point_sample(tgt_masks, coords[tgt_image])  # (T, P) 各物体を自分の画像の点で
    probs = logits.float().sigmoid() if activation == "sigmoid" else logits.float().softmax(-1)
    cost_class = -probs[:, :, tgt_labels]  # (B, Q, K+1)(sigmoid は (B, Q, K))-> (B, Q, T)
    # BCE の費用: 予測の値 x、正解 t で、点の平均の BCE = mean(softplus(x)) − mean(x t)
    cross = torch.einsum("bqp,tp->bqt", pm, tm)  # (B, Q, P) × (T, P) -> (B, Q, T) 点ごとの積の和
    cost_mask = rearrange(F.softplus(pm).mean(-1), "b q -> b q 1") - cross / N_POINTS  # (B, Q, T)
    p = pm.sigmoid()
    inter = torch.einsum("bqp,tp->bqt", p, tm)  # (B, Q, T) 予測の確率と正解の重なり
    cost_dice = 1 - (2 * inter + 1) / (rearrange(p.sum(-1), "b q -> b q 1") + tm.sum(-1) + 1)  # (B, Q, T)
    # (B, Q, T)。autocast の中で呼ばれる(DVIS-DAQ の追跡の部の中の割り当て)と einsum が bf16 になるので、float にしてから numpy に
    return (W_CLASS * cost_class + W_MASK * cost_mask + W_DICE * cost_dice).float().cpu().numpy()


@torch.no_grad()
def match(logits: torch.Tensor, masks: torch.Tensor, tgt_labels: torch.Tensor, tgt_masks: torch.Tensor,
          tgt_image: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """画像のまとめ(B 枚)の割り当て(ハンガリアン法)。

    Args:
        logits: match_cost と同じ。
        masks: match_cost と同じ。
        tgt_labels: match_cost と同じ。
        tgt_masks: match_cost と同じ。
        tgt_image: match_cost と同じ。

    Returns:
        画像ごとの (クエリの番号 (その画像の正解の数,), 画像の中での正解の番号 (同じ形,))。
    """
    b = logits.shape[0]
    if len(tgt_labels) == 0:
        return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))] * b
    cost = match_cost(logits, masks, tgt_labels, tgt_masks, tgt_image)  # (B, Q, T)
    owner = tgt_image.cpu().numpy()
    out = []
    for i in range(b):
        cols = (owner == i).nonzero()[0]  # この画像の正解の物体(全体の中の番号)
        qi, ti = linear_sum_assignment(cost[i][:, cols]) if len(cols) else ([], [])  # (Q, その画像の正解の数) の費用で割り当て
        out.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(ti, dtype=torch.long)))
    return out


class Mask2FormerCriterion(nn.Module):
    """全部の予測(最初のクエリ + 各層)に損失を掛けて足す。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
    """

    def __init__(self, num_classes: int):
        super().__init__()
        weight = torch.ones(num_classes + 1)
        weight[-1] = EOS_COEF
        self.register_buffer("class_weight", weight)
        self.num_classes = num_classes

    def _one(self, out: dict, labels: torch.Tensor, tmasks: torch.Tensor, owner: torch.Tensor, starts: list[int],
             indices: list | None = None) -> dict:
        """1 つの予測(1 つの層の出力)の損失。

        Args:
            out: {"logits": (B(画像の数), Q(クエリの数), K+1(クラスの数 + 物体なし)), "masks": (B, Q, h, w)}。
            labels: 全部の画像の正解のクラスをつなげたもの (T(正解の物体の数),)。
            tmasks: 同じ順の正解のマスク (T, H, W) の float。
            owner: 正解の物体がどの画像のものか (T,)。
            starts: 画像ごとの、つなげた正解の中での先頭の番号(長さ B)。
            indices: 画像ごとの割り当て [(クエリの番号, 画像の中での正解の番号)](None なら、この予測でハンガリアン法の割り当てをする。
                動画の VidEoMT の損失のように、割り当てを別の規則で決めるときに渡す)。

        Returns:
            {"class": クラスの損失, "mask": マスクの BCE, "dice": マスクの Dice の損失}(どれも 0 次元)。
        """
        logits, masks = out["logits"], out["masks"]  # (B, Q, K+1), (B, Q, h, w)
        b, q = logits.shape[:2]
        tgt_classes = torch.full((b, q), self.num_classes, dtype=torch.long, device=logits.device)  # (B, Q) 既定は「物体なし」
        src_masks, tgt_idx = [], []
        self.last_indices = []  # 画像ごとの (クエリの番号, 全体の中の正解の番号)。損失を足すサブクラス(iFAN)が使う
        if indices is None:
            indices = match(logits, masks, labels, tmasks, owner)
        for i, (qi, ti) in enumerate(indices):
            ti = ti.to(logits.device) + starts[i]  # 全体の中の番号に
            self.last_indices.append((qi.to(logits.device), ti))
            tgt_classes[i, qi] = labels[ti]
            src_masks.append(masks[i, qi])  # (その画像の正解の数, h, w)
            tgt_idx.append(ti)
        # cross_entropy はクラスの軸を 2 番目に求める: (B, Q, K+1) -> (B, K+1, Q)
        loss_class = F.cross_entropy(rearrange(logits.float(), "b q k -> b k q"), tgt_classes, weight=self.class_weight)
        src, tgt = torch.cat(src_masks), tmasks[torch.cat(tgt_idx)]  # 割り当てた組: (M, h, w), (M, H, W)
        n = max(len(src), 1)
        if len(src) == 0:
            zero = masks.sum() * 0
            return {"class": loss_class, "mask": zero, "dice": zero}
        with torch.no_grad():
            coords = uncertain_points(src.float())  # (M, P, 2)
            tp = point_sample(tgt, coords)  # (M, H, W) -> (M, P)
        sp = point_sample(src, coords)  # (M, h, w) -> (M, P)
        loss_mask = F.binary_cross_entropy_with_logits(sp, tp, reduction="none").mean(-1).sum() / n
        loss_dice = dice_loss(sp, tp).sum() / n
        return {"class": loss_class, "mask": loss_mask, "dice": loss_dice}

    def forward(self, outs: list[dict], targets: list[dict]) -> tuple[torch.Tensor, dict]:
        """全部の予測の損失を足す。

        Args:
            outs: モデルの予測の並び(Mask2Former.forward の返り値。各要素は "logits" (B, Q, K+1)・"masks" (B, Q, h, w))。
            targets: 画像ごとの {"masks": (その画像の正解の数, H, W) の 0/1, "labels": (その画像の正解の数,)}(長さ B)。

        Returns:
            (損失の合計(0 次元), 最終の層の内訳 {"class", "mask", "dice"} の float)。
        """
        # 正解は全部の画像の分をつなげ、マスクは 1 回だけ float にする(全部の予測で使い回す)
        labels = torch.cat([t["labels"] for t in targets])  # (T,)
        tmasks = torch.cat([t["masks"] for t in targets]).float()  # (T, H, W)
        counts = [len(t["labels"]) for t in targets]
        owner = torch.repeat_interleave(torch.arange(len(targets), device=labels.device),
                                        torch.as_tensor(counts, device=labels.device))  # (T,) 各物体の画像の番号
        starts = [sum(counts[:i]) for i in range(len(counts))]
        total, parts = 0.0, {}
        for k, out in enumerate(outs):
            l = self._one(out, labels, tmasks, owner, starts)
            total = total + W_CLASS * l["class"] + W_MASK * l["mask"] + W_DICE * l["dice"]
            if k == len(outs) - 1:
                parts = {key: float(v.detach()) for key, v in l.items()}  # 最終の層の内訳を記録する
        return total, parts
