"""自作の RF-DETR-Seg 系(models/rfdetr_seg.py)の割り当てと損失(公式の rfdetr の models/matcher.py・criterion.py と同じ式)。

1. 割り当て: クエリの組ごと・画像ごとにハンガリアン法。費用 =
       2 × クラス(focal の費用)+ 5 × 枠の L1 + 2 × (−GIoU)+ 5 × マスクの BCE + 5 × マスクの Dice
   マスクの費用は、画像ごとにランダムに選んだ点(1 マスクあたり mh·mw/16 個)で計算する
2. クラスの損失: IoU を考えた BCE(IA-BCE)。割り当てた組の正解の重み t = p^α × IoU^(1−α)(下限 0.01)、
   それ以外の負の重み p^γ(α = 0.25、γ = 2)
3. 枠の損失: 割り当てた組の L1(5)と 1 − GIoU(2)
4. マスクの損失: 割り当てた組だけマスクを作り、不確かな点を多めに選んだ点(1 マスクあたり mh·mw/16 個)で BCE(5)と Dice(5)。
   正解のマスクの点の値は、最近傍の画素を取る(公式と同じ。公式は画素のちょうど境目の点だけ別の計算で取り直すが、
   ここでは丸めの規則のまま取る。そうなる点はごく少ない)
5. 全部の層と two-stage の encoder の出力に、同じ重みで損失を掛けて足す。損失は「正解の物体の数 × 組の数」で割る

形の記号: B = 画像の数、M = 1 つの出力のクエリの数(学習では G·Q)、K = クラスの数、T = 正解の物体の数(全部の画像の分を
つなげた数)、P = 点の数、mh・mw = マスクの縦・横、S = 入力の一辺、A = 割り当てた組の数
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import box_convert, box_iou, generalized_box_iou, generalized_box_iou_loss, masks_to_boxes

W_CLASS_COST, W_BBOX, W_GIOU, W_MASK_CE, W_DICE = 2.0, 5.0, 2.0, 5.0, 5.0
W_CLASS = 1.0
ALPHA, GAMMA = 0.25, 2.0
POINT_RATIO = 16  # 1 マスクあたりの点の数 = マスクの画素の数 / POINT_RATIO
OVERSAMPLE, IMPORTANCE = 3.0, 0.75


def point_sample(x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """特徴やマスクの値を、点の位置で双線形に取る。

    Args:
        x: (N, C, h, w)。
        coords: 0〜1 の (x, y) の点の位置 (N, P, 2)。

    Returns:
        (N, C, P)。
    """
    grid = rearrange(2 * coords - 1, "n p xy -> n p 1 xy")
    return rearrange(F.grid_sample(x, grid.to(x.dtype), align_corners=False), "n c p 1 -> n c p")


def sample_target_nearest(masks: torch.Tensor, idx: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """正解のマスク(uint8 のまま)から、点の位置の最近傍の画素の値を取る(float に広げると大きすぎるため、画素の番号で取る)。

    Args:
        masks: 正解のマスク (T, S, S) の 0/1。
        idx: 取る物体の番号 (A,)。
        coords: 0〜1 の (x, y) の点の位置 (A, P, 2)。

    Returns:
        (A, P) の float の 0/1。
    """
    h, w = masks.shape[-2:]
    x = torch.round(coords[..., 0] * w - 0.5).long().clamp_(0, w - 1)  # grid_sample(align_corners=False)と同じ画素の位置
    y = torch.round(coords[..., 1] * h - 0.5).long().clamp_(0, h - 1)
    flat = rearrange(idx, "a -> a 1") * (h * w) + y * w + x  # (A, P)
    return masks.reshape(-1)[flat].float()


def uncertain_points(logits: torch.Tensor, n_points: int) -> torch.Tensor:
    """不確かな点(値が 0 に近い点)を多めに選んだ点の位置(Mask2Former・公式の get_uncertain_point_coords_with_randomness)。

    Args:
        logits: マスクの値 (A, 1, mh, mw)。
        n_points: 点の数 P。

    Returns:
        (A, P, 2)。
    """
    a = logits.shape[0]
    n_over = int(n_points * OVERSAMPLE)
    coords = torch.rand(a, n_over, 2, device=logits.device)
    unc = -point_sample(logits, coords)[:, 0].abs()  # (A, P_over)
    n_imp = int(n_points * IMPORTANCE)
    idx = unc.topk(n_imp, dim=1).indices  # (A, P_imp)
    imp = torch.gather(coords, 1, repeat(idx, "a p -> a p xy", xy=2))
    return torch.cat([imp, torch.rand(a, n_points - n_imp, 2, device=logits.device)], dim=1)


def targets_from_masks(targets: list[dict], size: tuple[int, int]) -> dict:
    """画像ごとの正解をつなげ、枠(0〜1 の cx, cy, w, h)をマスクから作る。

    Args:
        targets: 画像ごとの {"masks": (T_i, S, S) の 0/1, "labels": (T_i,)}。
        size: 入力の (縦, 横)。

    Returns:
        {"labels": (T,), "masks": (T, S, S), "boxes": (T, 4), "owner": (T,) 画像の番号, "counts": 画像ごとの物体の数}。
    """
    device = targets[0]["labels"].device
    labels = torch.cat([t["labels"] for t in targets])
    masks = torch.cat([t["masks"] for t in targets])
    counts = [len(t["labels"]) for t in targets]
    owner = torch.repeat_interleave(torch.arange(len(targets), device=device), torch.as_tensor(counts, device=device))
    if len(labels):
        xyxy = masks_to_boxes(masks.bool()).float()  # (T, 4) 画素の x0, y0, x1, y1(右下は含む画素)
        xyxy[:, 2:] += 1  # 画素の端までの枠にする
        scale = torch.tensor([size[1], size[0], size[1], size[0]], device=device, dtype=torch.float32)
        boxes = box_convert(xyxy / scale, "xyxy", "cxcywh")
    else:
        boxes = torch.zeros(0, 4, device=device)
    return {"labels": labels, "masks": masks, "boxes": boxes, "owner": owner, "counts": counts}


def to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """予測の枠 (..., 4) cx, cy, w, h を x0, y0, x1, y1 にする。幅・高さは 0 以上にする。

    枠を 4 辺のずれから作る方式(GTR 系)では幅・高さが負になりえて、そのまま IoU・GIoU を計算すると 1 を超える値になり、
    損失を下げるために枠を裏返すことを学んでしまう(公式の GTR も変換で 0 以上にしている)。

    Args:
        boxes: (..., 4) cx, cy, w, h。

    Returns:
        (..., 4) x0, y0, x1, y1。
    """
    return box_convert(torch.cat([boxes[..., :2], boxes[..., 2:].clamp(min=0)], -1), "cxcywh", "xyxy")


@torch.no_grad()
def full_mask_cost(parts: dict, i: int, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """D-FINE-seg と同じマスクの費用: 1 枚の画像の全部のクエリの予測のマスク(全部の画素)と正解の、focal の形の費用と Dice の費用。

    Args:
        parts: マスクの材料 {"spatial": (B, C, mh, mw), "query": (B, M, C), "bias"}。
        i: 画像の番号。
        target: この画像の正解のマスクを予測の大きさに双線形で縮めたもの (T_i, mh, mw)(0〜1)。
        alpha, gamma: focal の形の重み(D-FINE-seg の matcher と同じ 0.25・2)。

    Returns:
        (focal の形の費用 (M, T_i)(画素の数で割る), Dice の費用 (M, T_i))。
    """
    logit = torch.einsum("chw,mc->mhw", parts["spatial"][i].float(), parts["query"][i].float()) + parts["bias"].float()
    logit = rearrange(logit, "m h w -> m (h w)")  # (M, mh·mw)
    tm = rearrange(target.float(), "t h w -> t (h w)")  # (T_i, mh·mw)
    p = logit.sigmoid()
    neg = (1 - alpha) * p ** gamma * (-(1 - p + 1e-8).log())
    pos = alpha * (1 - p) ** gamma * (-(p + 1e-8).log())
    focal = (torch.einsum("mp,tp->mt", pos, tm) + torch.einsum("mp,tp->mt", neg, 1 - tm)) / logit.shape[1]
    dice = 1 - (2 * torch.einsum("mp,tp->mt", p, tm) + 1e-6) / (p.sum(-1, keepdim=True) + tm.sum(-1) + 1e-6)
    return focal, dice


@torch.no_grad()
def match(out: dict, tg: dict, groups: int,
          weights: tuple[float, float, float, float, float] = (W_CLASS_COST, W_BBOX, W_GIOU, W_MASK_CE, W_DICE),
          mask_cost: str = "points") -> list[tuple[torch.Tensor, torch.Tensor]]:
    """1 つの出力の割り当て(全部の画像・組の費用をまとめて計算し、CPU へは 1 回だけ移す)。

    Args:
        out: {"logits": (B, M, K), "boxes": (B, M, 4), "mask_parts": {"spatial": (B, C, mh, mw), "query": (B, M, C), "bias"}}。
            "mask_parts" がない(None の)出力はマスクの費用を使わない(two-stage の encoder の出力など)。
        tg: targets_from_masks の返り値。
        groups: クエリの組の数 G(M = G·Q)。
        weights: 費用の重み(クラス, 枠 L1, GIoU, マスク BCE(mask_cost="full_focal" では focal の形), Dice)。既定は RF-DETR の値。
        mask_cost: マスクの費用の計算。"points"(既定。画像ごとに選んだ点の BCE と Dice。RF-DETR・GTR と同じ)/
            "full_focal"(D-FINE-seg と同じ。全部の画素の focal の形の費用と Dice。正解のマスクは予測の大きさに双線形で縮める)。

    Returns:
        画像ごとの (クエリの番号 (A_i,), 画像の中での正解の番号 (A_i,))。組ごとに割り当てた結果をつなげたもの。
    """
    logits, boxes, parts = out["logits"], out["boxes"], out.get("mask_parts")
    w_cls, w_bbox, w_giou, w_ce, w_dice = weights
    b, m, _ = logits.shape
    if len(tg["labels"]) == 0:
        return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))] * b
    prob = logits.float().sigmoid()[:, :, tg["labels"]]  # (B, M, T)
    neg = (1 - ALPHA) * prob ** GAMMA * (-(1 - prob + 1e-8).log())
    pos = ALPHA * (1 - prob) ** GAMMA * (-(prob + 1e-8).log())
    cost_class = pos - neg  # (B, M, T)
    pb = rearrange(boxes.float(), "b m four -> (b m) four")
    cost_bbox = rearrange(torch.cdist(pb, tg["boxes"], p=1), "(b m) t -> b m t", b=b)  # (B, M, T)
    giou = generalized_box_iou(to_xyxy(pb), box_convert(tg["boxes"], "cxcywh", "xyxy"))
    cost_giou = -rearrange(giou, "(b m) t -> b m t", b=b)
    cost = w_cls * cost_class + w_bbox * cost_bbox + w_giou * cost_giou  # (B, M, T)
    if parts is not None and mask_cost == "full_focal":
        mh, mw = parts["spatial"].shape[-2:]
        key = ("masks_small", mh, mw)  # 正解ごとに 1 回だけ縮めて覚えておく(損失の box_crop と同じもの)
        if key not in tg:
            tg[key] = F.interpolate(rearrange(tg["masks"].float(), "t h w -> t 1 h w"), size=(mh, mw), mode="bilinear",
                                    align_corners=False)[:, 0].clamp(0, 1)
        target = tg[key]  # (T, mh, mw)
        extra = torch.zeros_like(cost)  # (B, M, T)。画像ごとに、その画像の正解の列だけ埋める(ほかの画像の列は割り当てで使わない)
        for i in range(b):
            cols = (tg["owner"] == i).nonzero()[:, 0]
            if len(cols):
                focal, dice = full_mask_cost(parts, i, target[cols])
                extra[i, :, cols] = w_ce * focal + w_dice * dice
        cost = cost + extra
    elif parts is not None:
        # マスクの費用: 画像ごとに点を選び、予測(画素の特徴を点で取ってクエリと内積)と正解を同じ点で比べる
        mh, mw = parts["spatial"].shape[-2:]
        n_points = mh * mw // POINT_RATIO
        coords = torch.rand(b, n_points, 2, device=logits.device)  # (B, P, 2)
        sp = point_sample(parts["spatial"].float(), coords)  # (B, C, P)
        pm = torch.einsum("bcp,bmc->bmp", sp, parts["query"].float()) + parts["bias"].float()  # (B, M, P)
        tm = sample_target_nearest(tg["masks"], torch.arange(len(tg["labels"]), device=logits.device),
                                   coords[tg["owner"]])  # (T, P)
        cost_ce = F.softplus(pm).mean(-1, keepdim=True) - torch.einsum("bmp,tp->bmt", pm, tm) / n_points  # (B, M, T)
        p = pm.sigmoid()
        cost_dice = 1 - (2 * torch.einsum("bmp,tp->bmt", p, tm) + 1) / (p.sum(-1, keepdim=True) + tm.sum(-1) + 1)
        cost = cost + w_ce * cost_ce + w_dice * cost_dice
    cost = cost.nan_to_num(1.0).cpu().numpy()  # (B, M, T)。公式と同じく NaN は 1 にする
    owner = tg["owner"].cpu().numpy()
    q = m // groups
    result = []
    for i in range(b):
        cols = (owner == i).nonzero()[0]
        qs, ts = [], []
        if len(cols):
            for g in range(groups):  # 組ごとに割り当てる
                qi, ti = linear_sum_assignment(cost[i, g * q:(g + 1) * q][:, cols])
                qs.append(qi + g * q)
                ts.append(ti)
        result.append((torch.as_tensor(sum(map(list, qs), []), dtype=torch.long),
                       torch.as_tensor(sum(map(list, ts), []), dtype=torch.long)))
    return result


class RFDETRCriterion(nn.Module):
    """自作の RF-DETR-Seg 系の損失。

    Args:
        groups: 学習のときのクエリの組の数 G。
    """

    def __init__(self, groups: int = 13):
        super().__init__()
        self.groups = groups

    def _one(self, out: dict, tg: dict, num_boxes: float, groups: int) -> dict:
        """1 つの出力(decoder の 1 層、または encoder の出力)の損失。

        Args:
            out: モデルの出力の 1 つ。
            tg: targets_from_masks の返り値。
            num_boxes: 損失を割る数(正解の物体の数 × 組の数)。
            groups: この出力のクエリの組の数(学習 G、評価 1)。

        Returns:
            {"class", "bbox", "giou", "mask", "dice"}(どれも 0 次元)。
        """
        logits, boxes, parts = out["logits"], out["boxes"], out["mask_parts"]
        indices = match(out, tg, groups)
        bi = torch.cat([torch.full_like(q, i) for i, (q, _) in enumerate(indices)]).to(logits.device)  # (A,) 画像の番号
        qi = torch.cat([q for q, _ in indices]).to(logits.device)  # (A,) クエリの番号
        starts = [sum(tg["counts"][:i]) for i in range(len(indices))]  # 画像ごとの、つなげた正解の中での先頭の番号
        ti = torch.cat([t + s for (_, t), s in zip(indices, starts)]).to(logits.device)  # (A,) 全体の正解の番号
        x = logits.float()  # (B, M, K)
        prob = x.sigmoid()
        pos_w = torch.zeros_like(x)
        neg_w = prob ** GAMMA
        if len(bi):
            src_boxes = boxes[bi, qi].float()  # (A, 4)
            tgt_boxes = tg["boxes"][ti]
            ious = box_iou(box_convert(src_boxes.detach(), "cxcywh", "xyxy"),
                           box_convert(tgt_boxes, "cxcywh", "xyxy")).diagonal()  # (A,) 割り当てた組の IoU
            labels = tg["labels"][ti]
            t = (prob[bi, qi, labels].pow(ALPHA) * ious.pow(1 - ALPHA)).clamp(min=0.01).detach()
            pos_w[bi, qi, labels] = t
            neg_w[bi, qi, labels] = 1 - t
        loss_class = (neg_w * x - F.logsigmoid(x) * (pos_w + neg_w)).sum() / num_boxes
        if not len(bi):
            zero = logits.sum() * 0 + parts["query"].sum() * 0 + parts["spatial"].sum() * 0 + boxes.sum() * 0
            return {"class": loss_class, "bbox": zero, "giou": zero, "mask": zero, "dice": zero}
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
        loss_giou = generalized_box_iou_loss(box_convert(src_boxes, "cxcywh", "xyxy"), box_convert(tgt_boxes, "cxcywh", "xyxy"),
                                             reduction="sum") / num_boxes
        # 割り当てた組だけマスクを作る(画像ごとに、その画像の画素の特徴とクエリの内積。画素の特徴を組の数だけ複製しないため)
        src_masks = torch.cat([torch.einsum("chw,ac->ahw", parts["spatial"][i], parts["query"][i, q.to(logits.device)])
                               for i, (q, _) in enumerate(indices) if len(q)]) + parts["bias"]  # (A, mh, mw)
        src_masks = rearrange(src_masks, "a h w -> a 1 h w").float()
        n_points = src_masks.shape[-2] * src_masks.shape[-1] // POINT_RATIO
        with torch.no_grad():
            coords = uncertain_points(src_masks, n_points)  # (A, P, 2)
            tp = sample_target_nearest(tg["masks"], ti, coords)  # (A, P)
        sp = point_sample(src_masks, coords)[:, 0]  # (A, 1, mh, mw) -> (A, P)
        loss_mask = F.binary_cross_entropy_with_logits(sp, tp, reduction="none").mean(-1).sum() / num_boxes
        p = sp.sigmoid()
        loss_dice = (1 - (2 * (p * tp).sum(-1) + 1) / (p.sum(-1) + tp.sum(-1) + 1)).sum() / num_boxes
        return {"class": loss_class, "bbox": loss_bbox, "giou": loss_giou, "mask": loss_mask, "dice": loss_dice}

    def forward(self, outs: dict, targets: list[dict]) -> tuple[torch.Tensor, dict]:
        """全部の出力の損失を足す。

        Args:
            outs: RFDETRSeg.forward の返り値(学習のとき)。
            targets: 画像ごとの {"masks": (T_i, S, S), "labels": (T_i,)}。枠は正解のマスクの大きさ S × S を基準に 0〜1 にする
                (学習の中で入力の画像だけを別の大きさに拡大縮小しても、0〜1 の枠と点の位置は変わらない)。

        Returns:
            (損失の合計(0 次元), 最終の層の内訳 {"class", "bbox", "giou", "mask", "dice"} の float)。
        """
        tg = targets_from_masks(targets, tuple(targets[0]["masks"].shape[-2:]))
        groups = self.groups if self.training else 1
        num_boxes = max(len(tg["labels"]) * groups, 1)
        total, parts = 0.0, {}
        outputs = outs["layers"] + [outs["enc"]]
        for k, out in enumerate(outputs):
            l = self._one(out, tg, num_boxes, groups)
            total = total + (W_CLASS * l["class"] + W_BBOX * l["bbox"] + W_GIOU * l["giou"] + W_MASK_CE * l["mask"]
                             + W_DICE * l["dice"])
            if k == len(outs["layers"]) - 1:
                parts = {key: float(v.detach()) for key, v in l.items()}
        return total, parts
