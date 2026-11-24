"""自作の YOLO26-seg 系の損失(論文 arXiv 2606.03748 の考え方。Ultralytics のコードは写さない)。

- 割り当て: TAL(Task-Aligned Assigner。TOOD の方式)。正解ごとに、中心が正解の枠に入る位置の中から、
  揃いの値 = (正解のクラスの確率)^α × (予測の枠と正解の枠の IoU)^β の大きい上位 k 個を正例にする(α 0.5、β 6.0)。
  1 つの位置が複数の正解に選ばれたら IoU の大きい正解にする。
  o2m(1 対多)は k = 10、o2o(1 対 1)は k = 7 の中からさらに最大の 1 個(論文の topk 7・topk2 1)。
  STAL: 候補の位置を選ぶときだけ、幅・高さが 8 画素より小さい正解の枠を 16 画素に広げる(小さい物体に正例がなくなるのを防ぐ)
- クラス: BCE。正例の目標の値は、揃いの値を正解ごとの最大で割り、その正解の正例の IoU の最大を掛けたもの
- 枠: CIoU の損失と、4 辺までの距離(stride の単位)の L1。どちらも正例の目標の値で重みを付ける
- マスク: 正例ごとに、係数と原型の線形結合のマスクを、正解の枠の中だけで BCE(枠の面積で割る)
- 意味のマスクの補助の損失(学習だけ): 正解のマスクをクラスごとに合わせたものとの BCE + Dice(同じ重み)
- ProgLoss: 損失 = α(t) × o2m の損失 + (1 − α(t)) × o2o の損失。α は学習の進み(0〜1)で 0.8 から 0.1 へ直線で下げる
- 重み: 枠(CIoU)9.83・クラス 0.65・L1 0.96(論文の YOLO26-s の COCO の値)、マスクは枠と同じ 9.83、意味のマスク 1.0
  (マスクと意味のマスクの重みは論文に書かれていないので、ここで決めた値)。合計に画像の数を掛ける(YOLO 系の慣例。
  SGD の学習率 0.01 がこの大きさを前提にしている)

形の記号: B = 画像の数、A = 位置の数、K = クラスの数、T = 正解の物体の数、P = 正例の数、M = 原型の数、S = 入力の一辺
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchvision.ops import box_iou, complete_box_iou_loss, masks_to_boxes

from segmentation.instance_segmentation.lib_common.models.yolo_seg import decode_boxes

ALPHA, BETA = 0.5, 6.0
W_BOX, W_CLS, W_L1, W_MASK, W_SEM = 9.83, 0.65, 0.96, 9.83, 1.0
STAL_MIN, STAL_SIZE = 8.0, 16.0


@torch.no_grad()
def assign(scores: torch.Tensor, boxes: torch.Tensor, anchors: torch.Tensor, gt_boxes: torch.Tensor,
           gt_labels: torch.Tensor, topk: int, topk2: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1 枚の画像の TAL の割り当て。

    Args:
        scores: 予測のクラスの確率 (A, K)。
        boxes: 予測の枠 (A, 4) 画素の x0, y0, x1, y1。
        anchors: 位置の中心 (A, 2)。
        gt_boxes: 正解の枠 (T, 4)。
        gt_labels: 正解のクラス (T,)。
        topk: 正解ごとに選ぶ位置の数。
        topk2: さらに絞る数(o2o は 1)。None なら絞らない。

    Returns:
        (正例か (A,) bool, 割り当てた正解の番号 (A,), 目標の値 (A,))。
    """
    a = anchors.shape[0]
    if len(gt_labels) == 0:
        return torch.zeros(a, dtype=torch.bool, device=anchors.device), torch.zeros(a, dtype=torch.long,
                                                                                    device=anchors.device), \
            torch.zeros(a, device=anchors.device)
    center, wh = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2, gt_boxes[:, 2:] - gt_boxes[:, :2]  # (T, 2)
    wh = torch.where(wh < STAL_MIN, torch.full_like(wh, STAL_SIZE), wh)  # STAL: 候補を選ぶときだけ広げる
    cand = torch.cat([center - wh / 2, center + wh / 2], dim=-1)  # (T, 4)
    inside = ((anchors[None] > cand[:, None, :2]) & (anchors[None] < cand[:, None, 2:])).all(-1)  # (T, A)
    iou = box_iou(gt_boxes, boxes).nan_to_num(0.0).clamp(min=0)  # (T, A) 面積 0 の予測の 0/0 は IoU 0 にする
    align = scores[:, gt_labels].T.pow(ALPHA) * iou.pow(BETA) * inside  # (A, K) -> (T, A)
    top = align.topk(min(topk, a), dim=1).indices  # (T, k)
    if topk2 is not None:
        top = top[:, :topk2]  # topk は大きい順なので先頭が最大
    pos = torch.zeros_like(inside).scatter_(1, top, True) & inside & (align > 0)  # (T, A)
    # 複数の正解に選ばれた位置は、IoU の最大の正解にする
    gt_idx = torch.where(pos, iou, torch.full_like(iou, -1)).argmax(0)  # (A,)
    fg = pos.any(0)  # (A,)
    pos = pos & (torch.arange(len(gt_labels), device=pos.device)[:, None] == gt_idx[None])  # (T, A) 1 つの位置に 1 つの正解
    # 目標の値: 揃いの値を正解ごとの最大で割り、その正解の正例の IoU の最大を掛ける
    pa, pi = align * pos, iou * pos
    norm = (pa / (pa.amax(1, keepdim=True) + 1e-9) * pi.amax(1, keepdim=True)).amax(0)  # (T, A) -> (A,)
    return fg, gt_idx, norm


class YOLOCriterion(nn.Module):
    """自作の YOLO26-seg 系の損失。

    Args:
        num_classes: クラスの数 K。
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.progress = 0.0  # 学習の進み(0〜1)。ProgLoss の重みに使う(学習の繰り返しでエポックごとに更新する)

    def _head(self, head: dict, out: dict, tg: list[dict], topk: int, topk2: int | None) -> dict:
        """1 組の head(o2m か o2o)の損失。

        Args:
            head: YOLOSeg の Head の出力。
            out: YOLOSeg.forward の返り値(anchors・strides・protos を使う)。
            tg: 画像ごとの {"boxes": (T, 4) 画素の xyxy, "labels": (T,), "masks": (T, S/4, S/4) float}。
            topk: assign の topk。
            topk2: assign の topk2。

        Returns:
            {"cls", "box", "l1", "mask"}(どれも 0 次元)。
        """
        logits, ltrb = head["logits"].float(), head["ltrb"].float()  # (B, A, K), (B, A, 4)
        anchors, strides = out["anchors"], out["strides"]
        boxes = decode_boxes(ltrb, anchors, strides)  # (B, A, 4)
        target = torch.zeros_like(logits)
        loss_box = loss_l1 = loss_mask = logits.sum() * 0
        tss, n_pos = 0.0, 0
        for b, t in enumerate(tg):
            fg, gi, score = assign(logits[b].sigmoid().detach(), boxes[b].detach(), anchors, t["boxes"], t["labels"],
                                   topk, topk2)
            if not fg.any():
                continue
            g = gi[fg]  # (P,) 正例の正解の番号
            w = score[fg]  # (P,)
            target[b, fg, t["labels"][g]] = w
            tss += float(w.sum())
            tb = t["boxes"][g]  # (P, 4)
            pb = boxes[b, fg]  # (P, 4)
            # 幅・高さが 0 以下の予測では CIoU の縦横の比の項が 0/0 になるので、右下を左上より少し右下にする
            pb = torch.cat([pb[:, :2], torch.maximum(pb[:, 2:], pb[:, :2] + 1e-3)], dim=-1)
            loss_box = loss_box + (complete_box_iou_loss(pb, tb, reduction="none") * w).sum()
            s, c = strides[fg, None], anchors[fg]  # (P, 1), (P, 2)
            t_ltrb = torch.cat([c - tb[:, :2], tb[:, 2:] - c], dim=-1) / s  # (P, 4) 正解の 4 辺までの距離
            loss_l1 = loss_l1 + ((ltrb[b, fg] - t_ltrb).abs().mean(-1) * w).sum()
            # マスク: 正例ごとの線形結合 (P, M) × (M, h, w) -> (P, h, w)、正解の枠の中だけの BCE を枠の面積で割る
            protos = out["protos"][b].float()
            pm = torch.einsum("pm,mhw->phw", head["coef"][b, fg].float(), protos)
            tm = t["masks"][g]  # (P, h, w)
            scale = protos.shape[-1] / out["size"]
            crop = box_crop_mask(tb * scale, pm.shape[-2:])  # (P, h, w)
            area = crop.sum((1, 2)).clamp(min=1)
            loss_mask = loss_mask + ((F.binary_cross_entropy_with_logits(pm, tm, reduction="none") * crop).sum((1, 2))
                                     / area).sum()
            n_pos += int(fg.sum())
        tss = max(tss, 1.0)
        loss_cls = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / tss
        return {"cls": loss_cls, "box": loss_box / tss, "l1": loss_l1 / tss, "mask": loss_mask / max(n_pos, 1)}

    def forward(self, out: dict, targets: list[dict]) -> tuple[torch.Tensor, dict]:
        """損失の合計(ProgLoss で o2m と o2o を混ぜる)。

        Args:
            out: YOLOSeg.forward の返り値(学習のとき)。
            targets: 画像ごとの {"masks": (T_i, S, S), "labels": (T_i,)}。

        Returns:
            (損失の合計(0 次元), o2o の内訳と意味のマスクの損失 {"cls", "box", "l1", "mask", "sem"} の float)。
        """
        size = targets[0]["masks"].shape[-1]
        out = out | {"size": size}
        ph, pw = out["protos"].shape[-2:]
        tg = []
        for t in targets:
            m = t["masks"]
            if len(m):
                xyxy = masks_to_boxes(m.bool()).float()
                xyxy[:, 2:] += 1  # 画素の端までの枠
                small = F.interpolate(rearrange(m.float(), "t h w -> t 1 h w"), size=(ph, pw), mode="bilinear",
                                      align_corners=False)
                small = (rearrange(small, "t 1 h w -> t h w") > 0.5).float()  # (T, S, S) -> (T, S/4, S/4)
            else:
                xyxy, small = torch.zeros(0, 4, device=m.device), torch.zeros(0, ph, pw, device=m.device)
            tg.append({"boxes": xyxy, "labels": t["labels"], "masks": small})
        o2m = self._head(out["o2m"], out, tg, topk=10, topk2=None)
        o2o = self._head(out["o2o"], out, tg, topk=7, topk2=1)
        alpha = 0.8 - 0.7 * self.progress
        weighted = lambda l: W_CLS * l["cls"] + W_BOX * l["box"] + W_L1 * l["l1"] + W_MASK * l["mask"]
        sem = semantic_loss(out["semantic"].float(), targets, self.num_classes)
        total = (alpha * weighted(o2m) + (1 - alpha) * weighted(o2o) + W_SEM * sem) * len(targets)
        return total, {**{k: float(v.detach()) for k, v in o2o.items()}, "sem": float(sem.detach())}


def box_crop_mask(boxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """枠の中を 1、外を 0 にした地図を作る。

    Args:
        boxes: (P, 4) 地図の画素の x0, y0, x1, y1。
        shape: 地図の (縦, 横)。

    Returns:
        (P, 縦, 横) の float。
    """
    h, w = shape
    xs = torch.arange(w, device=boxes.device, dtype=boxes.dtype)[None, None, :] + 0.5  # (1, 1, w) 画素の中心
    ys = torch.arange(h, device=boxes.device, dtype=boxes.dtype)[None, :, None] + 0.5  # (1, h, 1)
    x0, y0, x1, y1 = (rearrange(boxes[:, i], "p -> p 1 1") for i in range(4))
    return ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).float()


def semantic_loss(logits: torch.Tensor, targets: list[dict], num_classes: int) -> torch.Tensor:
    """意味のマスクの補助の損失(BCE + Dice、同じ重み)。

    Args:
        logits: (B, K, h, w)。
        targets: 画像ごとの {"masks": (T_i, S, S), "labels": (T_i,)}。
        num_classes: クラスの数 K。

    Returns:
        0 次元の損失。
    """
    b, k, h, w = logits.shape
    sem = torch.zeros_like(logits)
    for i, t in enumerate(targets):
        if len(t["labels"]):
            m = F.adaptive_max_pool2d(t["masks"].float()[:, None], (h, w))[:, 0]  # (T, S, S) -> (T, h, w) 小さい物体も残す
            sem[i].index_reduce_(0, t["labels"], m, "amax")  # クラスごとに合わせる (T, h, w) -> (K, h, w)
    bce = F.binary_cross_entropy_with_logits(logits, sem)
    p = rearrange(logits.sigmoid(), "b k h w -> b k (h w)")
    s = rearrange(sem, "b k h w -> b k (h w)")
    dice = 1 - (2 * (p * s).sum(-1) + 1) / (p.sum(-1) + s.sum(-1) + 1)  # (B, K)
    return bce + dice.mean()
