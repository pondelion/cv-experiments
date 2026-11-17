"""自作の GTR-Seg 系の損失(論文と公式の engine/gtr/det/criterion.py・configs/seg/gtrseg_base.yml を参考に、仕組みと重みを
読み取って自分で書いたもの。D-FINE の FGL・DDF、DEIM の MAL の考え方)。

出力ごとの損失(重みは公式の seg の設定):
    mal 2: Matchability-Aware Loss(DEIM)。割り当てたクエリの目標 = IoU^γ、重み = 正例 1・負例 p^γ(γ = 1.5)の BCE
    boxes: 枠の L1 1・GIoU 1(cx, cy, w, h、0〜1)
    local: FGL 0.15(4 辺までの距離の分布を、正解の距離の両隣の 2 つの値に、近さで割った重みで寄せる。重み = IoU)と
           DDF 1.5(途中の層の分布とクラスを、最後の層の分布に KL で寄せる。温度 5)
    masks: 割り当てたクエリのマスクの点の BCE 5・Dice 5(RF-DETR-Seg と同じ点の選び方)。mask_loss="box_crop" では D-FINE-seg の
           掛け方(正解の枠の中の全部の画素の BCE・Dice、重み 1・1。GTRCriterion の説明)
損失をかける出力: decoder の全層、最初の層の前の枠(pre。分布がないので local なし)、two-stage の候補(enc。マスクなし)、
denoising の全層と pre(割り当ては、作ったときの正解の番号そのまま)。
boxes と local は、全部の層・pre・enc の割り当ての和(1 つのクエリには最も多くの出力で選ばれた正解)を
使い、その数で割る(enc は組が複数なので自分の割り当て)。ほかは自分の割り当てを使い、正解の数 × 組の数で割る。
割り当て: ハンガリアン法。費用はクラス(focal)2・枠 L1 1・GIoU 1・マスク BCE 5・Dice 5(公式の seg の matcher)。

形の記号: B = 画像の数、M = クエリの数、K = クラスの数、A = 割り当ての数、R = 33(分布の値の数)、T = 正解の物体の数
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torchvision.ops import box_convert, box_iou, generalized_box_iou

from segmentation.instance_segmentation.lib_common.losses.rfdetr_criterion import (POINT_RATIO, match, point_sample,
                                                                                   sample_target_nearest,
                                                                                   targets_from_masks, to_xyxy,
                                                                                   uncertain_points)
from segmentation.instance_segmentation.lib_common.models.gtr_seg import REG_SCALE, distance_bins

W = {"mal": 2.0, "bbox": 1.0, "giou": 1.0, "fgl": 0.15, "ddf": 1.5, "mask": 5.0, "dice": 5.0}
MATCH_WEIGHTS = (2.0, 1.0, 1.0, 5.0, 5.0)  # クラス, L1, GIoU, マスク BCE, Dice
GAMMA = 1.5  # MAL の γ
DDF_T = 5.0  # DDF の温度


def paired_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """組になった枠どうしの IoU(a の i 番目と b の i 番目)。

    Args:
        a: 予測の枠 (A(組の数), 4) cx, cy, w, h(幅・高さが負なら 0 として扱う)。
        b: 正解の枠 (A, 4) cx, cy, w, h。

    Returns:
        (A,)。
    """
    if len(a) == 0:
        return a.new_zeros(0)
    return box_iou(to_xyxy(a), box_convert(b, "cxcywh", "xyxy")).diagonal()


def distance_targets(ref: torch.Tensor, xyxy: torch.Tensor, bins: torch.Tensor) -> tuple:
    """FGL の目標: 正解の枠の各辺のずれ(distance2bbox の逆)を、分布の隣り合う 2 つの値への重みに分ける。

    ずれ d が bins[l] と bins[l+1] の間なら、左 l に (bins[l+1] − d) / 幅、右 l+1 に (d − bins[l]) / 幅(線形補間)。
    範囲の外は端の値に全部の重みを置く。

    Args:
        ref: 参照の枠 (A, 4) cx, cy, w, h。
        xyxy: 正解の枠 (A, 4) x0, y0, x1, y1。
        bins: 分布の値 (R,)(小さい順)。

    Returns:
        (左の番号 (A·4,) の long, 左の重み (A·4,), 右の重み (A·4,))。
    """
    unit = ref[:, 2:] / REG_SCALE + 1e-16  # (A, 2)
    lt = (ref[:, :2] - xyxy[:, :2]) / unit - REG_SCALE / 2  # 左・上のずれ
    rb = (xyxy[:, 2:] - ref[:, :2]) / unit - REG_SCALE / 2  # 右・下のずれ
    d = torch.cat([lt, rb], -1).reshape(-1)  # (A·4,)
    left = (torch.bucketize(d, bins, right=True) - 1).clamp(0, len(bins) - 2)  # d 以下で最大の値の番号
    w_right = ((d - bins[left]) / (bins[left + 1] - bins[left])).clamp(0, 1)
    return left.detach(), (1 - w_right).detach(), w_right.detach()


def union_indices(all_indices: list[list[tuple]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """複数の出力の割り当てを 1 つにする: クエリごとに、選ばれた回数が最も多い正解を使う(同じ回数なら番号の小さい正解)。

    Args:
        all_indices: 出力ごとの、画像ごとの (クエリの番号, 正解の番号)。

    Returns:
        画像ごとの (クエリの番号, 正解の番号)(クエリの番号の順)。
    """
    out = []
    for per_image in zip(*all_indices):
        q = torch.cat([p[0] for p in per_image])
        t = torch.cat([p[1] for p in per_image])
        pairs, counts = torch.unique(torch.stack([q, t], 1), dim=0, return_counts=True)  # (U, 2) は (q, t) の小さい順
        # クエリの小さい順 → 回数の多い順 → 正解の小さい順に並べ、クエリごとの先頭を取る(安定な並べ替えを後ろの鍵から)
        order = torch.argsort(-counts, stable=True)
        order = order[torch.argsort(pairs[order, 0], stable=True)]
        pairs = pairs[order]
        first = torch.ones(len(pairs), dtype=torch.bool)
        first[1:] = pairs[1:, 0] != pairs[:-1, 0]
        out.append((pairs[first, 0], pairs[first, 1]))
    return out


class GTRCriterion(nn.Module):
    """自作の GTR-Seg 系の損失。

    Args:
        groups: 学習のときのクエリの組の数 G。
        mask_loss: マスクの損失の掛け方。"points"(既定。不確かな点を多めに選んだ点の BCE 5・Dice 5。公式の GTR と同じ)/
            "box_crop"(D-FINE-seg と同じ。予測のマスクの全部の画素のうち正解の枠の中だけの BCE(枠の面積で割る)と Dice、
            重みは 1・1。正解のマスクは予測の大きさに双線形で縮める。割り当てのマスクの費用も D-FINE-seg と同じく、全部の画素の
            focal の形の費用 1・Dice 1。クラス・枠の損失と費用は GTR のまま。exp00103 の B)。
        checkpoint: マスクの損失を gradient checkpointing で包むか(結果は同じで、GPU のメモリが減る)。
    """

    def __init__(self, groups: int = 3, mask_loss: str = "points", checkpoint: bool = False):
        super().__init__()
        self.groups, self.mask_loss, self.checkpoint = groups, mask_loss, checkpoint
        self.w = dict(W, mask=1.0, dice=1.0) if mask_loss == "box_crop" else W  # D-FINE-seg の重み
        # 割り当てのマスクの費用も D-FINE-seg に合わせる(全部の画素の focal の形の費用 1・Dice 1。クラス・枠の費用は GTR のまま)
        self.match_weights = MATCH_WEIGHTS[:3] + (1.0, 1.0) if mask_loss == "box_crop" else MATCH_WEIGHTS
        self.mask_cost = "full_focal" if mask_loss == "box_crop" else "points"
        self.register_buffer("bins", distance_bins(), persistent=False)

    @staticmethod
    def _flat(indices: list[tuple], tg: dict, device) -> tuple:
        """画像ごとの割り当てを、1 次元の番号の並びにする。

        Args:
            indices: 画像ごとの (クエリの番号, 画像の中での正解の番号)。
            tg: targets_from_masks の返り値("counts" を使う)。
            device: 置く装置。

        Returns:
            (画像の番号 (A,), クエリの番号 (A,), 全部の画像をつなげた中での正解の番号 (A,))(A = 割り当ての数)。
        """
        starts = [sum(tg["counts"][:i]) for i in range(len(indices))]
        bi = torch.cat([torch.full_like(q, i) for i, (q, _) in enumerate(indices)]).to(device)
        qi = torch.cat([q for q, _ in indices]).to(device)
        ti = torch.cat([t + s for (_, t), s in zip(indices, starts)]).to(device)
        return bi, qi, ti

    def _mal(self, out: dict, bi, qi, ti, tg: dict, ious: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """MAL(割り当てたクエリの目標 = IoU^γ、負例の重み = 確率^γ の BCE)。

        Args:
            out: 1 つの出力({"logits": (B, M, K), ...})。
            bi, qi, ti: _flat の返り値(画像・クエリ・正解の番号 (A,))。
            tg: targets_from_masks の返り値。
            ious: 割り当てた組の IoU (A,)。
            n: 損失を割る数。

        Returns:
            0 次元の損失。
        """
        x = out["logits"].float()  # (B, M, K)
        target = torch.zeros_like(x)
        pos = torch.zeros_like(x)
        if len(bi):
            target[bi, qi, tg["labels"][ti]] = ious.pow(GAMMA)
            pos[bi, qi, tg["labels"][ti]] = 1.0
        weight = x.sigmoid().detach().pow(GAMMA) * (1 - pos) + pos
        return F.binary_cross_entropy_with_logits(x, target, weight=weight, reduction="sum") / n

    def _boxes(self, out: dict, bi, qi, ti, tg: dict, n) -> tuple[torch.Tensor, torch.Tensor]:
        """割り当てた組の枠の L1 と GIoU の損失。

        Args:
            out: 1 つの出力({"boxes": (B, M, 4) cx, cy, w, h, ...})。
            bi, qi, ti: 画像・クエリ・正解の番号 (A,)。
            tg: targets_from_masks の返り値。
            n: 損失を割る数。

        Returns:
            (L1 の損失, GIoU の損失)(どちらも 0 次元)。
        """
        src, tgt = out["boxes"][bi, qi].float(), tg["boxes"][ti]
        if len(src) == 0:
            z = out["boxes"].sum() * 0
            return z, z
        l1 = F.l1_loss(src, tgt, reduction="sum") / n
        giou = (1 - generalized_box_iou(to_xyxy(src), box_convert(tgt, "cxcywh", "xyxy")).diagonal()).sum() / n
        return l1, giou

    def _fgl(self, out: dict, bi, qi, ti, tg: dict, ious, n, cache: dict, key: str) -> torch.Tensor:
        """FGL: 割り当てたクエリの 4 辺の分布を、正解のずれの両隣の値に寄せる交差エントロピー(重み = IoU)。

        参照の枠(最初の層の枠)と割り当て(全部の層の和)は全部の層で同じなので、目標は 1 回だけ作って cache に置く。

        Args:
            out: 1 つの出力({"corners": (B, M, 4·R), "refs": (B, M, 4), ...})。
            bi, qi, ti: 画像・クエリ・正解の番号 (A,)。
            tg: targets_from_masks の返り値。
            ious: 割り当てた組の IoU (A,)(重み)。
            n: 損失を割る数。
            cache: 同じ forward の中で目標を使い回す辞書。
            key: cache の鍵(検出と denoising で分ける)。

        Returns:
            0 次元の損失。
        """
        if len(bi) == 0:
            return out["corners"].sum() * 0
        logp = rearrange(out["corners"][bi, qi].float(), "a (four r) -> (a four) r", four=4).log_softmax(-1)  # (A·4, R)
        if key not in cache:
            cache[key] = distance_targets(out["refs"][bi, qi].detach().float(),
                                          box_convert(tg["boxes"][ti], "cxcywh", "xyxy"), self.bins)
        left, w_left, w_right = cache[key]
        nll = -(w_left * logp.gather(1, left[:, None])[:, 0] + w_right * logp.gather(1, (left + 1)[:, None])[:, 0])
        return (nll * ious.repeat_interleave(4)).sum() / n

    def _ddf(self, out: dict, teacher: dict, bi, qi, ious, cache: dict, is_dn: bool) -> torch.Tensor:
        """DDF(途中の層の分布を、最後の層の分布に KL で寄せる)。

        Args:
            out: 途中の層の出力({"corners": (B, M, 4·R), "boxes", ...})。
            teacher: 最後の層の出力({"corners", "logits": (B, M, K)})。
            bi, qi: 割り当てた組の画像・クエリの番号 (A,)。
            ious: 割り当てた組の IoU (A,)。
            cache: 正例・負例の数の重みを置く辞書(検出の出力で計算し、denoising で使い回す)。
            is_dn: denoising の出力か。

        Returns:
            0 次元の損失。
        """
        pred = rearrange(out["corners"].float(), "b m (four r) -> (b m four) r", four=4)
        tgt = rearrange(teacher["corners"].float(), "b m (four r) -> (b m four) r", four=4)
        if torch.equal(pred, tgt):
            return pred.sum() * 0
        w = teacher["logits"].float().sigmoid().max(-1).values  # (B, M)
        mask = torch.zeros_like(w, dtype=torch.bool)
        if len(bi):
            mask[bi, qi] = True
            w[bi, qi] = ious.to(w.dtype)
        w = w.detach().repeat_interleave(4, dim=-1).reshape(-1)  # (B·M·4,)
        mask = mask.repeat_interleave(4, dim=-1).reshape(-1)
        kl = F.kl_div(F.log_softmax(pred / DDF_T, -1), F.softmax(tgt.detach() / DDF_T, -1), reduction="none").sum(-1)
        loss = w * DDF_T ** 2 * kl
        if not is_dn:  # 正例・負例の数の重み(denoising は検出の出力のものを使う。公式と同じ)
            scale = 8 / out["boxes"].shape[0]
            cache["num_pos"], cache["num_neg"] = (mask.sum() * scale) ** 0.5, ((~mask).sum() * scale) ** 0.5
        mf = mask.float()
        l_pos = (loss * mf).sum() / mf.sum().clamp(min=1)
        l_neg = (loss * (1 - mf)).sum() / (1 - mf).sum().clamp(min=1)
        npos, nneg = cache["num_pos"], cache["num_neg"]
        return (l_pos * npos + l_neg * nneg) / (npos + nneg)

    def _masks(self, out: dict, indices: list[tuple], tg: dict, ti, n) -> tuple[torch.Tensor, torch.Tensor]:
        """割り当てたクエリのマスクの、点で取った BCE と Dice の損失。

        Args:
            out: 1 つの出力({"mask_parts": SegmentationHead.parts の 1 つ, ...})。
            indices: 画像ごとの (クエリの番号, 画像の中での正解の番号)。
            tg: targets_from_masks の返り値("masks" を使う)。
            ti: 全部の画像をつなげた中での正解の番号 (A,)。
            n: 損失を割る数。

        Returns:
            (BCE の損失, Dice の損失)(どちらも 0 次元)。
        """
        parts = out["mask_parts"]
        dev = parts["query"].device
        used = [(i, q.to(dev)) for i, (q, _) in enumerate(indices) if len(q)]
        if not used:
            z = parts["query"].sum() * 0 + parts["spatial"].sum() * 0
            return z, z
        qv = torch.cat([parts["query"][i, q] for i, q in used])  # (A, C) 割り当てたクエリの特徴
        args = (parts["spatial"], qv, parts["bias"], [i for i, _ in used], [len(q) for _, q in used], tg, ti, n)
        if self.checkpoint:
            # マスクを作るところから損失までを gradient checkpointing で包み、勾配を計算するときに計算し直す(結果は同じで、
            # GPU のメモリが減る。点の選び方の乱数も、計算し直すときに同じ値に戻る)
            return torch.utils.checkpoint.checkpoint(self._masks_from_parts, *args, use_reentrant=False)
        return self._masks_from_parts(*args)

    def _masks_from_parts(self, spatial: torch.Tensor, qv: torch.Tensor, bias: torch.Tensor, images: list[int],
                          counts: list[int], tg: dict, ti: torch.Tensor, n) -> tuple[torch.Tensor, torch.Tensor]:
        """マスクの材料から割り当てたクエリのマスクを作り、マスクの損失を計算する(mask_loss で掛け方を変える)。

        Args:
            spatial: 画素の特徴 (B, C, mh, mw)。
            qv: 割り当てたクエリの特徴を画像の順につなげたもの (A, C)。
            bias: マスクのバイアス (1,)。
            images: qv の区切りごとの画像の番号。
            counts: qv の区切りごとのクエリの数。
            tg: targets_from_masks の返り値。
            ti: 全部の画像をつなげた中での正解の番号 (A,)。
            n: 損失を割る数。

        Returns:
            (BCE の損失, Dice の損失)(どちらも 0 次元)。
        """
        pieces = [torch.einsum("chw,ac->ahw", spatial[i], q) for i, q in zip(images, qv.split(counts))]  # (A_i, mh, mw)
        logits = (torch.cat(pieces) + bias).float()  # (A, mh, mw)
        if self.mask_loss == "box_crop":
            return self._box_crop_masks(logits, tg, ti, n)
        src = rearrange(logits, "a h w -> a 1 h w")  # (A, 1, mh, mw)
        n_points = max(src.shape[-2], src.shape[-2] * src.shape[-1] // POINT_RATIO)
        with torch.no_grad():
            coords = uncertain_points(src, n_points)
            tp = sample_target_nearest(tg["masks"], ti, coords)
        sp = point_sample(src, coords)[:, 0]  # (A, P)
        ce = F.binary_cross_entropy_with_logits(sp, tp, reduction="none").mean(-1).sum() / n
        p = sp.sigmoid()
        dice = (1 - (2 * (p * tp).sum(-1) + 1) / (p.sum(-1) + tp.sum(-1) + 1)).sum() / n
        return ce, dice

    @staticmethod
    def _box_crop_masks(logits: torch.Tensor, tg: dict, ti: torch.Tensor, n) -> tuple[torch.Tensor, torch.Tensor]:
        """D-FINE-seg の掛け方のマスクの損失: 正解の枠の中の画素だけの BCE(枠の面積で割る)と Dice。

        Args:
            logits: 割り当てたクエリの予測のマスク (A, mh, mw)(sigmoid の前)。
            tg: targets_from_masks の返り値("masks" (T, S, S)・"boxes" (T, 4) の 0〜1 の cx, cy, w, h を使う)。
            ti: 全部の画像をつなげた中での正解の番号 (A,)。
            n: 損失を割る数。

        Returns:
            (BCE の損失, Dice の損失)(どちらも 0 次元)。
        """
        mh, mw = logits.shape[-2:]
        # 正解のマスクを予測の大きさに双線形で縮める。正解ごとに 1 回だけ縮めて覚えておき(出力ごと・割り当てごとに縮め直すと
        # 大きい一時的なメモリが要る)、割り当ての番号で取り出す: (T, S, S) -> (T, mh, mw) -> (A, mh, mw)
        key = ("masks_small", mh, mw)
        if key not in tg:
            tg[key] = F.interpolate(rearrange(tg["masks"].float(), "t h w -> t 1 h w"), size=(mh, mw), mode="bilinear",
                                    align_corners=False)[:, 0].clamp(0, 1)
        target = tg[key][ti]
        # 正解の枠を予測のマスクの画素の座標にして、枠の中を 1 にする: (A, mh, mw)
        x0, y0, x1, y1 = (to_xyxy(tg["boxes"][ti]) * logits.new_tensor([mw, mh, mw, mh])).unbind(-1)  # 4 × (A,)
        x0, y0 = x0.clamp(0, mw - 1), y0.clamp(0, mh - 1)
        x1, y1 = x1.clamp(1, mw), y1.clamp(1, mh)
        xs = torch.arange(mw, device=logits.device, dtype=logits.dtype)
        ys = torch.arange(mh, device=logits.device, dtype=logits.dtype)
        inside_x = (xs >= x0[:, None]) & (xs < x1[:, None])  # (A, mw)
        inside_y = (ys >= y0[:, None]) & (ys < y1[:, None])  # (A, mh)
        inside = (inside_y[:, :, None] & inside_x[:, None, :]).float()  # (A, mh, mw)
        area = ((x1 - x0) * (y1 - y0)).clamp(min=1.0)  # (A,)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # (A, mh, mw)
        ce = ((bce * inside).sum((1, 2)) / area).sum() / n
        p, t = (logits.sigmoid() * inside).flatten(1), (target * inside).flatten(1)  # (A, mh·mw)
        dice = (1 - (2 * (p * t).sum(-1) + 1e-6) / (p.sum(-1) + t.sum(-1) + 1e-6)).sum() / n
        return ce, dice

    def forward(self, outs: dict, targets: list[dict]) -> tuple[torch.Tensor, dict]:
        """全部の出力の損失を足す。

        Args:
            outs: GTRSeg.forward の返り値(学習のとき)。
            targets: 画像ごとの {"masks": (T_i, S, S), "labels": (T_i,)}。

        Returns:
            (損失の合計(0 次元), 最後の層の内訳 {"mal", "bbox", "giou", "fgl", "mask", "dice"} と "ddf"(途中の層の平均)の float)。
        """
        tg = targets_from_masks(targets, tuple(targets[0]["masks"].shape[-2:]))
        dev = tg["labels"].device
        g = self.groups if self.training else 1
        layers = outs["layers"]
        own = [match(o, tg, g, self.match_weights, self.mask_cost) for o in layers + [outs["pre"]]]  # 層ごと + pre
        enc_ind = match(outs["enc"], tg, g, self.match_weights, self.mask_cost)
        go = union_indices(own + [enc_ind])
        n = torch.tensor(max(len(tg["labels"]) * g, 1), dtype=torch.float, device=dev)
        n_go = torch.tensor(max(sum(len(q) for q, _ in go), 1), dtype=torch.float, device=dev)
        cache: dict = {}
        total, parts, ddfs = 0.0, {}, []

        def add(o: dict, indices, n_own, use_go: bool, teacher: dict | None, key: str, is_dn: bool = False,
                masks: bool = True) -> dict:
            """1 つの出力の損失を計算して合計に足す。

            Args:
                o: 1 つの出力。
                indices: 画像ごとの (クエリの番号, 画像の中での正解の番号)。
                n_own: 自分の割り当てで計算する損失を割る数。
                use_go: 枠の損失に、全部の出力の割り当ての和を使うか。
                teacher: DDF の先生(最後の層の出力)。None なら DDF なし。
                key: FGL の目標の cache の鍵。
                is_dn: denoising の出力か。
                masks: マスクの損失を計算するか。

            Returns:
                損失の内訳 {"mal", "bbox", "giou", ...}(どれも 0 次元)。
            """
            nonlocal total
            bi, qi, ti = self._flat(indices, tg, dev)
            ious = paired_iou(o["boxes"][bi, qi].detach().float(), tg["boxes"][ti])
            l = {"mal": self._mal(o, bi, qi, ti, tg, ious, n_own)}
            gb, gq, gt = (self._flat(go, tg, dev) if use_go else (bi, qi, ti))
            n_box = n_go if use_go else n_own
            l["bbox"], l["giou"] = self._boxes(o, gb, gq, gt, tg, n_box)
            if "corners" in o:
                gious = paired_iou(o["boxes"][gb, gq].detach().float(), tg["boxes"][gt])
                l["fgl"] = self._fgl(o, gb, gq, gt, tg, gious, n_box, cache, key)
                if teacher is not None:
                    l["ddf"] = self._ddf(o, teacher, gb, gq, gious, cache, is_dn)
            if masks and "mask_parts" in o:
                l["mask"], l["dice"] = self._masks(o, indices, tg, ti, n_own)
            total = total + sum(self.w[k] * v for k, v in l.items())
            return l

        last = len(layers) - 1
        for i, o in enumerate(layers):  # 途中の層は最後の層を先生にして DDF
            l = add(o, own[i], n, True, layers[last] if i < last else None, "fgl")
            if i == last:
                parts = {k: float(v.detach()) for k, v in l.items()}
            elif "ddf" in l:
                ddfs.append(float(l["ddf"].detach()))
        add(outs["pre"], own[-1], n, True, None, "fgl")
        add(outs["enc"], enc_ind, n, g == 1, None, "fgl", masks=False)
        meta = outs.get("dn_meta")
        if meta is not None:  # denoising: 正例のクエリと、作ったときの正解の番号をそのまま組にする
            tile = meta["num_group"] * g
            dn_ind = [(meta["positive"][i].cpu(), torch.arange(c).repeat(tile)) if c else
                      (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
                      for i, c in enumerate(tg["counts"])]
            n_dn = n * meta["num_group"]
            dn_layers = outs["dn_layers"]
            for o in dn_layers:
                add(o, dn_ind, n_dn, False, dn_layers[-1], "fgl_dn", is_dn=True)
            add(outs["dn_pre"], dn_ind, n_dn, False, None, "fgl_dn")
        parts["ddf"] = sum(ddfs) / max(len(ddfs), 1)
        return total, parts
