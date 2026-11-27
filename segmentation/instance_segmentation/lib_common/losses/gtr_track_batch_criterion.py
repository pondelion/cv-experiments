"""GTR 系 × VidEoMT 型(models/video/gtr_track.py)の損失を、micro-batch の区間をまとめて 1 回で計算する版。exp00502。

計算する値は、区間 1 本ずつの損失(gtr_track_criterion.GTRTrackCriterion.frame_loss)と同じ: 区間ごと(画像ごと)の損失の並び (A,)
を返す。区間をまとめて計算するのは、区間ごと・出力ごとに小さな GPU の処理を何度も起動すると、起動の時間で GPU が空くため。

区間 1 本ずつと同じ値にするための計算の仕方:
    - GTR 系の損失は、組の和を「画像の物体の数 × 組の数」で割る。ここでは各項を組ごと(MAL は画像のクエリごと)に計算し、画像ごとに
      足してから、その画像の割る数で割る
    - 区間ごとに違う追跡のクエリの数は埋めてそろえてある。埋めたクエリは MAL・DDF の対象(負例)から除く
    - DDF の正例・負例の数の重みは、画像ごとに数える(区間 1 本のときの scale = 8 / 1)
    - 正解の割り当て・損失の規則は GTRTrackCriterion と同じ(ファイルの説明はそちら)
マスクの損失は "points"(点を選んだ BCE・Dice)だけに対応する。

形の記号: A = まとめた区間(画像)の数、G = 組の数、K = 埋めた後の追跡のクエリの数、Q = 1 組の検出のクエリの数、M = G·(K + Q)、
N = まとめた正解の数(画像ごとの正解をつなげたもの)、P = 組の数(割り当て)。
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import box_convert, generalized_box_iou

from segmentation.instance_segmentation.lib_common.losses.gtr_criterion import (DDF_T, GAMMA, GTRCriterion, distance_targets,
                                                                                paired_iou, union_indices)
from segmentation.instance_segmentation.lib_common.losses.rfdetr_criterion import (POINT_RATIO, match, point_sample,
                                                                                   sample_target_nearest, to_xyxy,
                                                                                   uncertain_points)


def _scoped(name: str, fn):
    """fn を torch.profiler の名前の付いた範囲で包む(時間の内訳を測るため。測っていないときはほぼ時間がかからない)。

    Args:
        name: profiler に出す範囲の名前。
        fn: 包む関数。

    Returns:
        fn と同じ引数・返り値の関数。
    """
    def wrapped(*args, **kwargs):
        """fn を名前の付いた範囲の中で呼ぶ。

        Args:
            *args: fn の位置引数。
            **kwargs: fn のキーワード引数。

        Returns:
            fn の返り値。
        """
        with torch.profiler.record_function(name):
            return fn(*args, **kwargs)
    return wrapped


def _paired_iou_giou(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """組になった枠どうしの IoU と GIoU を、組ごとに直接計算する。

    gtr_criterion.paired_iou(torchvision の box_iou の対角)と、torchvision の generalized_box_iou の対角と同じ式。
    全部の組の組み合わせの行列(P × P)を作らないので、演算の数と Python の処理が少ない。予測の枠は幅・高さを 0 以上にする
    (rfdetr_criterion.to_xyxy と同じ)、正解の枠はそのまま。

    Args:
        pred: 予測の枠 (P, 4) cx, cy, w, h。
        gt: 正解の枠 (P, 4) cx, cy, w, h。

    Returns:
        (IoU (P,), GIoU (P,))。
    """
    pc, pw = pred[:, :2], pred[:, 2:].clamp(min=0)
    gc, gw = gt[:, :2], gt[:, 2:]
    p0, p1, g0, g1 = pc - 0.5 * pw, pc + 0.5 * pw, gc - 0.5 * gw, gc + 0.5 * gw  # 左上・右下 (P, 2)
    area_p = (p1 - p0).prod(-1)
    area_g = (g1 - g0).prod(-1)
    inter = (torch.minimum(p1, g1) - torch.maximum(p0, g0)).clamp(min=0).prod(-1)
    union = area_p + area_g - inter
    iou = inter / union
    enclose = (torch.maximum(p1, g1) - torch.minimum(p0, g0)).clamp(min=0).prod(-1)
    return iou, iou - (enclose - union) / enclose


def _cat_targets(tgs: list[dict]) -> dict:
    """画像ごとの正解(GTRTrack._frame_targets の形、画像は 1 枚)を、画像をまとめた形にする。

    Args:
        tgs: 画像ごとの正解の並び(長さ A)。

    Returns:
        {"labels" (N,), "masks" (N, S, S), "boxes" (N, 4), "owner" (N,)(どの画像の物体か), "counts"(画像ごとの物体の数)}。
    """
    dev = tgs[0]["labels"].device
    counts = [len(t["labels"]) for t in tgs]
    return {"labels": torch.cat([t["labels"] for t in tgs]), "masks": torch.cat([t["masks"] for t in tgs]),
            "boxes": torch.cat([t["boxes"] for t in tgs]),
            "owner": torch.repeat_interleave(torch.arange(len(tgs), device=dev), torch.tensor(counts, device=dev)),
            "counts": counts}


def _subset(tg: dict, idx: torch.Tensor, n_img: int) -> dict:
    """まとめた正解の一部(idx の物体)を取り出す(画像ごとの数も数え直す)。

    Args:
        tg: _cat_targets の形の正解。
        idx: 取り出す物体の番号 (n,)(まとめた中の番号)。
        n_img: 画像の数 A。

    Returns:
        同じ形の正解(物体は n 個、"counts" は画像ごとの数え直した数)。
    """
    owner = tg["owner"][idx]
    return {"labels": tg["labels"][idx], "masks": tg["masks"][idx], "boxes": tg["boxes"][idx], "owner": owner,
            "counts": torch.bincount(owner.cpu(), minlength=n_img).tolist()}


class GTRTrackBatchCriterion(GTRCriterion):
    """GTR 系 × VidEoMT 型の損失の、区間をまとめて計算する版(ファイルの先頭の規則)。

    Args:
        groups: 学習のときのクエリの組の数 G。
        mask_loss: マスクの損失("points" だけに対応する)。
        checkpoint: マスクの計算を gradient checkpointing にするか(GTRCriterion と同じ)。
    """

    def __init__(self, groups: int = 3, mask_loss: str = "points", checkpoint: bool = False):
        super().__init__(groups, mask_loss, checkpoint)
        assert mask_loss == "points", "区間をまとめる版はマスクの損失 points だけに対応する"

    def frame_loss_batch(self, outs: dict, tgs: list[dict], presents: list[torch.Tensor], track_objs: list[list[int]],
                         ks: list[int], k_max: int, num_det: int,
                         fixed_pairs: list | None = None) -> tuple[torch.Tensor, dict, list]:
        """1 フレームの、まとめた区間の損失。

        Args:
            outs: GTRSeg.forward の返り値(学習のとき。画像 A 枚、1 組のクエリは [追跡 K(うち先頭 ks[a] 個が本物) | 検出 Q])。
            tgs: 画像ごとの正解(GTRTrack._frame_targets の形)。
            presents: 画像ごとの、物体がこのフレームに写っているか (N_a,) の bool。
            track_objs: 画像ごとの、追跡のクエリ k が持つ物体の、その画像の正解の中での番号。
            ks: 画像ごとの本物の追跡のクエリの数。
            k_max: 埋めた後の追跡のクエリの数 K。
            num_det: Q。
            fixed_pairs: VidEoMT と同じ形(全部のクエリを次のフレームへ渡す。k_max = 0)のときの、出力ごと(層ごと + pre)・
                画像ごとの、前のフレームまでに決めた組 (クエリの番号 (·,)(G·Q の中), その画像の正解の中での物体の番号 (·,))。
                その組をそのまま使い、新しい物体はほかのクエリとだけハンガリアン法で割り当てる(公式の VidEoMT の割り当てと同じく、
                出力ごとに別々に決める)。None なら追跡のクエリの決まった組(track_objs)を使う。

        Returns:
            (画像ごとの損失 (A,)(GTRTrackCriterion.frame_loss の損失と同じ値), 最後の層の内訳の float(画像の平均),
             fixed_pairs が None なら、画像ごとの {組 0 の中でのクエリの番号(その画像の本物の追跡のクエリの数以上):
             その画像の正解の中での物体の番号}。fixed_pairs があれば、出力ごと・画像ごとの、このフレームの全部の組
             (クエリの番号, 物体の番号)(次のフレームの fixed_pairs))。
        """
        n_img = len(tgs)
        dev = tgs[0]["labels"].device
        g = self.groups if self.training else 1
        k, q, per = k_max, num_det, k_max + num_det
        tg = _cat_targets(tgs)
        starts = [0]
        for t in tgs[:-1]:
            starts.append(starts[-1] + len(t["labels"]))
        present_cpu = torch.cat(presents).cpu()  # (N,) CPU の上で使う(GPU の処理の終わりを待つのはここの 1 回だけ)
        n_per_img = torch.tensor([len(t["labels"]) for t in tgs], dtype=torch.float)  # (A,)
        # 新しい物体(このフレームで初めて写った物体)と、写っている物体(まとめた正解の中の番号)
        new_ids = torch.tensor([starts[a] + j for a in range(n_img) for j in range(len(tgs[a]["labels"]))
                                if present_cpu[starts[a] + j] and j not in set(track_objs[a])], dtype=torch.long)
        present_ids = present_cpu.nonzero()[:, 0]
        tg_new, tg_present = _subset(tg, new_ids.to(dev), n_img), _subset(tg, present_ids.to(dev), n_img)
        det_q = torch.cat([torch.arange(gi * per + k, (gi + 1) * per, device=dev) for gi in range(g)])  # (G·Q,)
        # 追跡のクエリの決まった組(画像ごと): 組 gi の k 番目の追跡のクエリ ↔ その画像の正解の中の物体
        fixed = [(torch.tensor([gi * per + j for gi in range(g) for j in range(ks[a])], dtype=torch.long),
                  torch.tensor(track_objs[a] * g, dtype=torch.long)) for a in range(n_img)]

        def local_ids(sub_owner_counts: list[int], ids: torch.Tensor) -> list[torch.Tensor]:
            """まとめた中の番号の並び ids を、画像ごとの「その画像の正解の中での番号」に分ける。

            Args:
                sub_owner_counts: 画像ごとの、ids の中の物体の数(長さ A)。
                ids: まとめた正解の中の番号 (n,)(画像の順に並んだもの)。

            Returns:
                画像ごとの、その画像の正解の中での番号の並び(長さ A)。
            """
            out, s = [], 0
            for a, c in enumerate(sub_owner_counts):
                out.append(ids[s: s + c] - starts[a])
                s += c
            return out

        new_local = local_ids(tg_new["counts"], new_ids)
        present_local = local_ids(tg_present["counts"], present_ids)

        def assign(o: dict, fx: list) -> list[tuple[torch.Tensor, torch.Tensor]]:
            """1 つの出力の割り当て(画像ごと): 決まった組 fx + 残りのクエリと新しい物体のハンガリアン法。

            Args:
                o: 1 つの出力(decoder の 1 層か pre)。
                fx: 画像ごとの決まった組 (クエリの番号 (·,), その画像の正解の中での物体の番号 (·,))。

            Returns:
                画像ごとの (クエリの番号, その画像の正解の中での物体の番号)(決まった組 + 新しい物体の組)。
            """
            ex = None if fixed_pairs is None else [f[0].numpy() for f in fx]  # 決まった組のクエリは割り当てに使わない
            m = match({"logits": o["logits"][:, det_q], "boxes": o["boxes"][:, det_q],
                       "mask_parts": o["mask_parts"] | {"query": o["mask_parts"]["query"][:, det_q]}},
                      tg_new, g, self.match_weights, self.mask_cost, exclude=ex)
            res = []
            for a, (qi, ti) in enumerate(m):
                res.append((torch.cat([fx[a][0], (qi // q) * per + k + qi % q]), torch.cat([fx[a][1], new_local[a][ti]])))
            return res

        layers = outs["layers"]
        assign = _scoped("loss/割り当て", assign)
        # 層ごと + pre(画像ごとの (クエリ, その画像の正解の番号))
        own = [assign(o, fixed if fixed_pairs is None else fixed_pairs[oi]) for oi, o in enumerate(layers + [outs["pre"]])]
        enc_ind = _scoped("loss/割り当て", match)(outs["enc"], tg_present, g, self.match_weights, self.mask_cost)
        enc_local = [(eq, present_local[a][et]) for a, (eq, et) in enumerate(enc_ind)]
        enc_dec = [(((eq // q) * per + k + eq % q), present_local[a][et]) for a, (eq, et) in enumerate(enc_ind)]
        go = union_indices(own + [enc_dec])

        def flat(indices) -> tuple:
            """画像ごとの組を、まとめた番号にして GPU へ 1 回で送る。

            Args:
                indices: 画像ごとの (クエリの番号, その画像の正解の中での物体の番号)。

            Returns:
                ((画像, クエリ, まとめた正解の番号), 写っているか, 写っている組だけの (画像, クエリ, 正解))。
            """
            bi = torch.cat([torch.full_like(qq, a) for a, (qq, _) in enumerate(indices)])
            qi = torch.cat([qq for qq, _ in indices])
            ti = torch.cat([tt + starts[a] for a, (_, tt) in enumerate(indices)])
            m = present_cpu[ti]
            data = torch.cat([bi, qi, ti, m.long(), bi[m], qi[m], ti[m]]).to(dev, non_blocking=True)
            s = data.split([len(bi)] * 4 + [int(m.sum())] * 3)
            return (s[0], s[1], s[2]), s[3].bool(), (s[4], s[5], s[6])

        go_present = flat(go)[2]
        # 割る数(画像ごと): n = 物体の数 × 組の数、枠の損失は割り当ての和のうち写っている組の数
        n_img_t = (n_per_img * g).clamp(min=1).to(dev)  # (A,)
        n_go = torch.bincount(go_present[0], minlength=n_img).float().clamp(min=1)  # (A,)
        valid = torch.ones(n_img, layers[0]["logits"].shape[1], dtype=torch.bool, device=dev)  # (A, M) 埋めていないクエリ
        for a in range(n_img):
            for gi in range(g):
                valid[a, gi * per + ks[a]: gi * per + k] = False
        cache: dict = {}
        total = torch.zeros(n_img, device=dev)
        parts, ddfs = {}, []

        def per_img(x: torch.Tensor, bi: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            """組ごとの値を画像ごとに足して、画像の割る数で割る。

            Args:
                x: 組ごとの値 (P,)。
                bi: 組ごとの画像の番号 (P,)。
                n: 画像ごとの割る数 (A,)。

            Returns:
                画像ごとの値 (A,)。
            """
            return torch.zeros(n_img, device=dev, dtype=torch.float).index_add(0, bi, x.float()) / n

        def mal(o, bi, qi, ti, ious, n, val) -> torch.Tensor:
            """MAL(GTRCriterion._mal と同じ。画像ごとに足して割る。val の外(埋めたクエリ)は除く)。

            Args:
                o: 1 つの出力({"logits": (A, M', K), ...})。
                bi: 組ごとの画像の番号 (P,)。
                qi: 組ごとのクエリの番号 (P,)。
                ti: 組ごとの正解の番号 (P,)(まとめた中の番号)。
                ious: 組ごとの予測の枠と正解の枠の IoU (P,)(MAL の目標の強さ。隠れている物体は 1)。
                n: 画像ごとの割る数 (A,)。
                val: 損失をかけるクエリ (A, M') の bool(None なら全部)。

            Returns:
                画像ごとの MAL (A,)。
            """
            x = o["logits"].float()  # (A, M', K)
            target = torch.zeros_like(x)
            pos = torch.zeros_like(x)
            if len(bi):
                target[bi, qi, tg["labels"][ti]] = ious.pow(GAMMA)
                pos[bi, qi, tg["labels"][ti]] = 1.0
            weight = x.sigmoid().detach().pow(GAMMA) * (1 - pos) + pos
            bce = F.binary_cross_entropy_with_logits(x, target, weight=weight, reduction="none")  # (A, M', K)
            if val is not None:
                bce = bce * val[..., None]
            return bce.sum((1, 2)) / n

        def boxes(o, bi, qi, ti, n) -> tuple[torch.Tensor, torch.Tensor]:
            """枠の L1 と GIoU の損失(GTRCriterion._boxes と同じ。画像ごとに足して割る)。

            Args:
                o: 1 つの出力({"boxes": (A, M', 4), ...})。
                bi: 組ごとの画像の番号 (P,)(写っている物体の組だけ)。
                qi: 組ごとのクエリの番号 (P,)。
                ti: 組ごとの正解の番号 (P,)(まとめた中の番号)。
                n: 画像ごとの割る数 (A,)。

            Returns:
                (画像ごとの L1 (A,), 画像ごとの GIoU の損失 (A,))。
            """
            src, tgt = o["boxes"][bi, qi].float(), tg["boxes"][ti]
            l1 = (src - tgt).abs().sum(-1)  # (P,)
            giou = 1 - _paired_iou_giou(src, tgt)[1]  # 組ごとの GIoU(P × P の行列を作らない)
            return per_img(l1, bi, n), per_img(giou, bi, n)

        def fgl(o, bi, qi, ti, ious, n, key) -> torch.Tensor:
            """FGL(枠の辺までの距離の分布の損失。GTRCriterion._fgl と同じ。画像ごとに足して割る)。

            Args:
                o: 1 つの出力({"corners": (A, M', 4·R), "refs": (A, M', 4), ...})。
                bi: 組ごとの画像の番号 (P,)(写っている物体の組だけ)。
                qi: 組ごとのクエリの番号 (P,)。
                ti: 組ごとの正解の番号 (P,)(まとめた中の番号)。
                ious: 組ごとの予測の枠と正解の枠の IoU (P,)(損失の重み)。
                n: 画像ごとの割る数 (A,)。
                key: 距離の目標を使い回す cache の鍵。

            Returns:
                画像ごとの FGL (A,)。
            """
            logp = rearrange(o["corners"][bi, qi].float(), "a (four r) -> (a four) r", four=4).log_softmax(-1)  # (P·4, R)
            if key not in cache:
                cache[key] = distance_targets(o["refs"][bi, qi].detach().float(),
                                              box_convert(tg["boxes"][ti], "cxcywh", "xyxy"), self.bins)
            left, w_left, w_right = cache[key]
            nll = -(w_left * logp.gather(1, left[:, None])[:, 0] + w_right * logp.gather(1, (left + 1)[:, None])[:, 0])
            per_pair = rearrange(nll * ious.repeat_interleave(4), "(p four) -> p four", four=4).sum(-1)  # (P,)
            return per_img(per_pair, bi, n)

        def ddf(o, teacher, bi, qi, ious, is_dn, val) -> torch.Tensor:
            """DDF(GTRCriterion._ddf と同じ。正例・負例の数と平均は画像ごと、埋めたクエリは除く)。

            Args:
                o: 1 つの出力({"corners": (A, M', 4·R), ...})。
                teacher: 先生の出力(最後の層。同じ形)。
                bi: 正例の組の画像の番号 (P,)。
                qi: 正例の組のクエリの番号 (P,)。
                ious: 正例の組の IoU (P,)(正例の重み)。
                is_dn: denoising の出力か。
                val: 損失をかけるクエリ (A, M') の bool(None なら全部)。

            Returns:
                画像ごとの DDF (A,)。
            """
            pred = rearrange(o["corners"].float(), "b m (four r) -> b (m four) r", four=4)
            tgt = rearrange(teacher["corners"].float(), "b m (four r) -> b (m four) r", four=4)
            w = teacher["logits"].float().sigmoid().max(-1).values  # (A, M')
            mask = torch.zeros_like(w, dtype=torch.bool)
            if len(bi):
                mask[bi, qi] = True
                w[bi, qi] = ious.to(w.dtype)
            w = w.detach().repeat_interleave(4, dim=-1)  # (A, M'·4)
            mask = mask.repeat_interleave(4, dim=-1)
            v = (torch.ones_like(mask) if val is None else val.repeat_interleave(4, dim=-1)).float()
            kl = F.kl_div(F.log_softmax(pred / DDF_T, -1), F.softmax(tgt.detach() / DDF_T, -1), reduction="none").sum(-1)
            loss = w * DDF_T ** 2 * kl  # (A, M'·4)
            mf = mask.float()
            if not is_dn:  # 正例・負例の数の重み(画像ごと。区間 1 本のときの scale = 8 / 1)
                cache["num_pos"] = (mf.sum(1) * 8) ** 0.5
                cache["num_neg"] = (((1 - mf) * v).sum(1) * 8) ** 0.5
            l_pos = (loss * mf).sum(1) / mf.sum(1).clamp(min=1)
            l_neg = (loss * (1 - mf) * v).sum(1) / ((1 - mf) * v).sum(1).clamp(min=1)
            npos, nneg = cache["num_pos"], cache["num_neg"]
            return (l_pos * npos + l_neg * nneg) / (npos + nneg)

        def masks_loss(o, indices, bi, ti, n) -> tuple[torch.Tensor, torch.Tensor]:
            """マスクの損失(点を選んだ BCE と Dice。GTRCriterion._masks と同じ。画像ごとに足して割る)。

            Args:
                o: 1 つの出力({"mask_parts": {"spatial", "query", "bias"}, ...})。
                indices: 画像ごとの (クエリの番号, その画像の正解の中での物体の番号)。
                bi: 組ごとの画像の番号 (P,)(flat の並び)。
                ti: 組ごとの正解の番号 (P,)(まとめた中の番号)。
                n: 画像ごとの割る数 (A,)。

            Returns:
                (画像ごとの BCE (A,), 画像ごとの Dice の損失 (A,))。
            """
            parts_ = o["mask_parts"]
            # 割り当てたクエリのマスク: 画像ごとに画素の特徴とクエリの内積(GTRCriterion._masks_from_parts と同じ。組の数だけ
            # 画素の特徴を複製しない)。並びは flat と同じ(画像の順、画像の中は割り当ての順) (P, mh, mw)
            pieces = [torch.einsum("chw,ac->ahw", parts_["spatial"][a], parts_["query"][a, qq.to(dev)])
                      for a, (qq, _) in enumerate(indices) if len(qq)]
            if not pieces:
                z = torch.zeros(n_img, device=dev) + parts_["query"].sum() * 0
                return z, z
            logits = (torch.cat(pieces) + parts_["bias"]).float()  # (P, mh, mw)
            src = rearrange(logits, "a h w -> a 1 h w")
            n_points = max(src.shape[-2], src.shape[-2] * src.shape[-1] // POINT_RATIO)
            with torch.no_grad():
                coords = uncertain_points(src, n_points)
                tp = sample_target_nearest(tg["masks"], ti, coords)
            spv = point_sample(src, coords)[:, 0]  # (P, Pt)
            ce = F.binary_cross_entropy_with_logits(spv, tp, reduction="none").mean(-1)
            p = spv.sigmoid()
            dice = 1 - (2 * (p * tp).sum(-1) + 1) / (p.sum(-1) + tp.sum(-1) + 1)
            return per_img(ce, bi, n), per_img(dice, bi, n)

        def add(o: dict, indices, n_own, use_go: bool, teacher: dict | None, key: str, is_dn: bool = False,
                has_masks: bool = True, val: torch.Tensor | None = None) -> dict:
            """1 つの出力の損失を計算して合計 total に足す。

            Args:
                o: 1 つの出力。
                indices: 画像ごとの自分の割り当て (クエリの番号, その画像の正解の中での物体の番号)。
                n_own: 自分の割り当てで計算する損失を割る数 (A,)。
                use_go: 枠の損失に、割り当ての和(go_present)を使うか。
                teacher: DDF の先生の出力(None なら DDF なし)。
                key: FGL の目標の cache の鍵。
                is_dn: denoising の出力か。
                has_masks: マスクの損失をかけるか(enc は False)。
                val: 損失をかけるクエリ (A, M') の bool(None なら全部)。

            Returns:
                項ごとの画像ごとの損失 {"mal", "bbox", "giou", ("fgl", "ddf", "mask", "dice")}(どれも (A,)。重みを掛ける前)。
            """
            nonlocal total
            (bi, qi, ti), pres, own_present = flat(indices)
            ious = _paired_iou_giou(o["boxes"][bi, qi].detach().float(), tg["boxes"][ti])[0]
            ious = torch.where(pres, ious, torch.ones_like(ious))  # 隠れている物体: MAL の目標は 1
            l = {"mal": mal(o, bi, qi, ti, ious, n_own, val)}
            gb, gq, gt = go_present if use_go else own_present  # 枠の損失は写っている物体の組だけ
            n_box = n_go if use_go else n_own
            l["bbox"], l["giou"] = boxes(o, gb, gq, gt, n_box)
            if "corners" in o:
                gious = _paired_iou_giou(o["boxes"][gb, gq].detach().float(), tg["boxes"][gt])[0]
                l["fgl"] = fgl(o, gb, gq, gt, gious, n_box, key + ("_go" if use_go else ""))
                if teacher is not None:
                    l["ddf"] = ddf(o, teacher, gb, gq, gious, is_dn, val)
            if has_masks and "mask_parts" in o:
                l["mask"], l["dice"] = masks_loss(o, indices, bi, ti, n_own)
            total = total + sum(self.w[kk] * v for kk, v in l.items())
            return l

        mal, boxes, fgl = _scoped("loss/MAL", mal), _scoped("loss/枠", boxes), _scoped("loss/FGL", fgl)
        ddf, masks_loss, flat = _scoped("loss/DDF", ddf), _scoped("loss/マスク", masks_loss), _scoped("loss/番号の準備", flat)

        def add_group(os: list[dict], inds: list, n_own, use_go: bool, ddf_teacher: dict | None, ddf_upto: int, key: str,
                      is_dn: bool, val: torch.Tensor | None) -> list[dict]:
            """同じ規則の出力の並び os の損失を、出力を積み重ねて 1 回で計算する(add を出力ごとに呼ぶのと同じ値)。

            Args:
                os: 出力の並び(L 個。どれも同じ形 (A, M', ·))。corners は先頭の層だけが持つ(pre は持たない)。
                inds: 出力ごとの自分の割り当て(画像ごとの (クエリ, その画像の正解の番号))。
                n_own: 自分の割り当てで計算する損失を割る数 (A,)。
                use_go: 枠の損失に、割り当ての和(go_present)を使うか。
                ddf_teacher: DDF の先生の出力(None なら DDF なし)。
                ddf_upto: DDF をかける出力の数(先頭からこの数まで)。
                key: FGL の目標の cache の鍵。
                is_dn: denoising の出力か。
                val: 埋めていないクエリ (A, M')(None なら全部)。

            Returns:
                出力ごとの損失の内訳 {"mal", "bbox", "giou", ("fgl", "ddf",) "mask", "dice"}(どれも (A,))。
            """
            nonlocal total
            n_l = len(os)
            res = [{} for _ in range(n_l)]
            # 出力ごとの自分の割り当てを、(出力, 画像, クエリ, 正解) の番号にまとめる
            fl = [flat(ix) for ix in inds]
            li = torch.cat([torch.full_like(f[0][0], j) for j, f in enumerate(fl)])
            bi, qi, ti = (torch.cat([f[0][c] for f in fl]) for c in range(3))
            pres = torch.cat([f[1] for f in fl])
            logits = torch.stack([o["logits"] for o in os]).float()  # (L, A, M', K)
            boxes_all = torch.stack([o["boxes"] for o in os])  # (L, A, M', 4)
            ious = _paired_iou_giou(boxes_all[li, bi, qi].detach().float(), tg["boxes"][ti])[0]
            ious = torch.where(pres, ious, torch.ones_like(ious))  # 隠れている物体: MAL の目標は 1
            # MAL(GTRCriterion._mal と同じ。画像ごとに足して割る。埋めたクエリは除く)
            target = torch.zeros_like(logits)
            pos = torch.zeros_like(logits)
            if len(bi):
                target[li, bi, qi, tg["labels"][ti]] = ious.pow(GAMMA)
                pos[li, bi, qi, tg["labels"][ti]] = 1.0
            weight = logits.sigmoid().detach().pow(GAMMA) * (1 - pos) + pos
            bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none")  # (L, A, M', K)
            if val is not None:
                bce = bce * val[None, ..., None]
            mal_l = bce.sum((2, 3)) / n_own  # (L, A)
            # 枠(L1・GIoU): use_go なら全部の出力で同じ割り当ての和、そうでなければ自分の割り当て(写っている組)
            if use_go:
                gb, gq, gt = go_present
                n_box = n_go
                src = boxes_all[:, gb, gq].float()  # (L, P, 4)
                gl = None
            else:
                own_p = [f[2] for f in fl]
                gl = torch.cat([torch.full_like(p[0], j) for j, p in enumerate(own_p)])
                gb, gq, gt = (torch.cat([p[c] for p in own_p]) for c in range(3))
                n_box = n_own
                src = boxes_all[gl, gb, gq].float()  # (P, 4)
            tgt_b = tg["boxes"][gt]

            def per_img_l(x: torch.Tensor, layer: torch.Tensor | None, b: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
                """組ごとの値を (出力, 画像) ごとに足して割る。

                Args:
                    x: 組ごとの値 (P,)。
                    layer: 組ごとの出力の番号 (P,)(None なら全部の出力で同じ組: x は (L, P))。
                    b: 組ごとの画像の番号 (P,)。
                    n: 画像ごとの割る数 (A,)。

                Returns:
                    (出力, 画像) ごとの値 (L, A)。
                """
                out = torch.zeros(n_l, n_img, device=dev, dtype=torch.float)
                if layer is None:
                    out.index_add_(1, b, x.float())
                else:
                    out.view(-1).index_add_(0, layer * n_img + b, x.float())
                return out / n

            if use_go:
                l1 = (src - tgt_b).abs().sum(-1)  # (L, P)
                iou_g, giou = _paired_iou_giou(src.reshape(-1, 4), tgt_b.repeat(n_l, 1))
                iou_g, giou = iou_g.view(n_l, -1), giou.view(n_l, -1)
            else:
                l1 = (src - tgt_b).abs().sum(-1)  # (P,)
                iou_g, giou = _paired_iou_giou(src, tgt_b)
            bbox_l, giou_l = per_img_l(l1, gl, gb, n_box), per_img_l(1 - giou, gl, gb, n_box)
            # FGL・DDF(分布 corners を持つ出力だけ。先頭の n_c 個)
            n_c = sum("corners" in o for o in os)
            fgl_l = ddf_l = None
            if n_c:
                corners = torch.stack([o["corners"] for o in os[:n_c]])  # (Lc, A, M', 4·R)
                iou_det = iou_g.detach()  # FGL・DDF の重み(その出力の枠と正解の IoU)
                if use_go:
                    cg, gious = corners[:, gb, gq].float(), iou_det[:n_c]  # (Lc, P, 4·R), (Lc, P)
                    layer_c = None
                else:
                    sel = gl < n_c
                    cg, gious = corners[gl[sel], gb[sel], gq[sel]].float(), iou_det[sel]  # (P', 4·R), (P',)
                    layer_c = gl[sel]
                logp = rearrange(cg, "... (four r) -> ... four r", four=4).log_softmax(-1)  # (…, P, 4, R)
                if key not in cache:
                    rb, rq, rt = (gb, gq, gt) if use_go else (gb[sel], gq[sel], gt[sel])
                    cache[key] = distance_targets(os[0]["refs"][rb, rq].detach().float(),
                                                  box_convert(tg["boxes"][rt], "cxcywh", "xyxy"), self.bins)
                left, w_left, w_right = cache[key]  # (P·4,)
                left4, wl4, wr4 = (rearrange(x_, "(p four) -> p four", four=4) for x_ in (left, w_left, w_right))
                lg = logp.gather(-1, left4.expand(logp.shape[:-1])[..., None])[..., 0]
                rg = logp.gather(-1, (left4 + 1).expand(logp.shape[:-1])[..., None])[..., 0]
                nll = -(wl4 * lg + wr4 * rg)  # (…, P, 4)
                per_pair = (nll * gious[..., None]).sum(-1)  # (Lc, P) か (P',)
                if use_go:
                    fgl_l = torch.zeros(n_c, n_img, device=dev).index_add_(1, gb, per_pair.float()) / n_box
                else:
                    fgl_l = torch.zeros(n_c * n_img, device=dev).index_add_(0, layer_c * n_img + gb[sel],
                                                                           per_pair.float()).view(n_c, n_img) / n_box
                if ddf_teacher is not None and ddf_upto:
                    # DDF(GTRCriterion._ddf と同じ。先生は ddf_teacher、正例・負例の数と平均は画像ごと、埋めたクエリは除く)
                    db, dq = (gb, gq) if use_go else (gb[sel], gq[sel])
                    pred = rearrange(corners[:ddf_upto].float(), "l b m (four r) -> l b (m four) r", four=4)
                    tgt_c = rearrange(ddf_teacher["corners"].float(), "b m (four r) -> b (m four) r", four=4)
                    w = ddf_teacher["logits"].float().sigmoid().max(-1).values  # (A, M')
                    w = w[None].repeat(ddf_upto, 1, 1)  # (Ld, A, M')
                    mask = torch.zeros_like(w[0], dtype=torch.bool)
                    if len(db):
                        mask[db, dq] = True
                        gious_d = gious[:ddf_upto] if use_go else None
                        if use_go:
                            w[:, db, dq] = gious_d.to(w.dtype)
                        else:  # 自分の割り当ての組(出力ごとに違う)
                            for j in range(ddf_upto):
                                mj = layer_c == j
                                w[j, gb[sel][mj], gq[sel][mj]] = gious[mj].to(w.dtype)
                    w = w.detach().repeat_interleave(4, dim=-1)  # (Ld, A, M'·4)
                    mf = mask.repeat_interleave(4, dim=-1).float()  # (A, M'·4)
                    v = (torch.ones_like(mf) if val is None else val.repeat_interleave(4, dim=-1).float())
                    kl = F.kl_div(F.log_softmax(pred / DDF_T, -1), F.softmax(tgt_c.detach()[None] / DDF_T, -1),
                                  reduction="none").sum(-1)  # (Ld, A, M'·4)
                    loss = w * DDF_T ** 2 * kl
                    if not is_dn:  # 正例・負例の数の重み(画像ごと。区間 1 本のときの scale = 8 / 1)
                        cache["num_pos"] = (mf.sum(1) * 8) ** 0.5
                        cache["num_neg"] = (((1 - mf) * v).sum(1) * 8) ** 0.5
                    l_pos = (loss * mf).sum(-1) / mf.sum(-1).clamp(min=1)  # (Ld, A)
                    l_neg = (loss * (1 - mf) * v).sum(-1) / ((1 - mf) * v).sum(-1).clamp(min=1)
                    npos, nneg = cache["num_pos"], cache["num_neg"]
                    ddf_l = (l_pos * npos + l_neg * nneg) / (npos + nneg)  # (Ld, A)
            # マスク: 出力ごと・画像ごとに内積でマスクを作り、点の選び方と損失は全部の組で 1 回
            pieces = []
            for j, (o, ix) in enumerate(zip(os, inds)):
                pp = o["mask_parts"]
                for a, (qq, _) in enumerate(ix):
                    if len(qq):
                        pieces.append(torch.einsum("chw,ac->ahw", pp["spatial"][a], pp["query"][a, qq.to(dev)]))
            if pieces:
                mlog = (torch.cat(pieces) + os[0]["mask_parts"]["bias"]).float()  # (P, mh, mw)。並びは li・bi と同じ
                srcm = rearrange(mlog, "a h w -> a 1 h w")
                n_points = max(srcm.shape[-2], srcm.shape[-2] * srcm.shape[-1] // POINT_RATIO)
                with torch.no_grad():
                    coords = uncertain_points(srcm, n_points)
                    tp = sample_target_nearest(tg["masks"], ti, coords)
                spv = point_sample(srcm, coords)[:, 0]
                ce = F.binary_cross_entropy_with_logits(spv, tp, reduction="none").mean(-1)
                pm = spv.sigmoid()
                dice = 1 - (2 * (pm * tp).sum(-1) + 1) / (pm.sum(-1) + tp.sum(-1) + 1)
                mask_l, dice_l = per_img_l(ce, li, bi, n_own), per_img_l(dice, li, bi, n_own)
            else:
                z = torch.zeros(n_l, n_img, device=dev) + os[0]["mask_parts"]["query"].sum() * 0
                mask_l = dice_l = z
            for j in range(n_l):
                r = {"mal": mal_l[j], "bbox": bbox_l[j], "giou": giou_l[j]}
                if fgl_l is not None and j < n_c:
                    r["fgl"] = fgl_l[j]
                    if ddf_l is not None and j < ddf_upto:
                        r["ddf"] = ddf_l[j]
                r["mask"], r["dice"] = mask_l[j], dice_l[j]
                res[j] = r
                total = total + sum(self.w[kk] * v for kk, v in r.items())
            return res

        add_group = _scoped("loss/まとめた計算", add_group)
        last = len(layers) - 1
        # decoder の層と pre(枠の損失は割り当ての和、DDF は最後の層を先生にして途中の層だけ)
        res = add_group(layers + [outs["pre"]], own, n_img_t, True, layers[last], last, "fgl_go", False, valid)
        parts = {kk: float(v.detach().mean()) for kk, v in res[last].items()}
        ddfs = [float(r["ddf"].detach().mean()) for r in res[:last] if "ddf" in r]
        l_enc = add(outs["enc"], enc_local, n_img_t, False, None, "fgl_enc", has_masks=False)
        res_dn = []
        meta = outs.get("dn_meta")
        if meta is not None:  # denoising: 正例のクエリと、作ったときの正解(写っている物体)の番号をそのまま組にする
            tile = meta["num_group"] * g
            dn_ind = [(meta["positive"][a].cpu(), present_local[a][torch.arange(len(present_local[a])).repeat(tile)])
                      for a in range(n_img)]
            n_dn = n_img_t * meta["num_group"]
            dl = outs["dn_layers"]
            # denoising の層(先生は最後の層。最後の層自身も先生と比べる。元の作りと同じ)と pre
            res_dn = add_group(dl + [outs["dn_pre"]], [dn_ind] * (len(dl) + 1), n_dn, False, dl[-1], len(dl), "fgl_dn", True,
                               None)
        parts["ddf"] = sum(ddfs) / max(len(ddfs), 1)
        # 損失の内訳(学習のログ用): 出力の種類(dec = decoder の全部の層 + pre、enc、dn = denoising の全部の層 + pre)ごと・
        # 項ごとの、重みを掛けた和。画像ごとの値 (A,)(全部を足すと画像ごとの損失 total になる)
        contrib = {}
        for name, rs in (("dec", res), ("enc", [l_enc]), ("dn", res_dn)):
            for r in rs:
                for kk, v in r.items():
                    contrib[f"{name}_{kk}"] = contrib.get(f"{name}_{kk}", 0) + self.w[kk] * v.detach()
        parts["contrib"] = contrib
        if fixed_pairs is not None:
            return total, parts, own
        # 次のフレームに渡す新しい追跡のクエリ: 最後の層の組 0 で、新しい物体と組になった検出のクエリ(組 0 の中の番号を、
        # 埋めたクエリを除いた並び [本物の追跡 ks[a] | 検出 Q] の番号にする)
        born = []
        for a in range(n_img):
            lq, lt = own[last][a]
            born.append({int(qq) - k + ks[a]: int(tt) for qq, tt in zip(lq.tolist(), lt.tolist()) if k <= qq < per})
        return total, parts, born
