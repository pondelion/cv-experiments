"""自作の GTR 系の動画版(models/video/gtr_track.py)の 1 フレーム分の損失。exp00502。

損失の中身と重みは画像の GTR 系(gtr_criterion.GTRCriterion)と同じ。違うのは正解の割り当ての作り方と、隠れている物体の扱いだけ。

1 フレームのクエリの並び: 組ごとに [追跡のクエリ K 個 | 検出のクエリ Q 個](組の数 G。全部で G·(K + Q) 個)。
正解の割り当て(VidEoMT の公式の割り当て(losses/video_consistent_criterion.py)を、追跡と検出のクエリを分ける形にしたもの):
    - 追跡のクエリ: 前のフレームまでに割り当てた物体を、決まった組としてそのまま持つ(どの組の複製も同じ物体)。物体がこの
      フレームに写っていない(隠れている)ときも同じ物体で、正解のクラスはその物体のクラス、正解のマスクは空(VidEoMT と同じ)
    - 検出のクエリ: このフレームで初めて写った物体(まだどのクエリも持っていない物体)とだけ、ハンガリアン法で組にする
      (画像の GTR 系と同じ費用。組ごとに割り当てる)。追跡のクエリが持つ物体は検出のクエリに割り当てない
      (= 物体を持つクエリは後から出た物体に使わない・新しい物体は空いたクエリで拾う、の VidEoMT の規則)
    - two-stage の候補(enc)と denoising は、画像の GTR 系と同じく、このフレームに写っている全部の物体に対してかける
      (どちらも検出のクエリを作る部分の学習なので、追跡とは関係しない)
隠れている物体の組の損失(私の判断。GTR 系の損失は枠がある前提なので):
    - 枠の損失(L1・GIoU・FGL・DDF)はかけない(正解の枠がない)
    - クラスの損失(MAL)の目標は、IoU の代わりに 1 にする(MAL の目標は予測の枠と正解の枠の IoU^γ だが、正解の枠がない)
    - マスクの損失は空のマスクを正解にしてかける(VidEoMT と同じ)

形の記号: G = 組の数、K = 追跡のクエリの数、Q = 1 組の検出のクエリの数、N = このフレームの正解の物体の数(写っている物体 + 追跡の
クエリが持つ隠れている物体)、A = 割り当ての数。
"""
import torch

from segmentation.instance_segmentation.lib_common.losses.gtr_criterion import GTRCriterion, paired_iou, union_indices
from segmentation.instance_segmentation.lib_common.losses.rfdetr_criterion import match


def _subset(tg: dict, idx: torch.Tensor) -> dict:
    """正解の一部(idx の物体)だけを取り出す(画像は 1 枚)。

    Args:
        tg: {"labels", "masks", "boxes", "owner", "counts"}(targets_from_masks と同じ形)。
        idx: 取り出す物体の番号 (n,)。

    Returns:
        同じ形の正解。
    """
    return {"labels": tg["labels"][idx], "masks": tg["masks"][idx], "boxes": tg["boxes"][idx],
            "owner": tg["owner"][idx], "counts": [len(idx)]}


def _slice_out(o: dict, idx: torch.Tensor) -> dict:
    """1 つの出力から、idx のクエリだけを取り出す(割り当ての費用の計算に使う)。

    Args:
        o: {"logits": (1, M, K), "boxes": (1, M, 4), "mask_parts"(なくてもよい)}。
        idx: クエリの番号 (m,)。

    Returns:
        同じ形の出力(クエリの数 m)。
    """
    out = {"logits": o["logits"][:, idx], "boxes": o["boxes"][:, idx]}
    if o.get("mask_parts") is not None:
        out["mask_parts"] = o["mask_parts"] | {"query": o["mask_parts"]["query"][:, idx]}
    return out


class GTRTrackCriterion(GTRCriterion):
    """GTR 系の動画版の 1 フレーム分の損失(ファイルの先頭の規則)。引数は GTRCriterion と同じ。"""

    def frame_loss(self, outs: dict, tg: dict, present: torch.Tensor, track_objs: list[int], num_track: int,
                   num_det: int) -> tuple[torch.Tensor, dict, dict[int, int]]:
        """1 フレーム(画像は 1 枚)の損失。

        Args:
            outs: GTRSeg.forward(track つき)の返り値(学習のとき)。denoising は present の物体から作ったもの。
            tg: このフレームの正解 N 個({"labels", "masks"(隠れている物体は空), "boxes"(隠れている物体は 0), "owner", "counts"})。
            present: 物体がこのフレームに写っているか (N,) の bool。
            track_objs: 追跡のクエリ k が持つ物体の、tg の中での番号(長さ K)。
            num_track: K。
            num_det: Q。

        Returns:
            (損失の合計(0 次元。N × G で割ったもの), 最後の層の内訳の float, 最後の層の組 0 で新しい物体と組になった
            検出のクエリ {組 0 の中でのクエリの番号(K 以上): tg の中での物体の番号})。
        """
        dev = tg["labels"].device
        g = self.groups if self.training else 1
        k, q, per = num_track, num_det, num_track + num_det
        assigned = set(track_objs)
        present_cpu = present.cpu()  # 写っているかは CPU の上で使う(GPU の処理の終わりを待つのはここの 1 回だけ)
        new_ids_cpu = torch.tensor([j for j in range(len(tg["labels"])) if present_cpu[j] and j not in assigned],
                                   dtype=torch.long)  # このフレームで初めて写った物体
        present_ids_cpu = present_cpu.nonzero()[:, 0]
        tg_new, tg_present = _subset(tg, new_ids_cpu.to(dev)), _subset(tg, present_ids_cpu.to(dev))
        det_q = torch.cat([torch.arange(gi * per + k, (gi + 1) * per, device=dev) for gi in range(g)])  # (G·Q,)
        fixed_q = torch.cat([torch.arange(gi * per, gi * per + k) for gi in range(g)]).long()  # (G·K,) 追跡のクエリ
        fixed_t = torch.tensor(track_objs * g, dtype=torch.long)

        def to_global(pairs: list[tuple], ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """検出のクエリだけで決めた割り当て(G·Q の中の番号、部分の正解の番号)を、全部のクエリ・tg の番号にする。"""
            qi, ti = pairs[0]
            gi, ql = qi // q, qi % q
            return gi * per + k + ql, ids[ti]  # ids は CPU の上の番号

        def assign(o: dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
            """1 つの出力の割り当て: 追跡のクエリの決まった組 + 検出のクエリと新しい物体のハンガリアン法。"""
            dq, dt = to_global(match(_slice_out(o, det_q), tg_new, g, self.match_weights, self.mask_cost), new_ids_cpu)
            return [(torch.cat([fixed_q, dq]), torch.cat([fixed_t, dt]))]

        layers = outs["layers"]
        own = [assign(o) for o in layers + [outs["pre"]]]  # 層ごと + pre
        # enc: 検出のクエリを作る候補(組ごとに Q 個)。写っている全部の物体に割り当てる(画像の GTR 系と同じ)
        enc_ind = match(outs["enc"], tg_present, g, self.match_weights, self.mask_cost)
        enc_q, enc_t = enc_ind[0]
        enc_local = [(enc_q, present_ids_cpu[enc_t])]  # enc の中での番号のまま(enc 自身の損失に使う)
        # 枠の損失の割り当ての和には、enc の候補 i を、それから作った decoder の検出のクエリの番号に直して入れる
        enc_dec = [(((enc_q // q) * per + k + enc_q % q), present_ids_cpu[enc_t])]
        go = union_indices(own + [enc_dec])
        n = torch.tensor(max(len(tg["labels"]) * g, 1), dtype=torch.float, device=dev)
        # 枠の損失を割る数: 割り当ての和のうち、写っている物体の組の数(画像の版の n_go と同じ考え方。隠れている物体は枠の損失がない)
        n_go = torch.tensor(max(int(present_cpu[go[0][1]].sum()), 1), dtype=torch.float, device=dev)
        cache: dict = {}
        total, parts, ddfs = 0.0, {}, []

        def to_dev(indices) -> tuple:
            """割り当て(画像は 1 枚)の番号を CPU の上で作り、GPU へは 1 回でまとめて送る(GPU の処理の終わりを待つ回数を減らす。
            GTRCriterion._flat と同じ番号)。

            Returns:
                (全部の組の (画像, クエリ, 正解) の番号, その正解が写っているか, 写っている物体の組だけの (画像, クエリ, 正解))。
            """
            q, t = indices[0][0].long(), indices[0][1].long()
            m = present_cpu[t]
            qp, tp = q[m], t[m]
            flat = torch.cat([torch.zeros_like(q), q, t, m.long(), torch.zeros_like(qp), qp, tp]).to(dev, non_blocking=True)
            bi, qi, ti, pres, gb, gq, gt = flat.split([len(q)] * 4 + [len(qp)] * 3)
            return (bi, qi, ti), pres.bool(), (gb, gq, gt)

        go_present = to_dev(go)[2]  # 枠の損失に使う割り当ての和(写っている物体の組だけ)。全部の層で同じなので 1 回だけ作る

        def add(o: dict, indices, n_own, use_go: bool, teacher: dict | None, key: str, is_dn: bool = False,
                masks: bool = True) -> dict:
            """1 つの出力の損失を計算して合計に足す(GTRCriterion.forward の add と同じ。隠れている物体の扱いだけ違う)。"""
            nonlocal total
            (bi, qi, ti), pres, own_present = to_dev(indices)
            ious = paired_iou(o["boxes"][bi, qi].detach().float(), tg["boxes"][ti])
            ious = torch.where(pres, ious, torch.ones_like(ious))  # 隠れている物体: MAL の目標は 1
            l = {"mal": self._mal(o, bi, qi, ti, tg, ious, n_own)}
            gb, gq, gt = go_present if use_go else own_present  # 枠の損失は写っている物体の組だけ
            n_box = n_go if use_go else n_own
            l["bbox"], l["giou"] = self._boxes(o, gb, gq, gt, tg, n_box)
            if "corners" in o:
                gious = paired_iou(o["boxes"][gb, gq].detach().float(), tg["boxes"][gt])
                l["fgl"] = self._fgl(o, gb, gq, gt, tg, gious, n_box, cache, key + ("_go" if use_go else ""))
                if teacher is not None:
                    l["ddf"] = self._ddf(o, teacher, gb, gq, gious, cache, is_dn)
            if masks and "mask_parts" in o:
                l["mask"], l["dice"] = self._masks(o, indices, tg, ti, n_own)
            total = total + sum(self.w[kk] * v for kk, v in l.items())
            return l

        last = len(layers) - 1
        for i, o in enumerate(layers):  # 途中の層は最後の層を先生にして DDF
            l = add(o, own[i], n, True, layers[last] if i < last else None, "fgl")
            if i == last:
                parts = {kk: float(v.detach()) for kk, v in l.items()}
            elif "ddf" in l:
                ddfs.append(float(l["ddf"].detach()))
        add(outs["pre"], own[-1], n, True, None, "fgl")
        add(outs["enc"], enc_local, n, False, None, "fgl_enc", masks=False)
        meta = outs.get("dn_meta")
        if meta is not None:  # denoising: 正例のクエリと、作ったときの正解(写っている物体)の番号をそのまま組にする
            tile = meta["num_group"] * g
            c = len(present_ids_cpu)
            dn_ind = [(meta["positive"][0].cpu(), present_ids_cpu[torch.arange(c).repeat(tile)])] if c else \
                [(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))]
            n_dn = n * meta["num_group"]
            for o in outs["dn_layers"]:
                add(o, dn_ind, n_dn, False, outs["dn_layers"][-1], "fgl_dn", is_dn=True)
            add(outs["dn_pre"], dn_ind, n_dn, False, None, "fgl_dn")
        parts["ddf"] = sum(ddfs) / max(len(ddfs), 1)
        # 次のフレームに渡す新しい追跡のクエリ: 最後の層の組 0 で、新しい物体と組になった検出のクエリ
        lq, lt = own[last][0]
        born = {int(a): int(b) for a, b in zip(lq.tolist(), lt.tolist()) if k <= a < per}
        return total, parts, born
