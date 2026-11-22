"""CAVIS の segmenter の損失(Lee ら ICCV 2025、"Context-Aware Video Instance Segmentation")。exp00502。

参考: 公式のコード(Seung-Hun-Lee/CAVIS の `cavis/cavis.py` の CAVIS_segmenter.forward・get_prototypical_contrastive_loss、
`cavis/criterion_cavis.py` の VideoSetCriterion.loss_ctxs・loss_reid、`cavis/data_video/dataset_mapper_vps.py`(VIPSeg の正解の作り方)、
`configs/VIPSeg/CAVIS_Segmenter_R50.yaml`)。コードは写さず、規則を読み取って書いた。

1. セグメンテーションの損失: 区間の各フレームを 1 枚の画像とみなし、画像の Mask2Former と同じ損失(全部の層。CTVIS と同じ)。
   正解は区間の全部の物体で、写っていないフレームはマスクが空(クラスはその物体。公式の VIPSeg の正解の作り方と同じ)
2. 文脈の対照学習(loss_ctx、論文の式 (8)): 1 の最後の層の割り当てで、各フレームの各物体に割り当てたクエリの reid の埋め込み
   (文脈の特徴から作る。models/mask2former.py の context_head)を物体ごとに集める(写っていないフレームの割り当ても含む。公式の
   VIPSeg の正解は物体の番号が全フレームで同じため)。物体ごとに、anchor = その物体の全部の埋め込み、正例 = 同じ、負例 = ほかの
   物体の全部の埋め込みとして、下の 4 の損失
3. 原型の対照学習(loss_proto、論文の式 (11)): マスクを作る特徴 (C, h, w) を、正解のマスクを (h, w) に bilinear で縮めた重みで
   平均した「フレームごとの物体の原型」を作る(マスクの和が 0 のフレームは作らない)。物体ごとに、anchor = 正例 = その物体の原型、
   負例 = ほかの物体の原型として、下の 4 の損失。区間に物体が 1 個だけか、区間を通してマスクが空の物体があれば、その区間は使わない(公式)
4. 損失(公式の loss_reid): anchor ごとに、内積で log(1 + Σ_正例 Σ_負例 exp(負例 − 正例)) を足し(anchor について和)、
   cos と正解(正例 1・負例 0)の差の 2 乗の平均を足す。組の数で割る。重みはどれも 2(ctx・ctx_aux・proto・proto_aux)
区間ごとに計算して区間の数で平均する(公式は GPU 1 枚あたり 1 区間で、勾配を GPU の数で平均するのと同じ。組のない区間は 0)。
私の判断の点: 公式は物体の番号が区間の中だけの番号(0 から)で、1 区間ずつなので区間をまたいで混ざらない。ここも区間ごとに分ける。

形の記号: B = 区間の数、T = フレームの数、Q = クエリの数、N = 区間の正解の物体の数、C = 埋め込みの次元。
"""
import torch
import torch.nn.functional as F

from segmentation.instance_segmentation.lib_common.losses.mask2former_criterion import Mask2FormerCriterion

CTX_WEIGHT = CTX_AUX_WEIGHT = PROTO_WEIGHT = PROTO_AUX_WEIGHT = 2.0  # 公式の weight_dict


def contrastive_pair_loss(anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1 組の損失(公式の loss_reid の 1 項)。Args: anchor (K, C)、pos (P, C)、neg (M, C)。Returns: (内積の損失, cos の損失)。"""
    dp, dn = anchor @ pos.T, anchor @ neg.T  # (K, P), (K, M)
    x = (dn[:, None, :] - dp[:, :, None]).flatten(1)  # (K, P·M) 全部の (正例, 負例) の組の 負例 − 正例
    contras = torch.logsumexp(F.pad(x, (0, 1)), dim=1).sum()  # anchor ごとの log(1 + Σ exp(負例 − 正例)) の和
    cos = F.normalize(anchor, dim=-1) @ F.normalize(torch.cat([pos, neg]), dim=-1).T  # (K, P + M)
    label = torch.cat([cos.new_ones(len(pos)), cos.new_zeros(len(neg))])
    return contras, ((cos - label) ** 2).mean()


def grouped_loss(groups: list[torch.Tensor], zero: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """物体ごとの埋め込みの集まり [(n_i, C)] から、anchor = 正例 = 自分、負例 = ほかの物体の全部で損失を平均する(組がなければ 0)。"""
    items = []
    for i, g in enumerate(groups):
        others = [h for j, h in enumerate(groups) if j != i]
        if others:
            items.append(contrastive_pair_loss(g, g, torch.cat(others)))
    if not items:
        return zero, zero
    return sum(c for c, _ in items) / len(items), sum(a for _, a in items) / len(items)


class CAVISSegCriterion(Mask2FormerCriterion):
    """CAVIS の segmenter の損失(上の 1〜3)。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        num_frames: 区間のフレームの数 T(公式の VIPSeg の segmenter は 3)。
    """

    def __init__(self, num_classes: int, num_frames: int):
        super().__init__(num_classes)
        self.num_frames = num_frames

    def forward(self, outs: list[dict], targets: list[dict], match_outs=None) -> tuple[torch.Tensor, dict]:
        """Args:
            outs: Mask2Former の層ごとの予測(画像は (b t) の順に B·T 枚)。最後の要素に "reid" (B·T, Q, C)・"mask_feat"。
            targets: 区間ごとの {"labels": (N,), "masks": (N, T, S, S)}(長さ B)。
            match_outs: 使わない(ほかの動画の損失と呼び方をそろえるため)。

        Returns:
            (損失の合計, 内訳 {"class", "mask", "dice", "ctx", "ctx_aux", "proto", "proto_aux"} の float)。
        """
        t = self.num_frames
        frame_targets = [{"labels": tg["labels"], "masks": tg["masks"][:, i]} for tg in targets for i in range(t)]
        seg_loss, parts = super().forward(outs, frame_targets)  # 1。self.last_indices に最後の層の割り当てが残る
        last = outs[-1]
        reid, feat = last["reid"].float(), last["mask_feat"].float()  # (B·T, Q, C), (B·T, C, h, w)
        zero = reid.sum() * 0 + feat.sum() * 0
        starts = [0]
        for tg in frame_targets:
            starts.append(starts[-1] + len(tg["labels"]))
        sums = {k: zero for k in ("ctx", "ctx_aux", "proto", "proto_aux")}
        for b, tg in enumerate(targets):
            n = len(tg["labels"])
            # 2. 文脈の対照学習: 物体ごとに、各フレームで割り当てたクエリの reid の埋め込み
            per_obj = [[] for _ in range(n)]
            for i in range(t):
                f = b * t + i
                qi, ti = self.last_indices[f]
                for q, g in zip(qi.tolist(), (ti - starts[f]).tolist()):
                    per_obj[g].append(reid[f, q])
            c, a = grouped_loss([torch.stack(p) for p in per_obj if p], zero)
            sums["ctx"], sums["ctx_aux"] = sums["ctx"] + c, sums["ctx_aux"] + a
            # 3. 原型の対照学習
            masks = tg["masks"].float()  # (N, T, S, S)
            if n > 1 and masks.flatten(1).sum(-1).min() > 0:
                fb = feat[b * t: (b + 1) * t]  # (T, C, h, w)
                m = F.interpolate(masks, size=fb.shape[-2:], mode="bilinear", align_corners=False)  # (N, T, h, w)
                area = m.flatten(2).sum(-1)  # (N, T)
                protos = []
                for j in range(n):
                    fr = [i for i in range(t) if area[j, i] > 0]
                    if fr:  # (T', C) その物体が写るフレームごとの原型
                        protos.append(torch.stack([(fb[i] * m[j, i]).flatten(1).sum(-1) / area[j, i] for i in fr]))
                c, a = grouped_loss(protos, zero)
                sums["proto"], sums["proto_aux"] = sums["proto"] + c, sums["proto_aux"] + a
        sums = {k: v / len(targets) for k, v in sums.items()}
        total = (seg_loss + CTX_WEIGHT * sums["ctx"] + CTX_AUX_WEIGHT * sums["ctx_aux"] + PROTO_WEIGHT * sums["proto"]
                 + PROTO_AUX_WEIGHT * sums["proto_aux"])
        return total, {**parts, **{k: float(v.detach()) for k, v in sums.items()}}
