"""iFAN(arXiv 2608.03216、"Inference-Aware Learning for Plain Mask Transformers")の学習の損失の自作の実装。exp00103 の D。

公式の学習のコードはまだ公開されていない(公式のリポジトリ Neesky163/iFAN は推論のコードだけ)ので、論文の式と値から書いた。
Mask2Former(EoMT)の損失に、次の 3 つを足す。

    マスクの質: 質の head の値 q̂(sigmoid)を、割り当てたクエリは予測のマスクと正解の soft-IoU、ほかのクエリは 0 を目標に BCE。
    APMR(順位づけ): 正解 k ごとに、割り当てたクエリ i を、紛らわしい外れのクエリ j(割り当てていないクエリのうち、点数が高い
        上位 N_H = 10 個)より上にする。点数の地図 Ŝ = (q̂ · p̂(c_k))^γ · M̂(γ = 2、p̂ はクラスの確率、M̂ はマスクの確率)。
        損失 = 正解のマスクの中の画素の平均の softplus(Ŝ_j − Ŝ_i)。p̂ と M̂ の勾配は止め、質の head だけを学習する。重み 0.1。
    CLSD(層をまたぐ自己蒸留): 正解 k ごとに、最後の層より前の層(最後の 3 層の前の予測)のうち、割り当てたクエリのマスクの
        soft-IoU が最も高い層を先生にし、最後の層より高いときだけ、正解のマスクの中の画素の平均の
        max(0, sg(Ŝ_先生) − Ŝ_最後) を損失にする(先生の勾配は止める)。重み 0.4。

論文に値がなく、こちらで決めたもの(README にも書く):
    - 質の BCE の重み: 1(論文は「拡張した分割の損失の一部として BCE で学習」とだけ書く)
    - 質の BCE と APMR は全部の予測(層)に掛ける(質の head は全部の層で共有と書かれている)。CLSD は最後の層だけ
    - 紛らわしい外れのクエリの選び方: 正解のマスクの中の Ŝ の平均が高い順(論文は「その正解のクラスの確率が高く、マスクの
      重なりが低い、Ŝ の上位 N_H 個」)
    - CLSD の重みの変え方: 学習の 1/3(論文の 12 エポックのうち 4 エポック目)までは 0、その後 0.4 から最後に 0 へ直線で下げる
      (論文は「4 エポック目から始め、その後は直線で下げる」)
    - soft-IoU と Ŝ は予測のマスクの大きさ(stride 4)で計算し、正解のマスクは双線形で縮める

形の記号: B = 画像の数、Q = クエリの数、K = クラスの数、T = 正解の物体の数、h・w = 予測のマスクの縦・横。
"""
import torch
import torch.nn.functional as F
from einops import rearrange

from segmentation.instance_segmentation.lib_common.losses.mask2former_criterion import (W_CLASS, W_DICE, W_MASK,
                                                                                         Mask2FormerCriterion)


def soft_iou(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """マスクの確率と正解の soft-IoU(Σ p·t / Σ (p + t − p·t))。

    Args:
        prob: (A, h, w) の 0〜1。
        target: (A, h, w) の 0〜1。

    Returns:
        (A,)。
    """
    inter = (prob * target).flatten(1).sum(1)
    return inter / ((prob + target).flatten(1).sum(1) - inter).clamp(min=1e-6)


class IFANCriterion(Mask2FormerCriterion):
    """Mask2Former の損失 + iFAN の 3 つの損失。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        gamma: 点数の地図の指数 γ(論文は 2)。
        n_hard: APMR の紛らわしい外れのクエリの数 N_H(論文は 10)。
        w_quality: 質の BCE の重み。
        w_rank: APMR の重み(論文は 0.1)。
        w_distill: CLSD の重み(論文は 0.4)。
        distill_start: CLSD を始める学習の進み方(論文の 12 エポックのうち 4 エポック目 = 1/3)。
        teacher_layers: CLSD の先生の候補にする、最後の層より前の層の数(論文は 3)。
    """

    def __init__(self, num_classes: int, gamma: float = 2.0, n_hard: int = 10, w_quality: float = 1.0,
                 w_rank: float = 0.1, w_distill: float = 0.4, distill_start: float = 1 / 3, teacher_layers: int = 3):
        super().__init__(num_classes)
        self.gamma, self.n_hard, self.w_quality, self.w_rank = gamma, n_hard, w_quality, w_rank
        self.w_distill, self.distill_start, self.teacher_layers = w_distill, distill_start, teacher_layers
        self.progress = 0.0

    def set_progress(self, progress: float) -> None:
        """学習の進み方(0〜1)を受け取る(CLSD の重みを決める)。

        Args:
            progress: 今までの更新の回数 / 全部の回数。
        """
        self.progress = progress

    def _distill_weight(self) -> float:
        """今の CLSD の重み(distill_start までは 0、その後 w_distill から 0 へ直線)。"""
        if self.progress < self.distill_start:
            return 0.0
        return self.w_distill * max(0.0, 1 - (self.progress - self.distill_start) / (1 - self.distill_start))

    def _layer_terms(self, out: dict, indices: list[tuple], labels: torch.Tensor, tsmall: torch.Tensor) -> dict:
        """1 つの予測(層)の、質の BCE と APMR の損失、と CLSD 用の値。

        Args:
            out: {"logits": (B, Q, K+1), "masks": (B, Q, h, w), "quality": (B, Q)}。
            indices: 画像ごとの (クエリの番号, 全体の中の正解の番号)(Mask2FormerCriterion._one の割り当て)。
            labels: 正解のクラス (T,)。
            tsmall: 予測の大きさに縮めた正解のマスク (T, h, w)。

        Returns:
            {"quality": 0 次元, "rank": 0 次元, "per_target": {正解の番号: (soft-IoU, Ŝ の地図 (h, w))}}。
            Ŝ の地図は、質・クラスの確率・マスクの確率のすべてに勾配が流れる(CLSD の生徒として使う)。
        """
        logits, masks, quality = out["logits"].float(), out["masks"].float(), out["quality"].float()
        prob_cls = logits.softmax(-1)  # (B, Q, K+1)
        mprob = masks.sigmoid()  # (B, Q, h, w)
        q = quality.sigmoid()  # (B, Q)
        target_q = torch.zeros_like(quality)  # (B, Q) 割り当てていないクエリは 0
        ranks, per_target = [], {}
        for b, (qi, ti) in enumerate(indices):
            if len(qi) == 0:
                continue
            iou = soft_iou(mprob[b, qi].detach(), tsmall[ti])  # (A_b,)
            target_q[b, qi] = iou
            unmatched = torch.ones(masks.shape[1], dtype=torch.bool, device=masks.device)
            unmatched[qi] = False
            t = tsmall[ti]  # (A_b, h, w) 割り当てた正解のマスク
            area = t.flatten(1).sum(1).clamp(min=1e-6)  # (A_b,)
            cls = labels[ti]  # (A_b,) 正解のクラス
            arange_a = torch.arange(len(qi), device=masks.device)
            # 正解ごとの、全部のクエリの係数 (q̂ · p̂(c_k))^γ: (Q, A_b)。APMR は質だけに勾配(クラスの確率を止める)、
            # CLSD の生徒はクラスの確率にも勾配を流す
            coef_rank = (q[b][:, None] * prob_cls[b][:, cls].detach()) ** self.gamma
            coef = (q[b][:, None] * prob_cls[b][:, cls]) ** self.gamma
            with torch.no_grad():
                # 紛らわしい外れのクエリ: 割り当てていないクエリのうち、正解の中の Ŝ の平均が高い上位 N_H 個: (N_H, A_b)
                inside = torch.einsum("qhw,ahw->qa", mprob[b].detach(), t) / area  # (Q, A_b)
                score = coef_rank.detach() * inside
                score[~unmatched] = -1
                hard = score.topk(min(self.n_hard, int(unmatched.sum())), dim=0).indices  # (N_H, A_b)
            s_i = coef_rank[qi, arange_a][:, None, None] * mprob[b, qi].detach()  # (A_b, h, w)
            s_j = coef_rank[hard, arange_a][..., None, None] * mprob[b][hard].detach()  # (N_H, A_b, h, w)
            # 正解のマスクの中の画素の平均の softplus(Ŝ_j − Ŝ_i): (N_H, A_b) を全部まとめて平均
            ranks.append(((F.softplus(s_j - s_i) * t).flatten(2).sum(-1) / area).flatten())
            s_final = coef[qi, arange_a][:, None, None] * mprob[b, qi]  # (A_b, h, w) CLSD 用(勾配はすべてに流れる)
            for a, k in enumerate(ti.tolist()):
                per_target[k] = (float(iou[a]), s_final[a])
        loss_q = F.binary_cross_entropy_with_logits(quality, target_q.detach())
        loss_rank = torch.cat(ranks).mean() if ranks else quality.sum() * 0
        return {"quality": loss_q, "rank": loss_rank, "per_target": per_target}

    def forward(self, outs: list[dict], targets: list[dict]) -> tuple[torch.Tensor, dict]:
        """Mask2Former の損失と iFAN の損失を足す。

        Args:
            outs: EoMT(quality_head=True)の予測の並び(最後が最終の予測)。
            targets: 画像ごとの {"masks": (T_i, H, W), "labels": (T_i,)}。

        Returns:
            (損失の合計(0 次元), 最終の層の内訳 {"class", "mask", "dice", "quality", "rank", "distill"} の float)。
        """
        labels = torch.cat([t["labels"] for t in targets])
        tmasks = torch.cat([t["masks"] for t in targets]).float()
        counts = [len(t["labels"]) for t in targets]
        owner = torch.repeat_interleave(torch.arange(len(targets), device=labels.device),
                                        torch.as_tensor(counts, device=labels.device))
        starts = [sum(counts[:i]) for i in range(len(counts))]
        h, w = outs[-1]["masks"].shape[-2:]
        # 正解のマスクを予測の大きさに双線形で縮める: (T, H, W) -> (T, h, w)
        tsmall = F.interpolate(rearrange(tmasks, "t hh ww -> t 1 hh ww"), size=(h, w), mode="bilinear",
                               align_corners=False)[:, 0].clamp(0, 1)
        total, parts, layer_targets = 0.0, {}, []
        for n, out in enumerate(outs):
            l = self._one(out, labels, tmasks, owner, starts)
            extra = self._layer_terms(out, self.last_indices, labels, tsmall)
            total = total + W_CLASS * l["class"] + W_MASK * l["mask"] + W_DICE * l["dice"] \
                + self.w_quality * extra["quality"] + self.w_rank * extra["rank"]
            layer_targets.append(extra["per_target"])
            if n == len(outs) - 1:
                parts = {key: float(v.detach()) for key, v in l.items()}
                parts |= {"quality": float(extra["quality"].detach()), "rank": float(extra["rank"].detach())}
        # CLSD: 最後の層を生徒に、その前の teacher_layers 層のうち soft-IoU が最も高い層を先生にする
        final, teachers = layer_targets[-1], layer_targets[-1 - self.teacher_layers:-1]
        distills = []
        for k, (iou_final, s_final) in final.items():
            cands = [tl[k] for tl in teachers if k in tl]
            if not cands:
                continue
            iou_t, s_teacher = max(cands, key=lambda x: x[0])
            if iou_t > iou_final:
                t = tsmall[k]
                distills.append(((s_teacher.detach() - s_final).clamp(min=0) * t).sum() / t.sum().clamp(min=1e-6))
        loss_distill = torch.stack(distills).mean() if distills else total * 0
        total = total + self._distill_weight() * loss_distill
        parts["distill"] = float(loss_distill.detach())
        return total, parts
