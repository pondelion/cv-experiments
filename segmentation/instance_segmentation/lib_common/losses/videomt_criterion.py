"""VidEoMT の損失(公式の videomt_online の学習と同じ規則。exp00502)。

公式(tue-mps/videomt の `VideoHungarianMatcher_Consistent` と `videomt_online.frame_decoder_loss_reshape`)の規則:
    1. 割り当て(区間ごと): 正解の物体を、区間の中で「最初に写ったフレーム」でだけクエリに割り当てる。フレームを時刻の順に見て、
       そのフレームで初めて写った物体だけを、そのフレームの予測との費用(画像の Mask2Former と同じ費用)でハンガリアン法で割り当てる。
       前のフレームで物体を割り当てたクエリは、後から写った物体には使わない(費用を大きくして除く)。
       こうして決めた (クエリ, 物体) の組を、区間の全フレームで使う(同じクエリが区間の最後まで同じ物体を持つ)。
    2. 損失はフレームごと: 各フレームを 1 枚の画像とみなし、割り当てたクエリに
         - クラス: その物体のクラス(物体がそのフレームに写っていなくても、隠れている間もクラスを持ち続けるように同じクラス)
         - マスク: そのフレームの正解のマスク(写っていなければ空のマスク)
       の損失をかける(割り当てのないクエリは「物体なし」)。損失の中身(交差エントロピー、点を選んだ BCE と Dice)は
       画像の Mask2Former の損失と同じ。
    3. 補助の損失: 層ごとの予測それぞれで 1 の割り当てをやり直し、同じ損失をかけて足す(公式と同じ)。

形の記号: B = 区間の数、T = 区間のフレームの数、Q = クエリの数、K = クラスの数、N = 区間の正解の物体の数、
h・w = 予測のマスクの縦・横、S = 正解のマスクの一辺。
"""
import numpy as np
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from segmentation.instance_segmentation.lib_common.losses.mask2former_criterion import (W_CLASS, W_DICE, W_MASK,
                                                                                       Mask2FormerCriterion, match_cost)


class VideoConsistentCriterion(Mask2FormerCriterion):
    """VidEoMT の損失(上の規則)。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
    """

    @torch.no_grad()
    def consistent_match(self, logits: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor, tmasks: torch.Tensor,
                         owner: torch.Tensor, starts: list[int], num_frames: int, counts: list[int]) -> list:
        """区間ごとに、物体が最初に写ったフレームでだけ割り当てて、その組を区間の全フレームで使う。

        Args:
            logits: フレームごとのクラスの値 (B·T, Q, K+1)。
            masks: フレームごとのマスクの値 (B·T, Q, h, w)。
            labels・tmasks・owner・starts: フレームを画像とみなしてつなげた正解(forward の中で作る)。
            num_frames: T。
            counts: 区間ごとの正解の物体の数 N(長さ B)。

        Returns:
            フレームごとの [(クエリの番号, そのフレームの中での正解の番号)](長さ B·T。同じ区間のフレームは同じ組)。
        """
        if len(labels) == 0:
            return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))] * logits.shape[0]
        cost = match_cost(logits, masks, labels, tmasks, owner)  # (B·T, Q, 全部のフレームの正解の数)
        present = tmasks.flatten(1).any(1).cpu().numpy()  # (全部のフレームの正解の数,) そのフレームに写っているか
        out = []
        for b, n in enumerate(counts):
            qi_all, ti_all, used = [], [], np.zeros(logits.shape[1], dtype=bool)
            seen = np.zeros(n, dtype=bool)
            for t in range(num_frames):
                f = b * num_frames + t
                new = [j for j in range(n) if present[starts[f] + j] and not seen[j]]  # このフレームで初めて写った物体
                if not new:
                    continue
                c = cost[f][:, [starts[f] + j for j in new]].copy()  # (Q, 新しい物体の数)
                c[used] = 1e6  # 前のフレームで割り当てたクエリは使わない
                qi, ti = linear_sum_assignment(c)
                used[qi] = True
                seen[new] = True
                qi_all += list(qi)
                ti_all += [new[k] for k in ti]
            pair = (torch.as_tensor(qi_all, dtype=torch.long), torch.as_tensor(ti_all, dtype=torch.long))
            out += [pair] * num_frames
        return out

    def forward(self, outs: list[dict], targets: list[dict]) -> tuple[torch.Tensor, dict]:
        """全部の層の予測の損失を足す。

        Args:
            outs: 層ごとの {"logits": (B, T, Q, K+1), "masks": (B, Q, T, h, w)}(VidEoMT.forward の返り値)。
            targets: 区間ごとの {"labels": (N,), "masks": (N, T, S, S) の 0/1}(長さ B)。

        Returns:
            (損失の合計(0 次元), 最終の層の内訳 {"class", "mask", "dice"} の float)。
        """
        t = outs[0]["logits"].shape[1]
        # 正解をフレームごとに分け、各フレームを 1 枚の画像とみなしてつなげる(区間の物体は全フレームに並べ、写っていない
        # フレームのマスクは空のまま)。順番は (区間 b, フレーム t) で、予測の (b t) と同じ
        labels = torch.cat([tg["labels"] for tg in targets for _ in range(t)])  # (Σ N × T,)
        tmasks = torch.cat([tg["masks"][:, i] for tg in targets for i in range(t)]).float()  # (Σ N × T, S, S)
        counts = [len(tg["labels"]) for tg in targets]
        frame_counts = [n for n in counts for _ in range(t)]  # フレームごとの正解の数
        owner = torch.repeat_interleave(torch.arange(len(frame_counts), device=labels.device),
                                        torch.as_tensor(frame_counts, device=labels.device))  # (Σ N × T,) どのフレームのものか
        starts = [sum(frame_counts[:i]) for i in range(len(frame_counts))]
        total, parts = 0.0, {}
        for k, out in enumerate(outs):
            frame_out = {"logits": rearrange(out["logits"], "b t q k -> (b t) q k"),
                         "masks": rearrange(out["masks"], "b q t h w -> (b t) q h w")}
            indices = self.consistent_match(frame_out["logits"], frame_out["masks"], labels, tmasks, owner, starts, t, counts)
            l = self._one(frame_out, labels, tmasks, owner, starts, indices)
            total = total + W_CLASS * l["class"] + W_MASK * l["mask"] + W_DICE * l["dice"]
            if k == len(outs) - 1:
                parts = {key: float(v.detach()) for key, v in l.items()}
        return total, parts
