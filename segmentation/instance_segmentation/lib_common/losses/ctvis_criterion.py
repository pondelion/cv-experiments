"""CTVIS の損失(Ying ら ICCV 2023、"CTVIS: Consistent Training for Online Video Instance Segmentation")。exp00502。

参考: 公式のコード(DVIS++ に入っている版。zhang-tao-whu/DVIS_Plus の `DVIS_DAQ/dvis_Plus/ctvis.py` の CTMinVIS・CTCLPlugin・
SimpleTrainMemory・TrainTracklet・loss_reid、`configs/dvis_Plus/VIPSeg/CTVIS_r50.yaml`)。コードは写さず、規則を読み取って書いた。

1. セグメンテーションの損失: 区間の各フレームを 1 枚の画像とみなし、画像の Mask2Former と同じ損失(ハンガリアン法の割り当て +
   クラス・点を選んだマスクの BCE と Dice、全部の層)。正解は区間の全部の物体で、写っていないフレームはマスクが空(クラスはその物体)。
   割り当てはフレームごとに独立(公式の VideoHungarianMatcher をフレームごとの形に直して使うのと同じ)
2. 対照学習の損失(reid): 最後の層の reid の埋め込みで、区間ごとに次を行う
   a. 割り当て: 各フレームで、最後の層の予測と正解をハンガリアン法で組にする(公式の image_matcher)
   b. 学習用の記憶のバンク(物体ごと)にフレームの順に入れる: 写っているフレームは、その物体のクエリの埋め込み(anchor)と、
      ほかのクエリから選んだ負例 num_negatives 個(公式は 99。クエリ 0〜99 から自分を除く)。写っていないフレームは anchor なし、
      負例はそのフレームの全部のクエリ。物体の「類似度で重みを付けて混ぜた埋め込み」(Similarity-Guided Feature Fusion)を
      更新する: 初めて写ったときはその埋め込み、2 回目からは β = max(0, 今までの埋め込みとの cos の平均) で (1 − β)・前 + β・今
   c. 2 フレーム目から、写っている物体ごとに: 正例 = 前に写っていれば、確率 0.5 で 1 つ前のフレームでの混ぜた埋め込み、
      それ以外は前に写った最後の埋め込み(前に写っていなければ、後で最初に写った埋め込み)。負例 = 1 つ前のフレームで入れた負例
   d. 損失(IDOL と同じ): anchor との内積で log(1 + Σ_正例 Σ_負例 exp(負例 − 正例)) と、cos と正解(正例 1・負例 0)の差の 2 乗の平均。
      全部の組で平均し、重み 2(reid)・3(aux)で足す
   埋め込みは記憶のバンクの中でも勾配を切らない(公式と同じ)。

形の記号: B = 区間の数、T = フレームの数、Q = クエリの数、K = クラスの数、N = 区間の正解の物体の数、C = 埋め込みの次元。
"""
import random

import numpy as np
import torch
import torch.nn.functional as F

from segmentation.instance_segmentation.lib_common.losses.mask2former_criterion import Mask2FormerCriterion, match

REID_WEIGHT, AUX_REID_WEIGHT = 2.0, 3.0  # 公式の CL_PLUGIN.REID_WEIGHT・AUX_REID_WEIGHT


class _Tracklet:
    """学習用の記憶のバンクの 1 物体分(公式の TrainTracklet。noise_embed は公式の既定の False)。"""

    def __init__(self):
        self.embeds, self.negatives, self.fused_list = [], [], []  # フレームごとの anchor(なければ None)・負例・混ぜた埋め込み
        self.fused, self.exist_frames = None, 0

    def update(self, embed: torch.Tensor | None, negative: torch.Tensor) -> None:
        """1 フレーム分を入れ、混ぜた埋め込みを更新する。

        Args:
            embed: その物体のクエリの埋め込み (1, C)(写っていないフレームは None)。
            negative: 負例 (負例の数, C)。
        """
        self.embeds.append(embed)
        self.negatives.append(negative)
        if embed is not None:
            if self.exist_frames == 0:
                self.fused = embed
            else:  # Similarity-Guided Feature Fusion: 今までの埋め込みとの cos の和 ÷ 今まで写ったフレームの数
                prev = torch.cat([e for e in self.embeds[:-1] if e is not None])
                sim = (F.normalize(prev, dim=-1) @ F.normalize(embed[0], dim=-1)).sum() / self.exist_frames
                beta = torch.clamp(sim, min=0)  # max(0, 類似度)
                self.fused = (1 - beta) * self.fused + beta * embed
            self.exist_frames += 1
        self.fused_list.append(self.fused)

    def samples(self, t: int, momentum: bool = True):
        """フレーム t の対照学習の組を作る。

        Args:
            t: フレームの番号。
            momentum: 確率 0.5 で、正例に 1 つ前のフレームでの混ぜた埋め込みを使うか(公式の既定 True)。

        Returns:
            (anchor (1, C), 正例 (1, C) か None(前にも後にも写っていない), 負例 (負例の数, C)(1 つ前のフレームで入れたもの))。
        """
        anchor, positive = self.embeds[t], None
        if any(e is not None for e in self.embeds[:t]):  # 前に写っている
            if momentum and np.random.rand() > 0.5:
                positive = self.fused_list[t - 1]
            else:
                positive = next(e for e in self.embeds[:t][::-1] if e is not None)
        elif any(e is not None for e in self.embeds[t + 1:]):
            positive = next(e for e in self.embeds[t + 1:] if e is not None)
        return anchor, positive, self.negatives[t - 1]


def reid_losses(items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], zero: torch.Tensor) -> dict:
    """対照学習の損失(公式の loss_reid と同じ。reduce=False)。

    Args:
        items: [(anchor (1, C), 正例 (P, C), 負例 (M, C))]。
        zero: 組がないときに返す 0(グラフにつながったもの)。

    Returns:
        {"reid": 内積の損失の平均, "aux_reid": cos の 2 乗の差の平均}。
    """
    if not items:
        return {"reid": zero, "aux_reid": zero}
    reid = aux = 0.0
    for anchor, pos, neg in items:
        dp, dn = (pos @ anchor[0]), (neg @ anchor[0])  # (P,), (M,) 内積
        x = (dn[None, :] - dp[:, None]).flatten()  # 全部の (正例, 負例) の組の 負例 − 正例
        reid = reid + torch.logsumexp(torch.cat([x, x.new_zeros(1)]), dim=0)  # log(1 + Σ exp(負例 − 正例))
        cos = F.normalize(torch.cat([pos, neg]), dim=-1) @ F.normalize(anchor[0], dim=-1)  # (P + M,)
        label = torch.cat([cos.new_ones(len(pos)), cos.new_zeros(len(neg))])
        aux = aux + ((cos - label) ** 2).mean()
    return {"reid": reid / len(items), "aux_reid": aux / len(items)}


class CTVISCriterion(Mask2FormerCriterion):
    """CTVIS の損失(上の 1 + 2)。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        num_frames: 区間のフレームの数 T。
        num_negatives: 負例の数(公式 99。クエリが 100 個なら、自分以外の全部のクエリ)。
        momentum_embed: 正例に、確率 0.5 で混ぜた埋め込みを使うか(公式の既定 True)。
    """

    def __init__(self, num_classes: int, num_frames: int, num_negatives: int = 99, momentum_embed: bool = True):
        super().__init__(num_classes)
        self.num_frames, self.num_negatives, self.momentum_embed = num_frames, num_negatives, momentum_embed

    def forward(self, outs: list[dict], targets: list[dict], match_outs=None) -> tuple[torch.Tensor, dict]:
        """区間ごとの損失(画像の損失 + reid の対照学習)。

        Args:
            outs: Mask2Former の層ごとの予測(画像は (b t) の順に B·T 枚)。最後の要素に "reid" (B·T, Q, C)。
            targets: 区間ごとの {"labels": (N,), "masks": (N, T, S, S)}(長さ B)。
            match_outs: 使わない(ほかの動画の損失と呼び方をそろえるため)。

        Returns:
            (損失の合計, 内訳 {"class", "mask", "dice", "reid", "aux_reid"} の float)。
        """
        t = self.num_frames
        # 1. フレームを画像とみなした正解(区間の全部の物体。写っていないフレームのマスクは空)
        frame_targets = [{"labels": tg["labels"], "masks": tg["masks"][:, i]} for tg in targets for i in range(t)]
        seg_loss, parts = super().forward(outs, frame_targets)
        # 2a. 最後の層の予測で、フレームごとに割り当てる(公式の image_matcher。損失の割り当てとは別に計算する)
        last = outs[-1]
        labels = torch.cat([tg["labels"] for tg in frame_targets])
        tmasks = torch.cat([tg["masks"] for tg in frame_targets]).float()
        owner = torch.repeat_interleave(torch.arange(len(frame_targets), device=labels.device),
                                        torch.as_tensor([len(tg["labels"]) for tg in frame_targets], device=labels.device))
        indices = match(last["logits"].detach(), last["masks"].detach(), labels, tmasks, owner)
        reid = last["reid"].float()  # (B·T, Q, C)
        assert reid.shape[1] > self.num_negatives, "負例はクエリ 0〜num_negatives から選ぶ(公式と同じ)ので、クエリはそれより多く要る"
        items = []
        for b, tg in enumerate(targets):
            n = len(tg["labels"])
            present = tg["masks"].flatten(2).any(-1)  # (N, T) そのフレームに写っているか
            bank = [_Tracklet() for _ in range(n)]
            for i in range(t):  # 2b. 記憶のバンクに入れる
                f = b * t + i
                qi, ti = indices[f]
                gt2query = {int(g): int(qq) for qq, g in zip(qi, ti)}
                for j in range(n):
                    if present[j, i]:
                        a = gt2query[j]
                        neg = sorted(random.sample(sorted(set(range(self.num_negatives + 1)) - {a}), self.num_negatives))
                        bank[j].update(reid[f, a][None], reid[f, neg])
                    else:
                        bank[j].update(None, reid[f])
            for i in range(1, t):  # 2c. 2 フレーム目から、写っている物体ごとに組を作る
                for j in range(n):
                    if not present[j, i]:
                        continue
                    anchor, pos, neg = bank[j].samples(i, self.momentum_embed)
                    if pos is not None:
                        items.append((anchor, pos, neg))
        rl = reid_losses(items, reid.sum() * 0)
        total = seg_loss + REID_WEIGHT * rl["reid"] + AUX_REID_WEIGHT * rl["aux_reid"]
        return total, {**parts, "reid": float(rl["reid"].detach()), "aux_reid": float(rl["aux_reid"].detach())}
