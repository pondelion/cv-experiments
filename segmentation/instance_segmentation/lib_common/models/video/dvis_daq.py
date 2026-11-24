"""DVIS-DAQ の online 版(Zhou ら ECCV 2024、"DVIS-DAQ: Improving Video Segmentation via Dynamic Anchor Queries")。exp00502。

参考: 公式のコード(zhang-tao-whu/DVIS_Plus の `DVIS_DAQ/dvis_daq/track_module.py` の VideoInstanceCutter・VideoInstanceSequence、
`slot_attention.py`、`matcher.py` の FrameMatcher・NewInsHungarianMatcher、`criterion.py` の DAQCriterion、
`meta_architecture.py` の DVIS_DAQ_online、`configs/dvis_daq/vipseg/DAQ_Online_R50.yaml`)。コードは写さず、規則を読み取って書いた。

MOT(動画全体での物体の ID の管理・識別)の形: **推論の規則でクエリを足し引きする形(dynamic anchor queries)**。
    - 追跡のクエリ(前のフレームから続く物体。1 本 = 1 つの ID)+ 新しい物体を探すクエリ(100 個。学習する 1 つの埋め込みを並べたもの)を
      毎フレーム同じ層(cross-attention → self-attention → FFN、6 層)で処理する。キー・値は今のフレームの segmenter のクエリ
    - 新しい ID: 新しい物体を探すクエリのうち、点数が閾値(0.01)を超えたものを、新しい ID の追跡のクエリとして次のフレームに足す
    - 消えた物体: 追跡のクエリの点数が閾値以下のフレームが続いても、8 フレームまでは同じ ID のまま持ち続ける(隠れた後に再び出れば
      同じ ID に戻る)。8 フレーム続いたら、その ID を終える(それより長く隠れて再び出ると、新しい物体を探すクエリが新しい ID を付ける)
    - 区別できる物体の数に決まった上限はない(クエリを足し引きするため)
    - 学習: 正解との割り当ては、前のフレームで追跡していた物体はそのクエリのまま、新しく写った物体だけを新しい物体を探すクエリに
      ハンガリアン法で割り当てる。消えた物体のクエリの正解は「物体なし」(点数を下げて、追跡を終えられるように)
    - slot: 追跡のクエリ + 背景の slot 5 個を、segmenter のクエリの特徴から slot attention で作り直す枝(消えた物体の扱いを学ぶ補助の損失)。
      学習の段階 3 では、追跡している物体を 1 つ無作為に選び、その物体の segmenter のクエリを slot から見えなくして、「消えた」ときの
      正解(物体なし)で学習させる
    - 位置の埋め込み: クエリのマスク(sigmoid > 0.5)の中のマスクを作る特徴の平均を 3 層の MLP に通したもの。追跡のクエリは、過去
      (最大 10 フレーム)の値を類似度で重みを付けて混ぜたもの(Similarity-Guided Feature Fusion)

学習の手順(公式の VIPSeg の設定): segmenter(CTVIS)を固定し、追跡の部だけを学習する。学習の最初の 10,000 / 90,000 step は、区間の 5 枚の
うち連続した 2 枚だけを使い(USING_FRAME_NUM [2, 5]、STEPS [10000])、段階 2(割り当てたクエリのうち点数の低い半分を追跡から外す)。
その後は 5 枚で段階 3(点数 0.01 を超えたクエリを追跡に残す)。

公式との違い(私の判断。理由つき):
    - 公式の段階 3 の「消えた物体をまねる」処理は、slot の正解から外す物体の list を、追跡の部の正解から外す物体の list と同じ物で
      共有していて、slot の側に足した物体が、追跡の部の側(その物体は見えている)の正解からも外れる。論文ではこの処理は slot の
      ためのものなので、slot の側だけに足す
    - 区間はまとめの中で 1 本ずつ処理する(公式は GPU 1 枚あたり 1 区間で、assert B == 1。同じ意味)

形の記号: Q = segmenter のクエリの数、P = 新しい物体を探すクエリの数(100)、R = 追跡のクエリの数(フレームで変わる)、
C = 次元(256)、K = クラスの数、N = 区間の正解の物体の数、h・w = マスクの縦・横(入力の 1/4)。
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from segmentation.instance_segmentation.lib_common.losses.mask2former_criterion import (EOS_COEF, W_CLASS, W_DICE, W_MASK,
                                                                                       dice_loss, match_cost, point_sample,
                                                                                       uncertain_points)
from segmentation.instance_segmentation.lib_common.models.mask2former import D_MODEL, Mask2Former


def _xavier(module: nn.Module) -> None:
    """2 次元以上の重みを xavier_uniform で初期化する(公式の各層の _reset_parameters と同じ)。

    Args:
        module: 初期化する層。
    """
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    """3 層の MLP(層の間は ReLU。公式の MLP と同じ)。

    Args:
        d_in: 入力の次元。
        d_hidden: 中間の次元。
        d_out: 出力の次元。

    Returns:
        nn.Sequential。
    """
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_hidden), nn.ReLU(inplace=True),
                         nn.Linear(d_hidden, d_out))


class TrackLayer(nn.Module):
    """追跡の部の 1 層: cross-attention(位置の埋め込みあり)→ self-attention → FFN(どれも残差 + LayerNorm の後置き)。

    Args:
        d: 次元。
        nhead: 注意の頭の数。
        ffn: FFN の中間の次元。
    """

    def __init__(self, d: int = D_MODEL, nhead: int = 8, ffn: int = 2048):
        super().__init__()
        self.cross = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, d))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        _xavier(self.cross); _xavier(self.self_attn); _xavier(self.ffn)

    def forward(self, x, memory, query_pos=None, pos=None):
        """1 層進める。

        Args:
            x: 追跡のクエリ (1, R, C)。
            memory: 今のフレームの segmenter のクエリ (1, Q, C)。
            query_pos: 追跡のクエリの位置の埋め込み (1, R, C) か None。
            pos: memory の位置の埋め込み (1, Q, C) か None。

        Returns:
            (1, R, C)。
        """
        q = x if query_pos is None else x + query_pos
        k = memory if pos is None else memory + pos
        x = self.norms[0](x + self.cross(q, k, memory, need_weights=False)[0])
        x = self.norms[1](x + self.self_attn(x, x, x, need_weights=False)[0])
        return self.norms[2](x + self.ffn(x))


class SlotLayer(nn.Module):
    """slot の枝の 1 層(公式の SlotCrossAttentionLayer + FFNLayer)。

    cross-attention の出力 a を入力、x + a をキーにした slot attention(slot どうしで入力を取り合う: softmax を slot の軸でとり、
    1e-6 を足して入力の軸で正規化した重みで入力を平均する)の結果を x に足して LayerNorm → FFN。

    Args:
        d: 次元。
        nhead: 注意の頭の数。
        ffn: FFN の中間の次元。
    """

    def __init__(self, d: int = D_MODEL, nhead: int = 8, ffn: int = 2048):
        super().__init__()
        self.nhead = nhead
        self.cross = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.norm_inputs = nn.LayerNorm(d)  # SlotAttention.norm_inputs
        self.project_q = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d, bias=False))
        self.project_k = nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, d))
        self.ffn_norm = nn.LayerNorm(d)
        _xavier(self.cross); _xavier(self.norm_inputs); _xavier(self.project_q); _xavier(self.project_k); _xavier(self.ffn)
        self.scale = d ** -0.5

    def forward(self, x, memory, query_pos, slot_query, memory_mask=None):
        """1 層進める。

        Args:
            x: slot の特徴 (1, S, C)(S = slot の数)。
            memory: 今のフレームの segmenter のクエリ (1, Q, C)。
            query_pos: slot の位置の埋め込み (1, S, C)。
            slot_query: slot attention の問い (1, S, C)。
            memory_mask: (S, Q) の bool(True = 見ない)か None。

        Returns:
            (1, S, C)。
        """
        mask = None if memory_mask is None else memory_mask.repeat(self.nhead, 1, 1)  # (頭の数, S, Q)
        a = self.cross(x + query_pos, memory, memory, attn_mask=mask, need_weights=False)[0]  # (1, S, C)
        k = self.project_k(self.norm_inputs(x + a))  # (1, S, C)
        q = self.project_q(slot_query)  # (1, S, C)
        attn = torch.einsum("bnc,bmc->bnm", k, q) * self.scale  # (1, 入力 S, slot S)
        attn = attn.softmax(-1) + 1e-6
        attn = attn / attn.sum(dim=1, keepdim=True)
        x = self.norm(x + torch.einsum("bnm,bnc->bmc", attn, a))
        return self.ffn_norm(x + self.ffn(x))


class InstanceSequence:
    """1 つの ID の記録(公式の VideoInstanceSequence のうち使う部分)。

    Args:
        start: ID が始まったフレームの番号。
        cache: 位置の埋め込みを混ぜるのに使う過去のフレームの数(公式は 10)。
    """

    def __init__(self, start: int, cache: int = 10):
        self.start, self.cache = start, cache
        self.pos, self.fused_pos = [], None  # 位置の埋め込みと、類似度で重みを付けて混ぜたもの
        self.embeds, self.logits, self.masks = [], [], []  # 推論の記録(フレームごと)
        self.invalid, self.dead = 0, False

    def update_pos(self, pos: torch.Tensor) -> None:
        """位置の埋め込みを足して、混ぜたものを更新する(過去の最大 cache フレームとの cos の平均を重みにする)。

        Args:
            pos: 今のフレームの位置の埋め込み (C,)。
        """
        self.pos.append(pos)
        if len(self.pos) == 1:
            self.fused_pos = pos
            return
        prev = torch.stack(self.pos[max(0, len(self.pos) - self.cache): -1])
        sim = (F.normalize(prev, dim=-1) @ F.normalize(pos, dim=-1)).sum() / len(prev)
        beta = torch.clamp(sim, min=0)
        self.fused_pos = (1 - beta) * self.fused_pos + beta * pos


class DAQTracker(nn.Module):
    """DVIS-DAQ の追跡の部(公式の VideoInstanceCutter)。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        layers: 層の数(公式 6)。
        num_new: 新しい物体を探すクエリの数(公式の VIPSeg 100)。
        num_slots: 背景の slot の数(公式 5)。
        train_thr: 学習の段階 3 で追跡に残す点数(公式の VIPSeg 0.01)。
        infer_thr: 推論で追跡に残す・新しい ID にする点数(公式の VIPSeg 0.01)。
        kick_out: 推論で、点数が閾値以下のフレームがこれだけ続いたら ID を終える(公式 8)。
    """

    def __init__(self, num_classes: int, layers: int = 6, num_new: int = 100, num_slots: int = 5, train_thr: float = 0.01,
                 infer_thr: float = 0.01, kick_out: int = 8):
        super().__init__()
        d = D_MODEL
        self.layers = nn.ModuleList([TrackLayer(d) for _ in range(layers)])
        self.slot_layers = nn.ModuleList([SlotLayer(d) for _ in range(layers)])
        self.decoder_norm = nn.LayerNorm(d)
        self.class_embed = nn.Linear(d, num_classes + 1)
        self.pos_embed = _mlp(d, d, d)
        self.mask_embed = _mlp(d, d, d)
        self.mask_feature_proj = nn.Conv2d(d, d, 1)
        self.new_ins_embeds = nn.Embedding(1, d)
        self.bg_slots = nn.Embedding(num_slots, d)
        self.num_new, self.train_thr, self.infer_thr, self.kick_out = num_new, train_thr, infer_thr, kick_out

    # ---- 共通の部品 ----
    def predict(self, x: torch.Tensor, mask_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """層ごとの出力からクラスとマスクを出す。

        Args:
            x: 層ごとの出力 (L, R, C)。
            mask_feat: マスクを作る特徴 (C, h, w)(射影の後)。

        Returns:
            (クラスの値 (L, R, K+1), マスクの値 (L, R, h, w))。
        """
        x = self.decoder_norm(x)
        return self.class_embed(x), torch.einsum("lrc,chw->lrhw", self.mask_embed(x), mask_feat)

    def mask_pos(self, mask_logits: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
        """位置の埋め込み: マスク(sigmoid > 0.5)の中の特徴の平均を MLP に通す。

        Args:
            mask_logits: マスクの値 (R, h, w)。
            mask_feat: マスクを作る特徴 (C, h, w)(射影の前)。

        Returns:
            (R, C)。
        """
        m = (mask_logits.sigmoid() > 0.5).to(mask_feat.dtype)
        pooled = torch.einsum("rhw,chw->rc", m, mask_feat) / (m.flatten(1).sum(-1, keepdim=True) + 1e-8)
        return self.pos_embed(pooled)

    def _track_layers(self, x, memory, query_pos=None, pos=None) -> torch.Tensor:
        """追跡の部の層を順にかける。

        Args:
            x: 追跡のクエリ (R, C)。
            memory: 今のフレームの segmenter のクエリ (Q, C)。
            query_pos: 追跡のクエリの位置の埋め込み (R, C) か None。
            pos: memory の位置の埋め込み (Q, C) か None。

        Returns:
            入力 + 各層の出力 (L + 1, R, C)。
        """
        outs = [x]
        y = x[None]
        for layer in self.layers:
            y = layer(y, memory[None], None if query_pos is None else query_pos[None], None if pos is None else pos[None])
            outs.append(y[0])
        return torch.stack(outs)

    def _slots(self, track_q, track_pos, memory, seg_query_feat, memory_mask=None) -> torch.Tensor:
        """slot の枝(追跡のクエリ + 背景の slot が、今のフレームのクエリを取り合う)。

        Args:
            track_q: 追跡のクエリ (R, C)。
            track_pos: 追跡のクエリの位置の埋め込み (R, C)。
            memory: 今のフレームの segmenter のクエリ (Q, C)。
            seg_query_feat: segmenter の学習するクエリの特徴 (Q, C)(slot の初期の特徴に使う)。
            memory_mask: (slot の数, Q) の bool(True = 見ない)か None。

        Returns:
            各層の出力 (L, R + 背景の slot の数, C)。
        """
        bg = self.bg_slots.weight
        pos = torch.cat([track_q, bg])  # 公式: slots_pos = [追跡のクエリ, 背景]
        slot_query = torch.cat([track_pos, bg])  # 公式: slots_query = [追跡の位置, 背景]
        # 初期の slot の特徴: 各 slot に、近い segmenter の学習するクエリの特徴を 1 つ(割り当てられたものはハンガリアン法の結果)
        cost = 1 - F.normalize(pos.detach(), dim=-1, eps=1e-6) @ F.normalize(seg_query_feat.detach(), dim=-1, eps=1e-6).T
        idx = cost.argmin(1)
        r, c = linear_sum_assignment(cost.float().cpu().numpy())
        idx[torch.as_tensor(r, device=idx.device)] = torch.as_tensor(c, device=idx.device)
        x = seg_query_feat[idx][None]
        outs = []
        for layer in self.slot_layers:
            x = layer(x, memory[None], pos[None], slot_query[None], memory_mask)
            outs.append(x[0])
        return torch.stack(outs)

    # ---- 学習 ----
    def train_clip(self, seg: dict, frame_targets: list[dict], stage: int, seg_query_feat: torch.Tensor,
                   aux_thr: float = 0.01) -> tuple[list[dict], list[dict]]:
        """1 本の区間を時刻の順に処理する(公式の VideoInstanceCutter.forward)。

        Args:
            seg: segmenter の出力(フレームごとに並べたもの){"embeds": (T, Q, C), "logits": (T, Q, K+1), "masks": (T, Q, h, w),
                "mask_feat": (T, C, h, w)}(どれも固定した segmenter の値)。
            frame_targets: フレームごとの {"labels": (N,), "masks": (N, H, W), "valid": (N,) の bool}(区間の全部の物体)。
            stage: 学習の段階(2: 割り当てたクエリの点数の低い半分を追跡から外す / 3: 点数 train_thr を超えたクエリを残す)。
            seg_query_feat: segmenter の学習するクエリの特徴 (Q, C)(slot の初期値)。
            aux_thr: segmenter の予測で「物体らしい」とみなす点数(公式の AUX_INFERENCE_SELECT_THRESHOLD)。

        Returns:
            (フレームごとの {"logits": [(R, K+1)](入力 + 各層), "masks": [(R, h, w)], "indices": (クエリ, 物体), "drop": 正解から外す物体},
             2 フレーム目からの slot の同じ形の出力)。
        """
        t_len = seg["embeds"].shape[0]
        mf = self.mask_feature_proj(seg["mask_feat"])  # (T, C, h, w)
        frame_idx, aux_idx = zip(*[frame_match(seg["logits"][i], seg["masks"][i], frame_targets[i], aux_thr)
                                   for i in range(t_len)])
        assert self.num_new == seg["embeds"].shape[1], "公式と同じく、新しい物体を探すクエリの数 = segmenter のクエリの数"
        outputs, slot_outputs = [], []
        prev_idx = track_q = track_pos = seq_ids = None
        hub, drop_next = {}, []
        for i in range(t_len):
            memory = seg["embeds"][i]
            if i == 0:
                ms = self._track_layers(memory, memory)  # (L + 1, Q, C)
                slots = None
            else:
                frame_pos = self.mask_pos(seg["masks"][i], seg["mask_feat"][i])  # (Q, C)
                new = self.new_ins_embeds.weight.expand(self.num_new, -1)
                ms = self._track_layers(torch.cat([track_q, new]), memory, torch.cat([track_pos, frame_pos]), frame_pos)
                # 段階 3: 追跡している物体を 1 つ選び、その物体の segmenter のクエリを slot から見えなくする(消えたときをまねる)
                hidden, hide_id = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device), None
                if stage == 3 and len(prev_idx[0]) > 3:
                    k = random.randrange(len(prev_idx[0]))
                    hide_id = int(prev_idx[1][k])
                    hidden = aux_idx[i][1] == hide_id
                n_slots = track_q.shape[0] + self.bg_slots.num_embeddings
                mask = hidden[None].expand(n_slots, -1) if hidden.any() else None
                slots = self._slots(track_q, track_pos, memory, seg_query_feat, mask)  # (L, R + 背景, C)
            lg, mk = self.predict(ms, mf[i])  # (L + 1, R, K+1), (L + 1, R, h, w)
            if i == 0:
                idx = frame_idx[0]
            else:
                idx = new_ins_match(lg[-1], mk[-1], frame_targets[i], prev_idx, self.num_new)
            outputs.append({"logits": list(lg), "masks": list(mk), "indices": idx, "drop": list(drop_next)})
            if slots is not None:
                slg, smk = self.predict(slots, mf[i])
                slot_drop = list(drop_next) + ([hide_id] if hide_id is not None else [])  # slot の側だけに足す(上の注)
                slot_outputs.append({"logits": list(slg), "masks": list(smk), "indices": prev_idx, "drop": slot_drop})
            # 次のフレームへ残す追跡のクエリ
            n_q = ms.shape[1]
            tgt_of = torch.full((n_q,), -1, dtype=torch.long, device=ms.device)
            tgt_of[idx[0]] = idx[1]
            if stage == 2:
                scores = lg[-1].softmax(-1)[:, :-1].max(-1).values[idx[0]]
                kick = idx[0][scores.argsort()[: len(scores) // 2]]
                keep = torch.zeros(n_q, dtype=torch.bool, device=ms.device)
                keep[idx[0]] = True
                keep[kick] = False
            else:
                keep = lg[-1].softmax(-1)[:, :-1].max(-1).values > self.train_thr
            track_q = ms[-1][keep]
            kept_tgt = tgt_of[keep]
            src = torch.nonzero(kept_tgt + 1).squeeze(-1)
            prev_idx = (src, kept_tgt[src])
            pos_now = self.mask_pos(mk[-1], seg["mask_feat"][i])  # (R, C)
            new_ids = []
            for k_, kept in enumerate(keep.tolist()):
                sid = seq_ids[k_] if seq_ids is not None and k_ < len(seq_ids) else object()  # 新しい ID(重ならない印)
                if kept:
                    hub.setdefault(sid, InstanceSequence(0)).update_pos(pos_now[k_])
                    new_ids.append(sid)
            seq_ids = new_ids
            track_pos = (torch.stack([hub[s].fused_pos for s in seq_ids]) if seq_ids
                         else ms.new_zeros(0, ms.shape[-1]))
            # 次のフレームで消える物体(追跡している物体のうち、次のフレームに写っていないもの)は、次のフレームの正解から外す
            nxt = frame_idx[i + 1 if i < t_len - 1 else -1][1].tolist()
            drop_next = [int(g) for g in prev_idx[1].tolist() if int(g) not in nxt]
        return outputs, slot_outputs

    # ---- 推論 ----
    @torch.no_grad()
    def infer_step(self, seg: dict, state: dict, t: int, seg_query_feat: torch.Tensor, aux_thr: float = 0.01) -> dict:
        """推論の 1 フレーム(公式の VideoInstanceCutter.inference の 1 フレーム分。OVIS_INFER は公式の VIPSeg と同じく False)。

        Args:
            seg: このフレームの segmenter の出力 {"embeds": (Q, C), "logits": (Q, K+1), "masks": (Q, h, w), "mask_feat": (C, h, w)}。
            state: {"seqs": {ID: InstanceSequence}, "ids": 追跡している ID の並び, "track_q": (R, C), "track_pos": (R, C)}(最初は空の dict)。
            t: フレームの番号。
            seg_query_feat: segmenter の学習するクエリの特徴 (Q, C)。
            aux_thr: 最初のフレームで segmenter の予測を物体とみなす点数。

        Returns:
            更新した state。
        """
        memory, mf = seg["embeds"], self.mask_feature_proj(seg["mask_feat"][None])[0]
        first = "ids" not in state
        if first:
            ms = self._track_layers(memory, memory)
        else:
            frame_pos = self.mask_pos(seg["masks"], seg["mask_feat"])
            new = self.new_ins_embeds.weight.expand(self.num_new, -1)
            ms = self._track_layers(torch.cat([state["track_q"], new]), memory, torch.cat([state["track_pos"], frame_pos]),
                                    frame_pos)
        lg, mk = self.predict(ms[-1:], mf)
        lg, mk = lg[0], mk[0]  # (R, K+1), (R, h, w)
        pos_now = self.mask_pos(mk, seg["mask_feat"])
        if first:
            valid = seg["logits"].softmax(-1)[:, :-1].max(-1).values > aux_thr
            state["seqs"], prev_ids = {}, []
        else:
            valid = lg.softmax(-1)[:, :-1].max(-1).values > self.infer_thr
            prev_ids = state["ids"]
        ids = []
        for k, v in enumerate(valid.tolist()):
            sid = prev_ids[k] if k < len(prev_ids) else len(state["seqs"])  # 新しい ID は通し番号
            if v:
                s = state["seqs"].setdefault(sid, InstanceSequence(t))
                s.invalid = 0
                s.update_pos(pos_now[k])
            elif k < len(prev_ids):  # 追跡していた ID の点数が閾値以下: kick_out フレーム続くまでは持ち続ける
                s = state["seqs"][sid]
                s.invalid += 1
                if s.invalid >= self.kick_out:
                    s.dead = True
                    continue
            else:
                continue
            s.embeds.append(ms[-1, k]); s.logits.append(lg[k].float()); s.masks.append(mk[k].float().cpu())
            ids.append(sid)
        state["ids"] = ids
        state["track_q"] = torch.stack([state["seqs"][s].embeds[-1] for s in ids]) if ids else memory.new_zeros(0, memory.shape[1])
        state["track_pos"] = (torch.stack([state["seqs"][s].fused_pos for s in ids]) if ids
                              else memory.new_zeros(0, memory.shape[1]))
        return state


# ---- 割り当て(公式の FrameMatcher・NewInsHungarianMatcher) ----
@torch.no_grad()
def frame_match(logits: torch.Tensor, masks: torch.Tensor, target: dict, aux_thr: float):
    """segmenter の予測と、そのフレームに写っている物体の割り当て(公式の FrameMatcher)。

    Args:
        logits: クラスの値 (Q, K+1)。
        masks: マスクの値 (Q, h, w)。
        target: そのフレームの正解 {"labels": (N,), "masks": (N, H, W), "valid": (N,)(そのフレームに写っているか)}。
        aux_thr: 物体らしい点数の閾値(公式は 0.01)。

    Returns:
        ((クエリ, 物体)(写っている物体だけ), (全部のクエリ, クエリごとの物体の番号(物体らしくないクエリは −1)))。
    """
    q, n, dev = logits.shape[0], len(target["labels"]), logits.device
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=dev)
        return (empty, empty), (torch.arange(q, device=dev), torch.full((q,), -1, dtype=torch.long, device=dev))
    cost = match_cost(logits[None], masks[None], target["labels"], target["masks"].float(),
                      torch.zeros(n, dtype=torch.long, device=dev))[0]  # (Q, N)
    valid = target["valid"].cpu().numpy()
    cost[:, ~valid] = 1e6
    src, tgt = linear_sum_assignment(cost)
    order = tgt.argsort()
    src, tgt = src[order][valid], tgt[order][valid]
    score = logits.softmax(-1)[:, :-1].max(-1).values
    keep = torch.zeros(q, dtype=torch.bool, device=dev)
    keep[torch.as_tensor(src, device=dev)] = True
    keep |= score > aux_thr
    aux_tgt = torch.as_tensor(cost.argmin(1), device=dev)
    aux_tgt[torch.as_tensor(src, device=dev)] = torch.as_tensor(tgt, device=dev)
    aux_tgt[~keep] = -1
    return ((torch.as_tensor(src, device=dev), torch.as_tensor(tgt, device=dev)),
            (torch.arange(q, device=dev), aux_tgt))


@torch.no_grad()
def new_ins_match(logits: torch.Tensor, masks: torch.Tensor, target: dict, prev_idx, num_new: int):
    """2 フレーム目からの割り当て: 追跡していた物体は前のフレームのクエリのまま、新しく写った物体だけを新しい物体を探すクエリに
    ハンガリアン法で割り当てる(公式の NewInsHungarianMatcher)。

    Args:
        logits: クラスの値 (R + P, K+1)。
        masks: マスクの値 (R + P, h, w)。
        target: frame_match と同じ。
        prev_idx: 前のフレームの (追跡のクエリ, 物体)。
        num_new: 新しい物体を探すクエリの数 P(並びの最後の P 個)。

    Returns:
        (クエリ, 物体)。
    """
    dev = logits.device
    new = target["valid"].clone()
    new[prev_idx[1]] = False
    old_src, old_tgt = prev_idx[0].cpu().numpy(), prev_idx[1].cpu().numpy()
    if not new.any():
        return prev_idx
    n = len(target["labels"])
    cost = match_cost(logits[None], masks[None], target["labels"], target["masks"].float(),
                      torch.zeros(n, dtype=torch.long, device=dev))[0]  # (R + P, N)
    new_np = new.cpu().numpy()
    cost[:, ~new_np] = 1e6
    cost[:-num_new, :] = 1e6
    src, tgt = linear_sum_assignment(cost)
    order = tgt.argsort()
    src, tgt = src[order], tgt[order]
    if len(src) == len(new_np):
        new_tgt, new_src = tgt[new_np], src[new_np]
    else:
        is_new = new_np[tgt]
        new_tgt, new_src = tgt[is_new], src[is_new]
    assert np.all(new_src >= logits.shape[0] - num_new)
    return (torch.as_tensor(np.concatenate([old_src, new_src]), dtype=torch.long, device=dev),
            torch.as_tensor(np.concatenate([old_tgt, new_tgt]), dtype=torch.long, device=dev))


# ---- 損失(公式の DAQCriterion) ----
def daq_loss(outputs: list[dict], frame_targets: list[dict], class_weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """区間のフレームごとの出力の損失(入力 + 各層。割り当ては最後の層で決めたものを全部の層で使う)。

    クラス: 全部のフレームのクエリをつなげて 1 回の交差エントロピー(割り当てたクエリのうち、そのフレームに写っていて正解から外さない
    物体はそのクラス、ほかは物体なし)。マスク: 同じ組の点を選んだ BCE と Dice を、全部のフレームの割り当ての数で割る。

    Args:
        outputs: train_clip の出力(またはその slot の出力)。
        frame_targets: 同じフレームの正解。
        class_weight: クラスの損失の重み (K+1,)(物体なしは 0.1)。

    Returns:
        (損失の合計, 最後の層の内訳 {"class", "mask", "dice"})。
    """
    n_masks = max(sum(len(o["indices"][1]) for o in outputs), 1)
    total, parts = 0.0, {}
    n_layers = len(outputs[0]["logits"])
    for layer in range(n_layers):
        logits, tgt_cls, src_m, tgt_m = [], [], [], []
        for o, tg in zip(outputs, frame_targets):
            lg, mk = o["logits"][layer], o["masks"][layer]
            src, tgt = o["indices"]
            ok = tg["valid"][tgt].clone()
            for d in o["drop"]:
                ok[tgt == d] = False
            src, tgt = src[ok], tgt[ok]
            if lg.shape[0] == 0:
                continue
            c = torch.full((lg.shape[0],), class_weight.numel() - 1, dtype=torch.long, device=lg.device)
            c[src] = tg["labels"][tgt]
            logits.append(lg); tgt_cls.append(c)
            src_m.append(mk[src]); tgt_m.append(tg["masks"][tgt].float())
        loss_class = (F.cross_entropy(torch.cat(logits).float(), torch.cat(tgt_cls), weight=class_weight)
                      if logits else outputs[0]["logits"][layer].sum() * 0)
        src_m, tgt_m = torch.cat(src_m), torch.cat(tgt_m)
        if len(src_m):
            with torch.no_grad():
                coords = uncertain_points(src_m.float())
                tp = point_sample(tgt_m, coords)
            sp = point_sample(src_m, coords)
            loss_mask = F.binary_cross_entropy_with_logits(sp, tp, reduction="none").mean(-1).sum() / n_masks
            loss_dice = dice_loss(sp, tp).sum() / n_masks
        else:
            loss_mask = loss_dice = outputs[0]["masks"][layer].sum() * 0
        total = total + W_CLASS * loss_class + W_MASK * loss_mask + W_DICE * loss_dice
        if layer == n_layers - 1:
            parts = {"class": float(loss_class.detach()), "mask": float(loss_mask.detach()), "dice": float(loss_dice.detach())}
    return total, parts


class DVISDAQOnline(nn.Module):
    """DVIS-DAQ の online 版(固定した segmenter + DAQTracker)。

    Args:
        segmenter: 学習済みの Mask2Former(公式は DVIS++ の segmenter = CTVIS)。
        num_classes: クラスの数。
        first_stage_frac: 学習の最初のこの割合の step は 2 枚・段階 2(公式の VIPSeg は 10,000 / 90,000)。
        noise_frame_num: 推論で、これより短く、動画の最後まで続かない ID を除く(公式の VIPSeg 5)。
        **tracker_kw: DAQTracker の引数。
    """

    def __init__(self, segmenter: Mask2Former, num_classes: int, first_stage_frac: float = 10000 / 90000,
                 noise_frame_num: int = 5, **tracker_kw):
        super().__init__()
        # slot の初期値に segmenter の学習するクエリの特徴(Mask2Former の query_feat)を使うので、Mask2Former 系に限る
        # (GTR 系は two-stage でフレームごとにクエリを選び、学習するクエリの特徴を持たない)
        assert isinstance(segmenter, Mask2Former), "DVIS-DAQ の segmenter は Mask2Former 系(MinVIS・CTVIS の学習の重み)"
        self.segmenter = segmenter
        for p in self.segmenter.parameters():
            p.requires_grad_(False)
        self.tracker = DAQTracker(num_classes, **tracker_kw)
        self.first_stage_frac, self.noise_frame_num = first_stage_frac, noise_frame_num
        self.register_buffer("class_weight", torch.cat([torch.ones(num_classes), torch.tensor([EOS_COEF])]))

    def train(self, mode: bool = True):
        """学習の状態を切り替える(segmenter は公式と同じく eval のまま)。

        Args:
            mode: True で学習の状態。

        Returns:
            self。
        """
        super().train(mode)
        self.segmenter.eval()  # 公式と同じく segmenter は eval のまま
        return self

    @torch.no_grad()
    def _segment(self, frames: torch.Tensor) -> dict:
        """固定した segmenter をフレームにかける。

        Args:
            frames: (M, 3, S, S)。

        Returns:
            最後の層の {"embeds", "logits", "masks", "mask_feat"}(勾配なし)。
        """
        out = self.segmenter(frames)[-1]
        return {k: out[k].detach() for k in ("embeds", "logits", "masks", "mask_feat")}

    def train_loss(self, clips: torch.Tensor, targets: list[dict], progress: float) -> tuple[torch.Tensor, dict]:
        """学習の損失(区間ごとに追跡の部を通し、追跡の出力と slot の出力の損失を足す)。

        Args:
            clips: (B, T, 3, S, S)。
            targets: 区間ごとの {"labels": (N,), "masks": (N, T, S, S)}。
            progress: 学習の進み(0〜1。最初の first_stage_frac は段階 2 の 2 枚で学習する)。

        Returns:
            (損失の合計(区間で平均), 内訳)。
        """
        first = progress < self.first_stage_frac
        stage, use_t = (2, 2) if first else (3, clips.shape[1])
        total, parts = 0.0, {}
        q_feat = self.segmenter.query_feat.weight.detach()
        for b in range(clips.shape[0]):
            t_all = clips.shape[1]
            s = 0 if use_t == t_all else random.randint(0, t_all - use_t)  # 区間の中の連続した use_t 枚(公式の select_fi_set)
            seg = self._segment(clips[b, s: s + use_t])
            masks = targets[b]["masks"][:, s: s + use_t]  # (N, t, S, S)
            present = masks.flatten(2).any(-1)  # (N, t)
            keep = present.any(-1)  # 使う枚数のどこにも写っていない物体は除く(公式の prepare_targets)
            labels, masks, present = targets[b]["labels"][keep], masks[keep], present[keep]
            frame_t = [{"labels": labels, "masks": masks[:, i], "valid": present[:, i]} for i in range(use_t)]
            outs, slot_outs = self.tracker.train_clip(seg, frame_t, stage, q_feat)
            loss, p = daq_loss(outs, frame_t, self.class_weight)
            if slot_outs:
                sl, sp = daq_loss(slot_outs, frame_t[1:], self.class_weight)
                loss = loss + sl
                p.update({f"slot_{k}": v for k, v in sp.items()})
            total = total + loss / clips.shape[0]
            parts = p
        return total, {**parts, "stage": stage}

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor, chunk: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """1 本の動画を時刻の順に処理し、ID ごとの予測にする(公式の run_window_inference。segmenter は 5 枚ずつ)。

        Args:
            frames: (T, 3, S, S)。
            chunk: segmenter でまとめて処理するフレームの数(公式は 5)。

        Returns:
            (ID ごとの動画のクラスの値 (R, K+1)(記録したフレームの平均。公式と同じ), マスクの値 (R, T, h, w)(記録のないフレームは −1e4))。
        """
        state, t_all = {}, len(frames)
        q_feat = self.segmenter.query_feat.weight
        for s in range(0, t_all, chunk):
            seg = self._segment(frames[s: s + chunk])
            for i in range(len(seg["embeds"])):
                state = self.tracker.infer_step({k: v[i] for k, v in seg.items()}, state, s + i, q_feat)
        logits, masks = [], []
        h, w = seg["masks"].shape[-2:]
        for sq in state.get("seqs", {}).values():
            if len(sq.masks) < self.noise_frame_num and sq.start + len(sq.masks) < t_all:
                continue  # 短く、動画の最後まで続かない ID は雑音として除く(公式と同じ)
            full = torch.full((t_all, h, w), -1e4)
            full[sq.start: sq.start + len(sq.masks)] = torch.stack(sq.masks)
            logits.append(torch.stack(sq.logits).mean(0))
            masks.append(full)
        if not logits:
            k = self.class_weight.numel()
            return torch.zeros(0, k, device=frames.device), torch.zeros(0, t_all, h, w, device=frames.device)
        return torch.stack(logits), torch.stack(masks).to(frames.device)
