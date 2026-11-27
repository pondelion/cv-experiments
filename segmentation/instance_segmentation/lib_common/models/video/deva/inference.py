"""DEVA の推論: 画像のモデルの検出 + クラスによらない伝搬を、数フレームの投票と統合でつなぐ(半 online)。exp00502。

参考: 公式のコード(hkchengrex/Tracking-Anything-with-DEVA の `deva/inference/inference_core.py`・`consensus_automatic.py`・
`consensus_associated.py`(spatial_alignment)・`segment_merging.py`・`object_manager.py`・`object_info.py`・`image_feature_store.py`・
`evaluation/eval_with_detections.py` の semionline の手順・`result_utils.py`)。コードは写さず、規則を読み取って書いた。
設定は公式の VIPSeg の評価と同じ(detection_every 5、num_voting_frames 3、max_missed_detection_count 5、mem_every 5、top_k 30、
長期記憶 10・5・128・10,000)。ただし画像のモデルの検出の確信度の閾値(公式は 0.8)は使わず、確信度の上位 100 個の検出を使う
(AP の評価の前に確信度で予測を切らない。ほかの方式と同じ。DET_TOP_K)。

MOT(動画全体での物体の ID の管理・識別)の形: **推論の規則(伝搬と検出の統合)+ 学習した伝搬の記憶**。1 つの ID = 1 つの物体の伝搬。
    - 5 フレームごとに、その先の 3 フレームの検出を最初のフレームへ伝搬で運び、IoU > 0.5 で互いに支え合う検出の組を、整数線形計画
      (支えの IoU の和 × 2 − 選んだ数 を最大に、IoU > 0.49 で重なる 2 つは同時に選ばない)で選んで、そのフレームの検出にする(投票)
    - 伝搬している物体と、投票した検出を IoU > 0.5 で統合する(面積の大きい順に塗る)。統合できなかった検出は新しい ID。
      統合できなかった伝搬の物体は「見逃し」を 1 つ数え、5 回を超えたら記憶から消す(その ID を終える)。統合できれば数え直し
    - その間のフレームは記憶から伝搬するだけ(記憶は 5 フレームごと)
    物体のクラスと点数は、統合した検出のクラスの最頻値と点数の平均(公式の vote_category_id・vote_score)。

公式との違い(exp の条件・私の判断):
    - 検出の入力: 公式は画像のモデルのパノプティックの出力(重ならない領域の地図)。ここではインスタンスセグメンテーションのモデル
      (ImageSegmenter)の出力を、Mask2Former の公式のパノプティックの後処理(things だけ: 点数 > 0.8 のクエリを、点数 × マスクの確率の
      argmax で塗り、自分の元の領域(確率 ≥ 0.5)の 0.8 以上を占めるものだけ残す)で同じ形にする
    - 整数線形計画の解き方: 公式は gurobi か pulp(CBC)。ここは SciPy の milp(HiGHS)。同じ問題なので最適解は同じ(同点のときの選び方だけ
      違いうる)
    - 動画インスタンスセグメンテーションの出力: ID ごとに 1 つのクラスが要るので、最後の多数決のクラスと平均の点数を使う
      (公式の VIPSeg の出力は、フレームごとのその時点の多数決)

形の記号: N = 物体の数、S = 入力の一辺(16 の倍数)、h・w = stride 16 の特徴の縦・横。
"""
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from segmentation.instance_segmentation.lib_common.models.video.deva.memory import MemoryManager
from segmentation.instance_segmentation.lib_common.models.video.deva.network import DEVA, do_softmax, get_similarity

DETECTION_EVERY, NUM_VOTING_FRAMES, MAX_MISSED, MEM_EVERY = 5, 3, 5, 5
# OVERLAP_THRESHOLD: Mask2Former の VIPSeg の推論の設定(パノプティックの後処理。ほかの物体に塗りつぶされて元のマスクの 0.8 未満しか
# 残らない検出は使わない)。公式の確信度の閾値 OBJECT_MASK_THRESHOLD(0.8)は、AP の評価の前に確信度で予測を切らないため使わない。
# 代わりに、ほかの方式の上位 100 個と同じく、確信度の上位 DET_TOP_K 個の検出を使う
OVERLAP_THRESHOLD, DET_TOP_K = 0.8, 100


class ObjectInfo:
    """物体の情報(公式の ObjectInfo)。"""

    def __init__(self, oid: int, category_id=None, isthing=None, score=None):
        """物体の情報を作る。

        Args:
            oid: 物体の ID。
            category_id: クラス(0 始まり)か None。
            isthing: 物体のクラスか(None は不明)。
            score: 検出の点数か None。
        """
        self.id, self.category_ids, self.scores, self.isthing, self.poke_count = oid, [category_id], [score], isthing, 0

    def poke(self):
        """検出と合わなかった回を 1 つ足す。"""
        self.poke_count += 1

    def unpoke(self):
        """検出と合ったので、合わなかった回を 0 に戻す。"""
        self.poke_count = 0

    def merge(self, other):
        """ほかの物体の情報のクラスと点数を足す(多数決・平均に使う)。

        Args:
            other: ObjectInfo。
        """
        self.category_ids.extend(other.category_ids)
        self.scores.extend(other.scores)

    def copy_meta_info(self, other):
        """ほかの物体の情報のクラス・点数・isthing を写す。

        Args:
            other: ObjectInfo。
        """
        self.category_ids, self.scores, self.isthing = other.category_ids, other.scores, other.isthing

    def vote_category_id(self):
        """クラスの多数決(同数なら小さい方。scipy.stats.mode と同じ)。

        Returns:
            クラス(0 始まり)か None(クラスの情報がない)。
        """
        c = [x for x in self.category_ids if x is not None]
        if not c:
            return None
        cnt = Counter(c)
        best = max(cnt.values())
        return min(k for k, v in cnt.items() if v == best)  # scipy.stats.mode と同じく、同数なら小さい方

    def vote_score(self):
        """点数の平均。

        Returns:
            平均の点数か None(点数の情報がない)。
        """
        s = [x for x in self.scores if x is not None]
        return float(np.mean(s)) if s else None

    def __hash__(self):
        """物体の ID で hash をとる(dict のキーに使う)。

        Returns:
            物体の ID の hash。
        """
        return hash(self.id)

    def __eq__(self, other):
        """物体の ID が同じかを比べる。

        Args:
            other: ObjectInfo。

        Returns:
            物体の ID が同じなら True。
        """
        return self.id == other.id


class ObjectManager:
    """物体の ID(変わらない)と、テンソルの中の位置の番号(tmp id。物体を消すと詰める)の対応(公式の ObjectManager)。"""

    def __init__(self):
        self.obj_to_tmp_id, self.tmp_id_to_obj, self.all_ids = {}, {}, set()
        self.next_id = 1
        self.all_seen: list[ObjectInfo] = []  # 消した物体も含む全部(最後のクラスの多数決に使う)

    def add_new_objects(self, objs: list[ObjectInfo]):
        """新しい物体を足す(ID は通し番号で、情報は渡したものを写す)。

        Args:
            objs: 足す物体の情報。

        Returns:
            (足した物体の tmp id の並び, 物体の ID の並び)。
        """
        tmp_ids, obj_ids = [], []
        for o in objs:
            while self.next_id in self.all_ids:  # 公式は乱数で重ならない ID を選ぶ。ここは通し番号(ID の値は評価に影響しない)
                self.next_id += 1
            new = ObjectInfo(self.next_id)
            new.copy_meta_info(o)
            t = len(self.obj_to_tmp_id) + 1
            self.obj_to_tmp_id[new], self.tmp_id_to_obj[t] = t, new
            self.all_ids.add(new.id)
            self.all_seen.append(new)
            tmp_ids.append(t); obj_ids.append(new.id)
        return tmp_ids, obj_ids

    def delete_objects(self, ids: list[int]):
        """物体を消し、残りの tmp id を 1 から詰め直す。

        Args:
            ids: 消す物体の ID。
        """
        new_o2t, new_t2o, t = {}, {}, 1
        for k in range(1, len(self.obj_to_tmp_id) + 1):
            o = self.tmp_id_to_obj[k]
            if o.id not in ids:
                new_o2t[o], new_t2o[t] = t, o
                t += 1
        self.obj_to_tmp_id, self.tmp_id_to_obj = new_o2t, new_t2o

    def purge_inactive_objects(self, max_missed: int):
        """検出と合わなかった回が max_missed を超えた物体を消す(公式の purge_inactive_objects)。

        Args:
            max_missed: 残す合わなかった回の上限(公式は 5)。

        Returns:
            (消した物体があるか, 残した物体の消す前の tmp id の並び, 残した物体の ID の並び)。
        """
        drop = [o.id for o in self.obj_to_tmp_id if o.poke_count > max_missed]
        keep_tmp = [self.obj_to_tmp_id[o] for o in self.obj_to_tmp_id if o.poke_count <= max_missed]
        keep_obj = [o.id for o in self.obj_to_tmp_id if o.poke_count <= max_missed]
        if drop:
            self.delete_objects(drop)
        return bool(drop), keep_tmp, keep_obj

    def make_one_hot(self, cls_mask: torch.Tensor) -> torch.Tensor:
        """物体の ID の地図を、tmp id の順の物体ごとのマスクにする。

        Args:
            cls_mask: 物体の ID の地図 (S, S)。

        Returns:
            (N, S, S) の bool。
        """
        if not self.tmp_id_to_obj:
            return torch.zeros((0, *cls_mask.shape), dtype=torch.bool, device=cls_mask.device)
        return torch.stack([cls_mask == o.id for o in self.tmp_id_to_obj.values()])

    def realize_dict(self, d: dict) -> torch.Tensor:
        """物体の ID ごとの値を、tmp id の順に重ねる。

        Args:
            d: {物体の ID: テンソル}。

        Returns:
            (N, ...)。
        """
        return torch.stack([d[o.id] for o in self.tmp_id_to_obj.values()])

    @property
    def all_obj_ids(self) -> list[int]:
        """今いる物体の ID(tmp id の順)。"""
        return [o.id for o in self.obj_to_tmp_id]

    @property
    def num_obj(self) -> int:
        """今いる物体の数。"""
        return len(self.obj_to_tmp_id)


def _solve_ilp(pairwise_iou: np.ndarray, indicator: np.ndarray, n: int) -> list[bool]:
    """投票で残す検出を 0/1 の整数線形計画で選ぶ(公式の solve_with_pulp と同じ問題。scipy の milp で解く)。

    目的: 選んだ検出ごとの「ほかのフレームの検出との IoU の和 × 2 − 1」の和を最大にする。制約: 重なる 2 つは同時に選ばない。

    Args:
        pairwise_iou: 検出どうしの IoU (n, n)(閾値に届かない組は 0)。
        indicator: 重なる組の印 (n, n) の bool。
        n: 検出の数。

    Returns:
        検出ごとに選んだか。
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    c = -(pairwise_iou.sum(0) * 2 - 1)  # 最小化に直す
    rows = [(i, j) for i in range(n) for j in range(i + 1, n) if indicator[i, j]]
    cons = None
    if rows:
        a = np.zeros((len(rows), n))
        for r, (i, j) in enumerate(rows):
            a[r, i] = a[r, j] = 1
        cons = LinearConstraint(a, -np.inf, 1)
    res = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1))
    return [bool(x > 0.5) for x in res.x]


class DEVAInferenceCore:
    """1 本の動画の DEVA の推論の状態(公式の DEVAInferenceCore)。

    Args:
        network: DEVA(eval)。
        count_long_term_usage: 長期記憶の使われた量も数えるか(公式: 動画の長さで決める)。
        top_k: 記憶の読み出しの softmax に使う上位の数(公式は 30)。
    """

    def __init__(self, network: DEVA, count_long_term_usage: bool, top_k: int = 30):
        self.network, self.top_k = network, top_k
        self.memory = MemoryManager(top_k=top_k, count_long_term_usage=count_long_term_usage)
        self.objects = ObjectManager()
        self.curr_ti, self.last_mem_ti = -1, 0
        self.last_mask = None
        self.store = {}  # 画像の特徴の cache(公式の ImageFeatureStore)
        self.frame_buffer = []
        self.given_to_internal = {}  # step で与えた物体の ID → 内部の ID(J&F の評価で使う)

    def _features(self, ti: int, image: torch.Tensor):
        """フレームの画像の特徴と key を作る(同じフレームは cache から返す。公式の ImageFeatureStore)。

        Args:
            ti: フレームの番号。
            image: (1, 3, S, S)。

        Returns:
            (画素の encoder の 3 段の特徴, key, shrinkage, selection)。
        """
        if ti not in self.store:
            ms, feat = self.network.encode_image(image)
            k, s, e = self.network.transform_key(feat)
            self.store[ti] = (ms, k, s, e)
        return self.store[ti]

    def _add_memory(self, image, ms, prob, k, s, e, deep: bool = True):
        """今のフレームのマスクを value にして記憶に足す(公式の _add_memory)。

        Args:
            image: (1, 3, S, S)。
            ms: 画素の encoder の 3 段の特徴。
            prob: 物体ごとのマスク (1, N, S, S)(物体がなければ何もしない)。
            k: key。
            s: shrinkage。
            e: selection。
            deep: 感覚の記憶も更新するか(deep update)。
        """
        if prob.shape[1] == 0:
            return
        ids = self.objects.all_obj_ids
        self.memory.initialize_sensory_if_needed(k, ids)
        value, sensory = self.network.encode_mask(image, ms, self.memory.get_sensory(ids), prob, is_deep_update=deep)
        self.memory.add_memory(k, s, value, ids, e)
        self.last_mem_ti = self.curr_ti
        if deep:
            self.memory.update_sensory(sensory, ids)

    def _segment(self, k, e, ms, update_sensory: bool = True) -> torch.Tensor:
        """記憶を読み出して今のフレームのマスクを出す(記憶が空なら背景だけ)。

        Args:
            k: 今の key。
            e: 今の selection。
            ms: 画素の encoder の 3 段の特徴。
            update_sensory: 感覚の記憶を更新するか。

        Returns:
            背景つきの確率 (N + 1, S, S)(記憶が空なら (1, S, S) の 0)。
        """
        if not self.memory.engaged:
            return torch.zeros((1, k.shape[-2] * 16, k.shape[-1] * 16), device=k.device)
        readout = self.objects.realize_dict(self.memory.match_memory(k, e))[None]
        ids = self.objects.all_obj_ids
        sensory, _, prob = self.network.segment(ms, readout, self.memory.get_sensory(ids), self.last_mask,
                                                update_sensory=update_sensory)
        if update_sensory:
            self.memory.update_sensory(sensory, ids)
        return prob[0]  # (N + 1, S, S)

    def step(self, image: torch.Tensor, end: bool = False, mask: torch.Tensor | None = None,
             objects: list[int] | None = None) -> torch.Tensor:
        """1 フレーム進める(公式の step。hard_mask)。

        Args:
            image: (3, S, S)。
            end: 動画の最後のフレームか(記憶を足さず、感覚の記憶も更新しない)。
            mask: このフレームで与える物体の ID の地図 (S, S) か None。
            objects: mask の中の物体の ID。

        Returns:
            背景つきの確率 (N + 1, S, S)。
        """
        self.curr_ti += 1
        ti, image = self.curr_ti, image[None]
        mem_frame = ((self.curr_ti - self.last_mem_ti >= MEM_EVERY) or mask is not None) and not end
        need_segment = mask is None or (not all(o in self.given_to_internal for o in objects) and self.objects.num_obj > 0)
        ms, k, s, e = self._features(ti, image)
        if need_segment:
            prob = self._segment(k, e, ms, update_sensory=not end)
        if mask is not None:  # 与えたマスクの物体を足し、その画素は与えたマスクにする(公式の hard_mask の手順)
            tmp_ids, obj_ids = self.objects.add_new_objects([ObjectInfo(o, isthing=True) for o in objects])
            self.given_to_internal.update(dict(zip(objects, obj_ids)))
            if need_segment:
                no_bg = prob[1:]
                no_bg[:, mask > 0] = 0
                new = []
                for mi, t_id in enumerate(tmp_ids):
                    this = (mask == objects[mi]).to(no_bg.dtype)
                    if t_id - 1 >= no_bg.shape[0]:
                        new.append(this[None])
                    else:
                        no_bg[t_id - 1] = this
                stacked = torch.cat([no_bg, *new], 0)
            else:
                stacked = torch.stack([(mask == objects[mi]) for mi in range(len(tmp_ids))]).float()
            prob = torch.softmax(self.network.aggregate(stacked, dim=0), dim=0)
        self.last_mask = prob[1:][None]
        if mem_frame:
            self._add_memory(image, ms, self.last_mask, k, s, e)
        self.store.pop(ti, None)
        return prob

    def incorporate_detection(self, image: torch.Tensor, new_mask: torch.Tensor, infos: list[ObjectInfo]) -> torch.Tensor:
        """投票した検出を伝搬と統合し、合わない回が続いた物体を消して、記憶に足す(公式の incorporate_detection)。

        Args:
            image: (3, S, S)。
            new_mask: 投票した検出の ID の地図 (S, S)。
            infos: 投票した検出の物体の情報。

        Returns:
            統合したマスクの背景つきの確率 (N + 1, S, S)。
        """
        self.curr_ti += 1
        ti, image = self.curr_ti, image[None]
        ms, k, s, e = self._features(ti, image)
        forward = torch.argmax(self._segment(k, e, ms), 0) if self.memory.engaged else torch.zeros_like(new_mask)
        merged = match_and_merge(forward, new_mask, self.objects, infos)
        purged, keep_tmp, keep_obj = self.objects.purge_inactive_objects(MAX_MISSED)
        if purged:
            self.memory.purge_except(keep_obj)
            merged = merged[[i - 1 for i in keep_tmp]]
        self.last_mask = merged[None].to(k.dtype)
        self._add_memory(image, ms, self.last_mask, k, s, e)
        self.store.pop(ti, None)
        return F.softmax(self.network.aggregate(self.last_mask[0], dim=0), dim=0)

    # ---- 投票(公式の find_consensus_auto_association と spatial_alignment) ----
    def _spatial_alignment(self, src_ti, src_image, src_mask, tar_ti, tar_image) -> torch.Tensor:
        """あるフレームの検出のマスクを、別のフレームに 1 回の伝搬で運ぶ(公式の spatial_alignment)。

        Args:
            src_ti: 運ぶ元のフレームの番号。
            src_image: 元の画像 (3, S, S)。
            src_mask: 元の物体ごとのマスク (n, S, S)。
            tar_ti: 運ぶ先のフレームの番号。
            tar_image: 先の画像 (3, S, S)。

        Returns:
            先のフレームでの背景つきの確率 (n + 1, S, S)。
        """
        n = src_mask.shape[0]
        src_ms, src_k, src_s, _ = self._features(src_ti, src_image[None])
        tar_ms, tar_k, _, tar_e = self._features(tar_ti, tar_image[None])
        h, w = src_k.shape[-2:]
        sensory = torch.zeros((1, n, self.memory.sensory_dim, h, w), device=src_k.device)
        value, sensory = self.network.encode_mask(src_image[None], src_ms, sensory, src_mask[None], is_deep_update=True)
        aff = do_softmax(get_similarity(src_k.flatten(2), src_s.flatten(2), tar_k.flatten(2), tar_e.flatten(2)), self.top_k)
        readout = (value.flatten(1, 2).flatten(2) @ aff.to(value.dtype)).view(1, n, -1, h, w)
        _, _, tar_mask = self.network.segment(tar_ms, readout, sensory, src_mask[None], update_sensory=False)
        return tar_mask[0]  # (n + 1, S, S)

    def vote_in_buffer(self):
        """バッファーのフレームの検出を最初のフレームに運んで投票する(公式の find_consensus_auto_association)。

        Returns:
            (残した検出の ID の地図 (S, S), 残した検出の物体の情報の並び)。
        """
        frames = self.frame_buffer
        masks, infos_per_frame, mappings, all_infos = [], [], [], {}
        bookkeeper = 0
        for f in frames:
            one_hot, mapping, infos = [], {}, []
            for si, inf in enumerate(f["infos"]):
                bookkeeper += 1
                ni = ObjectInfo(bookkeeper)
                ni.copy_meta_info(inf)
                all_infos[bookkeeper] = ni
                one_hot.append(f["mask"] == inf.id)
                mapping[si] = bookkeeper
                infos.append(ni)
            masks.append(torch.stack(one_hot).float() if one_hot else None)
            mappings.append(mapping); infos_per_frame.append(infos)
        key = frames[0]
        projected, areas, seg_mask = [], {}, {}
        for f, m, mp in zip(frames, masks, mappings):
            if m is None:
                projected.append(None)
                continue
            if f is key:
                pm = torch.cat([torch.ones_like(m[0:1]) * 0.5, m], 0)
            else:
                pm = self._spatial_alignment(f["ti"], f["image"], m, key["ti"], key["image"])
            pm = torch.argmax(pm, 0)
            remap = torch.zeros_like(pm)
            for ch, oid in mp.items():
                this = pm == (ch + 1)
                remap[this] = oid
                areas[oid] = this.sum().item()
                seg_mask[oid] = this
            projected.append(remap.long())
        n = bookkeeper
        if n == 0:
            return torch.zeros_like(frames[0]["mask"]), []
        scaling = 4096
        pairwise = np.zeros((n, n), np.float32)
        matching = {i: [] for i in range(1, n + 1)}
        for i in range(len(frames)):
            if projected[i] is None:
                continue
            m1 = projected[i] * scaling
            for j in range(i + 1, len(frames)):
                if projected[j] is None:
                    continue
                comb = m1 + projected[j]
                for status in (None, False, True):
                    matched2 = set()
                    for o1 in infos_per_frame[i]:
                        if o1.isthing != status:
                            continue
                        for o2 in infos_per_frame[j]:
                            if o2.isthing != status or o2.id in matched2:
                                continue
                            inter = (comb == o1.id * scaling + o2.id).sum().item()
                            if inter == 0:
                                continue
                            iou = inter / (areas[o1.id] + areas[o2.id] - inter)
                            if iou > 0.5:
                                matching[o1.id].append(o2.id); matching[o2.id].append(o1.id)
                                matched2.add(o2.id)
                                pairwise[o1.id - 1, o2.id - 1] = iou
                                break
        pairwise = pairwise + pairwise.T
        indicator = pairwise > 0.49
        pairwise = pairwise * indicator
        chosen = _solve_ilp(pairwise, indicator, n)
        out_mask = torch.zeros_like(frames[0]["mask"])
        out_infos, chosen_area = [], {}
        for ch, sel in enumerate(chosen):
            if sel:
                oid = ch + 1
                chosen_area[oid] = areas[oid]
                inf = all_infos[oid]
                for other in matching[oid]:
                    inf.merge(all_infos[other])
                out_infos.append(inf)
        for oid, _ in sorted(chosen_area.items(), key=lambda x: x[1], reverse=True):
            out_mask[seg_mask[oid]] = oid
        return out_mask, out_infos


def match_and_merge(our_mask: torch.Tensor, new_mask: torch.Tensor, objects: ObjectManager, new_infos: list[ObjectInfo]):
    """伝搬と検出を IoU > 0.5 で統合する(公式の match_and_merge・merge_by_iou。incremental なし)。

    合った物体は同じ ID のまま検出のマスクも足し、合わない回を 0 に戻す。合わない検出は新しい物体として objects に足す。
    検出と合わない伝搬の物体は、合わない回を 1 つ足す。

    Args:
        our_mask: 伝搬の tmp id の地図 (S, S)。
        new_mask: 検出の ID の地図 (S, S)。
        objects: 物体の管理(新しい物体を足し、合わない回を更新する)。
        new_infos: 検出の物体の情報。

    Returns:
        統合した物体ごとのマスク (N, S, S) の bool(tmp id の順)。
    """
    our_mask, new_mask = our_mask.long(), new_mask.long()
    ours = {o: our_mask == t for o, t in objects.obj_to_tmp_id.items()}
    news = {o: new_mask == o.id for o in new_infos}
    our_sum = {o: m.sum() for o, m in ours.items()}
    new_sum = {o: m.sum() for o, m in news.items()}
    merged = torch.zeros_like(our_mask)
    for status in (None, False, True):
        matching, area, new_objs = {}, {}, []
        for no in new_infos:
            if no.isthing != status:
                continue
            for oo in list(objects.obj_to_tmp_id):
                if oo.isthing != status or oo in matching:
                    continue
                inter = (news[no] & ours[oo]).sum()
                if inter < 1e-3:
                    continue
                union = new_sum[no] + our_sum[oo] - inter
                if inter / union > 0.5:
                    matching[oo] = no
                    area[(oo, False)] = union
                    break
            else:
                new_objs.append(no)
                area[(no, True)] = new_sum[no]
        for oo in list(objects.obj_to_tmp_id):
            if oo.isthing == status and oo not in matching:
                area[(oo, False)] = our_sum[oo]
        for (o, is_new), _ in sorted(area.items(), key=lambda x: float(x[1]), reverse=True):
            if is_new:
                _, ids = objects.add_new_objects([o])
                merged[news[o]] = ids[0]
            elif o in matching:
                merged[ours[o]] = o.id
                merged[news[matching[o]]] = o.id
                o.merge(matching[o])
                o.unpoke()
            else:
                merged[ours[o]] = o.id
                o.poke()
    return objects.make_one_hot(merged)


def detections_to_panoptic(out: dict, i: int, size: int) -> tuple[torch.Tensor, list[ObjectInfo]]:
    """画像のモデルの 1 フレームの出力を、重ならない ID の地図と物体の情報にする(Mask2Former の公式の panoptic_inference の things の規則)。

    Args:
        out: 画像のモデルの出力(ImageSegmenter.forward)。
        i: フレームの番号(out の中の何枚目か)。
        size: 入力の一辺 S。

    Returns:
        (ID の地図 (S, S)(0 は背景), 物体の情報 [ObjectInfo(id, category_id(0 始まり), isthing, score)])。
    """
    probs = out["probs"][i]  # (Q, K)(物体なしを除いたクラスの確率)
    scores, labels = probs.max(-1)
    # 確信度の上位 DET_TOP_K 個のクエリ(確信度の閾値・「物体なしが最大のクエリを除く」規則は、確信度で予測を切ることになるので使わない)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep[scores.topk(min(DET_TOP_K, len(scores))).indices] = True
    pan = torch.zeros((size, size), dtype=torch.long, device=probs.device)
    infos = []
    if keep.sum() == 0:
        return pan, infos
    masks = F.interpolate(out["masks"][i, keep][None].float(), size=(size, size), mode="bilinear", align_corners=False)[0].sigmoid()
    sc, lb = scores[keep], labels[keep]
    ids = (sc[:, None, None] * masks).argmax(0)
    seg = 0
    for k in range(len(sc)):
        area = (ids == k).sum()
        orig = (masks[k] >= 0.5).sum()
        m = (ids == k) & (masks[k] >= 0.5)
        if area > 0 and orig > 0 and m.sum() > 0:
            if area / orig < OVERLAP_THRESHOLD:
                continue
            seg += 1
            pan[m] = seg
            infos.append(ObjectInfo(seg, int(lb[k]), True, float(sc[k])))
    return pan, infos


class DEVAVideo(torch.nn.Module):
    """DEVA の動画インスタンスセグメンテーション(半 online。公式の eval_with_detections の semionline と同じ手順)。

    Args:
        network: 学習した DEVA の伝搬のネットワーク。
        detector: 学習した画像のモデル(ImageSegmenter)。
    """

    def __init__(self, network: DEVA, detector):
        super().__init__()
        self.network, self.detector = network, detector

    @torch.no_grad()
    def track_video(self, frames: torch.Tensor, valid: tuple[int, int], orig: tuple[int, int]) -> list[dict]:
        """1 本の動画を追跡する(5 フレームごとに 3 フレームの検出で投票して統合し、その間は伝搬)。

        Args:
            frames: (T, 3, S, S)(評価の変換。右と下を埋めたもの)。
            valid: 入力の中の画像の部分の (縦, 横)。
            orig: 元の大きさ (縦, 横)。

        Returns:
            ID ごとの {"oid", "label", "score", "masks": {t: 元の大きさの RLE}}(公式の ResultSaver と同じく、背景つきの確率を
            元の大きさに bilinear で拡大してから argmax で ID を決める)。
        """
        from pycocotools import mask as mask_util
        t_all, s = len(frames), frames.shape[-1]
        # 公式: 長期記憶の使われた量を数えるのは、動画が長期記憶の上限を超えそうなときだけ
        count_usage = t_all / (10 - 5) * 128 >= 10000
        core = DEVAInferenceCore(self.network, count_usage)
        dets = []
        for c in range(0, t_all, 4):
            out = self.detector(frames[c: c + 4])
            dets += [detections_to_panoptic(out, i, s) for i in range(len(frames[c: c + 4]))]
        results = {}

        def save(prob):
            """今のフレーム(core.curr_ti_saved)の確率を元の大きさにして、ID ごとの RLE を results に入れる。

            Args:
                prob: 背景つきの確率 (N + 1, S, S)。
            """
            p = F.interpolate(prob[None, :, : valid[0], : valid[1]].float(), size=orig, mode="bilinear", align_corners=False)[0]
            idx = p.argmax(0)
            for t_id, o in core.objects.tmp_id_to_obj.items():
                m = (idx == t_id).cpu().numpy()
                if m.any():
                    r = mask_util.encode(np.asfortranarray(m))
                    r["counts"] = r["counts"].decode("ascii")
                    results.setdefault(o.id, {})[core.curr_ti_saved] = r

        next_vote = NUM_VOTING_FRAMES - 1
        for ti in range(t_all):
            if ti + NUM_VOTING_FRAMES > next_vote:
                core.frame_buffer.append({"ti": ti, "image": frames[ti], "mask": dets[ti][0], "infos": dets[ti][1]})
                if ti == next_vote:
                    mask, infos = core.vote_in_buffer()
                    first = core.frame_buffer[0]
                    prob = core.incorporate_detection(first["image"], mask, infos)
                    core.curr_ti_saved = first["ti"]; save(prob)
                    next_vote += DETECTION_EVERY
                    if next_vote >= t_all:
                        next_vote = t_all + NUM_VOTING_FRAMES
                    for f in core.frame_buffer[1:]:
                        prob = core.step(f["image"], end=(f["ti"] == t_all - 1))
                        core.curr_ti_saved = f["ti"]; save(prob)
                    for f in core.frame_buffer:
                        core.store.pop(f["ti"], None)
                    core.frame_buffer = []
            else:
                prob = core.step(frames[ti], end=(ti == t_all - 1))
                core.curr_ti_saved = ti; save(prob)
        infos = {o.id: o for o in core.objects.all_seen}
        return [{"oid": oid, "label": infos[oid].vote_category_id(), "score": infos[oid].vote_score(), "masks": m}
                for oid, m in results.items() if infos[oid].vote_category_id() is not None]

    @torch.no_grad()
    def to_predictions(self, tracks: list[dict], video_id: int, cat_ids: list[int], size: int, valid: tuple[int, int],
                       orig: tuple[int, int], t_all: int) -> list[dict]:
        """ID ごとの予測を評価の形にする(ID の地図から作ったので重なりはない。マスクは track_video で元の大きさの RLE 済み)。

        Args:
            tracks: track_video の返り値。
            video_id: 動画の ID。
            cat_ids: クラスの番号 → カテゴリーの id。
            size: 入力の一辺(使わない。ほかの方式と呼び方をそろえるため)。
            valid: 入力の中の画像の部分(使わない。同上)。
            orig: 元の大きさ(使わない。同上)。
            t_all: 動画のフレームの数。

        Returns:
            評価の予測 [{"video_id", "category_id", "score", "segmentations", "query"}]。
        """
        preds = []
        for tr in tracks:
            segs = [tr["masks"].get(t) for t in range(t_all)]
            if any(x is not None for x in segs):
                preds.append({"video_id": video_id, "category_id": cat_ids[tr["label"]], "score": tr["score"],
                              "segmentations": segs, "query": tr["oid"]})
        return preds


@torch.no_grad()
def propagate_vos(network: DEVA, frames: torch.Tensor, first_masks: dict[int, tuple[int, torch.Tensor]]) -> dict[int, torch.Tensor]:
    """マスクを与えて追う(伝搬だけの出来の J&F 用)。各物体は初めて写ったフレームで正解のマスクを与える(公式の step の hard_mask)。

    Args:
        network: 学習した DEVA の伝搬のネットワーク。
        frames: (T, 3, S, S)。
        first_masks: {物体の ID: (始めるフレーム, マスク (S, S) の bool)}。

    Returns:
        {物体の ID: フレームごとのマスク (T, S, S)(1 = その物体。背景つきの確率の argmax で各画素を 1 つの物体に割り当てる(公式の
        出力と同じ)。始めるフレームより前は 0)}。
    """
    t_all = len(frames)
    count_usage = t_all / (10 - 5) * 128 >= 10000
    core = DEVAInferenceCore(network, count_usage)
    out = {o: torch.zeros(t_all, *frames.shape[-2:], device=frames.device) for o in first_masks}
    for t in range(t_all):
        starting = [o for o, (st, _) in first_masks.items() if st == t]
        if starting:
            mask = torch.zeros(frames.shape[-2:], dtype=torch.long, device=frames.device)
            for o in starting:
                mask[first_masks[o][1]] = o
            prob = core.step(frames[t], end=(t == t_all - 1), mask=mask, objects=starting)
        else:
            prob = core.step(frames[t], end=(t == t_all - 1))
        idx = prob.argmax(0)
        tmp_of = {x.id: t_id for x, t_id in core.objects.obj_to_tmp_id.items()}
        for o, internal in core.given_to_internal.items():
            if internal in tmp_of:
                out[o][t] = (idx == tmp_of[internal]).float()
    return out
