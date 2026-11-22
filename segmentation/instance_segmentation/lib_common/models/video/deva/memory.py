"""DEVA の記憶の管理(作業記憶・長期記憶・感覚の記憶)。exp00502。

参考: 公式のコード(hkchengrex/Tracking-Anything-with-DEVA の `deva/inference/kv_memory_store.py`・`memory_manager.py`)。
コードは写さず、規則を読み取って書いた(XMem と同じ考え方に、物体のまとまり(bucket)ごとの長期記憶と物体の削除を足したもの)。

規則(公式と同じ):
    - 同じフレームで初めて出た物体は 1 つの bucket にまとめ、bucket ごとに key(と shrinkage・selection)を、物体ごとに value を持つ
    - 作業記憶が max_mid_term_frames(10)フレーム分に達したら、最初のフレームと最近の min_mid_term_frames(5)フレームを残し、
      その間を「使われた量 ÷ 経過」の上位 num_prototypes(128)個の prototype にまとめて長期記憶に移す(prototype の value・shrinkage は
      候補との類似度の softmax で平均する)。長期記憶が上限(10,000 − 128)を超えたら、使われた量の少ないものから消す
    - 読み出しは、bucket ごとに(長期 + 作業)記憶との類似度の上位 top_k(30)の softmax で value を平均する

形の記号: C = チャンネル、N = 記憶の要素の数(フレーム × 画素)、HW = 1 フレームの画素の数(stride 16)。
"""
import torch

from segmentation.instance_segmentation.lib_common.models.video.deva.network import do_softmax, get_similarity


class KeyValueMemoryStore:
    """key・value の記憶(作業記憶・長期記憶)。"""

    def __init__(self, save_selection: bool = False, save_usage: bool = False):
        self.save_selection, self.save_usage = save_selection, save_usage
        self.global_bucket_id = 0
        self.buckets: dict[int, list[int]] = {}
        self.k, self.v, self.s, self.e, self.use_cnt, self.life_cnt = {}, {}, {}, {}, {}, {}

    def add(self, key, values: dict, shrinkage, selection, supposed_bucket_id: int = -1) -> None:
        """Args: key (CK, N)、values {物体: (CV, N)}、shrinkage (1, N)、selection (CK, N) か None。"""
        if supposed_bucket_id >= 0:
            enabled = {supposed_bucket_id}
            exist = supposed_bucket_id in self.buckets
            for obj, val in values.items():
                self.v[obj] = torch.cat([self.v[obj], val], -1) if exist else val
            self.buckets[supposed_bucket_id] = list(values.keys())
        else:
            new_bucket, enabled = None, set()
            for obj, val in values.items():
                if obj in self.v:
                    self.v[obj] = torch.cat([self.v[obj], val], -1)
                    enabled.add(next(b for b, ids in self.buckets.items() if obj in ids))
                else:
                    self.v[obj] = val
                    if new_bucket is None:
                        new_bucket = self.global_bucket_id
                        self.global_bucket_id += 1
                        self.buckets[new_bucket] = []
                    self.buckets[new_bucket].append(obj)
                    enabled.add(new_bucket)
        n = key.shape[1]
        for b in self.buckets:
            if b not in enabled:
                continue
            new_cnt = torch.zeros(n, device=key.device)
            new_life = torch.zeros(n, device=key.device) + 1e-7
            if b in self.k:
                self.k[b], self.s[b] = torch.cat([self.k[b], key], -1), torch.cat([self.s[b], shrinkage], -1)
                if self.save_selection:
                    self.e[b] = torch.cat([self.e[b], selection], -1)
                if self.save_usage:
                    self.use_cnt[b], self.life_cnt[b] = torch.cat([self.use_cnt[b], new_cnt]), torch.cat([self.life_cnt[b], new_life])
            else:
                self.k[b], self.s[b] = key, shrinkage
                if self.save_selection:
                    self.e[b] = selection
                if self.save_usage:
                    self.use_cnt[b], self.life_cnt[b] = new_cnt, new_life

    def update_bucket_usage(self, b: int, usage: torch.Tensor) -> None:
        if self.save_usage:
            self.use_cnt[b] += usage.view_as(self.use_cnt[b])
            self.life_cnt[b] += 1

    def sieve_by_range(self, b: int, start: int, end: int, min_size: int) -> None:
        """[start, end) を除く(end は負。公式と同じ)。"""
        if self.k[b].shape[-1] <= min_size:
            return
        cut = lambda x: torch.cat([x[..., :start], x[..., end:]], -1)
        self.k[b], self.s[b] = cut(self.k[b]), cut(self.s[b])
        if self.save_selection:
            self.e[b] = cut(self.e[b])
        if self.save_usage:
            self.use_cnt[b], self.life_cnt[b] = cut(self.use_cnt[b]), cut(self.life_cnt[b])
        for obj in self.buckets[b]:
            self.v[obj] = cut(self.v[obj])

    def remove_obsolete_features(self, b: int, max_size: int) -> None:
        usage = self.get_usage(b).flatten()
        values, _ = torch.topk(usage, k=self.size(b) - max_size, largest=False, sorted=True)
        keep = usage > values[-1]
        self.k[b], self.s[b] = self.k[b][:, keep], self.s[b][:, keep]
        if self.save_selection:
            self.e[b] = self.e[b][:, keep]
        for obj in self.buckets[b]:
            self.v[obj] = self.v[obj][:, keep]
        self.use_cnt[b], self.life_cnt[b] = self.use_cnt[b][keep], self.life_cnt[b][keep]

    def get_usage(self, b: int) -> torch.Tensor:
        return self.use_cnt[b] / self.life_cnt[b]

    def get_all_sliced(self, b: int, start: int, end: int):
        sl = (lambda x: x[..., start:]) if end == 0 else (lambda x: x[..., start:end])
        return (sl(self.k[b]), sl(self.s[b]), sl(self.e[b]) if self.save_selection else None,
                {o: sl(self.v[o]) for o in self.buckets[b]}, sl(self.get_usage(b)) if self.save_usage else None)

    def purge_except(self, keep: list[int]) -> None:
        keep = set(keep)
        gone = []
        for b, ids in self.buckets.items():
            self.buckets[b] = [o for o in ids if o in keep]
            if not self.buckets[b]:
                gone.append(b)
        self.v = {o: v for o, v in self.v.items() if o in keep}
        for b in gone:
            del self.buckets[b], self.k[b], self.s[b]
            if self.save_selection:
                del self.e[b]
            if self.save_usage:
                del self.use_cnt[b], self.life_cnt[b]

    def size(self, b: int) -> int:
        return self.k[b].shape[-1] if b in self.k else 0

    def engaged(self, b: int | None = None) -> bool:
        return len(self.buckets) > 0 if b is None else b in self.buckets


class MemoryManager:
    """作業記憶・長期記憶・感覚の記憶の管理(公式の MemoryManager)。

    Args:
        top_k・max_mid_term_frames・min_mid_term_frames・num_prototypes・max_long_term_elements: 公式の設定(30・10・5・128・10,000)。
        count_long_term_usage: 長期記憶の使われた量も数えるか(公式は動画の長さで決める。DEVAVideo を参照)。
    """

    def __init__(self, top_k: int = 30, max_mid_term_frames: int = 10, min_mid_term_frames: int = 5, num_prototypes: int = 128,
                 max_long_term_elements: int = 10000, count_long_term_usage: bool = False, value_dim: int = 512):
        self.top_k, self.max_mem, self.min_mem = top_k, max_mid_term_frames, min_mid_term_frames
        self.num_prototypes, self.max_long, self.count_long = num_prototypes, max_long_term_elements, count_long_term_usage
        self.sensory_dim = value_dim
        self.sensory: dict[int, torch.Tensor] = {}
        self.work_mem = KeyValueMemoryStore(save_selection=True, save_usage=True)
        self.long_mem = KeyValueMemoryStore(save_usage=count_long_term_usage)
        self.hw = None
        self.engaged = False

    def match_memory(self, qk: torch.Tensor, qe: torch.Tensor) -> dict[int, torch.Tensor]:
        """Args: qk・qe (1, CK, H, W)。Returns: {物体: 読み出し (CV, H, W)}。"""
        h, w = qk.shape[-2:]
        qk, qe = qk[0].flatten(1), qe[0].flatten(1)
        out = {}
        for b, objs in self.work_mem.buckets.items():
            if self.long_mem.engaged(b):
                n_long = self.long_mem.size(b)
                mk = torch.cat([self.long_mem.k[b], self.work_mem.k[b]], -1)
                ms = torch.cat([self.long_mem.s[b], self.work_mem.s[b]], -1)
                aff, usage = do_softmax(get_similarity(mk[None], ms[None], qk[None], qe[None]), self.top_k, return_usage=True)
                self.work_mem.update_bucket_usage(b, usage[:, n_long:].flatten())
                if self.count_long:
                    self.long_mem.update_bucket_usage(b, usage[:, :n_long].flatten())
            else:
                aff, usage = do_softmax(get_similarity(self.work_mem.k[b][None], self.work_mem.s[b][None], qk[None], qe[None]),
                                        self.top_k, return_usage=True)
                self.work_mem.update_bucket_usage(b, usage.flatten())
            v = torch.stack([self.work_mem.v[o] for o in objs])  # (n, CV, N)
            if objs[0] in self.long_mem.v:
                v = torch.cat([torch.stack([self.long_mem.v[o] for o in objs]), v], -1)
            n, cv, _ = v.shape
            r = (v.reshape(n * cv, -1) @ aff[0].to(v.dtype)).view(n, cv, h, w)
            for i, o in enumerate(objs):
                out[o] = r[i]
        return out

    def add_memory(self, key, shrinkage, value, objects: list[int], selection) -> None:
        """Args: key (1, CK, H, W)、shrinkage (1, 1, H, W)、value (1, n, CV, H, W)、selection (1, CK, H, W)。"""
        self.engaged = True
        if self.hw is None:
            self.hw = value.shape[-2] * value.shape[-1]
            self.max_work, self.min_work = self.max_mem * self.hw, self.min_mem * self.hw
        k, s, e = key[0].flatten(1), shrinkage[0].flatten(1), selection[0].flatten(1)
        v = value[0].flatten(2)
        self.work_mem.add(k, {o: v[i] for i, o in enumerate(objects)}, s, e)
        for b in list(self.work_mem.buckets.keys()):
            if self.work_mem.size(b) >= self.max_work:
                if self.long_mem.size(b) >= self.max_long - self.num_prototypes:
                    self.long_mem.remove_obsolete_features(b, self.max_long - self.num_prototypes)
                self._compress(b)

    def _compress(self, b: int) -> None:
        hw = self.hw
        pk, pv, ps = self._consolidate(*self.work_mem.get_all_sliced(b, hw, -self.min_work + hw))
        self.work_mem.sieve_by_range(b, hw, -self.min_work + hw, min_size=self.min_work + hw)
        self.long_mem.add(pk, pv, ps, None, supposed_bucket_id=b)

    def _consolidate(self, ck, cs, ce, cv: dict, usage):
        _, idx = torch.topk(usage, k=self.num_prototypes, dim=-1, sorted=True)
        idx = idx.flatten()
        pk, pe = ck[:, idx], ce[:, idx]
        aff = do_softmax(get_similarity(ck[None], cs[None], pk[None], pe[None]))[0]  # (候補, prototype)
        pv = {o: v @ aff.to(v.dtype) for o, v in cv.items()}
        ps = cs @ aff.to(cs.dtype)
        return pk, pv, ps

    def purge_except(self, keep: list[int]) -> None:
        self.work_mem.purge_except(keep)
        if self.long_mem.engaged():
            self.long_mem.purge_except(keep)
        self.sensory = {o: s for o, s in self.sensory.items() if o in keep}
        if not self.work_mem.engaged():
            self.engaged = False

    def initialize_sensory_if_needed(self, sample_key, ids: list[int]) -> None:
        for o in ids:
            if o not in self.sensory:
                self.sensory[o] = torch.zeros((self.sensory_dim, *sample_key.shape[-2:]), device=sample_key.device)

    def update_sensory(self, sensory, ids: list[int]) -> None:
        for i, o in enumerate(ids):
            self.sensory[o] = sensory[0, i]

    def get_sensory(self, ids: list[int]) -> torch.Tensor:
        return torch.stack([self.sensory[o] for o in ids])[None]
