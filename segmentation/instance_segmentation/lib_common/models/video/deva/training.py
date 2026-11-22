"""DEVA の伝搬のネットワークの学習(静止画の段階 = 公式の stage 0 と、動画の段階 = 公式の stage 3)。exp00502。
静止画の段階のデータは static_data.py。forward・損失は両方の段階で同じ(物体の数・参照のフレームの数だけが違う)。

参考: 公式のコード(hkchengrex/Tracking-Anything-with-DEVA の `deva/model/trainer.py` の do_pass、`deva/model/losses.py`、
`deva/dataset/vos_dataset.py` の __getitem__、`deva/train.py` の skip の切り替え、`deva/utils/configuration.py` の既定)。
コードは写さず、規則を読み取って書いた。

1 サンプル(公式と同じ): 動画から 8 フレームを選ぶ(最初の 1 枚を無作為に選び、選んだどれかのフレームから max_jump 以内のフレームを
重ならないように足していく。時刻の順に並べ、確率 0.5 で逆順)。最初のフレームに写っている物体から最大 3 個を無作為に選ぶ
(いなければ 5 回まで選び直す)。
forward(公式の do_pass): 最初のフレームの正解のマスクを記憶に入れ、2 フレーム目から順に、記憶(最初の 3 フレームまでは全部、
それより後は無作為な 3 フレーム)を読み出してマスクを出し、最後のフレーム以外は出したマスクを記憶に足す(感覚の記憶の deep update は
確率 0.2)。
損失(公式と同じ): フレームごと・サンプルごとに、背景つきの交差エントロピー(bootstrapped: 10,000〜35,000 step で、難しい画素の
割合を 100% から 30% へ。それより後は 30%)+ 補助の予測の交差エントロピー × 0.1 + Dice。
max_jump の切り替え(公式): 5 → 学習の 10% で 10 → 30% で 15 → 80% で 5。公式は DAVIS(全フレーム)に max_jump、注釈が 5 フレーム
おきの YouTube-VOS に max_jump // 5 を使う。VIPSeg の 720P のフレームは元の動画の 3 フレームおき(ファイル名の番号が 3 ずつ)なので、
max_jump // 3 を使う(私の判断)。

形の記号: B = 区間の数、T = フレームの数(8)、N = 物体の数(最大 3)、S = 入力の一辺。
"""
import numpy as np
import torch
import torch.nn.functional as F

from segmentation.instance_segmentation.lib_common.models.video.deva.network import DEVA

NUM_FRAMES, NUM_REF_FRAMES, MAX_NUM_OBJ, DEEP_UPDATE_PROB = 8, 3, 3, 0.2
START_WARM, END_WARM, TOP_P = 10000, 35000, 0.3
SKIP_VALUES, SKIP_FRACTIONS, VIPSEG_SKIP_DIV = (5, 10, 15, 5), (0.1, 0.3, 0.8), 3


def max_jump_at(progress: float) -> int:
    """学習の進み(0〜1)での max_jump(公式の 5 → 10 → 15 → 5 を、VIPSeg のフレームの間隔で割ったもの。最小 1)。"""
    k = sum(progress >= f for f in SKIP_FRACTIONS)
    return max(1, SKIP_VALUES[k] // VIPSEG_SKIP_DIV)


def sample_frames(length: int, max_jump: int, rng: np.random.Generator, num_frames: int = NUM_FRAMES) -> list[int]:
    """公式の VOSDataset のフレームの選び方。動画が num_frames より短いときは全部 + 重ねて足す(公式は短い動画を使わない)。"""
    if length < num_frames:
        idx = sorted(list(range(length)) + rng.choice(length, num_frames - length).tolist())
        return idx[::-1] if rng.random() < 0.5 else idx
    jump = min(length, max_jump)
    idx = [int(rng.integers(length))]
    ok = set(range(max(0, idx[-1] - jump), min(length, idx[-1] + jump + 1))) - set(idx)
    while len(idx) < num_frames:
        i = int(rng.choice(sorted(ok)))
        idx.append(i)
        ok = (ok | set(range(max(0, i - jump), min(length, i + jump + 1)))) - set(idx)
    idx = sorted(idx)
    return idx[::-1] if rng.random() < 0.5 else idx


def select_objects(masks: torch.Tensor, rng: np.random.Generator, max_obj: int = MAX_NUM_OBJ, empty_ok: bool = False):
    """最初のフレームに写っている物体から最大 max_obj 個(公式の stage 3 は 3、stage 0 は 1)。Args: masks (N_all, T, S, S)。

    empty_ok: 公式の stage 0 は、変換で物体が最初のフレームから外れても選び直さず、物体の数 1(正解は全部背景)として学習する。
    Returns: (クラスの地図 (T, S, S)(0 背景、1〜n), 最初のフレームの正解 (max_obj, S, S), selector (max_obj,), n) か
        None(物体なし。empty_ok のときは返さない)。
    """
    t, s = masks.shape[1], masks.shape[-1]
    visible = torch.nonzero(masks[:, 0].flatten(1).any(-1)).squeeze(-1).tolist() if len(masks) else []
    if not visible and not empty_ok:
        return None
    if len(visible) > max_obj:
        visible = rng.choice(visible, max_obj, replace=False).tolist()
    cls_gt = torch.zeros((t, s, s), dtype=torch.long, device=masks.device)
    first = torch.zeros((max_obj, s, s), device=masks.device)
    for i, o in enumerate(visible):
        m = masks[o].bool()
        cls_gt[m] = i + 1  # 公式と同じく、後の物体が上書きする
        first[i] = m[0].float()
    n = max(1, len(visible))  # 公式の num_objects = max(1, 物体の数)
    sel = torch.tensor([1.0 if i < n else 0.0 for i in range(max_obj)], device=masks.device)
    return cls_gt, first, sel, n


def train_forward(net: DEVA, frames: torch.Tensor, first_gt: torch.Tensor, selector: torch.Tensor,
                  num_ref_frames: int = NUM_REF_FRAMES) -> dict:
    """公式の do_pass の forward。Args: frames (B, T, 3, S, S)、first_gt (B, N, S, S)、selector (B, N)、
    num_ref_frames: 読み出す記憶のフレームの数(公式の stage 3 は 3、stage 0 は 2)。"""
    b, t = frames.shape[:2]
    n = first_gt.shape[1]
    sel = selector[:, :, None, None]
    ms, feat = net.encode_image(frames[:, 0])
    k, s, _ = net.transform_key(feat, need_ek=False)
    sensory = torch.zeros((b, n, 512, *k.shape[-2:]), device=frames.device)
    v16, sensory = net.encode_mask(frames[:, 0], ms, sensory, first_gt)
    masks = first_gt
    keys, shrinks, values = k.unsqueeze(2), s.unsqueeze(2), v16.unsqueeze(3)  # (B, CK, T, H, W)、(B, N, CV, T, H, W)
    out = {}
    for ti in range(1, t):
        ms, feat = net.encode_image(frames[:, ti])
        k, s, e = net.transform_key(feat)
        keys, shrinks = torch.cat([keys, k.unsqueeze(2)], 2), torch.cat([shrinks, s.unsqueeze(2)], 2)
        if ti <= num_ref_frames:
            rv, rk, rs = values, keys[:, :, :ti], shrinks[:, :, :ti]
        else:
            ind = [torch.randperm(ti)[:num_ref_frames] for _ in range(b)]
            rv = torch.stack([values[i, :, :, ind[i]] for i in range(b)])
            rk = torch.stack([keys[i, :, ind[i]] for i in range(b)])
            rs = torch.stack([shrinks[i, :, ind[i]] for i in range(b)])
        readout = net.read_memory(k, e, rk, rs, rv)
        sensory, logits, masks, aux_logits, _ = net.segment(ms, readout, sensory, masks, selector=sel, need_aux=True)
        masks = masks[:, 1:]
        if ti < t - 1:
            deep = np.random.rand() < DEEP_UPDATE_PROB
            v16, sensory = net.encode_mask(frames[:, ti], ms, sensory, masks, is_deep_update=deep)
            values = torch.cat([values, v16.unsqueeze(3)], 3)
        out[f"masks_{ti}"], out[f"logits_{ti}"], out[f"aux_logits_{ti}"] = masks, logits, aux_logits
    return out


def _bootstrapped_ce(logits, target, it: int):
    if it < START_WARM:
        return F.cross_entropy(logits, target)
    raw = F.cross_entropy(logits, target, reduction="none").flatten()
    p = TOP_P if it > END_WARM else TOP_P + (1 - TOP_P) * ((END_WARM - it) / (END_WARM - START_WARM))
    loss, _ = torch.topk(raw, int(raw.numel() * p), sorted=False)
    return loss.mean()


def deva_loss(out: dict, cls_gt: torch.Tensor, num_objects: list[int], it: int) -> tuple[torch.Tensor, dict]:
    """公式の LossComputer.compute。Args: cls_gt (B, T, S, S)、num_objects: 区間ごとの物体の数。"""
    b, t = cls_gt.shape[:2]
    total, parts = 0.0, {"ce": 0.0, "aux": 0.0, "dice": 0.0}
    for ti in range(1, t):
        ce = aux = 0.0
        for bi in range(b):
            n = num_objects[bi]
            ce = ce + _bootstrapped_ce(out[f"logits_{ti}"][bi: bi + 1, : n + 1].float(), cls_gt[bi: bi + 1, ti], it) / b
            aux = aux + F.cross_entropy(out[f"aux_logits_{ti}"][bi: bi + 1, : n + 1, 0].float(), cls_gt[bi: bi + 1, ti]) / b
        m = out[f"masks_{ti}"]  # (B, N, S, S)(背景を除いた確率)
        dl = []
        for i in range(m.shape[1]):
            p, g = m[:, i].flatten(1).float(), (cls_gt[:, ti] == i + 1).float().flatten(1)
            dl.append(1 - (2 * (p * g).sum(-1) + 1) / (p.sum(-1) + g.sum(-1) + 1))
        dice = torch.cat(dl).mean()
        total = total + ce + 0.1 * aux + dice
        parts["ce"] += float(ce.detach()); parts["aux"] += float(aux.detach()); parts["dice"] += float(dice.detach())
    return total, parts
