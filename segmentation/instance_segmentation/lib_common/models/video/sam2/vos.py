"""SAM 2 の伝搬だけの推論(マスクを与えて追う。J&F の評価用)。exp00502。

各物体は、初めて写ったフレームで正解のマスクをプロンプト(条件のフレーム)として与え、そこから動画の最後まで時刻の順に伝搬する
(公式の SAM2VideoPredictor の add_new_mask → propagate_in_video と同じ順。物体ごとに記憶を持つ)。同じフレームから始まる物体は
まとめて処理する。画像の特徴は動画の全フレームで 1 回だけ計算する。

形の記号: T = フレームの数、O = 物体の数、S = 入力の一辺、H・W = 元の大きさ。
"""
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

from segmentation.instance_segmentation.lib_common.models.video.sam2.model import SAM2


@torch.no_grad()
def propagate(model: SAM2, frames: torch.Tensor, prompts: dict[int, torch.Tensor], chunk: int = 8) -> dict[int, torch.Tensor]:
    """マスクを与えた物体を伝搬する。

    Args:
        model: SAM2(eval)。
        frames: (T, 3, S, S)。
        prompts: {始めるフレーム: その時刻に与える物体のマスク (O, S, S) の bool}。
        chunk: 画像の encoder にまとめて通すフレームの数。

    Returns:
        {始めるフレーム: 高解像度のマスクの値 (T, O, S, S)(始めるフレームより前は −1024)}。
    """
    t_all = len(frames)
    fpn, pos = None, None
    for s in range(0, t_all, chunk):  # 画像の特徴を全フレーム分(stride 4・8・16)
        f, p = model.forward_image(frames[s: s + chunk])
        fpn = f if fpn is None else [torch.cat([a, b]) for a, b in zip(fpn, f)]
        pos = p if pos is None else [torch.cat([a, b]) for a, b in zip(pos, p)]
    out = {}
    for start, masks in prompts.items():
        n = masks.shape[0]
        res = torch.full((t_all, n, *frames.shape[-2:]), -1024.0, device=frames.device)
        out_dict = {"cond": {}, "non_cond": {}}
        for t in range(start, t_all):
            feats = [f[t: t + 1].expand(n, -1, -1, -1) for f in fpn]
            ps = [p[t: t + 1].expand(n, -1, -1, -1) for p in pos]
            is_cond = t == start
            cur = model.track_step(t, is_cond, feats, ps, None, masks[:, None].float() if is_cond else None, out_dict, t_all)
            (out_dict["cond"] if is_cond else out_dict["non_cond"])[t] = cur
            res[t] = cur["pred_masks_high_res"][:, 0].float()
        out[start] = res
    return out


def to_input_mask(rle: dict, valid: tuple[int, int], size: int, device) -> torch.Tensor:
    """元の大きさの RLE を、評価の変換(長い辺を S に縮めて右と下を埋める)と同じ形のマスクにする。

    Args:
        rle: 元の大きさの RLE。
        valid: 入力の中の画像の部分 (縦, 横)。
        size: 入力の一辺 S。
        device: 作る device。

    Returns:
        (S, S) の bool。
    """
    m = torch.as_tensor(mask_util.decode(rle), device=device)[None, None].float()
    m = F.interpolate(m, size=valid, mode="nearest")[0, 0] > 0.5
    out = torch.zeros(size, size, dtype=torch.bool, device=device)
    out[: valid[0], : valid[1]] = m
    return out


def to_orig_rles(logits: torch.Tensor, valid: tuple[int, int], orig: tuple[int, int]) -> list:
    """入力の大きさのマスクの値を、元の大きさのフレームごとの RLE にする。

    Args:
        logits: マスクの値 (T, S, S)。
        valid: 入力の中の画像の部分 (縦, 横)。
        orig: 元の大きさ (縦, 横)。

    Returns:
        フレームごとの RLE(物体がなければ None)の並び。
    """
    m = logits[:, None, : valid[0], : valid[1]]
    m = F.interpolate(m, size=orig, mode="bilinear", align_corners=False)[:, 0] > 0
    out = []
    for f in m.cpu().numpy():
        if f.any():
            r = mask_util.encode(np.asfortranarray(f))
            r["counts"] = r["counts"].decode("ascii")
            out.append(r)
        else:
            out.append(None)
    return out
