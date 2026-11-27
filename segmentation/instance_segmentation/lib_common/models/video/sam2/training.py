"""SAM 2 の学習の手順と損失。exp00502。

参考: 公式のコード(facebookresearch/sam2 の `training/model/sam2.py` の SAM2Train(prepare_prompt_inputs・forward_tracking・
track_step・_iter_correct_pt_sampling)、`training/loss_fns.py` の MultiStepMultiMasksAndIous、`training/dataset/vos_sampler.py` の
RandomUniformSampler、設定 `sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`)。コードは写さず、規則を読み取って書いた。

学習の 1 サンプル(公式と同じ): 動画から連続した 8 フレーム、最初のフレームに写っている物体から最大 3 個を無作為に選ぶ。
    - プロンプト: 確率 0.5 で点(そのうち確率 0.5 で箱、ほかは誤りの領域から一様に 1 点)、それ以外は正解のマスク
    - 条件のフレーム(プロンプトを与えるフレーム): 最初のフレーム + 確率で 1 枚(1〜2 枚を一様に)
    - 修正の点(点のプロンプトのときだけ): 条件のフレームを含む 1〜2 枚(一様に)に、予測の誤りの領域から 1 点ずつ 7 回足す
      (確率 0.1 で、予測ではなく正解の中から選ぶ)。修正の点を足したフレームは条件のフレームになる(add_all_frames_to_correct_as_cond)
    - 処理の順: 条件のフレーム → 残りのフレームを時刻の順
損失(公式と同じ。重み mask 20・dice 1・iou 1・class 1): 各フレームの各段(最初 + 修正の点ごと)で、
    - マスク: focal(α 0.25・γ 2、画素の平均)と Dice。複数のマスク(3)を出した段は、focal × 20 + Dice が最小のマスクだけに損失
    - IoU の予測: 予測のマスク(> 0)と正解の実際の IoU との L1。複数のマスクは全部の平均(supervise_all_iou)
    - 物体らしさ: その物体がフレームに写っているかの BCE(focal の α −1・γ 0)
    - マスク・Dice・IoU は、物体が写っているフレームだけ(写っていないフレームは物体らしさだけ)。物体の数で割る

形の記号: T = フレームの数、O = 物体の数、S = 入力の一辺、M = マスクの候補の数(3 か 1)。
"""
import numpy as np
import torch
import torch.nn.functional as F

from segmentation.instance_segmentation.lib_common.models.video.sam2.model import SAM2, sample_box_points, sample_points_from_errors

# 公式の SAM2Train の学習の設定(MOSE の fine-tune の設定。SA-V の学習と同じ値)
PROB_PT, PROB_BOX, PROB_GT_PT = 0.5, 0.5, 0.1
NUM_FRAMES_TO_CORRECT, NUM_INIT_COND = 2, 2
WEIGHTS = {"mask": 20.0, "dice": 1.0, "iou": 1.0, "class": 1.0}
NUM_FRAMES, MAX_OBJECTS = 8, 3


def train_forward(model: SAM2, frames: torch.Tensor, gt: torch.Tensor, rng: np.random.Generator,
                  obj_clip: torch.Tensor | None = None) -> list[dict]:
    """区間の学習の forward(公式の SAM2Train.forward)。複数の区間の物体をまとめて 1 回で処理できる。

    まとめるとき(公式の batch と同じ扱い): 全部の区間の物体を物体の向きに並べ、各物体は自分の区間のフレームの特徴を使う
    (公式の flat_obj_to_img_idx)。プロンプトの種類・条件のフレーム・修正の点を足すフレームは、公式の prepare_prompt_inputs と同じく
    1 回だけ選んで全部の物体で共有する。

    Args:
        model: SAM2(train のまま)。
        frames: (T, 3, S, S)(区間 1 本)か (B, T, 3, S, S)(B 本をまとめる)。
        gt: 選んだ物体の正解のマスク (T, O, S, S) の bool(まとめるときは全部の区間の物体を並べたもの)。
        rng: 乱数(プロンプトの選び方)。
        obj_clip: 物体ごとの区間の番号 (O,)(まとめるとき)。None なら区間 1 本。

    Returns:
        フレームごとの出力(track_step の返り値)。
    """
    t_all, n_obj = gt.shape[:2]
    if frames.dim() == 4:
        frames = frames[None]
    # 画像の encoder は区間ごとに通す(backbone の BatchNorm が学習中は batch の統計を使うので、区間 1 本ずつのときと同じ統計にする)。
    # 並びは (区間, フレーム) の順 (B·T, ·, ·, ·)
    per_clip = [model.forward_image(f) for f in frames]
    fpn = [torch.cat([c[0][lv] for c in per_clip]) for lv in range(len(per_clip[0][0]))]
    pos = [torch.cat([c[1][lv] for c in per_clip]) for lv in range(len(per_clip[0][1]))]
    if obj_clip is None:
        obj_clip = torch.zeros(n_obj, dtype=torch.long, device=gt.device)
    obj_clip = obj_clip.to(gt.device)
    use_pt = rng.random() < PROB_PT
    n_init = int(rng.integers(1, NUM_INIT_COND, endpoint=True)) if NUM_INIT_COND > 1 else 1  # rand_init_cond_frames
    n_correct = NUM_FRAMES_TO_CORRECT
    if use_pt and n_correct > n_init:  # rand_frames_to_correct
        n_correct = int(rng.integers(n_init, n_correct, endpoint=True))
    init = [0] + (rng.choice(range(1, t_all), n_init - 1, replace=False).tolist() if n_init > 1 else [])
    rest = [t for t in range(t_all) if t not in init]
    masks_in, points_in = {}, {}
    for t in init:
        g = gt[t][:, None]  # (O, 1, S, S)
        if not use_pt:
            masks_in[t] = g
        elif rng.random() < PROB_BOX:
            p, l = sample_box_points(g)
            points_in[t] = {"point_coords": p, "point_labels": l}
        else:
            p, l = sample_points_from_errors(g, None)
            points_in[t] = {"point_coords": p, "point_labels": l}
    if not use_pt:
        to_correct = []
    elif n_correct == n_init:
        to_correct = init
    else:
        to_correct = init + rng.choice(rest, n_correct - n_init, replace=False).tolist()
    out_dict, outs = {"cond": {}, "non_cond": {}}, {}
    for t in init + rest:
        img = obj_clip * t_all + t  # (O,) 物体ごとに、自分の区間のこのフレームの特徴の番号(公式の flat_obj_to_img_idx)
        feats = [f[img] for f in fpn]
        ps = [p[img] for p in pos]
        cur = model.track_step(t, t in init, feats, ps, points_in.get(t), masks_in.get(t), out_dict, t_all,
                               gt_masks=gt[t][:, None], correct=t in to_correct, rng=rng, prob_gt_pt=PROB_GT_PT)
        (out_dict["cond"] if (t in init or t in to_correct) else out_dict["non_cond"])[t] = cur
        outs[t] = cur
    return [outs[t] for t in range(t_all)]


def _focal(x, y, alpha: float, gamma: float):
    """sigmoid の focal loss(要素ごと)。

    Args:
        x: 値(logit)。
        y: 正解(0/1)。
        alpha: 正例の重み(0 未満なら重みなし)。
        gamma: 当てやすい要素を軽くする指数。

    Returns:
        x と同じ形の損失。
    """
    p = x.sigmoid()
    ce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    loss = ce * (1 - (p * y + (1 - p) * (1 - y))) ** gamma
    if alpha >= 0:
        loss = (alpha * y + (1 - alpha) * (1 - y)) * loss
    return loss


def sam2_loss(outs: list[dict], gt: torch.Tensor, obj_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """SAM 2 の損失(公式の MultiStepMultiMasksAndIous。ファイルの先頭に書いた損失)。

    Args:
        outs: train_forward の返り値(フレームごとの出力)。
        gt: 正解のマスク (T, O, S, S)。
        obj_weight: 物体ごとの重み (O,)。None なら 1 / O(区間 1 本の公式の割り方)。区間をまとめたときは
            1 /(その物体の区間の物体の数 × 区間の数)にすると、区間ごとの損失の平均と同じ値になる。

    Returns:
        (損失の合計, 内訳 {"mask", "dice", "iou", "class"})。
    """
    n_obj = max(gt.shape[1], 1)
    w = torch.full((gt.shape[1],), 1.0 / n_obj, device=gt.device) if obj_weight is None else obj_weight.to(gt.device)
    w = w[:, None]  # (O, 1) 物体ごとの割る数の代わり
    sums = {"mask": 0.0, "dice": 0.0, "iou": 0.0, "class": 0.0}
    for o, g in zip(outs, gt):
        target = g[:, None].float()  # (O, 1, S, S)
        present = target.flatten(1).any(-1, keepdim=True).float()  # (O, 1)
        for masks, ious, obj in zip(o["multistep_pred_multimasks_high_res"], o["multistep_pred_ious"],
                                    o["multistep_object_score_logits"]):
            tgt = target.expand_as(masks)
            l_mask = _focal(masks, tgt, 0.25, 2.0).flatten(2).mean(-1) * w  # (O, M)
            pr = masks.flatten(2).sigmoid()
            tg = tgt.flatten(2)
            l_dice = (1 - (2 * (pr * tg).sum(-1) + 1) / (pr.sum(-1) + tg.sum(-1) + 1)) * w  # (O, M)
            pm, gm = masks.flatten(2) > 0, tg > 0
            actual = (pm & gm).sum(-1).float() / torch.clamp((pm | gm).sum(-1).float(), min=1.0)
            l_iou = F.l1_loss(ious.float(), actual, reduction="none") * w  # (O, M)
            l_class = (_focal(obj.float(), present, -1.0, 0.0).mean(1) * w[:, 0]).sum()
            if l_mask.shape[1] > 1:
                best = (l_mask * WEIGHTS["mask"] + l_dice * WEIGHTS["dice"]).argmin(-1)
                ar = torch.arange(len(best), device=best.device)
                l_mask, l_dice = l_mask[ar, best][:, None], l_dice[ar, best][:, None]
                l_iou = l_iou.mean(-1, keepdim=True)
            sums["mask"] = sums["mask"] + (l_mask * present).sum()
            sums["dice"] = sums["dice"] + (l_dice * present).sum()
            sums["iou"] = sums["iou"] + (l_iou * present).sum()
            sums["class"] = sums["class"] + l_class
    total = sum(WEIGHTS[k] * v for k, v in sums.items())
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in sums.items()}


def select_objects(masks: torch.Tensor, rng: np.random.Generator, max_objects: int = MAX_OBJECTS) -> torch.Tensor | None:
    """最初のフレームに写っている物体から最大 max_objects 個を無作為に選ぶ(公式の RandomUniformSampler)。

    Args:
        masks: 区間の正解のマスク (N, T, S, S)。
        rng: 乱数。
        max_objects: 選ぶ物体の数の上限。

    Returns:
        選んだ物体の番号 (O,) か None(最初のフレームに物体がない)。
    """
    visible = torch.nonzero(masks[:, 0].flatten(1).any(-1)).squeeze(-1).tolist()
    if not visible:
        return None
    return torch.as_tensor(sorted(rng.choice(visible, min(len(visible), max_objects), replace=False).tolist()))
