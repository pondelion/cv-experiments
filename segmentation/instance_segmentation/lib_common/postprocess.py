"""クエリ方式のモデルの出力を、COCO の結果の形式の予測にする。

to_predictions: Mask2Former 系(Mask2Former の公式と同じ手順)。rfdetr_to_predictions: RF-DETR 系(RF-DETR の公式と同じ手順)。
yolo_to_predictions: YOLO 系(1 対 1 の head の上位をそのまま使う。NMS なし)。

to_predictions の手順:

1. クラスの確率(物体なしを除く)を Q × K 個並べ、上位 TOP_K 個(クエリ, クラス)を選ぶ
2. 選んだクエリのマスクを入力の大きさに戻し、評価の変換で埋めた部分を除いて、元の画像の大きさに戻す
3. スコア = クラスの確率 × マスクの確からしさ(マスクの内側の画素の確率の平均)

形の記号: B = 画像の数、Q = クエリの数、K = クラスの数、h・w = 予測のマスクの縦・横、S = 入力の一辺、
         H・W = 元の画像の縦・横、R = 選んだ予測の数(TOP_K)
"""
import torch
import torch.nn.functional as F
from einops import rearrange

from lib_common.masks import mask_to_rle

TOP_K = 100


@torch.no_grad()
def to_predictions(logits: torch.Tensor, masks: torch.Tensor, metas: list[dict], size: int,
                   quality: torch.Tensor | None = None) -> list[dict]:
    """モデルの最終の予測を、COCO の結果の形式の予測の並びにする。

    Args:
        logits: クラスの値 (B(画像の数), Q(クエリの数), K+1(クラスの数 + 物体なし))。
        masks: マスクの値 (B, Q, h, w)。
        metas: 画像ごとの image_id・orig_size(data.collate の meta)。画像は transforms.eval_transforms で変換したもの。
        size: 入力の一辺の画素数 S。
        quality: マスクの質の値 (B, Q)(sigmoid の前。iFAN)。あれば、クラスの確率の代わりに (質 × クラスの確率)^2 を使う
            (iFAN の推論の点数。γ = 2)。

    Returns:
        予測の並び。各要素は {"image_id", "category_id"(1 始まり), "segmentation"(RLE), "score"}。
    """
    preds = []
    for i, meta in enumerate(metas):
        prob = logits[i].float().softmax(-1)[:, :-1]  # (Q, K+1) -> (Q, K) 物体なしを除く
        if quality is not None:
            prob = (quality[i].float().sigmoid()[:, None] * prob) ** 2  # (Q, K)
        k = prob.shape[1]
        scores, idx = rearrange(prob, "q k -> (q k)").topk(min(TOP_K, prob.numel()))  # (Q × K,) から上位 R 個
        qi, labels = idx // k, idx % k  # (R,) どのクエリの、どのクラスか
        m = rearrange(masks[i, qi].float(), "r h w -> 1 r h w")  # interpolate は (まとめ, チャンネル, 縦, 横) を求める
        m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)  # (1, R, h, w) -> (1, R, S, S)
        h, w = meta["orig_size"]
        scale = size / max(h, w)  # 評価の変換(transforms.eval_transforms)で長い辺を S にそろえた倍率
        rh, rw = round(h * scale), round(w * scale)  # 左上に寄せた画像の部分の大きさ
        # 埋めた部分を除いて元の大きさに: (1, R, rh, rw) -> (1, R, H, W) -> (R, H, W)
        m = rearrange(F.interpolate(m[:, :, :rh, :rw], size=(h, w), mode="bilinear", align_corners=False),
                      "1 r h w -> r h w")
        binary = m > 0  # (R, H, W)
        inside = rearrange(m.sigmoid() * binary, "r h w -> r (h w)").sum(1)  # (R,) マスクの内側の確率の和
        mask_score = inside / rearrange(binary, "r h w -> r (h w)").sum(1).clamp(min=1)  # (R,)
        final = scores * mask_score  # (R,)
        for b, s, c in zip(binary.cpu().numpy(), final.tolist(), labels.tolist()):
            if b.any():
                preds.append({"image_id": meta["image_id"], "category_id": c + 1, "segmentation": mask_to_rle(b), "score": s})
    return preds


@torch.no_grad()
def maskrcnn_to_predictions(outs: list[dict], metas: list[dict], size: int) -> list[dict]:
    """Mask R-CNN(models/maskrcnn.py。torchvision の後処理で NMS・上位 100 個まで済み)の予測を、COCO の結果の形式にする。

    マスクの確率(入力の大きさ)の、埋めた部分を除いて元の画像の大きさに戻し、0.5 より大きい所を物体にする(detectron2・torchvision と
    同じ閾値)。スコアは枠のスコア。

    Args:
        outs: 画像ごとの {"boxes": (N, 4) 入力の画素の x0, y0, x1, y1, "labels": (N,)(1 始まり), "scores": (N,),
            "masks": (N, 1, S, S) の確率}。
        metas: 画像ごとの image_id・orig_size(画像は transforms.eval_transforms で変換したもの)。
        size: 入力の一辺 S。

    Returns:
        予測の並び。各要素は {"image_id", "category_id", "segmentation", "score", "bbox"}。
    """
    preds = []
    for o, meta in zip(outs, metas):
        h, w = meta["orig_size"]
        scale = size / max(h, w)
        rh, rw = round(h * scale), round(w * scale)
        m = o["masks"][:, :, :rh, :rw].float()  # (N, 1, rh, rw) 埋めた部分を除く
        binary = rearrange(F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False), "n 1 h w -> n h w") > 0.5
        xywh = o["boxes"].float() / scale
        xywh[:, 2:] -= xywh[:, :2]  # x0, y0, x1, y1 -> x, y, 幅, 高さ
        for b, s, c, bb in zip(binary.cpu().numpy(), o["scores"].tolist(), o["labels"].tolist(), xywh.tolist()):
            if b.any():
                preds.append({"image_id": meta["image_id"], "category_id": c, "segmentation": mask_to_rle(b),
                              "score": s, "bbox": bb})
    return preds


@torch.no_grad()
def yolo_to_predictions(logits: torch.Tensor, boxes: torch.Tensor, coef: torch.Tensor, protos: torch.Tensor,
                        metas: list[dict], size: int) -> list[dict]:
    """YOLO 系(o2o の head、NMS なし)の予測を、COCO の結果の形式にする。

    1. クラスの確率(sigmoid)を A × K 個並べ、上位 TOP_K 個(位置, クラス)を選ぶ。スコア = その確率
    2. 係数 × 原型のマスクを入力の大きさに拡大し、予測の枠の外を消して、値 > 0 を物体にする
    3. 埋めた部分を除いて元の画像の大きさに戻す(入力は transforms.eval_transforms)

    Args:
        logits: (B, A, K)。
        boxes: (B, A, 4) 入力の画素の x0, y0, x1, y1。
        coef: (B, A, M)。
        protos: (B, M, h, w)。
        metas: 画像ごとの image_id・orig_size。
        size: 入力の一辺 S。

    Returns:
        予測の並び。各要素は {"image_id", "category_id", "segmentation", "score", "bbox"}。
    """
    from segmentation.instance_segmentation.lib_common.losses.yolo_criterion import box_crop_mask
    preds = []
    for i, meta in enumerate(metas):
        prob = logits[i].float().sigmoid()  # (A, K)
        k = prob.shape[1]
        scores, idx = rearrange(prob, "a k -> (a k)").topk(min(TOP_K, prob.numel()))
        ai, labels = idx // k, idx % k  # (R,)
        bx = boxes[i, ai].float().clamp(0, size)  # (R, 4)
        m = torch.einsum("rm,mhw->rhw", coef[i, ai].float(), protos[i].float())  # (R, h, w)
        m = rearrange(F.interpolate(rearrange(m, "r h w -> 1 r h w"), size=(size, size), mode="bilinear",
                                    align_corners=False), "1 r h w -> r h w")  # -> (R, S, S)
        m = torch.where(box_crop_mask(bx, (size, size)) > 0, m, torch.full_like(m, -1.0))  # 枠の外を消す
        h, w = meta["orig_size"]
        scale = size / max(h, w)
        rh, rw = round(h * scale), round(w * scale)
        m = F.interpolate(rearrange(m[:, :rh, :rw], "r h w -> 1 r h w"), size=(h, w), mode="bilinear",
                          align_corners=False)  # (1, R, rh, rw) -> (1, R, H, W)
        binary = rearrange(m, "1 r h w -> r h w") > 0
        xywh = bx / scale
        xywh[:, 2:] -= xywh[:, :2]  # x0, y0, x1, y1 -> x, y, 幅, 高さ
        for b, s, c, bb in zip(binary.cpu().numpy(), scores.tolist(), labels.tolist(), xywh.tolist()):
            preds.append({"image_id": meta["image_id"], "category_id": c + 1, "segmentation": mask_to_rle(b),
                          "score": s, "bbox": bb})
    return preds


@torch.no_grad()
def rfdetr_to_predictions(logits: torch.Tensor, boxes: torch.Tensor, masks: torch.Tensor, metas: list[dict],
                          pad_size: int | None = None, crop_to_box: bool = False) -> list[dict]:
    """RF-DETR 系(クラスが sigmoid)の最終の予測を、COCO の結果の形式にする(公式の PostProcess と同じ手順)。

    1. クラスの確率(sigmoid)を Q × K 個並べ、上位 TOP_K 個(クエリ, クラス)を選ぶ。スコア = その確率
    2. 選んだクエリのマスクを元の画像の大きさに拡大縮小し(入力は transforms.square_eval_transforms の正方形)、値 > 0 を物体にする
    3. 枠(0〜1 の cx, cy, w, h)を元の画像の画素の x, y, 幅, 高さにする

    Args:
        logits: クラスの値 (B, Q, K)。
        boxes: 枠 (B, Q, 4)。0〜1 の cx, cy, w, h。
        masks: マスクの値 (B, Q, h, w)。
        metas: 画像ごとの image_id・orig_size(data.collate の meta)。
        pad_size: None なら入力は transforms.square_eval_transforms の正方形。数なら transforms.eval_transforms
            (長い辺をこの S にして右と下を埋めた入力。GTR 系など)。
        crop_to_box: True なら、各マスクの予測の枠の外を 0 にする(D-FINE-seg の cleanup_masks と同じ。枠の中だけでマスクの
            損失を掛ける学習(GTR 系の mask_loss="box_crop")では、枠の外を学習しないため)。

    Returns:
        予測の並び。各要素は {"image_id", "category_id"(1 始まり), "segmentation"(RLE), "score", "bbox"}。
    """
    preds = []
    for i, meta in enumerate(metas):
        prob = logits[i].float().sigmoid()  # (Q, K)
        k = prob.shape[1]
        scores, idx = rearrange(prob, "q k -> (q k)").topk(min(TOP_K, prob.numel()))  # (Q × K,) から上位 R 個
        qi, labels = idx // k, idx % k  # (R,)
        h, w = meta["orig_size"]
        m = rearrange(masks[i, qi].float(), "r h w -> 1 r h w")
        if pad_size is None:  # 正方形に拡大縮小した入力: そのまま元の大きさへ
            m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)  # (1, R, h, w) -> (1, R, H, W)
            xywh = boxes[i, qi].float() * torch.tensor([w, h, w, h], device=boxes.device)  # (R, 4) 画素の cx, cy, w, h
        else:  # 長い辺を S にして埋めた入力: S × S に戻し、埋めた部分を除いてから元の大きさへ
            scale = pad_size / max(h, w)
            rh, rw = round(h * scale), round(w * scale)
            m = F.interpolate(m, size=(pad_size, pad_size), mode="bilinear", align_corners=False)[:, :, :rh, :rw]
            m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)  # (1, R, rh, rw) -> (1, R, H, W)
            xywh = boxes[i, qi].float() * pad_size / scale  # 0〜1(S × S の中)→ 元の画像の画素
        binary = rearrange(m, "1 r h w -> r h w") > 0  # (R, H, W)
        xywh[:, 2:] = xywh[:, 2:].clamp(min=0)  # 幅・高さが負の枠(GTR 系で起こりうる)は 0 にする
        xywh[:, :2] -= xywh[:, 2:] / 2  # 中心 → 左上
        if crop_to_box:  # 枠(x0 ≤ x < x1、y0 ≤ y < y1)の外を 0 にする: (R, H, W)
            xs = torch.arange(w, device=binary.device)[None, None, :]
            ys = torch.arange(h, device=binary.device)[None, :, None]
            x0, y0 = xywh[:, 0, None, None], xywh[:, 1, None, None]
            x1, y1 = x0 + xywh[:, 2, None, None], y0 + xywh[:, 3, None, None]
            binary = binary & (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        for b, s, c, bb in zip(binary.cpu().numpy(), scores.tolist(), labels.tolist(), xywh.tolist()):
            preds.append({"image_id": meta["image_id"], "category_id": c + 1, "segmentation": mask_to_rle(b),
                          "score": s, "bbox": bb})
    return preds
