"""動画インスタンスセグメンテーションの推論の後処理(方式によらず共通): フレームごとの予測を、評価の予測の形にする。

入力は 1 本の動画の、クエリごと・フレームごとの予測(クエリの番号が物体の ID になる online の方式の出力。VidEoMT・MinVIS など):
    logits: (T, Q, K+1)(フレームごとのクラスの値。最後は「物体なし」)
    masks:  (Q, T, h, w)(フレームごとのマスクの値。sigmoid の前。入力の 1/4 の大きさ)
手順(Mask2Former-VIS・MinVIS の公式と同じ考え方):
    1. クエリごとに、クラスの確率(softmax、「物体なし」を除く)をフレームで平均する
    2. (クエリ, クラス) の組を確率の高い順に top_k 個選ぶ。点数 = その確率
    3. 選んだクエリのマスクを入力の大きさに拡大し、画像の部分を切り出して元の大きさに拡大し、値 > 0 を物体にする(画像の段階の
       後処理と同じ)。フレームごとに RLE にする(物体のないフレームは None)

形の記号: T = フレームの数、Q = クエリの数、K = クラスの数、h・w = 予測のマスクの縦・横、S = 入力の一辺、H・W = 元の大きさ。
"""
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util


@torch.no_grad()
def video_predictions(logits: torch.Tensor, masks: torch.Tensor, video_id: int, cat_ids: list[int], size: int,
                      valid_size: tuple[int, int], orig_size: tuple[int, int], top_k: int = 30) -> list[dict]:
    """1 本の動画の予測を、評価の予測の形にする。

    Args:
        logits: (T, Q, K+1)。
        masks: (Q, T, h, w)。
        video_id: 動画の id。
        cat_ids: 0 始まりのクラスの番号 → カテゴリーの id。
        size: 入力の一辺 S。
        valid_size: 入力の中の画像の部分の (縦, 横)(残りは埋めた部分)。
        orig_size: 元の (H, W)。
        top_k: 動画ごとに残す (クエリ, クラス) の組の数。

    Returns:
        [{"video_id", "category_id", "score", "segmentations": [フレームごとの RLE または None], "query"}]。
    """
    prob = logits.float().softmax(-1)[:, :, :-1].mean(0)  # (T, Q, K+1) -> (Q, K)
    q, k = prob.shape
    scores, idx = prob.flatten().topk(min(top_k, q * k))
    preds = []
    for score, i in zip(scores.tolist(), idx.tolist()):
        qi, ki = divmod(i, k)
        m = masks[qi].float()[None]  # (1, T, h, w)
        m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)[:, :, : valid_size[0], : valid_size[1]]
        m = F.interpolate(m, size=orig_size, mode="bilinear", align_corners=False)[0] > 0  # (T, H, W)
        segs = []
        for frame in m.cpu().numpy():
            if frame.any():
                r = mask_util.encode(np.asfortranarray(frame))
                r["counts"] = r["counts"].decode("ascii")
                segs.append(r)
            else:
                segs.append(None)
        preds.append({"video_id": video_id, "category_id": cat_ids[ki], "score": score, "segmentations": segs,
                      "query": qi})  # query: クエリの番号(= 動画を通した物体の ID。見本の色分けに使う。評価では使わない)
    return preds
