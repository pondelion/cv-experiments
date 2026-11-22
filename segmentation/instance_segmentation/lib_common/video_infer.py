"""動画インスタンスセグメンテーションの推論の後処理(方式によらず共通): フレームごとの予測を、評価の予測の形にする。

入力は 1 本の動画の、クエリごと・フレームごとの予測(クエリの番号が物体の ID になる online の方式の出力。VidEoMT・MinVIS など):
    logits: (T, Q, K+1)(フレームごとのクラスの値。最後は「物体なし」)
    masks:  (Q, T, h, w)(フレームごとのマスクの値。sigmoid の前。入力の 1/4 の大きさ)
手順(Mask2Former-VIS・MinVIS の公式と同じ考え方):
    1. クエリごとに、クラスの値(softmax の前)をフレームで平均し、softmax をかけて「物体なし」を除く(公式の VidEoMT・MinVIS と同じ順)
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
    prob = logits.float().mean(0).softmax(-1)[:, :-1]  # (T, Q, K+1) -> (Q, K+1) -> (Q, K)
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


class MinVIS(torch.nn.Module):
    """MinVIS(Huang ら NeurIPS 2022)の推論: 画像の Mask2Former をフレームごとにかけ、隣のフレームのクエリを照合して ID をつなぐ。

    MOT(動画全体での物体の ID の管理)の形: 推論の照合。クエリの枠 = ID。フレーム t のクエリを、1 つ前のフレームの(並べ替えた後の)
    クエリと、最後の decoder 層のクエリの埋め込み(LayerNorm の後)の cos の近さでハンガリアン法で 1 対 1 に組にし、前のフレームの
    並びにそろえる(公式の NVlabs/MinVIS の match_from_embds・post_processing と同じ規則)。全部のクエリを毎フレーム組にするので、
    隠れている物体の枠は「物体なし」のクエリとつながり、再び写ったときに埋め込みが近ければ同じ枠に戻る(戻す規則があるわけではない)。
    新しい物体は、前のフレームで「物体なし」だった枠に入る。区別できる物体はクエリの数まで。学習は画像の学習だけ(動画の正解は使わない)。

    Args:
        model: 画像の Mask2Former(forward の最後の要素に "embeds" (B, Q, C) を持つ)。
        chunk: まとめて処理するフレームの数(GPU のメモリとの兼ね合い)。
    """

    def __init__(self, model: torch.nn.Module, chunk: int = 8):
        super().__init__()
        self.model, self.chunk = model, chunk

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """VidEoMT.predict_video と同じ形で返す。

        Args:
            frames: (T, 3, S, S)。

        Returns:
            (並べ替えた後のフレームごとのクラスの値 (T, Q, K+1), マスクの値 (Q, T, S/4, S/4))。
        """
        from scipy.optimize import linear_sum_assignment
        logits, masks, embeds = [], [], []
        for s in range(0, len(frames), self.chunk):
            out = self.model(frames[s: s + self.chunk])[-1]
            logits.append(out["logits"]); masks.append(out["masks"]); embeds.append(out["embeds"])
        logits, masks, embeds = torch.cat(logits), torch.cat(masks), torch.cat(embeds)  # (T, Q, K+1), (T, Q, h, w), (T, Q, C)
        emb = F.normalize(embeds.float(), dim=-1)
        order = [torch.arange(emb.shape[1], device=emb.device)]
        for t in range(1, len(emb)):
            prev = emb[t - 1][order[-1]]  # 1 つ前のフレームの、並べ替えた後の埋め込み (Q, C)
            cost = (1 - emb[t] @ prev.T).float().cpu().numpy()  # (今のクエリ, 前の枠) の 1 − cos(autocast の中でも float にする)
            # (前の枠, 今のクエリ) の費用で割り当てる。正方形なので、返り値の 1 つ目は前の枠の 0〜Q−1 の順、2 つ目がその枠に入る今のクエリ
            _, cur = linear_sum_assignment(cost.T)
            order.append(torch.as_tensor(cur, device=emb.device))
        idx = torch.stack(order)  # (T, Q) フレーム t の枠 q に入る今のクエリの番号
        logits = torch.gather(logits, 1, idx[..., None].expand_as(logits))
        masks = masks[torch.arange(len(idx), device=idx.device)[:, None], idx]  # (T, Q, h, w)
        return logits, masks.transpose(0, 1)  # (T, Q, K+1), (Q, T, h, w)
