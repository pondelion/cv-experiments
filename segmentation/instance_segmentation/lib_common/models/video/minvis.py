"""MinVIS(Huang ら NeurIPS 2022、"MinVIS: A Minimal Video Instance Segmentation Framework without Video-based Training")。exp00502。

参考: 公式のコード(NVlabs/MinVIS の `minvis/video_maskformer_model.py` の match_from_embds・post_processing)。

MOT(動画全体での物体の ID の管理・識別)の形: **推論の照合**。クエリの枠 = ID。
    - 画像の Mask2Former(フレームを独立に学習したもの。動画の正解は使わない)をフレームごとにかける
    - フレーム t のクエリを、1 つ前のフレームの(並べ替えた後の)クエリと、最後の decoder 層のクエリの埋め込み(LayerNorm の後)の
      cos の近さでハンガリアン法で 1 対 1 に組にし、前のフレームの並び(枠)にそろえる
    - 全部のクエリを毎フレーム組にするので、隠れている物体の枠は「物体なし」のクエリとつながり、再び写ったときに埋め込みが近ければ
      同じ枠に戻る(戻す規則があるわけではない)。新しい物体は、前のフレームで「物体なし」だった枠に入る
    - 区別できる物体はクエリの数まで。照合は隣のフレームどうしだけ(それより前の記憶は持たない)

形の記号: T = フレームの数、Q = クエリの数、K = クラスの数、C = クエリの次元、h・w = マスクの縦・横(入力の 1/4)。
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def match_embeds(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """前のフレームの枠ごとに、今のフレームのクエリを 1 つずつ割り当てる(cos の近さ、ハンガリアン法)。

    公式の MinVIS の match_from_embds、DVIS++ の Noiser.match_embds と同じ向き(DVIS の初版の ReferringTracker.match_embds は
    cos の行列の向きが逆で、並べ替えが逆の置換になる。後継の DVIS++ で直っているので、こちらに合わせる)。

    Args:
        prev: 前のフレームの、並べ替えた後のクエリの埋め込み (Q, C)。
        cur: 今のフレームのクエリの埋め込み (Q, C)。

    Returns:
        枠 j に入る今のクエリの番号 (Q,)(cur[返り値] で前のフレームの並びにそろう)。
    """
    cos = F.normalize(cur.float(), dim=-1) @ F.normalize(prev.float(), dim=-1).T  # (今のクエリ, 前の枠)
    # (DVIS と同じく NaN は 0 に)。autocast の中では行列の積が bf16 になるので、numpy にする前に float にする
    cost = torch.nan_to_num(1 - cos, nan=0.0).float().cpu().numpy()
    # (前の枠, 今のクエリ) の費用で割り当てる。正方形なので、返り値の 1 つ目は前の枠の 0〜Q−1 の順、2 つ目がその枠に入る今のクエリ
    _, idx = linear_sum_assignment(cost.T)
    return torch.as_tensor(idx, device=cur.device)


class MinVIS(torch.nn.Module):
    """MinVIS の推論(画像の Mask2Former + 隣のフレームのクエリの照合)。

    Args:
        model: 画像の Mask2Former(forward の最後の要素に "embeds" (B, Q, C) を持つ。models/mask2former.py)。
        chunk: まとめて処理するフレームの数(GPU のメモリとの兼ね合い)。
        embed_key: 照合に使う埋め込み("embeds": MinVIS と同じクエリの埋め込み / "reid": CTVIS の reid の埋め込み。公式の
            CTMinVIS.post_processing は pred_reid_embed で照合する)。
    """

    def __init__(self, model: torch.nn.Module, chunk: int = 8, embed_key: str = "embeds"):
        super().__init__()
        from segmentation.instance_segmentation.lib_common.models.video.image_adapter import as_segmenter
        # Mask2Former をそのまま渡しても、GTR 系を ImageSegmenter で包んで渡してもよい(image_adapter.py)
        self.model, self.chunk, self.embed_key = as_segmenter(model), chunk, embed_key
        self.activation = "sigmoid" if self.model.arch == "gtr_seg" else "softmax"  # クラスの値の付け方(後処理に渡す)

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """VidEoMT.predict_video と同じ形で返す(lib_common/video_infer.py の video_predictions にそのまま渡せる)。

        Args:
            frames: (T, 3, S, S)。

        Returns:
            (並べ替えた後のフレームごとのクラスの値 (T, Q, K+1)(GTR 系は (T, Q, K)。self.activation), マスクの値 (Q, T, h, w))。
        """
        logits, masks, embeds = [], [], []
        for s in range(0, len(frames), self.chunk):
            out = self.model(frames[s: s + self.chunk])
            logits.append(out["logits"]); masks.append(out["masks"]); embeds.append(out[self.embed_key])
        logits, masks, embeds = torch.cat(logits), torch.cat(masks), torch.cat(embeds)  # (T, Q, K+1), (T, Q, h, w), (T, Q, C)
        order = [torch.arange(embeds.shape[1], device=embeds.device)]
        for t in range(1, len(embeds)):
            order.append(match_embeds(embeds[t - 1][order[-1]], embeds[t]))
        idx = torch.stack(order)  # (T, Q) フレーム t の枠 q に入る今のクエリの番号
        logits = torch.gather(logits, 1, idx[..., None].expand_as(logits))
        masks = masks[torch.arange(len(idx), device=idx.device)[:, None], idx]  # (T, Q, h, w)
        return logits, masks.transpose(0, 1)  # (T, Q, K+1), (Q, T, h, w)
