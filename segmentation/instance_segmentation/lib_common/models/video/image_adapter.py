"""画像のインスタンスセグメンテーションのモデル(Mask2Former・GTR 系)を、動画の方式から同じ形で呼ぶための包み。exp00502。

動画の方式のうち、画像のモデルをそのまま使うもの(MinVIS の照合・DVIS の segmenter・SAM 3 型の検出器)は、モデルによって出力の形と
クラスの点数の付け方が違う。ここで次の形にそろえる(どれもフレームを画像とみなした 1 枚ずつの値):
    "logits": クラスの値 (n, Q, K+1)(softmax の方式)か (n, Q, K)(sigmoid の方式)。"activation" に "softmax" か "sigmoid"
    "probs": クラスの確率 (n, Q, K)(物体なしを除く。画像の評価の点数と同じ付け方)
    "masks": マスクの値 (n, Q, h, w)(入力の 1/4)
    "embeds": 照合に使うクエリの埋め込み (n, Q, C)(Mask2Former: 最後の層の LayerNorm の後。GTR: 最後の層の出力のクエリ)
    "mask_feat": マスクを作る画素の特徴 (n, c, h, w)(Mask2Former: c = 256。GTR: マスクの head の spatial)
検出(detections)は、画像の評価の後処理(postprocess.py の to_predictions・rfdetr_to_predictions)と同じく、(クエリ, クラス) の上位
TOP_K 個を点数の順に選ぶ。GTR の box_crop の学習のモデルは、予測の枠の外のマスクを消す(画像の評価と同じ規則。画像の評価は元の大きさの
画素で、ここはマスクの解像度で消す)。

形の記号: n = フレームの数、Q = クエリの数、K = クラスの数、R = 検出の数、h・w = マスクの縦・横、S = 入力の一辺。
"""
import json
from pathlib import Path

import torch
from torch import nn

from segmentation.instance_segmentation.lib_common.postprocess import TOP_K


class ImageSegmenter(nn.Module):
    """画像のモデルの包み。

    Args:
        model: Mask2Former か GTRSeg。
        arch: "mask2former" か "gtr_seg"。
        crop_to_box: GTR の box_crop の学習のモデルなら True(検出のマスクを予測の枠の中だけにする)。
    """

    def __init__(self, model: nn.Module, arch: str, crop_to_box: bool = False):
        super().__init__()
        assert arch in ("mask2former", "gtr_seg"), f"対応していない画像のモデル: {arch}"
        self.model, self.arch, self.crop_to_box = model, arch, crop_to_box

    @classmethod
    def from_run(cls, path: str) -> "ImageSegmenter":
        """画像の学習(exp00101 の train.py・exp00502 の train_image.py)の重みから作る(同じ場所の config.json で作り方を読む)。"""
        from segmentation.instance_segmentation.lib_common.train import TrainConfig, build_model
        cfg = json.loads((Path(path).parent / "config.json").read_text())
        names = TrainConfig.__dataclass_fields__.keys()
        tc = TrainConfig(**{k: v for k, v in cfg.items() if k in names})
        n_classes = len(json.loads(Path(tc.train_json).read_text())["categories"])
        model = build_model(tc, n_classes)
        model.load_state_dict(torch.load(path, map_location="cuda"))
        crop = tc.arch == "gtr_seg" and tc.gtr_mask_loss == "box_crop"
        return cls(model, tc.arch, crop).eval()

    def forward(self, frames: torch.Tensor) -> dict:
        """Args: frames (n, 3, S, S)(評価の変換。長い辺を S にして右と下を埋めたもの)。Returns: 上の形の dict。"""
        if self.arch == "mask2former":
            o = self.model(frames)[-1]
            out = {"logits": o["logits"], "probs": o["logits"].float().softmax(-1)[..., :-1], "masks": o["masks"],
                   "embeds": o["embeds"], "activation": "softmax"}
            # reid は CTVIS・CAVIS の segmenter だけ、embeds_ctx は CAVIS の segmenter だけ
            out.update({k: o[k] for k in ("embeds_raw", "mask_feat", "reid", "embeds_ctx") if k in o})
            return out
        from segmentation.instance_segmentation.lib_common.models.rfdetr_seg import masks_from_parts
        last = self.model(frames)["layers"][-1]
        parts = last["mask_parts"]
        masks = masks_from_parts(parts)
        if self.crop_to_box:  # 予測の枠(0〜1 の cx, cy, w, h。S × S の中)の外を消す(枠の中だけで学習したマスクのため)
            h, w = masks.shape[-2:]
            cx, cy, bw, bh = last["boxes"].float().unbind(-1)  # (n, Q)
            xs = ((torch.arange(w, device=masks.device) + 0.5) / w)[None, None, None, :]
            ys = ((torch.arange(h, device=masks.device) + 0.5) / h)[None, None, :, None]
            e = lambda v: v[..., None, None]
            inside = (xs >= e(cx - bw / 2)) & (xs < e(cx + bw / 2)) & (ys >= e(cy - bh / 2)) & (ys < e(cy + bh / 2))
            masks = torch.where(inside, masks, torch.full_like(masks, -1e4))
        return {"logits": last["logits"], "probs": last["logits"].float().sigmoid(), "masks": masks,
                "embeds": last["hidden"], "embeds_raw": last["hidden"], "mask_feat": parts["spatial"], "boxes": last["boxes"],
                "activation": "sigmoid"}

    @torch.no_grad()
    def detections(self, out: dict) -> list[dict]:
        """forward の出力から、フレームごとの検出 {"masks": (R, h, w) の値, "scores": (R,), "labels": (R,)(0 始まり)} を作る。"""
        res = []
        for i in range(out["probs"].shape[0]):
            prob = out["probs"][i]  # (Q, K)
            k = prob.shape[1]
            scores, idx = prob.flatten().topk(min(TOP_K, prob.numel()))
            qi, labels = idx // k, idx % k
            res.append({"masks": out["masks"][i, qi].float(), "scores": scores, "labels": labels})  # (R, h, w)(box_crop は forward で済み)
        return res


def as_segmenter(model: nn.Module) -> ImageSegmenter:
    """Mask2Former をそのまま渡したときも ImageSegmenter として扱う(MinVIS の学習中の評価など)。"""
    return model if isinstance(model, ImageSegmenter) else ImageSegmenter(model, "mask2former")
