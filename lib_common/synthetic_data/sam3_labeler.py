"""SAM 3 で、画像の中の指定したクラスの物体を全部、マスク付きで見つける(評価・疑似ラベル・合成画像の確認に共通で使う)。

SAM 3 は「文章で指定した概念の物体を全部、マスクで出す」モデル。1 枚の画像について、画像の特徴を 1 回だけ計算し、
複数のクラスの名前をまとめて 1 回の推論で処理する(公式の Sam3Processor は 1 つずつ処理するが、モデル自体は
「どの画像 × どの文章」の組を並べてまとめて処理できる。公式の sam3_multiplex_detector.py と同じ使い方)。
1 回にまとめるクラスの数は 20 まで(80 クラスを一度に処理すると 32GB の VRAM があふれて極端に遅くなる)。

- 重み: 公式の sam3.pt(Hugging Face の facebook/sam3。vendor/sam3 の公式コードで読む。キャッシュから読み、ネットワークには出ない)
- 入力: 公式の前処理のまま(1008×1008 に縮小・拡大)、bf16。マスクは元の画像の大きさに戻す

使い方:
    labeler = Sam3Labeler(["dog", "person"])
    per_class = labeler(image)  # [(masks (N, H, W) bool, scores (N,), boxes (N, 4) x0 y0 x1 y1), ...] クラスの順
形の記号: K = 1 回にまとめて処理するクラスの数、Qd = SAM 3 の検出のクエリの数、N = 残した物体の数、
         h・w = SAM 3 のマスクの縦・横、H・W = 元の画像の縦・横
"""
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image
from sam3.model.data_misc import FindStage
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from torch.nn.functional import interpolate

CKPT_REPO = "facebook/sam3"
MAX_CLASSES_PER_PASS = 20


def load_sam3(score_floor: float = 0.01) -> Sam3Processor:
    """公式の SAM 3 を読み込み、前処理を受け持つ Sam3Processor を返す(model は processor.model)。

    Args:
        score_floor: これ以上のスコアの予測を残す(Sam3Processor の confidence_threshold)。

    Returns:
        Sam3Processor。
    """
    model = build_sam3_image_model(checkpoint_path=hf_hub_download(CKPT_REPO, "sam3.pt", local_files_only=True),
                                   load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=score_floor)


@torch.inference_mode()
def predict_classes(processor: Sam3Processor, state: dict, text_out: dict, score_floor: float) -> list:
    """1 枚の画像について、text_out に入っている全クラスの文章をまとめて 1 回の推論で処理する。

    後処理は Sam3Processor._forward_grounding と同じ(スコア = 物体ごとの確率 × その概念が画像にあるかの確率)。

    Args:
        processor: set_image を済ませた Sam3Processor。
        state: set_image が返した状態。
        text_out: クラスの名前を backbone.forward_text に通した結果(最初に 1 回だけ計算しておく)。
        score_floor: これ以上のスコアの予測を残す。

    Returns:
        クラスの順に、(masks (物体の数, H, W) bool, scores (物体の数,), boxes (物体の数, 4) x0 y0 x1 y1) のリスト。
    """
    model = processor.model
    k = text_out["language_features"].shape[1]  # クラスの数
    dev = processor.device
    find = FindStage(img_ids=torch.zeros(k, dtype=torch.long, device=dev), text_ids=torch.arange(k, device=dev),
                     input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                     input_points=None, input_points_mask=None)
    state["backbone_out"].update(text_out)
    out = model.forward_grounding(backbone_out=state["backbone_out"], find_input=find,
                                  geometric_prompt=model._get_dummy_prompt(num_prompts=k), find_target=None)
    presence = rearrange(out["presence_logit_dec"].sigmoid(), "k o -> k 1 o")  # (K, 1) -> (K, 1, 1) その概念が画像にある確率
    probs = rearrange(out["pred_logits"].sigmoid() * presence, "k q 1 -> k q")  # (K, Qd, 1) -> (K, Qd)
    h, w = state["original_height"], state["original_width"]
    scale = torch.tensor([w, h, w, h], device=dev)
    results = []
    for c in range(k):
        keep = probs[c] > score_floor  # (Qd,)
        boxes = out["pred_boxes"][c][keep]  # (物体, 4) 中心 x, 中心 y, 幅, 高さ(0〜1)
        boxes = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=1) * scale  # (N, 4) x0 y0 x1 y1 の画素
        # interpolate は (まとめ, チャンネル, 縦, 横) を求める: (N, h, w) -> (N, 1, h, w) -> (N, 1, H, W) -> (N, H, W)
        logits = interpolate(rearrange(out["pred_masks"][c][keep], "n h w -> n 1 h w"), (h, w), mode="bilinear",
                             align_corners=False)
        results.append(((rearrange(logits, "n 1 h w -> n h w") > 0).cpu().numpy(), probs[c][keep].float().cpu().numpy(),
                        boxes.float().cpu().numpy()))
    return results


class Sam3Labeler:
    """決まったクラスの集合について、画像ごとに SAM 3 で物体を見つける。

    Args:
        class_names: 探すクラスの名前(文章として SAM 3 に与える)。
        score_floor: これ以上のスコアの予測を残す。
        processor: 読み込み済みの Sam3Processor(複数の Sam3Labeler で同じモデルを使うとき)。
    """

    def __init__(self, class_names: list[str], score_floor: float = 0.01, processor: Sam3Processor | None = None):
        self.processor = processor or load_sam3(score_floor)
        self.score_floor = score_floor
        chunks = [class_names[i: i + MAX_CLASSES_PER_PASS] for i in range(0, len(class_names), MAX_CLASSES_PER_PASS)]
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            self.text_outs = [self.processor.model.backbone.forward_text(c, device="cuda") for c in chunks]

    def __call__(self, image: Image.Image) -> list:
        """画像 1 枚について、物体を見つける。

        Args:
            image: RGB の画像。

        Returns:
            クラスの順に、(masks (N, H, W) bool, scores (N,), boxes (N, 4) x0 y0 x1 y1) のリスト。
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            return [r for t in self.text_outs for r in predict_classes(self.processor, state, t, self.score_floor)]
