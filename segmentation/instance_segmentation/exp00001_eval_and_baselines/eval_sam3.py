"""SAM 3 をそのまま(学習し直さずに)動かして、評価のセットで評価する。

SAM 3 は「文章で指定した概念の物体を全部、マスクで出す」モデル。1 枚の画像について、
画像の特徴を 1 回だけ計算し、20 クラスの名前を 1 つずつ文章として与えて、それぞれのクラスの物体のマスクとスコアを出す。

- 重み: 公式の sam3.pt(Hugging Face の facebook/sam3。vendor/sam3 の公式コードで読む)
- 入力: 公式の前処理のまま(1008×1008 に縮小・拡大)。マスクは元の画像の大きさに戻してから評価する
- スコア: 公式は 0.5 以上だけを返すが、AP には低いスコアの予測も要るので、--score-floor 以上を全部残す

出力:
    runs/sam3_{セット}_predictions.json  予測(COCO の結果の形式)
    results/sam3_{セット}.json           評価の結果と、1 枚あたりの時間

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00001_eval_and_baselines/eval_sam3.py --set coco_val_dev
    (test・ほかのセットは、dev で選んだスコアの閾値を --threshold-from で渡す)
"""
import argparse
import json

import common
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from tqdm import tqdm

from lib_common import DATASETS_DIR
from lib_common.datasets import load_set
from lib_common.latency import Timer, summarize
from lib_common.masks import mask_to_rle
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics
from lib_common.synthetic_data.sam3_labeler import CKPT_REPO, predict_classes  # 複数のクラスをまとめて推論する処理(共通)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="coco_val_dev")
    parser.add_argument("--limit", type=int, default=None, help="先頭の何枚だけ使うか(動作確認用)")
    parser.add_argument("--score-floor", type=float, default=0.01, help="これ以上のスコアの予測を残す")
    parser.add_argument("--threshold-from", default=None, help="PQ などのスコアの閾値を、この結果の JSON から取る")
    parser.add_argument("--ap-only", action="store_true", help="AP 系だけを計算する(公式の値との比較用。Boundary AP・画素単位の指標は省く)")
    parser.add_argument("--class-chunk", type=int, default=20,
                        help="1 回の推論でまとめて処理するクラスの数(80 クラスを一度に処理すると 32GB の VRAM があふれて極端に遅くなる)")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    model = build_sam3_image_model(checkpoint_path=hf_hub_download(CKPT_REPO, "sam3.pt"), load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=args.score_floor)

    data = load_set(args.set)
    images = data["images"][: args.limit]
    classes = [(c["id"], c["name"]) for c in data["categories"]]
    predictions, times = [], {"image": [], "classes": [], "to_rle": []}
    chunks = [classes[i: i + args.class_chunk] for i in range(0, len(classes), args.class_chunk)]
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        # クラスの組ごとの文章の特徴(最初に 1 回だけ計算しておく)
        text_outs = [model.backbone.forward_text([name for _, name in chunk], device="cuda") for chunk in chunks]

    for im in tqdm(images, desc=f"SAM 3 {args.set}"):
        image = Image.open(DATASETS_DIR / im["file_name"]).convert("RGB")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with Timer() as t_img:
                state = processor.set_image(image)
            with Timer() as t_cls:
                per_class = [r for text_out in text_outs
                             for r in predict_classes(processor, state, text_out, args.score_floor)]
        times["image"].append(t_img.ms)
        times["classes"].append(t_cls.ms)
        with Timer() as t_rle:
            for (cat_id, _), (masks, scores, boxes) in zip(classes, per_class):
                for m, s, b in zip(masks, scores, boxes):
                    predictions.append({
                        "image_id": im["id"], "category_id": cat_id, "segmentation": mask_to_rle(m),
                        "score": float(s), "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                    })
        times["to_rle"].append(t_rle.ms)

    suffix = f"_limit{args.limit}" if args.limit else ""
    common.RUNS_DIR.mkdir(exist_ok=True)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(common.RUNS_DIR / f"sam3_{args.set}{suffix}_predictions.json", "w") as f:
        json.dump(predictions, f)

    threshold = None
    if args.threshold_from:
        with open(args.threshold_from) as f:
            threshold = json.load(f)["pq"]["score_threshold"]
    # 動作確認(--limit)のときは、全部の画像に正解があるとして評価しないよう、使った画像だけに絞った評価はしない
    if args.limit:
        print(f"予測 {len(predictions)} 個、1 枚あたりの予測の数 {len(predictions) / len(images):.1f}")
        print("スコアの分布(分位点 0.5 / 0.9 / 0.99):",
              np.quantile([p["score"] for p in predictions], [0.5, 0.9, 0.99]).round(3) if predictions else "-")
        metrics = {}
    else:
        metrics = evaluate(args.set, predictions, boundary=not args.ap_only, pixel=not args.ap_only,
                           score_threshold=threshold)
    metrics["model"] = "SAM 3 (facebook/sam3, sam3.pt)"
    metrics["score_floor"] = args.score_floor
    warmup = min(5, len(images) - 1)
    metrics["time_ms"] = {
        "image_encoder": summarize(times["image"], warmup),
        "all_classes_grounding": summarize(times["classes"], warmup),  # 20 クラスまとめての検出・マスク(元の大きさに戻すまで)
        "to_rle": summarize(times["to_rle"], warmup),
        "per_image_20_classes": summarize((np.array(times["image"]) + np.array(times["classes"])).tolist(), warmup),
    }
    save_metrics(common.RESULTS_DIR / f"sam3_{args.set}{suffix}.json", metrics)
    print(json.dumps({k: metrics[k] for k in ["time_ms"]}, ensure_ascii=False, indent=1))
    if not args.limit:
        print(f"mask AP {metrics['segm']['AP']:.3f} box AP {metrics['bbox']['AP']:.3f}"
              + ("" if args.ap_only else f" SQ {metrics['pq']['SQ']:.3f} (閾値 {metrics['pq']['score_threshold']})"))


if __name__ == "__main__":
    main()
