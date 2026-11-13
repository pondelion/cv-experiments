"""公式の学習済みの RF-DETR-Seg(COCO の 80 クラスで学習済み)をそのまま動かして、評価のセットで評価する。

- 重み: rfdetr ライブラリの公式の重み(初回に自動でダウンロードされる)
- 予測: 1 枚あたり、公式の設定の上位 num_select 個(サイズにより 100〜300 個)をスコアの閾値なしで受け取り、
  VOC の 20 クラスに当たる予測だけを残して、この実験のクラスの ID に変える
- 推論: ライブラリが勧める予測器(model.inference()。fp32 のまま、入力の大きさを固定して CUDA Graph で組み立てたもの)を使い、
  精度と時間を同じ予測器で測る
- 時間: 予測器の predict(PIL の画像 → 元の大きさのマスク。前処理・後処理込み)を測る。
  公式の速度の値(T4 + TensorRT fp16)とは条件が違う。TensorRT での速度はフェーズ D で測る

出力:
    runs/rfdetr_{サイズ}_{セット}_predictions.json  予測(COCO の結果の形式)
    results/rfdetr_{サイズ}_{セット}.json           評価の結果と、1 枚あたりの時間

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00001_eval_and_baselines/eval_rfdetr.py --size nano --set coco_val_dev
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import common
import numpy as np
import rfdetr
import torch
from PIL import Image
from tqdm import tqdm

from lib_common import DATASETS_DIR
from rfdetr.assets.coco_classes import COCO_CLASSES
from lib_common.datasets import load_set
from lib_common.latency import Timer, summarize
from lib_common.masks import mask_to_rle
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics

SIZES = {"nano": rfdetr.RFDETRSegNano, "small": rfdetr.RFDETRSegSmall, "medium": rfdetr.RFDETRSegMedium,
         "large": rfdetr.RFDETRSegLarge, "xlarge": rfdetr.RFDETRSegXLarge, "2xlarge": rfdetr.RFDETRSeg2XLarge}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="nano", choices=list(SIZES))
    parser.add_argument("--set", default="coco_val_dev")
    parser.add_argument("--limit", type=int, default=None, help="先頭の何枚だけ使うか(動作確認用。評価はしない)")
    parser.add_argument("--threshold-from", default=None, help="PQ などのスコアの閾値を、この結果の JSON から取る")
    parser.add_argument("--ap-only", action="store_true", help="AP 系だけを計算する(公式の値との比較用。Boundary AP・画素単位の指標は省く)")
    args = parser.parse_args()

    model = SIZES[args.size]()
    cfg = model.model_config
    params_m = sum(p.numel() for p in model.model.model.parameters()) / 1e6
    predictor = model.inference()  # 推論用に最適化した予測器
    data = load_set(args.set)
    # RF-DETR の出すクラスの番号(COCO の category_id)→ このセットの ID。クラス名で対応させる(20 クラスでも 80 クラスでも同じ)
    ours = {c["name"]: c["id"] for c in data["categories"]}
    to_ours = {coco_id: ours[name] for coco_id, name in COCO_CLASSES.items() if name in ours}
    images = data["images"][: args.limit]
    predictions, times = [], []

    for im in tqdm(images, desc=f"RF-DETR-Seg {args.size} {args.set}"):
        image = Image.open(DATASETS_DIR / im["file_name"]).convert("RGB")
        with Timer() as t:
            det = predictor.predict(image, threshold=0.0, include_source_image=False)
        times.append(t.ms)
        assert det.mask.shape[1:] == (im["height"], im["width"]), "マスクが元の画像の大きさになっていない"
        for m, s, c, b in zip(det.mask, det.confidence, det.class_id, det.xyxy):
            if int(c) in to_ours:
                predictions.append({
                    "image_id": im["id"], "category_id": to_ours[int(c)], "segmentation": mask_to_rle(m),
                    "score": float(s), "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                })

    suffix = f"_limit{args.limit}" if args.limit else ""
    name = f"rfdetr_{args.size}_{args.set}{suffix}"
    common.RUNS_DIR.mkdir(exist_ok=True)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(common.RUNS_DIR / f"{name}_predictions.json", "w") as f:
        json.dump(predictions, f)

    threshold = None
    if args.threshold_from:
        with open(args.threshold_from) as f:
            threshold = json.load(f)["pq"]["score_threshold"]
    if args.limit:
        print(f"予測(20 クラス){len(predictions)} 個、1 枚あたり {len(predictions) / len(images):.1f} 個")
        metrics = {}
    else:
        metrics = evaluate(args.set, predictions, boundary=not args.ap_only, pixel=not args.ap_only,
                           score_threshold=threshold)
    metrics["model"] = f"RF-DETR-Seg {args.size} (rfdetr {version("rfdetr")}, {Path(str(cfg.pretrain_weights)).name})"  # ファイル名だけ(環境のパスを残さない)
    metrics["resolution"] = cfg.resolution
    metrics["num_select"] = cfg.num_select
    metrics["params_M"] = params_m
    metrics["time_ms"] = {"predict_end_to_end": summarize(times, warmup=min(10, len(times) - 1))}
    metrics["gpu"] = torch.cuda.get_device_name(0)
    save_metrics(common.RESULTS_DIR / f"{name}.json", metrics)
    print(json.dumps(metrics["time_ms"], indent=1), f"params {metrics['params_M']:.1f}M")
    if not args.limit:
        print(f"mask AP {metrics['segm']['AP']:.3f} box AP {metrics['bbox']['AP']:.3f}"
              + ("" if args.ap_only else f" SQ {metrics['pq']['SQ']:.3f} (閾値 {metrics['pq']['score_threshold']})"))


if __name__ == "__main__":
    main()
