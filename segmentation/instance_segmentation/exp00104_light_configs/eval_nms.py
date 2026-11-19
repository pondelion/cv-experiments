"""exp00104: 学習済みのモデルの予測に、推論の後でマスクの NMS をかけたときの精度と、重複した予測の数を測る。

DETR 系の方式は 1 対 1 の割り当てで学習するので NMS が要らないとされるが、同じ物体への重複した予測が残ることがある。その量と、
NMS で消したときの精度の変わり方を見る。

マスクの NMS: 画像・クラスごとに、予測をスコアの高い順に見て、それより前に残した予測とのマスクの IoU が閾値を超えるものを消す。
重複の数: スコアが --dup-score 以上の予測のうち、同じ画像・同じクラスで、マスクの IoU が --dup-iou を超える組の数。
評価は学習のときの最後の評価と同じ手順(dev で PQ の閾値を選び、test・voc で測る)で、NMS の条件ごとに閾値を選び直す。
結果は `results/{名前}_nms.json` に保存する(学習のときの結果のファイルと見本の画像は変えない)。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00104_light_configs/eval_nms.py --name gtr_g1_r50_cls8_2k_ep100
"""
import argparse
import dataclasses
import json
from collections import defaultdict

import common

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

from lib_common.datasets import load_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.train import TrainConfig, build_model, predict  # noqa: E402


def group(preds: list[dict]) -> dict:
    """予測を (画像, クラス) ごとに、スコアの高い順に並べて分ける。

    Args:
        preds: COCO の結果の形式の予測の並び。

    Returns:
        {(image_id, category_id): [予測, ...]}。
    """
    out = defaultdict(list)
    for p in preds:
        out[(p["image_id"], p["category_id"])].append(p)
    return {k: sorted(v, key=lambda p: -p["score"]) for k, v in out.items()}


def mask_nms(preds: list[dict], iou: float) -> list[dict]:
    """画像・クラスごとのマスクの NMS(スコアの高い予測を残し、それとのマスクの IoU が iou を超える予測を消す)。

    Args:
        preds: COCO の結果の形式の予測の並び(segmentation は RLE)。
        iou: 消すマスクの IoU の閾値。

    Returns:
        残した予測の並び。
    """
    kept = []
    for ps in group(preds).values():
        keep = []
        for p in ps:
            if keep and mask_utils.iou([p["segmentation"]], [k["segmentation"] for k in keep], [0] * len(keep)).max() > iou:
                continue
            keep.append(p)
        kept += keep
    return kept


def count_duplicates(preds: list[dict], score: float, iou: float) -> tuple[int, int]:
    """重複した予測の組の数を数える。

    Args:
        preds: 予測の並び。
        score: このスコア以上の予測だけを見る。
        iou: 重複とみなすマスクの IoU の閾値。

    Returns:
        (重複した組の数, 見た予測の数)。
    """
    pairs, n = 0, 0
    for ps in group([p for p in preds if p["score"] >= score]).values():
        n += len(ps)
        if len(ps) > 1:
            m = mask_utils.iou([p["segmentation"] for p in ps], [p["segmentation"] for p in ps], [0] * len(ps))
            pairs += int((np.triu(m, k=1) > iou).sum())
    return pairs, n


def main() -> None:
    """NMS の条件ごとに評価し、重複の数とあわせて保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="学習の名前(runs/{名前}/)")
    p.add_argument("--nms-iou", type=float, nargs="+", default=[0.7, 0.5], help="NMS で消すマスクの IoU の閾値(複数可)")
    p.add_argument("--dup-score", type=float, default=0.3, help="重複を数える予測のスコアの下限(デモの既定の閾値と同じ)")
    p.add_argument("--dup-iou", type=float, default=0.7, help="重複とみなすマスクの IoU")
    a = p.parse_args()
    saved = json.loads((common.RUNS_DIR / a.name / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in saved.items() if k in fields})
    model = build_model(cfg, len(load_set(cfg.eval_sets[0])["categories"]))
    model.load_state_dict(torch.load(common.RUNS_DIR / a.name / "best.pt", map_location="cuda"))
    raw = {s: predict(model, s, cfg)[0] for s in cfg.eval_sets}
    conditions = [("none", None)] + [(f"nms{t}", t) for t in a.nms_iou]
    results = {}
    for name, t in conditions:
        threshold, results[name] = None, {}
        for s in cfg.eval_sets:  # 最初のセット(dev)で PQ の閾値を選び、ほかのセットにはその値を使う
            preds = raw[s] if t is None else mask_nms(raw[s], t)
            m = evaluate(s, preds, score_threshold=threshold)
            threshold = threshold if threshold is not None else m["pq"]["score_threshold"]
            dup, n = count_duplicates(preds, a.dup_score, a.dup_iou)
            results[name][s] = m | {"duplicate_pairs": dup, "n_scored": n}
            print(f"[{name}] {s}: mask AP {m['segm']['AP']:.4f} | box AP {m['bbox']['AP']:.4f} | PQ {m['pq']['PQ']:.4f} "
                  f"(閾値 {threshold:.2f}) | 予測 {len(preds)} | スコア {a.dup_score} 以上 {n} 個のうち重複の組 {dup}", flush=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(common.RESULTS_DIR / f"{a.name}_nms.json",
                 {"name": a.name, "dup_score": a.dup_score, "dup_iou": a.dup_iou, "results": results})


if __name__ == "__main__":
    main()
