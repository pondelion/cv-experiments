"""インスタンスセグメンテーションの評価(COCO 形式の mask AP・box AP・Boundary AP)。

予測は COCO の結果の形式(1 物体 = 1 つの辞書)のリストで渡す:
    {"image_id": int, "category_id": int(1〜20), "segmentation": RLE, "score": float, "bbox": [x, y, w, h](省略可)}
bbox を省略すると、マスクを囲む枠で box AP を計算する。

指標(どれも 0〜1。論文の表の値はこれを 100 倍したもの):
    AP:       IoU の閾値を 0.50〜0.95 で 0.05 刻みに変えた AP の平均(COCO の主な指標)
    AP50/75:  IoU の閾値 0.50 / 0.75 での AP
    APs/m/l:  物体の面積が 32² 未満 / 32²〜96² / 96² 以上のものだけでの AP
    boundary: Boundary AP。マスクの輪郭付近の帯どうしの IoU も使う、境界の正確さに厳しい AP
画素単位の指標(正解ごとの最良の IoU の平均・PQ・SQ・RQ・semantic mIoU・前景の IoU)は pixel_metrics.py で計算する。
定義と計算の手順は instance_segmentation/README.md の「評価の設定」に書いてある。
"""
import copy
import json

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

from lib_common.datasets import set_path
from segmentation.instance_segmentation.lib_common.pixel_metrics import PixelMetrics

_KEYS = {"AP_all": "AP", "AP_50": "AP50", "AP_75": "AP75", "AP_small": "APs", "AP_medium": "APm", "AP_large": "APl",
         "AR_third": "AR100"}


def _evaluate_one(gt: COCO, predictions: list[dict], iou_type: str) -> tuple[dict, dict]:
    """1 種類の IoU(segm / bbox / boundary)で評価し、全体の指標とクラスごとの AP を返す。

    Args:
        gt: 正解の COCO。
        predictions: COCO の結果の形式の予測のリスト。
        iou_type: "segm"・"bbox"・"boundary" のどれか。

    Returns:
        (全体の指標 {"AP", "AP50", ...}, クラスごとの AP {クラス名: AP})。
    """
    dt = gt.loadRes(copy.deepcopy(predictions))  # loadRes は予測の辞書を書き換えるので複製を渡す
    ev = COCOeval_faster(gt, dt, iouType=iou_type, print_function=lambda *a, **k: None)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = {_KEYS[k]: float(v) for k, v in ev.stats_as_dict.items() if k in _KEYS}
    # precision: (IoU の閾値 T, recall の点 R, クラス K, 面積の区分 A, 出す数の上限 M)。-1 は該当する正解なし
    precision = ev.eval["precision"][:, :, :, 0, -1]  # 面積は全部、上限は 100 個 → (T, R, K)
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        p = precision[:, :, k]
        per_class[gt.cats[cat_id]["name"]] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return stats, per_class


def evaluate(set_name: str, predictions: list[dict], boundary: bool = True, pixel: bool = True,
             score_threshold: float | None = None) -> dict:
    """保存した評価のセットに対して予測を評価する。

    Args:
        set_name: lib_common/datasets.py のセット名(例: "coco_val_dev")。
        predictions: COCO の結果の形式の予測のリスト(モジュールの説明を参照)。
        boundary: Boundary AP も計算するか(マスクの輪郭を作るぶん時間がかかる)。
        pixel: 画素単位の指標(pixel_metrics.py)も計算するか。
        score_threshold: PQ・semantic mIoU に使うスコアの閾値。None ならこのセットで PQ が最大になる閾値を選ぶ
            (dev のセットのときだけ None にし、test のセットには dev で選んだ閾値を渡す)。

    Returns:
        {"segm": {...}, "bbox": {...}, "boundary": {...}, "per_class_segm_AP": {クラス名: AP},
         "best_iou_mean": 値, "pq": {...}, "semantic": {...}, "n_predictions": 数}。
    """
    gt = COCO(str(set_path(set_name)))
    result = {"set": set_name, "n_predictions": len(predictions)}
    if not predictions:
        raise ValueError("予測が 1 つもない(評価できない)")
    result["segm"], result["per_class_segm_AP"] = _evaluate_one(gt, predictions, "segm")
    result["bbox"], _ = _evaluate_one(gt, predictions, "bbox")
    if boundary:
        result["boundary"], result["per_class_boundary_AP"] = _evaluate_one(gt, predictions, "boundary")
    if pixel:
        pm = PixelMetrics(gt, predictions)
        t = pm.best_threshold() if score_threshold is None else score_threshold
        result["best_iou_mean"] = pm.best_iou_mean()
        result["pq"] = pm.pq(t)
        result["semantic"] = pm.semantic(t)
    return result


def save_metrics(path, metrics: dict) -> None:
    """評価の結果を JSON に保存する(NaN は null にする)。

    Args:
        path: 保存先。
        metrics: 評価の結果の辞書。
    """
    def _clean(x):
        """辞書の中の NaN を None に変える(JSON は NaN を書けないため)。

        Args:
            x: 値(辞書なら中の値も変える)。

        Returns:
            NaN を None にした値。
        """
        if isinstance(x, dict):
            return {k: _clean(v) for k, v in x.items()}
        return None if isinstance(x, float) and np.isnan(x) else x
    with open(path, "w") as f:
        json.dump(_clean(metrics), f, ensure_ascii=False, indent=1)
