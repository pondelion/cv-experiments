"""評価の仕組みが正しく動くかを、答えのわかる予測で確かめる(CPU だけで動く)。

確かめること:
    1. 正解のマスクをそのまま予測として渡すと、どの指標も 1.0 になる
    2. 予測を半分に減らすと、AP・AR が約 0.5、RQ が約 0.67(= 0.5 ÷ 0.75)になり、SQ は 1.0 のまま(見つけた物体の形は正確なので)
    3. 予測のクラスをずらすと、AP がほぼ 0 になる(クラスの対応が評価に効いている)
    4. 正解のマスクを 2 画素縮めると、AP50 はほぼ 1 のまま、AP・SQ・最良の IoU の平均は少し下がる(形のずれが連続的に効く)

使い方: .venv/bin/python segmentation/instance_segmentation/exp00001_eval_and_baselines/check_eval.py
"""
import common  # noqa: F401  リポジトリ直下を import の探索パスに追加する
import cv2
import numpy as np
from faster_coco_eval import COCO

from lib_common.datasets import set_path
from lib_common.masks import mask_to_rle, rle_to_mask
from segmentation.instance_segmentation.lib_common.evaluate import evaluate


def gt_as_predictions(set_name: str) -> list[dict]:
    """正解の物体(群衆の領域は除く)を、スコア 1.0 の予測の形にする。"""
    gt = COCO(str(set_path(set_name)))
    return [
        {"image_id": a["image_id"], "category_id": a["category_id"], "segmentation": gt.annToRLE(a),
         "bbox": a["bbox"], "score": 1.0}
        for a in gt.dataset["annotations"] if not a["iscrowd"]
    ]


def shrink(p: dict, pixels: int = 2) -> dict:
    """予測のマスクを、3×3 の縮小で pixels 画素だけ縮める(枠はマスクから計算し直させるため外す)。"""
    mask = cv2.erode(rle_to_mask(p["segmentation"]).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=pixels)
    return {k: v for k, v in p.items() if k != "bbox"} | {"segmentation": mask_to_rle(mask)}


def summary(r: dict) -> str:
    """結果の主な値を 1 行にする。"""
    s = f"mask AP {r['segm']['AP']:.3f} AP50 {r['segm']['AP50']:.3f} AR100 {r['segm']['AR100']:.3f}"
    if "pq" in r:
        s += (f" | 最良IoU平均 {r['best_iou_mean']:.3f} PQ {r['pq']['PQ']:.3f} SQ {r['pq']['SQ']:.3f} "
              f"RQ {r['pq']['RQ']:.3f} | semantic mIoU {r['semantic']['mIoU']:.3f} 前景IoU {r['semantic']['foreground_IoU']:.3f}")
    return s


def main() -> None:
    for set_name in ["coco_val_dev", "voc_val"]:
        preds = gt_as_predictions(set_name)
        perfect = evaluate(set_name, preds)
        print(f"[{set_name}] 予測 {len(preds)} 個")
        print(f"  正解そのまま: {summary(perfect)} | box AP {perfect['bbox']['AP']:.3f} "
              f"Boundary AP {perfect['boundary']['AP']:.3f} クラスごとの mask AP の最小 "
              f"{min(perfect['per_class_segm_AP'].values()):.3f}")
        print(f"  半分に減らす: {summary(evaluate(set_name, preds[::2], boundary=False))}")
        shifted = [dict(p, category_id=p["category_id"] % 20 + 1) for p in preds]
        print(f"  クラスをずらす: {summary(evaluate(set_name, shifted, boundary=False, pixel=False))}")
        print(f"  2 画素縮める: {summary(evaluate(set_name, [shrink(p) for p in preds], boundary=False))}")


if __name__ == "__main__":
    main()
