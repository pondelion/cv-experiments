"""動画インスタンスセグメンテーションの評価(YouTube-VIS と同じ計算の動画の AP)。

pycocotools の COCOeval を土台にし、公式の YouTube-VIS の評価(youtubevos/cocoapi の ytvoseval.py)と同じになるように 2 点だけ変える:
    1. 物体どうしの IoU: フレームごとの重なりの面積の和 ÷ 和集合の面積の和(片方にしか写っていないフレームは、和集合にだけ足す)
    2. 物体の大きさ(小・中・大の AP の区別): 写っているフレームのマスクの面積の平均(avg_area)。境目は公式と同じ 128²・256²
COCOeval の「画像」を動画、「物体」を動画を通した物体(フレームごとのマスクの並び)として扱う。AP の積み上げ方(IoU の閾値 0.50〜0.95、
最大の検出の数 1・10・100)は COCOeval のまま。

正解は共通の形式の JSON(lib_common/video_data.py)、予測は
    [{"video_id", "category_id", "score", "segmentations": [フレームごとの RLE または None]}]
"""
import contextlib
import io

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _avg_area(segs: list) -> float:
    """写っているフレームのマスクの面積の平均(公式の avg_area と同じ。どのフレームにも写っていなければ 0)。"""
    a = [float(mask_util.area(s)) for s in segs if s]
    return float(np.mean(a)) if a else 0.0


def video_iou(d_seq: list, g_seq: list) -> float:
    """2 つの物体(フレームごとの RLE の並び)の、動画を通した IoU。

    Args:
        d_seq: 予測のフレームごとの RLE(写っていないフレームは None)。
        g_seq: 正解のフレームごとの RLE。

    Returns:
        フレームごとの重なりの面積の和 ÷ 和集合の面積の和(どちらも 0 なら 0)。
    """
    inter = union = 0.0
    for d, g in zip(d_seq, g_seq):
        if d and g:
            inter += float(mask_util.area(mask_util.merge([d, g], intersect=True)))
            union += float(mask_util.area(mask_util.merge([d, g], intersect=False)))
        elif g:
            union += float(mask_util.area(g))
        elif d:
            union += float(mask_util.area(d))
    return inter / union if union > 0 else 0.0


def _to_coco(gt: dict, anns: list[dict]) -> COCO:
    """動画を画像、物体を 1 つの注釈とした pycocotools の COCO を作る(COCOeval に渡すため)。"""
    c = COCO()
    c.dataset = {"images": [{"id": v["id"]} for v in gt["videos"]], "categories": gt["categories"],
                 "annotations": [dict(a, image_id=a["video_id"], area=_avg_area(a["segmentations"]), iscrowd=a.get("iscrowd", 0),
                                      id=i + 1) for i, a in enumerate(anns)]}
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


class VideoCOCOeval(COCOeval):
    """COCOeval の IoU の計算だけを、動画を通した IoU にしたもの。"""

    def computeIoU(self, img_id, cat_id):  # noqa: N802(COCOeval の名前に合わせる)
        p = self.params
        gt = self._gts[img_id, cat_id] if p.useCats else [g for c in p.catIds for g in self._gts[img_id, c]]
        dt = self._dts[img_id, cat_id] if p.useCats else [d for c in p.catIds for d in self._dts[img_id, c]]
        if not gt and not dt:
            return []
        dt = sorted(dt, key=lambda d: -d["score"])[: p.maxDets[-1]]
        ious = np.zeros((len(dt), len(gt)))
        for i, j in np.ndindex(ious.shape):
            ious[i, j] = video_iou(dt[i]["segmentations"], gt[j]["segmentations"])
        return ious


def evaluate_vis(gt: dict, predictions: list[dict]) -> dict:
    """動画の AP を計算する。

    Args:
        gt: 共通の形式の正解(dict)。
        predictions: 予測の並び(上の形)。

    Returns:
        {"AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "per_class_AP": {クラス名: AP}}(値は 0〜1。予測がなければ 0)。
    """
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10"]
    if not predictions:
        return {**{k: 0.0 for k in keys}, "per_class_AP": {c["name"]: 0.0 for c in gt["categories"]}}
    cg, cd = _to_coco(gt, gt["annotations"]), _to_coco(gt, predictions)
    # iouType="segm" だと COCOeval が前処理で画像ごとのマスクに直そうとする(画像の大きさが要る)。IoU は computeIoU で動画を通した
    # 値に差し替えるので、前処理をしない "bbox" で作る(AP の積み上げ方は iouType によらず同じ)
    e = VideoCOCOeval(cg, cd, iouType="bbox")
    # 小・中・大の物体の境目は、公式の YouTube-VIS の評価と同じ 128²・256²(COCO の画像の 32²・96² ではない)
    e.params.areaRng = [[0, 1e5 ** 2], [0, 128 ** 2], [128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = e.stats  # COCOeval の並び: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl
    precision = e.eval["precision"]  # (IoU の閾値, recall, クラス, 大きさ, 最大の検出の数)
    per_class = {}
    for k, cat_id in enumerate(e.params.catIds):
        pk = precision[:, :, k, 0, -1]
        per_class[cg.cats[cat_id]["name"]] = float(pk[pk > -1].mean()) if (pk > -1).any() else float("nan")
    return {**{k: float(v) for k, v in zip(keys, s[:8])}, "per_class_AP": per_class}


def evaluate_frames(gt: dict, predictions: list[dict]) -> dict:
    """画像 AP: 各フレームを 1 枚の画像とみなし、画像の段階と同じ COCO の mask AP を計算する(物体の ID は使わない)。

    動画の AP(evaluate_vis)は、動画を通した物体ごとに 1 回だけ対応させるので、フレームごとのセグメンテーションの良さが、
    追跡の切れや入れ替わりの影響に埋もれやすい。画像 AP は、同じ予測をフレームごとに分けて評価するので、フレームごとの
    セグメンテーションの良さを見られ、画像のモデル(フレーム独立)の mask AP とも同じ土台で並べられる。予測のスコアは、その物体の
    動画の予測のスコアをそのまま使う(推論の追加はない)。

    Args:
        gt: 共通の形式の正解(dict)。
        predictions: evaluate_vis と同じ予測の並び。

    Returns:
        {"image_AP", "image_AP50", "image_AP75", "image_APs", "image_APm", "image_APl", "image_per_class_AP": {クラス名: AP}}。
    """
    keys = ["image_AP", "image_AP50", "image_AP75", "image_APs", "image_APm", "image_APl"]
    if not predictions:
        return {**{k: 0.0 for k in keys}, "image_per_class_AP": {c["name"]: 0.0 for c in gt["categories"]}}
    frame_id = lambda vid, t: vid * 1000 + t  # フレームの画像の id(1 本の動画は 1,000 フレーム未満)
    images = [{"id": frame_id(v["id"], t), "height": v["height"], "width": v["width"]}
              for v in gt["videos"] for t in range(v["length"])]
    def per_frame(objs: list[dict], with_score: bool) -> list[dict]:
        out = []
        for o in objs:
            for t, rle in enumerate(o["segmentations"]):
                if rle is None:
                    continue
                d = {"id": len(out) + 1, "image_id": frame_id(o["video_id"], t), "category_id": o["category_id"],
                     "segmentation": rle, "area": float(mask_util.area(rle)), "iscrowd": o.get("iscrowd", 0),
                     "bbox": [float(x) for x in mask_util.toBbox(rle)]}
                if with_score:
                    d["score"] = o["score"]
                out.append(d)
        return out
    def coco(anns: list[dict]) -> COCO:
        c = COCO()
        c.dataset = {"images": images, "categories": gt["categories"], "annotations": anns}
        with contextlib.redirect_stdout(io.StringIO()):
            c.createIndex()
        return c
    cg, cd = coco(per_frame(gt["annotations"], False)), coco(per_frame(predictions, True))
    e = COCOeval(cg, cd, iouType="segm")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    precision = e.eval["precision"]
    per_class = {}
    for k, cat_id in enumerate(e.params.catIds):
        pk = precision[:, :, k, 0, -1]
        per_class[cg.cats[cat_id]["name"]] = float(pk[pk > -1].mean()) if (pk > -1).any() else float("nan")
    return {**{k: float(v) for k, v in zip(keys, e.stats[:6])}, "image_per_class_AP": per_class}
