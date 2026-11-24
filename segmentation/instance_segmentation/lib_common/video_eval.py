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
    """写っているフレームのマスクの面積の平均(公式の avg_area と同じ)。

    Args:
        segs: フレームごとの RLE(写っていないフレームは None)。

    Returns:
        面積の平均(どのフレームにも写っていなければ 0)。
    """
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
    """動画を画像、物体を 1 つの注釈とした pycocotools の COCO を作る(COCOeval に渡すため)。

    Args:
        gt: 正解のセット(videos・categories を使う)。
        anns: 物体の並び(正解か予測)。

    Returns:
        pycocotools の COCO。
    """
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
        """1 本の動画・1 クラスの、予測と正解の動画の IoU(video_iou。点数の上位 maxDets 個の予測)。

        Args:
            img_id: 動画の ID。
            cat_id: クラスの ID。

        Returns:
            (予測の数, 正解の数) の IoU(どちらもなければ [])。
        """
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
        """物体ごとの動画の注釈を、フレームごとの画像の注釈に分ける。

        Args:
            objs: 物体の並び(正解か予測)。
            with_score: 予測なら True(点数を写す)。

        Returns:
            フレームの画像ごとの COCO の注釈の並び。
        """
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
        """フレームの画像の注釈から pycocotools の COCO を作る。

        Args:
            anns: per_frame の返り値。

        Returns:
            pycocotools の COCO。
        """
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


# ---- J&F(マスクを与えて追う方式の、伝搬だけの出来。DAVIS の公式の評価 davis2017-evaluation と同じ計算) ----
def _disk(r: int) -> np.ndarray:
    """半径 r の円の構造要素(skimage.morphology.disk と同じ: x² + y² ≤ r² の格子点)。

    Args:
        r: 半径(画素)。

    Returns:
        (2r + 1, 2r + 1) の uint8。
    """
    ax = np.arange(-r, r + 1)
    x, y = np.meshgrid(ax, ax)
    return (x ** 2 + y ** 2 <= r ** 2).astype(np.uint8)


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """マスクの輪郭の 1 画素の幅の地図(公式の _seg2bmap。大きさは変えない場合)。

    Args:
        seg: (H, W) の 0/1。

    Returns:
        (H, W) の bool。
    """
    seg = seg.astype(bool)
    e, s, se = np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg)
    e[:, :-1], s[:-1, :], se[:-1, :-1] = seg[:, 1:], seg[1:, :], seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def j_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """J: 領域の IoU(公式の db_eval_iou)。

    Args:
        gt: 正解 (H, W) の 0/1。
        pred: 予測 (H, W) の 0/1。

    Returns:
        IoU(両方とも空なら 1)。
    """
    gt, pred = gt.astype(bool), pred.astype(bool)
    union = np.sum(gt | pred)
    return 1.0 if np.isclose(union, 0) else float(np.sum(gt & pred) / union)


def f_score(gt: np.ndarray, pred: np.ndarray, bound_th: float = 0.008) -> float:
    """F: 輪郭の一致度(輪郭の画素の適合率と再現率の F 値。公式の f_measure)。

    Args:
        gt: 正解 (H, W) の 0/1。
        pred: 予測 (H, W) の 0/1。
        bound_th: 輪郭の一致を許す幅(1 未満なら対角線に対する割合で、画素は切り上げ。公式は 0.008)。

    Returns:
        F 値。
    """
    import cv2
    bound_pix = bound_th if bound_th >= 1 else int(np.ceil(bound_th * np.linalg.norm(pred.shape)))
    fg_b, gt_b = _seg2bmap(pred), _seg2bmap(gt)
    k = _disk(bound_pix)
    fg_dil, gt_dil = cv2.dilate(fg_b.astype(np.uint8), k), cv2.dilate(gt_b.astype(np.uint8), k)
    n_fg, n_gt = fg_b.sum(), gt_b.sum()
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision, recall = float((fg_b * gt_dil).sum() / n_fg), float((gt_b * fg_dil).sum() / n_gt)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate_jf(gt: dict, pred_tracks: dict) -> dict:
    """J&F(マスクを与えて追う方式の評価)。

    各物体は、初めて写ったフレームで正解のマスクを与え(そのフレームは評価から除く。DAVIS の公式が最初のフレームを除くのと同じ)、
    その後のフレームで予測を評価する。DAVIS の公式と同じく、動画の最後のフレームも除く。物体ごとにフレームの平均をとり、
    全部の物体で平均する。J&F = (J + F) / 2。

    Args:
        gt: 共通の形式の正解(dict)。
        pred_tracks: {正解の物体の id: [フレームごとの RLE か None](予測。与えたフレームより前は使わない)}。

    Returns:
        {"J&F", "J", "F", "n_objects"}(0〜1)。
    """
    videos = {v["id"]: v for v in gt["videos"]}
    js, fs = [], []
    for a in gt["annotations"]:
        segs = a["segmentations"]
        first = next((t for t, s in enumerate(segs) if s), None)
        if first is None or a["id"] not in pred_tracks:
            continue
        v = videos[a["video_id"]]
        pred = pred_tracks[a["id"]]
        jj, ff = [], []
        for t in range(first + 1, v["length"] - 1):
            g = mask_util.decode(segs[t]) if segs[t] else np.zeros((v["height"], v["width"]), np.uint8)
            p = mask_util.decode(pred[t]) if pred[t] else np.zeros((v["height"], v["width"]), np.uint8)
            jj.append(j_score(g, p)); ff.append(f_score(g, p))
        if jj:  # 公式の db_statistics の平均(nanmean)
            js.append(float(np.nanmean(jj))); fs.append(float(np.nanmean(ff)))
    j, f = (float(np.mean(js)) if js else 0.0), (float(np.mean(fs)) if fs else 0.0)
    return {"J&F": (j + f) / 2, "J": j, "F": f, "n_objects": len(js)}
