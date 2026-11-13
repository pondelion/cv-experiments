"""インスタンスセグメンテーションの、画素単位の(閾値で当たり・外れを数えない)指標。

AP 系の指標(evaluate.py)は IoU を閾値で区切って当たり・外れを数えるので、マスクの形がどれだけ正確かは直接は見えない。
ここでは、マスクの IoU そのものを平均する指標を計算する。定義は instance_segmentation/README.md の「評価の設定」を参照。

    best_iou_mean: 正解ごとの最良の IoU の平均(スコアの閾値なし)
    PQ・SQ・RQ:    予測と正解を 1 対 1 で対応させた Panoptic Quality(物体の部分だけ)。スコアの閾値が要る
    semantic mIoU・前景の IoU: 予測を画素ごとのクラスの地図にしたときの IoU。スコアの閾値が要る

群衆の領域(iscrowd=1)の扱い:
    - best_iou_mean・PQ の正解には含めない
    - PQ: 予測の画素の半分以上が同じクラスの群衆の領域に入っていれば、誤検出に数えない(COCO の panoptic の評価と同じ決まり)
    - semantic mIoU・前景の IoU: 群衆の領域の画素は計算に入れない(正解で別のクラスの物体が重なる画素も入れない)

形の記号: D = (画像, クラス) ごとの予測の数、G = 同じく正解の数、k = クラスの数 + 1(背景)、h・w = 画像の縦・横
"""
from collections import defaultdict

import numpy as np
from einops import rearrange
from faster_coco_eval import COCO
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

THRESHOLD_CANDIDATES = np.round(np.arange(0.05, 1.0, 0.05), 2)  # PQ が最大になるスコアの閾値を選ぶときの候補
MAX_PER_CLASS = 100  # 1 画像・1 クラスあたりに使う予測の数の上限(AP と同じ)


class PixelMetrics:
    """1 つの評価のセットと予測について、画素単位の指標を計算する。

    初めに、画像 × クラスごとに全部の予測と正解のマスクの IoU を計算しておき、スコアの閾値を変えて何度でも PQ を計算できるようにする。

    Args:
        gt: 正解の COCO。
        predictions: COCO の結果の形式の予測のリスト(segmentation は RLE)。
    """

    def __init__(self, gt: COCO, predictions: list[dict]):
        self.gt = gt
        self.cat_ids = sorted(gt.cats)
        gts, crowds = defaultdict(list), defaultdict(list)  # (画像, クラス) → 正解の RLE のリスト
        for ann in gt.dataset["annotations"]:
            (crowds if ann["iscrowd"] else gts)[(ann["image_id"], ann["category_id"])].append(gt.annToRLE(ann))
        dts = defaultdict(list)  # (画像, クラス) → 予測のリスト(スコアの高い順、上位 MAX_PER_CLASS 個)
        for p in predictions:
            dts[(p["image_id"], p["category_id"])].append(p)
        for key in dts:
            dts[key] = sorted(dts[key], key=lambda p: -p["score"])[:MAX_PER_CLASS]
        self.gts, self.crowds, self.dts = gts, crowds, dts

        # 画像 × クラスごとの、予測の数 D × 正解の数 G の IoU と、各予測の画素のうち群衆の領域に入る割合
        self.pairs = {}
        for key in set(gts) | set(dts):
            d = [p["segmentation"] for p in dts.get(key, [])]
            g = gts.get(key, [])
            ious = np.asarray(mask_utils.iou(d, g, [0] * len(g))) if d and g else np.zeros((len(d), len(g)))  # (D, G)
            in_crowd = np.zeros(len(d))
            if d and key in crowds:
                crowd = mask_utils.merge(crowds[key])
                # 群衆扱いの IoU = 重なり ÷ 予測の面積: (D, 1) -> (D,)
                in_crowd = rearrange(np.asarray(mask_utils.iou(d, [crowd], [1])), "d 1 -> d")
            scores = np.array([p["score"] for p in dts.get(key, [])])
            self.pairs[key] = (ious, scores, in_crowd)

    def best_iou_mean(self) -> float:
        """正解ごとに、同じクラスの予測との IoU の最大値を取り、全部の正解で平均する(予測がなければ 0)。

        Returns:
            最大の IoU の平均(0〜1)。
        """
        best = [ious.max(axis=0) if len(ious) else np.zeros(ious.shape[1]) for ious, _, _ in self.pairs.values()]
        best = np.concatenate([b for b in best if len(b)])
        return float(best.mean())

    def pq(self, score_threshold: float) -> dict:
        """スコアが閾値以上の予測だけを使って、PQ・SQ・RQ を計算する(クラスごとに計算してクラスで平均)。

        Args:
            score_threshold: 使う予測のスコアの下限。

        Returns:
            {"PQ", "SQ", "RQ"(クラスの平均), "per_class_PQ": {クラス名: PQ}, "TP", "FP", "FN"(全クラスの合計)}。
        """
        tp, fp, fn, iou_sum = (defaultdict(float) for _ in range(4))
        for (_, cat), (ious, scores, in_crowd) in self.pairs.items():
            keep = scores >= score_threshold
            sub = ious[keep]  # (閾値以上の予測の数, 正解の数)
            matched = np.zeros(len(sub), dtype=bool)
            if sub.size:
                rows, cols = linear_sum_assignment(sub, maximize=True)  # IoU の合計が最大になる 1 対 1 の対応
                ok = sub[rows, cols] > 0.5
                matched[rows[ok]] = True
                tp[cat] += ok.sum()
                iou_sum[cat] += sub[rows[ok], cols[ok]].sum()
            fn[cat] += ious.shape[1] - matched.sum()
            fp[cat] += (~matched & (in_crowd[keep] <= 0.5)).sum()

        per_class = {}
        for cat in self.cat_ids:
            if tp[cat] + fn[cat] == 0:  # 正解が 1 つもないクラスは平均に入れない
                continue
            denom = tp[cat] + 0.5 * fp[cat] + 0.5 * fn[cat]
            sq = iou_sum[cat] / tp[cat] if tp[cat] else 0.0
            rq = tp[cat] / denom
            per_class[cat] = (sq * rq, sq, rq)
        values = np.array(list(per_class.values()))  # (クラスの数, 3) 列は PQ・SQ・RQ
        return {
            "score_threshold": float(score_threshold),
            "PQ": float(values[:, 0].mean()), "SQ": float(values[:, 1].mean()), "RQ": float(values[:, 2].mean()),
            "per_class_PQ": {self.gt.cats[c]["name"]: float(v[0]) for c, v in per_class.items()},
            "TP": int(sum(tp.values())), "FP": int(sum(fp.values())), "FN": int(sum(fn.values())),
        }

    def best_threshold(self) -> float:
        """THRESHOLD_CANDIDATES のうち、PQ が最大になるスコアの閾値を返す(dev のセットで選ぶ用)。

        Returns:
            スコアの閾値。
        """
        return float(max(THRESHOLD_CANDIDATES, key=lambda t: self.pq(t)["PQ"]))

    def semantic(self, score_threshold: float) -> dict:
        """予測を画素ごとのクラスの地図にして、クラスごとの IoU(semantic mIoU)と前景の IoU を計算する。

        予測の地図は、スコアが閾値以上の予測を、スコアの低い順に塗る(重なった所はスコアの高い予測のクラスになる)。
        正解の地図は、群衆以外の正解の物体を塗る。群衆の領域の画素と、正解で 2 つ以上のクラスの物体が重なる画素
        (COCO の物体の画素の約 1.4%)は、クラスが決まらないので計算に入れない。
        データセット全体で、正解のクラス × 予測のクラスの画素の数を足し合わせてから IoU を計算する。

        Args:
            score_threshold: 使う予測のスコアの下限。

        Returns:
            {"mIoU"(正解に出てくるクラスの平均), "foreground_IoU", "per_class_IoU": {クラス名: IoU}}。
        """
        k = len(self.cat_ids) + 1  # 0 = 背景、1〜20 = クラス
        confusion = np.zeros((k, k), dtype=np.int64)  # (正解のクラス, 予測のクラス) の画素の数
        by_image = defaultdict(list)
        for (img, cat), preds in self.dts.items():
            by_image[img] += [(p["score"], cat, p["segmentation"]) for p in preds if p["score"] >= score_threshold]
        for img_id, info in self.gt.imgs.items():
            h, w = info["height"], info["width"]
            gt_map = np.zeros((h, w), dtype=np.int64)
            ignore = np.zeros((h, w), dtype=bool)
            for cat in self.cat_ids:
                for rle in self.gts.get((img_id, cat), []):
                    m = mask_utils.decode(rle).astype(bool)
                    ignore |= m & (gt_map > 0) & (gt_map != cat)  # 別のクラスの正解と重なる画素はクラスが決まらない
                    gt_map[m] = cat
                for rle in self.crowds.get((img_id, cat), []):
                    ignore |= mask_utils.decode(rle).astype(bool)
            pred_map = np.zeros((h, w), dtype=np.int64)
            for _, cat, rle in sorted(by_image.get(img_id, []), key=lambda x: x[0]):
                pred_map[mask_utils.decode(rle).astype(bool)] = cat
            valid = ~ignore
            counts = np.bincount(gt_map[valid] * k + pred_map[valid], minlength=k * k)
            confusion += rearrange(counts, "(g p) -> g p", g=k)  # (k²,) → (正解のクラス, 予測のクラス)

        inter = np.diag(confusion)
        union = confusion.sum(axis=0) + confusion.sum(axis=1) - inter
        present = confusion.sum(axis=1) > 0  # 正解に出てくるクラス
        per_class = {self.gt.cats[c]["name"]: float(inter[c] / union[c]) for c in self.cat_ids if present[c]}
        fg_inter = confusion[1:, 1:].sum()  # 正解も予測も物体(クラスは問わない)
        fg_union = confusion.sum() - confusion[0, 0]
        return {"score_threshold": float(score_threshold), "mIoU": float(np.mean(list(per_class.values()))),
                "foreground_IoU": float(fg_inter / fg_union), "per_class_IoU": per_class}
