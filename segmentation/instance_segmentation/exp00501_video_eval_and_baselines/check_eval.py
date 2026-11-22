"""exp00501: 動画の AP の評価(lib_common/video_eval.py)の計算を確かめる。

確かめること:
    1. 正解をそのまま予測にすると AP = 1
    2. 同じクラスの 2 つの物体の ID を動画の途中から入れ替えると下がる
    3. 物体ごとに半分のフレームのマスクを抜くと下がる
    4. 正解をわざと崩した予測(マスクを数画素ずらす・点数をばらつかせる・一部を消す)で、公式の YouTube-VIS の評価のコード
       (ytvos.py・ytvoseval.py。--official-dir で置き場所を渡したときだけ)と値が一致する

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00501_video_eval_and_baselines/check_eval.py --set vipseg_vis_val \
        [--official-dir 公式の ytvis_api のフォルダ]
"""
import argparse
import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile

import common

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from lib_common import DATASETS_DIR  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_eval import evaluate_vis  # noqa: E402


def gt_as_pred(gt: dict) -> list[dict]:
    """正解をそのまま予測の形にする(点数は全部 1)。"""
    return [{"video_id": a["video_id"], "category_id": a["category_id"], "score": 1.0, "segmentations": a["segmentations"]}
            for a in gt["annotations"]]


def shift_rle(rle: dict, dx: int) -> dict:
    """RLE のマスクを横に dx 画素ずらす(端ははみ出た分を捨てる)。"""
    m = mask_util.decode(rle)
    out = np.zeros_like(m)
    if dx >= 0:
        out[:, dx:] = m[:, : m.shape[1] - dx]
    else:
        out[:, :dx] = m[:, -dx:]
    r = mask_util.encode(np.asfortranarray(out))
    r["counts"] = r["counts"].decode("ascii")
    return r


def perturbed(gt: dict, seed: int = 0) -> list[dict]:
    """正解を崩した予測: 物体の 20% を消し、残りはマスクを 0〜12 画素ずらし、点数を乱数にし、偽の物体を 10% 足す。"""
    rng = random.Random(seed)
    preds = []
    for a in gt["annotations"]:
        if rng.random() < 0.2:
            continue
        dx = rng.randint(0, 12)
        segs = [shift_rle(s, dx) if s else None for s in a["segmentations"]]
        preds.append({"video_id": a["video_id"], "category_id": a["category_id"], "score": rng.random(), "segmentations": segs})
    for a in rng.sample(gt["annotations"], len(gt["annotations"]) // 10):  # 別のクラスの偽の物体
        other = rng.choice([c["id"] for c in gt["categories"] if c["id"] != a["category_id"]])
        preds.append({"video_id": a["video_id"], "category_id": other, "score": rng.random() * 0.5, "segmentations": a["segmentations"]})
    return preds


def official_ap(gt: dict, preds: list[dict], official_dir: str) -> list[float]:
    """公式の YouTube-VIS の評価のコードで AP などを計算する(比べる相手として使うだけ)。"""
    sys.path.insert(0, official_dir)
    spec = importlib.util.spec_from_file_location("ytvos", f"{official_dir}/ytvos.py")
    ytvos = importlib.util.module_from_spec(spec); spec.loader.exec_module(ytvos)
    sys.modules["ytvos"] = ytvos
    spec = importlib.util.spec_from_file_location("ytvoseval", f"{official_dir}/ytvoseval.py")
    ytvoseval = importlib.util.module_from_spec(spec); spec.loader.exec_module(ytvoseval)
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f, contextlib.redirect_stdout(io.StringIO()):
        json.dump(gt, f); f.flush()
        g = ytvos.YTVOS(f.name)
        d = g.loadRes(preds)
        e = ytvoseval.YTVOSeval(g, d, "segm")
        e.evaluate(); e.accumulate(); e.summarize()
    return [float(x) for x in e.stats[:8]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="vipseg_vis_val")
    p.add_argument("--official-dir", default=None)
    a = p.parse_args()
    gt = json.loads((DATASETS_DIR / "sets" / f"{a.set}.json").read_text())
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10"]
    r = evaluate_vis(gt, gt_as_pred(gt))
    print("1. 正解をそのまま:", {k: round(r[k], 4) for k in keys})
    # 2. 同じ動画・同じクラスの 2 つの物体の、後半のフレームのマスクを入れ替える
    preds = gt_as_pred(gt)
    by = {}
    for i, q in enumerate(preds):
        by.setdefault((q["video_id"], q["category_id"]), []).append(i)
    swapped = 0
    for idx in by.values():
        if len(idx) >= 2:
            i, j = idx[:2]
            t = len(preds[i]["segmentations"]) // 2
            si, sj = preds[i]["segmentations"], preds[j]["segmentations"]
            preds[i] = dict(preds[i], segmentations=si[:t] + sj[t:])
            preds[j] = dict(preds[j], segmentations=sj[:t] + si[t:])
            swapped += 1
    print(f"2. ID を途中で入れ替え({swapped} 組):", {k: round(v, 4) for k, v in evaluate_vis(gt, preds).items() if k in keys})
    preds = [dict(q, segmentations=[s if k % 2 == 0 else None for k, s in enumerate(q["segmentations"])]) for q in gt_as_pred(gt)]
    print("3. 半分のフレームを抜く:", {k: round(v, 4) for k, v in evaluate_vis(gt, preds).items() if k in keys})
    preds = perturbed(gt)
    mine = evaluate_vis(gt, preds)
    print("4. 崩した予測(この実装):", {k: round(mine[k], 4) for k in keys})
    if a.official_dir:
        off = official_ap(gt, preds, a.official_dir)
        print("   崩した予測(公式のコード):", {k: round(v, 4) for k, v in zip(keys, off)})
        print("   差の最大:", max(abs(mine[k] - v) for k, v in zip(keys, off)))


if __name__ == "__main__":
    main()
