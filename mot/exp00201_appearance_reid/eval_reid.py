"""段階 1: 追跡のデータの正解の切り出しで、見た目の特徴の ReID の評価(Rank-1・mAP・mINP)を測る(README の「評価」)。

切り出しは stride フレームおきのフレームの正解の物体(枠の幅・高さ 16 画素以上)。問いと探す先の間は gap フレーム以上あける
(主に見る 15 フレームと、長い隠れの 60 フレーム)。KITTI MOTS は車と歩行者を分けて測る(ID の付け方がクラスごと)。
結果: results/reid_{特徴の名前}.json。
"""
import argparse
import json
import time

import numpy as np
import torch

import common
from mot.lib_common.reid.data import load_instances, sample_instances
from mot.lib_common.reid.features import DINOv3Feature, extract
from mot.lib_common.reid.metrics import retrieval_metrics
from mot.lib_common.sets import load_sequences


def build_model(name: str) -> torch.nn.Module:
    """特徴の名前から、特徴を出すモデルを作る。

    Args:
        name: 特徴の名前("dinov3_vitb": F2)。

    Returns:
        切り出し (B, 3, H, W) → 特徴 (B, C) のモデル(crop_size の属性を持つ)。
    """
    if name == "dinov3_vitb":
        return DINOv3Feature("vit_base_patch16_dinov3.lvd1689m")
    raise ValueError(f"知らない特徴: {name}")


def main() -> None:
    """各セットの ReID の評価を測って保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True)
    p.add_argument("--sets", nargs="+", default=["mot17_val", "dancetrack_val", "kitti_mots_val"])
    p.add_argument("--stride", type=int, default=5, help="切り出しに使うフレームの間隔")
    p.add_argument("--gaps", type=int, nargs="+", default=[15, 60])
    p.add_argument("--max-per-set", type=int, default=None, help="動作確認用: セットごとの切り出しの数の上限")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    model = build_model(a.feature).to(a.device)
    result = {"feature": a.feature, "stride": a.stride, "gaps": a.gaps, "sets": {}}
    for s in a.sets:
        t0 = time.time()
        paths = {q.name: q.image_paths() for q in load_sequences(s)}
        ins = sorted(sample_instances(load_instances(s), a.stride), key=lambda i: (i.seq, i.frame))[: a.max_per_set]
        feats = extract(model, [(paths[i.seq][i.frame], i.box) for i in ins], model.crop_size, device=a.device)
        res = {}
        for cls in sorted({i.cls for i in ins}):
            sel = np.array([i.cls == cls for i in ins])
            frames = np.array([i.frame for i in ins])[sel]
            ids = np.array([i.track_id for i in ins])[sel]
            seqs = [i.seq for i, k in zip(ins, sel) if k]
            res[cls] = {f"gap{g}": retrieval_metrics(feats[sel], seqs, frames, ids, g) for g in a.gaps}
            res[cls]["n_crops"] = int(sel.sum())
        res["s_total"] = round(time.time() - t0, 1)
        result["sets"][s] = res
        print(s, json.dumps(res, ensure_ascii=False), flush=True)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    out = common.RESULTS_DIR / f"reid_{a.feature}{'_SMOKETEST' if a.max_per_set else ''}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
