"""exp00501: 動画インスタンスセグメンテーションの評価・学習のセットを、共通の形式(YouTube-VIS と同じ JSON)で作る。

VIPSeg(物体の 58 クラス)から次の 3 つを作り、`datasets/sets/` に置く:
    vipseg_vis_train: 公式の train から dev を除いた残り(学習)
    vipseg_vis_dev:   公式の train から seed 0 で選んだ 200 本(学習の途中の評価・最良の重みの選択。val への過適合を防ぐ)
    vipseg_vis_val:   公式の val の 343 本(節目の評価)
公式の test(387 本)は正解がないので作らない。フレームは公式の前処理と同じく高さ 720 に縮めて `VIPSeg_720P/` に保存する
(lib_common/video_data.py)。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00501_video_eval_and_baselines/make_sets.py
    (動作確認: --limit 5 で各セットの先頭 5 本だけ、セット名に _first5 を付けて作る)
"""
import argparse
import json
import random

import common

from lib_common import DATASETS_DIR  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_data import convert_vipseg  # noqa: E402

DEV_VIDEOS = 200  # dev の本数(train 2,806 本の約 7%。1 本あたり中央値 15 フレームで約 3,000 フレーム)


def main() -> None:
    """VIPSeg のセットを作って保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="動作確認用: 各セットの先頭この本数だけ")
    a = p.parse_args()
    root = DATASETS_DIR / "vipseg" / "VIPSeg"
    read = lambda s: [l.strip() for l in (root / f"{s}.txt").read_text().splitlines() if l.strip()]
    train, val = read("train"), read("val")
    dev = sorted(random.Random(0).sample(train, DEV_VIDEOS))
    splits = {"vipseg_vis_train": sorted(set(train) - set(dev)), "vipseg_vis_dev": dev, "vipseg_vis_val": val}
    if a.limit:
        splits = {f"{k}_first{a.limit}": v[: a.limit] for k, v in splits.items()}
    for name, d in convert_vipseg(root, splits, a.workers).items():
        path = DATASETS_DIR / "sets" / f"{name}.json"
        path.write_text(json.dumps(d))
        n_frames = sum(v["length"] for v in d["videos"])
        print(f"{name}: 動画 {len(d['videos'])} 本・フレーム {n_frames} 枚・物体 {len(d['annotations'])} 個 → {path.name}", flush=True)


if __name__ == "__main__":
    main()
