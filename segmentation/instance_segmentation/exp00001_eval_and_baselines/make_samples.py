"""結果のページに載せる見本の画像を作る(正解と、評価した各モデルの予測)。

runs/ に保存した予測を読み、results/ の評価の結果にある閾値(dev で PQ が最大になる値)以上の予測を描く。
評価を済ませたモデルの分だけ作るので、評価の途中でも動かせる。

使い方: .venv/bin/python segmentation/instance_segmentation/exp00001_eval_and_baselines/make_samples.py
"""
import json

import common

from segmentation.instance_segmentation.lib_common.samples import SAMPLE_COUNTS, render

SAMPLES_DIR = common.RESULTS_DIR / "samples"
# RF-DETR-Seg は比較対象なので、見本はファイルの大きさを抑えるため 3 サイズ(一番小さい・中くらい・一番大きい)だけにする
MODELS = ["sam3"] + [f"rfdetr_{s}" for s in ["nano", "large", "2xlarge"]]


def main() -> None:
    """セットごとに、正解と各モデルの予測の見本の画像を作る。"""
    for set_name in SAMPLE_COUNTS:
        render(SAMPLES_DIR, set_name, "gt")
        for model in MODELS:
            pred_path = common.RUNS_DIR / f"{model}_{set_name}_predictions.json"
            result_path = common.RESULTS_DIR / f"{model}_{set_name}.json"
            if not (pred_path.exists() and result_path.exists()):
                continue
            threshold = json.loads(result_path.read_text())["pq"]["score_threshold"]
            render(SAMPLES_DIR, set_name, model, json.loads(pred_path.read_text()), threshold)
            print(f"{set_name} {model}: 閾値 {threshold}")


if __name__ == "__main__":
    main()
