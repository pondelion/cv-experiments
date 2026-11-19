"""exp00104: 学習済みのモデルを、学習し直さずに別の入力の大きさで評価する(入力を小さくしたときの精度の下がり方を見る)。

学習の記録(`runs/{名前}/config.json`)と評価に使った重み(`best.pt`)を読み、入力の一辺だけを変えて、学習のときの最後の評価と同じ手順
(最初のセット(dev)で PQ の閾値を選び、test・voc で測る)で評価する。データの変換(長い辺を一辺にそろえて埋める)も、その一辺で行う。
結果は `results/{名前}_eval{一辺}.json` に保存する(学習のときの結果のファイルと見本の画像は変えない)。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00104_light_configs/eval_size.py \
        --name gtr_g2_r50_cls8_2k_ep100 --size 384
"""
import argparse
import dataclasses
import json

import common

import torch  # noqa: E402

from lib_common.datasets import load_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.train import TrainConfig, build_model, predict  # noqa: E402


def main() -> None:
    """入力の一辺を変えて評価し、結果を保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="学習の名前(runs/{名前}/)")
    p.add_argument("--size", type=int, required=True, help="評価に使う入力の一辺")
    a = p.parse_args()
    saved = json.loads((common.RUNS_DIR / a.name / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in saved.items() if k in fields} | {"size": a.size})
    model = build_model(cfg, len(load_set(cfg.eval_sets[0])["categories"]))
    model.load_state_dict(torch.load(common.RUNS_DIR / a.name / "best.pt", map_location="cuda"))
    results, threshold = {}, None
    for s in cfg.eval_sets:  # 学習のときの最後の評価(train.evaluate_final)と同じ手順。見本の画像は書き出さない
        preds, ms = predict(model, s, cfg)
        m = evaluate(s, preds, score_threshold=threshold)
        threshold = threshold if threshold is not None else m["pq"]["score_threshold"]
        results[s] = m | {"ms_per_image_batch": ms}
        print(f"[final] {s}: mask AP {m['segm']['AP']:.4f} AP50 {m['segm']['AP50']:.4f} | box AP {m['bbox']['AP']:.4f} "
              f"| PQ {m['pq']['PQ']:.4f}", flush=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(common.RESULTS_DIR / f"{a.name}_eval{a.size}.json",
                 {"name": a.name, "train_size": saved["size"], "eval_size": a.size, "eval": results})


if __name__ == "__main__":
    main()
