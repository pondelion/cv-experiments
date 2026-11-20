"""exp00105: 学習済みの重み(best.pt)で予測し直して、クラスごとの mask AP と boundary AP(マスクの輪郭の正確さに厳しい AP)を出す。

学習のときの最後の評価の結果(results/{名前}.json)は上書きせず、results/{名前}_boundary.json に保存する。予測は学習のときの
最後の評価と同じ手順(train.predict)。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00105_backbone/eval_boundary.py \
        --runs segmentation/instance_segmentation/exp00105_backbone/runs/gtr_light_hgv2b2_cls8_2k_ep100 ...
"""
import argparse
import dataclasses
import json
from pathlib import Path

import common

import torch  # noqa: E402

from lib_common.datasets import load_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.train import TrainConfig, build_model, predict  # noqa: E402


def main() -> None:
    """学習ごとに予測して評価し、保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="学習の出力のフォルダ(config.json と best.pt がある)")
    p.add_argument("--set", default="coco_val_test_cls8")
    a = p.parse_args()
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    for run in map(Path, a.runs):
        cfg = TrainConfig(**{k: v for k, v in json.loads((run / "config.json").read_text()).items() if k in fields})
        model = build_model(cfg, len(load_set(a.set)["categories"]))
        model.load_state_dict(torch.load(run / "best.pt", map_location="cuda"))
        model.eval()
        preds, _ = predict(model, a.set, cfg)
        m = evaluate(a.set, preds)
        out = {"run": run.name, "set": a.set, "segm_AP": m["segm"]["AP"], "boundary_AP": m["boundary"]["AP"],
               "per_class_segm_AP": m["per_class_segm_AP"], "per_class_boundary_AP": m["per_class_boundary_AP"]}
        res_dir = Path(cfg.results_dir)
        save_metrics(res_dir / f"{run.name}_boundary.json", out)
        print(run.name, f"mask AP {100 * out['segm_AP']:.1f} boundary AP {100 * out['boundary_AP']:.1f}",
              {k: f"{100 * out['per_class_segm_AP'][k]:.1f}/{100 * v:.1f}" for k, v in out["per_class_boundary_AP"].items()},
              flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
