"""途中で止めた画像の学習(train_image.py)の、最後の評価だけを行って results/{名前}.json を作る。

学習の最後(lib_common/train.py の train の最後)と同じこと: 最良の重み(runs/{名前}/best.pt。MinVIS は dev の動画の AP、画像の
モデル単体は dev のフレームの mask AP が最良だった重み)で eval_sets の最後の評価(evaluate_final)と、動画の評価がある学習(MinVIS)
だけ動画の test のセットの MinVIS の評価(evaluate_video)を行い、学習の途中の評価の記録
(train_log.jsonl の eval・test・video_eval)を history に並べて保存する。止めたエポック("stopped_at_epoch")も書く。

使い方(リポジトリ直下で):
    .venv/bin/python -u segmentation/instance_segmentation/exp00502_video_arch_comparison/finalize_stopped.py \
        --name minvis_cnxtb_dinov3_cls8_ep100_const --max-vram-gb 6
"""
import argparse
import json
import os
from dataclasses import asdict

import common
import torch

from lib_common import REPO_ROOT  # noqa: E402
from lib_common.datasets import load_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.train import (TrainConfig, build_model, class_image_counts,  # noqa: E402
                                                                 evaluate_final, evaluate_video)


def history_from_log(path) -> tuple[list[dict], int]:
    """train_log.jsonl の途中の評価の記録を、train の history と同じ並び(エポックごと)にする。

    Args:
        path: train_log.jsonl。

    Returns:
        (history, 最後に評価したエポック)。
    """
    hist, by_epoch = [], {}
    for line in open(path):
        d = json.loads(line)
        if "eval" in d:
            rec = dict(d["eval"]); by_epoch[rec["epoch"]] = rec; hist.append(rec)
        elif "test" in d:
            t = dict(d["test"]); e, s = t.pop("epoch"), t.pop("set")
            by_epoch[e].setdefault("test", {})[s] = t
        elif "video_eval" in d:
            v = d["video_eval"]
            by_epoch[v["epoch"]]["video" if "dev" in v["set"] else "video_test"] = v
    return hist, max(by_epoch) if by_epoch else 0


def main() -> None:
    """最良の重みで最後の評価を行い、結果を保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--max-vram-gb", type=float, default=6.0)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    out_dir = common.RUNS_DIR / a.name
    cfg_d = json.loads((out_dir / "config.json").read_text())
    names = TrainConfig.__dataclass_fields__.keys()
    cfg = TrainConfig(**{k: v for k, v in cfg_d.items() if k in names})
    history, last_epoch = history_from_log(out_dir / "train_log.jsonl")
    # train と同じ選び方: 動画の評価がある学習(MinVIS)は dev の動画の AP、ない学習(画像のモデル単体)は dev のフレームの mask AP
    if any("video" in h for h in history):
        best = max((h for h in history if "video" in h), key=lambda h: h["video"]["AP"])
    else:
        best = max(history, key=lambda h: h["segm"]["AP"])
    # 2026-11-23 より前の学習中の動画の評価は、画像のセットと動画のセットのクラスの並びの違い(person と car が入れ替わる)を
    # 直す前の値なので、history に残さない(画像の評価 eval・test は正しい)。最後の動画の評価(下の evaluate_video)は直した後の値
    for h in history:
        h.pop("video", None), h.pop("video_test", None)
    model = build_model(cfg, len(load_set(cfg.eval_sets[0])["categories"])).cuda()
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cuda"))
    model.eval()
    config = {k: os.path.relpath(v, REPO_ROOT) if isinstance(v, str) and v.startswith(str(REPO_ROOT)) else v
              for k, v in asdict(cfg).items()}
    result = {"name": cfg.name, "config": config, "history": history, "best_epoch": best["epoch"],
              "stopped_at_epoch": last_epoch, "eval": evaluate_final(model, cfg),
              "train_class_images": class_image_counts(cfg.train_json)}
    if cfg.video_test_set:  # 動画の評価がある学習(MinVIS)だけ
        with open(out_dir / "train_log.jsonl", "a") as log:
            result["video_eval"] = evaluate_video(model, cfg, cfg.video_test_set, best["epoch"], log)
    save_metrics(common.RESULTS_DIR / f"{cfg.name}.json", result)
    print("best_epoch", best["epoch"], "stopped", last_epoch,
          {s: round(m["segm"]["AP"] * 100, 1) for s, m in result["eval"].items() if "segm" in m},
          "video", round(result["video_eval"]["AP"] * 100, 1) if "video_eval" in result else None, flush=True)


if __name__ == "__main__":
    main()
