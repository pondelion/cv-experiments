"""exp00502: 動画インスタンスセグメンテーションの方式を学習する(lib_common/video_train.py の共通の学習のループ)。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00502_video_arch_comparison/train.py --arch videomt \
        --name videomt_deit3s_vipseg
    (動作確認: --max-steps 3 --size 256 --clip-len 2 --batch-size 1 --dev-n 2 --eval-sets vipseg_vis_dev --max-vram-gb 3)
"""
import argparse
import dataclasses

import common

from segmentation.instance_segmentation.lib_common.video_train import VideoTrainConfig, train_video  # noqa: E402


def main() -> None:
    """引数で VideoTrainConfig の項目を上書きして学習する(名前の - は _ に読み替える)。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    for f in dataclasses.fields(VideoTrainConfig):
        if f.name in ("name", "out_dir", "results_dir"):
            continue
        if f.name == "eval_sets":
            p.add_argument("--eval-sets", nargs="+", default=None)
        else:
            to_bool = lambda s: s.lower() in ("1", "true", "yes")  # --norm-queries false などの真偽の項目
            typ = {int: int, float: float, str: str, bool: to_bool}.get(type(f.default), str)
            none_typ = str if f.name in ("init_weights", "init_model", "test_set") else float  # 既定が None の項目: パス・セット名は文字列、ほかは数
            p.add_argument("--" + f.name.replace("_", "-"), type=typ if f.default is not None else none_typ, default=None)
    a = p.parse_args()
    over = {k: v for k, v in vars(a).items() if v is not None and k != "name"}
    for k in ("max_steps", "dev_n", "epochs"):  # None が既定の整数の項目
        if k in over:
            over[k] = int(over[k])
    cfg = VideoTrainConfig(name=a.name, out_dir=str(common.RUNS_DIR), results_dir=str(common.RESULTS_DIR), **over)
    print(train_video(cfg)["eval"], flush=True)


if __name__ == "__main__":
    main()
