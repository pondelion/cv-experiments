"""exp00502: 画像のインスタンスセグメンテーションのモデル(MinVIS・DVIS の土台の Mask2Former)を、VIPSeg のフレームで学習する。

中身は exp00101 の train.py と同じで、出力だけを exp00502 の runs/・results/ にする(exp00105 の train.py と同じやり方)。
引数も exp00101 の train.py と同じ。

使い方(例):
    .venv/bin/python -u segmentation/instance_segmentation/exp00502_video_arch_comparison/train_image.py --arch mask2former \
        --backbone convnext_base.dinov3_lvd1689m --name m2f_cnxtb_dinov3_frames_cls8 --train-set vipseg_frames_cls8_train \
        --eval-sets vipseg_frames_cls8_dev vipseg_frames_cls8_val --eval-n 4699 --test-sets vipseg_frames_cls8_val \
        --epochs 22 --eval-every 2 --test-every 2 --schedule constant --no-compile
"""
import runpy

import common

runpy.run_path(str(common.EXP_DIR.parent / "exp00101_arch_comparison" / "train.py"), run_name="__main__")
