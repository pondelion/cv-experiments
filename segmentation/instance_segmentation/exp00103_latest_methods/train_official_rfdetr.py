"""exp00103: 公式の RF-DETR-Seg の学習と評価。中身は exp00101 の train_official_rfdetr.py と同じで、出力だけを exp00103 の
runs/・results/ にする(exp00101 の train.py を使う train.py と同じやり方)。

exp00103 では、ViT-S の方式(EoMT・PMT)が低かったのが ViT の backbone のためかを確かめるため、公式の RF-DETR-Seg の backbone を
同じ ViT-S/16(ImageNet-1K)にして学習する。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00103_latest_methods/train_official_rfdetr.py \
        --name rfdetr_seg_vits_in1k_cls8_2k_ep100 --epochs 100 --encoder timm/deit3_small_patch16_224.fb_in1k \
        --out-feature-indexes 2 5 8 11 --batch-size 4 --grad-accum 4
"""
import runpy

import common

runpy.run_path(str(common.EXP_DIR.parent / "exp00101_arch_comparison" / "train_official_rfdetr.py"), run_name="__main__")
