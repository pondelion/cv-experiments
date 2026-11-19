#!/usr/bin/env bash
# exp00105: 速度バランス型の backbone の学習を、引数の順に 1 本ずつ流す(この列を 2 つ同時に動かす)。
# 方式は exp00104 の G1 + E2(64) + E3。batch 16 のまま、1 本あたりの GPU のメモリの上限を 15GB にする。
# メモリが足りずに止まったときだけ、batch を 8 × 2 回の勾配の累積にしてやり直す(1 回の更新あたり 16 枚のまま)。
# 使い方(リポジトリ直下で): bash segmentation/instance_segmentation/exp00105_backbone/run_light.sh 短い名前=timm の名前 ...
E=segmentation/instance_segmentation/exp00105_backbone
KW='{"dim": 192, "ffn": 512, "mask_dim": 64, "enc_block": "dw"}'
for pair in "$@"; do
  short=${pair%%=*}; bb=${pair#*=}; n=gtr_light_${short}_cls8_2k_ep100
  train() { .venv/bin/python -u $E/train.py --arch gtr_seg --gtr-encoder hybrid --epochs 100 --backbone "$bb" --name $n \
              --max-vram-gb 15 --model-kwargs "$KW" "$@" > $E/runs/$n.log 2>&1; }
  train; code=$?
  if [ $code -ne 0 ] && grep -q OutOfMemoryError $E/runs/$n.log; then
    mv $E/runs/$n.log $E/runs/${n}_oom.log; rm -rf "${E:?}/runs/$n"
    train --grad-accum 2; code=$?
  fi
  echo "$n exit=$code $(grep '\[final\] coco_val_test_cls8: mask\|\[final\] voc_val_cls8: mask' $E/runs/$n.log | cut -c1-90 | tr '\n' ' ')"
done
