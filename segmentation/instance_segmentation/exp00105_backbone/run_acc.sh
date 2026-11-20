#!/usr/bin/env bash
# exp00105: 精度重視の backbone の学習を、引数の順に 1 本ずつ流す(速度バランス型の run_light.sh と同時に動かせる)。
# 方式は exp00103 の A のまま(GTR 系 + hybrid encoder、後段は既定の構成)。decoder の自己注意は、計算が同じで速い形
# (sdpa_self_attn)にする。batch 16 のまま、1 本あたりの GPU のメモリの上限を 15GB にし、メモリが足りずに止まったときだけ、
# 1 回に載せる枚数を 8 × 2 回 → 4 × 4 回 → 2 × 8 回の勾配の累積に順に下げてやり直す(1 回の更新あたり 16 枚のまま)。
# encoder に BN があるので、載せる枚数は載る範囲で大きい方を使う(使った累積の回数はログの名前 {名前}_oom_accum{n}.log でわかる)。
# 使い方(リポジトリ直下で): bash segmentation/instance_segmentation/exp00105_backbone/run_acc.sh 短い名前=timm の名前 ...
E=segmentation/instance_segmentation/exp00105_backbone
KW='{"sdpa_self_attn": true}'
for pair in "$@"; do
  short=${pair%%=*}; bb=${pair#*=}; n=gtr_${short}_cls8_2k_ep100
  train() { .venv/bin/python -u $E/train.py --arch gtr_seg --gtr-encoder hybrid --epochs 100 --backbone "$bb" --name $n \
              --max-vram-gb 15 --model-kwargs "$KW" "$@" > $E/runs/$n.log 2>&1; }
  train; code=$?
  for accum in 2 4 8; do  # メモリが足りないときだけ、載せる枚数を半分ずつにする
    [ $code -ne 0 ] && grep -q OutOfMemoryError $E/runs/$n.log || break
    mv $E/runs/$n.log $E/runs/${n}_oom_accum$((accum / 2)).log; rm -rf "${E:?}/runs/$n"
    train --grad-accum $accum; code=$?
  done
  echo "$n exit=$code $(grep '\[final\] coco_val_test_cls8: mask\|\[final\] voc_val_cls8: mask' $E/runs/$n.log | cut -c1-90 | tr '\n' ' ')"
done
