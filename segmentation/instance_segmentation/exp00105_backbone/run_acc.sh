#!/usr/bin/env bash
# exp00105: 精度重視の backbone の学習を、引数の順に 1 本ずつ流す(速度バランス型の run_light.sh と同時に動かせる)。
# 方式は exp00103 の A のまま(GTR 系 + hybrid encoder、後段は既定の構成)。decoder の自己注意は、計算が同じで速い形
# (sdpa_self_attn)にする。batch 16 のまま、1 本あたりの GPU のメモリの上限を 15GB にし、メモリが足りずに止まったときだけ、
# micro-batch size × gradient accumulation steps を 8 × 2 → 4 × 4 → 2 × 8 に順に下げてやり直す(batch size は 16 のまま)。
# encoder に BN があるので、micro-batch size は載る範囲で大きい方を使う(OOM で止まった回のログは {名前}_oom_accum{n}.log)。
# 使い方(リポジトリ直下で): bash segmentation/instance_segmentation/exp00105_backbone/run_acc.sh 短い名前=timm の名前 ...
# 大きい backbone(Qwen3-VL の画像の部分)を 1 本だけで流すとき: VRAM=28 EXTRA=--grad-checkpoint bash .../run_acc.sh ...
E=segmentation/instance_segmentation/exp00105_backbone
KW='{"sdpa_self_attn": true}'
VRAM=${VRAM:-15}  # 1 本あたりの GPU のメモリの上限(GB)
for pair in "$@"; do
  short=${pair%%=*}; bb=${pair#*=}; n=gtr_${short}_cls8_2k_ep100
  train() { .venv/bin/python -u $E/train.py --arch gtr_seg --gtr-encoder hybrid --epochs 100 --backbone "$bb" --name $n \
              --max-vram-gb "$VRAM" --model-kwargs "$KW" $EXTRA "$@" > $E/runs/$n.log 2>&1; }
  train; code=$?
  for accum in 2 4 8; do  # メモリが足りないときだけ、micro-batch size を半分ずつにする
    [ $code -ne 0 ] && grep -q OutOfMemoryError $E/runs/$n.log || break
    mv $E/runs/$n.log $E/runs/${n}_oom_accum$((accum / 2)).log; rm -rf "${E:?}/runs/$n"
    train --grad-accum $accum; code=$?
  done
  echo "$n exit=$code $(grep '\[final\] coco_val_test_cls8: mask\|\[final\] voc_val_cls8: mask' $E/runs/$n.log | cut -c1-90 | tr '\n' ' ')"
done
