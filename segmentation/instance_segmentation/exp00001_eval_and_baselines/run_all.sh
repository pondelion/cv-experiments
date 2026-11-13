#!/bin/bash
# exp00001 の比較対象の評価を、GPU を 1 つずつ使って順に流す(遅延を正しく測るため、同時には動かさない)。
# test・voc では、PQ などのスコアの閾値に dev で選んだ値を使う。
# 使い方(リポジトリ直下で): bash segmentation/instance_segmentation/exp00001_eval_and_baselines/run_all.sh
set -e
cd "$(dirname "$0")/../../.."
EXP=segmentation/instance_segmentation/exp00001_eval_and_baselines
PY=".venv/bin/python -u"

$PY $EXP/eval_sam3.py --set coco_val_dev
for s in coco_val_test voc_val; do
  $PY $EXP/eval_sam3.py --set $s --threshold-from $EXP/results/sam3_coco_val_dev.json
done

for size in nano small medium large xlarge 2xlarge; do
  $PY $EXP/eval_rfdetr.py --size $size --set coco_val_dev
  for s in coco_val_test voc_val; do
    $PY $EXP/eval_rfdetr.py --size $size --set $s --threshold-from $EXP/results/rfdetr_${size}_coco_val_dev.json
  done
done
