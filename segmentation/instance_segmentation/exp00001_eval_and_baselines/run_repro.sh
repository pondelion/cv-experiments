#!/bin/bash
# 公式の値が再現できるかの確認: COCO val2017 の 5,000 枚・80 クラスで、AP 系だけを評価する。
# 公式の値: RF-DETR-Seg の mask AP(README の表)、SAM 3 の box AP(論文の表 1)。
# SAM 3 は 80 クラスだと予測が多くなりすぎるので、スコアの下限を 0.05 にする(20 クラスの評価は 0.01)。
# 使い方(リポジトリ直下で): bash segmentation/instance_segmentation/exp00001_eval_and_baselines/run_repro.sh
# 精度だけを見るので、3 本を同時に流す(GPU・メモリに余裕があるため)。同時に動かすので、ここで記録される遅延は使わない。
# ログは runs/run_repro_{rfdetr_a,rfdetr_b,sam3}.log。
set -e
cd "$(dirname "$0")/../../.."
EXP=segmentation/instance_segmentation/exp00001_eval_and_baselines
PY=".venv/bin/python -u"
rfdetr() {
  for size in "$@"; do
    $PY $EXP/eval_rfdetr.py --size $size --set coco_val2017_80cls --ap-only
  done
}
rfdetr nano small medium > $EXP/runs/run_repro_rfdetr_a.log 2>&1 &
rfdetr large xlarge 2xlarge > $EXP/runs/run_repro_rfdetr_b.log 2>&1 &
$PY $EXP/eval_sam3.py --set coco_val2017_80cls --ap-only --score-floor 0.05 > $EXP/runs/run_repro_sam3.log 2>&1 &
wait
