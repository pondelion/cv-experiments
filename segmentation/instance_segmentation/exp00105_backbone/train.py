"""exp00105: backbone を差し替えた学習。中身は exp00101 の train.py と同じで、出力だけを exp00105 の runs/・results/ にする。

exp00101 の train.py を、この exp の `common`(出力の場所)を読んだ状態で実行する(exp00101 の train.py の `import common` は、
先に読んだこの exp の `common` を使う)。引数は exp00101 の train.py と同じ。backbone は --backbone(timm の名前)、方式の構成は
--model-kwargs で渡す。速度バランス型の構成は exp00104 の G1 + E2(64) + E3。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00105_backbone/train.py --arch gtr_seg --gtr-encoder hybrid \
        --epochs 100 --backbone mobilenetv4_conv_medium.e500_r256_in1k --name gtr_light_mnv4cm_cls8_2k_ep100 \
        --model-kwargs '{"dim": 192, "ffn": 512, "mask_dim": 64, "enc_block": "dw"}'
    (動作確認: --max-steps 30 --eval-n 20)
"""
import runpy

import common

runpy.run_path(str(common.EXP_DIR.parent / "exp00101_arch_comparison" / "train.py"), run_name="__main__")
