"""exp00104: 軽い構成の学習。中身は exp00101 の train.py と同じで、出力だけを exp00104 の runs/・results/ にする。

exp00101 の train.py を、この exp の `common`(出力の場所)を読んだ状態で実行する(exp00101 の train.py の `import common` は、
先に読んだこの exp の `common` を使う)。引数は exp00101 の train.py と同じ。構成の値は --model-kwargs で渡す。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00104_light_configs/train.py --arch gtr_seg \
        --gtr-encoder hybrid --epochs 100 --name gtr_g2_r50_cls8_2k_ep100 \
        --model-kwargs '{"dim": 128, "ffn": 512, "enc_expansion": 0.34, "enc_depth": 2, "enc_ffn": 512, "dec_layers": 3}'
    (動作確認: --max-steps 30 --eval-n 20)
"""
import runpy

import common

runpy.run_path(str(common.EXP_DIR.parent / "exp00101_arch_comparison" / "train.py"), run_name="__main__")
