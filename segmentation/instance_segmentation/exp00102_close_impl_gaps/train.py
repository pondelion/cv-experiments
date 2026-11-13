"""exp00102: 自作のモデルの学習。中身は exp00101 の train.py と同じで、出力だけを exp00102 の runs/・results/ にする。

exp00101 の train.py を、この exp の `common`(出力の場所)を読んだ状態で実行する(exp00101 の train.py の `import common` は、
先に読んだこの exp の `common` を使う)。引数は exp00101 の train.py と同じ。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00102_close_impl_gaps/train.py --arch mask2former \
        --pixel-decoder fpn_m2f --epochs 100 --name m2f_fpn_m2f_r50_cls8_2k_ep100
    (動作確認: --max-steps 30 --eval-n 20)
"""
import runpy

import common

runpy.run_path(str(common.EXP_DIR.parent / "exp00101_arch_comparison" / "train.py"), run_name="__main__")
