"""exp00302: 量子化した ONNX の精度の評価。中身は exp00301 の eval_onnx.py と同じで、出力だけを exp00302 の results/ にする。

使い方は exp00301 の eval_onnx.py と同じ(引数も同じ)。
"""
import runpy

import common

runpy.run_path(str(common.EXP_DIR.parent / "exp00301_export_runtime" / "eval_onnx.py"), run_name="__main__")
