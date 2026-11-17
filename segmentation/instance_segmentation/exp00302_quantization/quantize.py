"""exp00302: exp00301 で書き出した fp32 の ONNX を、静的な INT8 に量子化する(ONNX Runtime の quantize_static)。

形式: QDQ(量子化と戻しの組を、量子化する演算の前後に入れる形。ONNX Runtime の CPU・CUDA・Web のどれでも読める)。
重みは出力のチャンネルごとの INT8(per_channel)、途中の値も INT8。途中の値の範囲は、調整の画像を通したときの最小・最大で決める
(MinMax)。量子化するのは ONNX Runtime が量子化に対応する演算(畳み込み・行列の掛け算など)だけで、GridSample・TopK などは
fp32 のまま。量子化の前に quant_pre_process(形の推定と、グラフの整理)を通す(ONNX Runtime が勧める手順)。

調整の画像(--calib):
    synthetic: exp00201 で生成した合成画像(runs/q2 の paste・inpaint_full_empty・inpaint_full_noclass、40 場面 × 3)。
               インスタンスセグメンテーションの README の計画どおり、実データなしで量子化できるかを見る
    real:      学習データ(coco_train_cls8_2k)の先頭の画像(比べるため。評価のデータは使わない)
前処理は評価のときと同じ(transforms.eval_transforms)。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00302_quantization/quantize.py \
        --onnx segmentation/instance_segmentation/exp00301_export_runtime/runs/gtr_webgpu/model.onnx \
        --calib synthetic --name gtr_webgpu_int8_synthetic --keep-fp32-ops Gather,GatherElements
"""
import argparse
import tempfile
from pathlib import Path

import common

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quant_pre_process,  # noqa: E402
                                      quantize_static)
from onnxruntime.quantization.registry import QDQRegistry  # noqa: E402

from lib_common import DATASETS_DIR  # noqa: E402
from lib_common.datasets import load_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.transforms import eval_transforms  # noqa: E402

SYNTHETIC_DIR = common.EXP_DIR.parent / "exp00201_gt_mask_pilot" / "runs" / "q2"


def calibration_images(kind: str, n: int) -> list[Path]:
    """調整に使う画像のパス。

    Args:
        kind: "synthetic" / "real"。
        n: 枚数の上限。

    Returns:
        画像のパスの並び。
    """
    if kind == "synthetic":
        paths = sorted(p for d in ("paste", "inpaint_full_empty", "inpaint_full_noclass")
                       for p in (SYNTHETIC_DIR / d).glob("*.png"))
    else:
        paths = [DATASETS_DIR / im["file_name"] for im in load_set("coco_train_cls8_2k")["images"]]
    return paths[:n]


class Reader(CalibrationDataReader):
    """ONNX Runtime の量子化に、調整の画像を 1 枚ずつ渡す。

    Args:
        paths: 画像のパス。
        size: 入力の一辺 S。
    """

    def __init__(self, paths: list[Path], size: int):
        self.paths, self.transform, self.i = paths, eval_transforms(size), 0

    def get_next(self) -> dict | None:
        """次の入力を返す(なくなったら None)。

        Returns:
            {"images": (1, 3, S, S) の float32} か None。
        """
        if self.i >= len(self.paths):
            return None
        img = cv2.cvtColor(cv2.imread(str(self.paths[self.i])), cv2.COLOR_BGR2RGB)  # (h, w, 3)
        self.i += 1
        return {"images": self.transform(image=img)["image"][None].numpy().astype(np.float32)}


def main() -> None:
    """量子化して、exp00302 の runs/{名前}/model.onnx に保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="fp32 の ONNX(exp00301 で書き出したもの)")
    p.add_argument("--calib", choices=["synthetic", "real"], required=True)
    p.add_argument("--n", type=int, default=120, help="調整の画像の枚数")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--name", required=True)
    p.add_argument("--keep-fp32-ops", default="",
                   help="量子化せずに fp32 のまま残す演算の種類(カンマ区切り)。GTR 系は Gather,GatherElements"
                        "(上位のクエリの特徴と参照の枠を選び出す部分。量子化するとスコアが壊れる。README)")
    a = p.parse_args()
    out_dir = common.RUNS_DIR / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = calibration_images(a.calib, a.n)
    out = out_dir / "model.onnx"
    with tempfile.TemporaryDirectory() as tmp:  # 前処理の途中のファイル(重みが別のファイルになる)は一時フォルダに置いて消す
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(a.onnx, str(pre), skip_symbolic_shape=True, save_as_external_data=True,
                          all_tensors_to_one_file=True)
        keep = {s for s in a.keep_fp32_ops.split(",") if s}
        ops = [o for o in QDQRegistry if o not in keep] if keep else None  # None = 既定(量子化に対応する演算の全部)
        quantize_static(str(pre), str(out), Reader(paths, a.size), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, op_types_to_quantize=ops)
    size = sum(f.stat().st_size for f in out_dir.glob("model.onnx*")) / 1e6
    print(f"量子化した: {out}({size:.1f}MB、調整の画像 {len(paths)} 枚: {a.calib})", flush=True)


if __name__ == "__main__":
    main()
