"""exp00301: 学習したモデルの、ONNX にする前(PyTorch)と後(ONNX Runtime fp32・重みだけ INT8)の CPU での 1 コマの時間を測る。

どれも batch 1・入力 512・モデルだけ(前後の処理は含まない)。PyTorch は export_onnx.py の build で作る書き出し前のモデル(GTRExport
で包んだもの。出力は ONNX と同じ)を fp32 で動かす。重みだけ INT8 の ONNX は、ブラウザのデモと同じく `session.disable_quant_qdq = 1`
を入れる。スレッドの数ごとに、最初の warmup 回を捨て、続く runs 回の中央値と p90 を出す。結果は `results/{名前}.json` に保存する。

backbone だけの時間も測る(PyTorch はモデルの backbone、ONNX Runtime は backbone だけを同じ学習の重みで書き出した fp32 の ONNX。
`runs/{名前}/backbone.onnx` に 1 つのファイルで残し、ブラウザでも同じものを測る)。backbone より後ろの時間 = 全体 − backbone。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00301_export_runtime/bench_cpu.py \
        --run segmentation/instance_segmentation/exp00105_backbone/runs/gtr_light_hgv2b2_cls8_2k_ep100 \
        --onnx-fp32 segmentation/instance_segmentation/exp00301_export_runtime/runs/gtr_light_hgv2b2_webgpu/model.onnx \
        --onnx-int8 segmentation/instance_segmentation/exp00302_quantization/runs/gtr_light_hgv2b2_webgpu_int8_weights/model.onnx \
        --name cpu_latency_gtr_light_hgv2b2
"""
import argparse
import platform
import subprocess
import time
from pathlib import Path

import common

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from export_onnx import build  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import save_metrics  # noqa: E402


def timed(fn, warmup: int, runs: int) -> tuple[float, float]:
    """fn() の時間を測る。

    Args:
        fn: 1 コマ分の推論をする関数。
        warmup: 最初に捨てる回数。
        runs: 測る回数。

    Returns:
        (中央値, p90)(ミリ秒)。
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def ort_fn(path: str, threads: int, disable_qdq: bool, x: np.ndarray):
    """ONNX Runtime の CPU のセッションを作り、1 コマの推論をする関数を返す。

    Args:
        path: ONNX のパス。
        threads: スレッドの数(intra_op)。
        disable_qdq: `session.disable_quant_qdq = 1` を入れるか(重みだけ INT8 の ONNX)。
        x: 入力 (1, 3, S, S)。

    Returns:
        引数なしの関数。
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    if disable_qdq:
        so.add_session_config_entry("session.disable_quant_qdq", "1")
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return lambda: sess.run(None, {"images": x})


def main() -> None:
    """3 つの形式をスレッドの数ごとに測り、保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="学習の出力のフォルダ(config.json と best.pt)")
    p.add_argument("--onnx-fp32", required=True)
    p.add_argument("--onnx-int8", required=True, help="重みだけ INT8 の ONNX(exp00302 の quantize_weights.py)")
    p.add_argument("--name", required=True, help="結果の名前(results/{名前}.json)")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=15)
    a = p.parse_args()
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    cpu = next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), platform.processor())
    model, cfg, _ = build(Path(a.run))
    size = cfg["size"]
    x = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)
    xt = torch.from_numpy(x)
    backbone = model.model.backbone  # stride 8・16・32 の 3 段の特徴を出す timm の backbone
    bb_path = common.RUNS_DIR / a.name / "backbone.onnx"
    bb_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(backbone, (xt,), str(bb_path), dynamo=True, opset_version=18, input_names=["images"],
                      external_data=False)
    results = {}
    for t in a.threads:
        torch.set_num_threads(t)

        def torch_fn(m=model):
            """PyTorch で 1 回推論する(遅延を測る対象)。

            Args:
                m: 測るモデル(既定は全体のモデル)。
            """
            with torch.inference_mode():
                m(xt)

        fns = {"pytorch_fp32": torch_fn, "pytorch_fp32_backbone": lambda: torch_fn(backbone),
               "ort_fp32": ort_fn(a.onnx_fp32, t, False, x), "ort_int8_weights": ort_fn(a.onnx_int8, t, True, x),
               "ort_fp32_backbone": ort_fn(str(bb_path), t, False, x)}
        for k, fn in fns.items():
            med, p90 = timed(fn, a.warmup, a.runs)
            results.setdefault(k, {})[f"threads{t}"] = {"median_ms": med, "p90_ms": p90}
            print(f"{k}: スレッド {t} で中央値 {med:.1f} ms(p90 {p90:.1f})", flush=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(common.RESULTS_DIR / f"{a.name}.json",
                 {"run": Path(a.run).name, "cpu": cpu, "torch": torch.__version__, "onnxruntime": ort.__version__,
                  "input": size, "warmup": a.warmup, "runs": a.runs, "results": results})


if __name__ == "__main__":
    main()
