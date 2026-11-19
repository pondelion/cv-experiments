"""exp00104: 構成ごとの、ONNX Runtime の CPU での 1 コマの時間(モデルだけ。前後の処理は含まない)を測る。

時間は重みの値によらないので、重みは読まずに各構成を作り、exp00301 の書き出しと同じ形(GTRExport、batch 1・入力 512、opset 18)で
ONNX に書き出して測る。E1 は `mask_topk`、E4 は RepVGG の枝をまとめてから書き出す(exp00301 の `--mask-topk`・`--fuse-rep` と同じ)。
スレッドの数ごとに、最初の warmup 回を捨て、続く runs 回の中央値と p90 を出す。fp32 のまま(INT8 にしたときの時間は別に測る)。
書き出した ONNX は一時フォルダに置いて、測った後に消す。結果は `results/cpu_latency.json` に保存する。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00104_light_configs/bench_cpu.py --threads 1 4
"""
import argparse
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import common

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(common.EXP_DIR.parent / "exp00301_export_runtime"))
from export_onnx import GTRExport  # noqa: E402

from segmentation.instance_segmentation.lib_common.evaluate import save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.models.gtr_seg import GTRSeg  # noqa: E402

G1 = {"dim": 192, "ffn": 512}
G2 = {"dim": 128, "ffn": 512, "enc_expansion": 0.34, "enc_depth": 2, "enc_ffn": 512, "dec_layers": 3}
# (名前, GTRSeg の引数, E1 の上位のクエリの数(0 ならなし), E4(RepVGG の枝をまとめる))
CONFIGS = [
    ("G0", {}, 0, False),
    ("G1", G1, 0, False),
    ("G1 + E1", G1, 100, False),
    ("G1 + E1 + E4", G1, 100, True),
    ("G1 + E2(64)", G1 | {"mask_dim": 64}, 0, False),
    ("G1 + E2(32)", G1 | {"mask_dim": 32}, 0, False),
    ("G1 + E3", G1 | {"enc_block": "dw"}, 0, False),
    ("G1 + E2(64) + E3", G1 | {"mask_dim": 64, "enc_block": "dw"}, 0, False),
    ("G1 + E1 + E2(64) + E3", G1 | {"mask_dim": 64, "enc_block": "dw"}, 100, False),
    ("G1 + E1 + E2(32) + E3", G1 | {"mask_dim": 32, "enc_block": "dw"}, 100, False),
    ("G1 + decoder 2 層", G1 | {"dec_layers": 2}, 0, False),
    ("G2", G2, 0, False),
]


def export(kwargs: dict, topk: int, fuse: bool, path: Path) -> None:
    """構成を作って、推論の形のまま ONNX に書き出す(重みは読まない)。

    マスクの head の材料は推論の形(最後の層の分だけ)で書き出す。E1 の (2) を入れる前の形にしても、書き出すときに使われない計算は
    消されるので、ONNX のグラフは同じになる(E1 の (2) は PyTorch のまま動かすときだけ計算が減る)。

    Args:
        kwargs: GTRSeg の引数。
        topk: E1 の上位のクエリの数(0 ならなし)。
        fuse: E4 を入れるか。
        path: 書き出す先。
    """
    model = GTRSeg(8, "resnet50.a1_in1k", pretrained=False, encoder="hybrid", **kwargs).eval()
    if fuse:
        for m in model.modules():
            if hasattr(m, "fuse_"):
                m.fuse_()
    torch.onnx.export(GTRExport(model, topk).eval(), (torch.randn(1, 3, 512, 512),), str(path), dynamo=True,
                      opset_version=18, input_names=["images"], external_data=False)


def export_backbone(path: Path) -> None:
    """backbone(R50)だけを、GTR 系が使う stride 8・16・32 の 3 段の特徴を出す形で ONNX に書き出す(backbone より後ろの時間 =
    全体の時間 − この時間、にするため)。

    Args:
        path: 書き出す先。
    """
    import timm
    m = timm.create_model("resnet50.a1_in1k", pretrained=False, features_only=True, out_indices=(2, 3, 4)).eval()
    torch.onnx.export(m, (torch.randn(1, 3, 512, 512),), str(path), dynamo=True, opset_version=18, input_names=["images"],
                      external_data=False)


def bench(path: Path, threads: int, warmup: int, runs: int) -> tuple[float, float]:
    """1 コマの時間を測る。

    Args:
        path: ONNX のパス。
        threads: ONNX Runtime の CPU のスレッドの数(intra_op)。
        warmup: 最初に捨てる回数。
        runs: 測る回数。

    Returns:
        (中央値, p90)(ミリ秒)。
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = {"images": np.random.randn(1, 3, 512, 512).astype(np.float32)}
    for _ in range(warmup):
        sess.run(None, x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def main() -> None:
    """全部の構成を書き出して測り、保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--only", nargs="+", default=None, help="測る構成の名前(CONFIGS の名前。省略すると全部)")
    p.add_argument("--backbone", action="store_true", help="backbone(R50)だけの時間も測る")
    p.add_argument("--out", default="cpu_latency.json", help="結果のファイル名(results/ の下)")
    a = p.parse_args()
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    cpu = next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), platform.processor())
    results = {}
    configs = [c for c in CONFIGS if a.only is None or c[0] in a.only]
    with tempfile.TemporaryDirectory() as tmp:
        jobs = ([("R50 backbone だけ", None, 0, False)] if a.backbone else []) + configs
        for name, kw, topk, fuse in jobs:
            path = Path(tmp) / "m.onnx"
            if kw is None:
                export_backbone(path)
            else:
                export(kw, topk, fuse, path)
            results[name] = {}
            for t in a.threads:
                med, p90 = bench(path, t, a.warmup, a.runs)
                results[name][f"threads{t}"] = {"median_ms": med, "p90_ms": p90}
                print(f"{name}: スレッド {t} で中央値 {med:.1f} ms(p90 {p90:.1f})", flush=True)
            path.unlink()
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(common.RESULTS_DIR / a.out,
                 {"cpu": cpu, "onnxruntime": ort.__version__, "input": 512, "dtype": "fp32", "backbone": "resnet50",
                  "warmup": a.warmup, "runs": a.runs, "results": results})


if __name__ == "__main__":
    main()
