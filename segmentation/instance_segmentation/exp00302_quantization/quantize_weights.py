"""exp00302: exp00301 で書き出した fp32 の ONNX の重みだけを INT8 にする(途中の値は fp32 のまま。調整の画像は使わない)。

ブラウザの ONNX Runtime Web の WebGPU は、INT8 から戻す演算(DequantizeLinear)には対応するが、fp32 から INT8 にする演算
(QuantizeLinear)には対応しない(onnxruntime-web 1.31.1 の docs/webgpu-operators.md)。quantize.py の静的な量子化は途中の値も INT8 に
するので QuantizeLinear が層ごとに入り、WebGPU では CPU で計算されて遅い。そこで重みだけを INT8 で持ち、使うときに
DequantizeLinear で fp32 に戻して、計算は fp32 で行う形にする(ファイルの大きさは重みが 1/4 になる分だけ小さくなる)。

やり方: 畳み込み(Conv・ConvTranspose)と行列の掛け算(MatMul・Gemm)の重みの初期値(2 次元以上・要素が min_size 個以上)を、
出力のチャンネルごとの対称な INT8(scale = そのチャンネルの絶対値の最大 / 127、zero point 0)にし、元の名前を出力にする
DequantizeLinear を足す(使う側の演算は変えない)。出力のチャンネルの軸: Conv は 0、ConvTranspose は 1、MatMul の B は最後、
Gemm の B は transB なら 0・でなければ 1。ONNX Runtime の量子化の機能には「重みだけを QDQ で INT8 にする」形がないので、
onnx のグラフを直接書き換える。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00302_quantization/quantize_weights.py \
        --onnx segmentation/instance_segmentation/exp00301_export_runtime/runs/gtr_webgpu/model.onnx \
        --name gtr_webgpu_int8_weights
"""
import argparse

import common

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402


def channel_axis(node: onnx.NodeProto, ndim: int) -> int | None:
    """重み(入力の 2 番目)の、出力のチャンネルの軸。

    Args:
        node: 演算。
        ndim: 重みの次元の数。

    Returns:
        軸(量子化しない演算なら None)。
    """
    if node.op_type == "Conv":
        return 0
    if node.op_type == "ConvTranspose":
        return 1
    if node.op_type == "MatMul":
        return ndim - 1
    if node.op_type == "Gemm":
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        return 0 if trans_b else 1
    return None


def quantize_weights(model: onnx.ModelProto, min_size: int = 1024) -> tuple[int, int]:
    """モデルの重みをその場で INT8 + DequantizeLinear に書き換える。

    Args:
        model: fp32 の ONNX のモデル(重みを読み込んだもの)。
        min_size: これより要素の少ない重みはそのまま(小さい重みは大きさに効かない)。

    Returns:
        (INT8 にした重みの数, その要素の数)。
    """
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    done, new_nodes, count, elems = set(), [], 0, 0
    for node in graph.node:
        if len(node.input) < 2 or node.input[1] in done or node.input[1] not in inits:
            continue
        w_init = inits[node.input[1]]
        if w_init.data_type != onnx.TensorProto.FLOAT or len(w_init.dims) < 2:
            continue
        axis = channel_axis(node, len(w_init.dims))
        w = numpy_helper.to_array(w_init)
        if axis is None or w.size < min_size:
            continue
        # 出力のチャンネルごとの scale: 軸以外の向きの絶対値の最大 / 127
        others = tuple(i for i in range(w.ndim) if i != axis)
        scale = (np.abs(w).max(axis=others) / 127.0).astype(np.float32)  # (チャンネルの数,)
        scale[scale == 0] = 1.0
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        name = node.input[1]
        graph.initializer.remove(w_init)
        graph.initializer.extend([numpy_helper.from_array(q, name + "_int8"),
                                  numpy_helper.from_array(scale, name + "_scale"),
                                  numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "_zero")])
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_int8", name + "_scale", name + "_zero"], [name],
                                          axis=axis, name=name + "_dequantize"))
        done.add(name)
        count += 1
        elems += w.size
    for i, n in enumerate(new_nodes):  # 重みを使う演算より前に置く(入力は初期値だけなので先頭でよい)
        graph.node.insert(i, n)
    return count, elems


def main() -> None:
    """重みを INT8 にして、exp00302 の runs/{名前}/model.onnx に保存する(1 つのファイル)。"""
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="fp32 の ONNX(exp00301 で書き出したもの)")
    p.add_argument("--name", required=True)
    a = p.parse_args()
    model = onnx.load(a.onnx)  # 別のファイルの重みも読む
    count, elems = quantize_weights(model)
    out = common.RUNS_DIR / a.name / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    onnx.checker.check_model(str(out))
    print(f"重みを INT8 にした: {out}({out.stat().st_size / 1e6:.1f}MB、{count} 個の重み・{elems / 1e6:.1f}M 要素)", flush=True)


if __name__ == "__main__":
    main()
