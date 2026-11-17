"""exp00301・exp00302: 書き出した ONNX を ONNX Runtime で動かして、学習のときと同じ評価で精度を測る。

学習のときの最後の評価(lib_common/train.py の evaluate_final)と同じ手順: 評価のセットを順に予測し、最初のセット(dev)で PQ などの
スコアの閾値を選び、ほかのセット(test・voc)にはその値を使う。前処理は transforms.eval_transforms、後処理は方式ごとの
lib_common/postprocess.py の関数(学習のときの評価と同じ)。ONNX は batch 1 なので 1 枚ずつ入れる。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00301_export_runtime/eval_onnx.py \
        --run segmentation/instance_segmentation/exp00103_latest_methods/runs/gtr_hybrid_r50_cls8_2k_ep100 \
        --onnx segmentation/instance_segmentation/exp00301_export_runtime/runs/gtr_hybrid_r50_cls8_2k_ep100/model.onnx \
        --provider cuda --tf32 0 --name gtr_hybrid_ort_cuda_fp32
"""
import argparse
import json
from pathlib import Path

import common

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lib_common.datasets import set_path  # noqa: E402
from segmentation.instance_segmentation.lib_common.data import InstanceDataset, collate  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.postprocess import (rfdetr_to_predictions,  # noqa: E402
                                                                      yolo_to_predictions)
from segmentation.instance_segmentation.lib_common.transforms import eval_transforms  # noqa: E402


def make_session(path: str, provider: str, tf32: bool):
    """ONNX Runtime の推論のセッションを作る。

    Args:
        path: ONNX のパス。
        provider: "cuda" / "cpu"。
        tf32: CUDA で TF32(少し粗い精度の行列の掛け算)を使うか。

    Returns:
        onnxruntime.InferenceSession。
    """
    import onnxruntime as ort
    try:
        ort.preload_dlls()  # torch の CUDA・cuDNN を ONNX Runtime の CUDA で使う
    except AttributeError:
        pass
    providers = [("CUDAExecutionProvider", {"use_tf32": "1" if tf32 else "0"})] if provider == "cuda" \
        else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)


def predict(sess, arch: str, set_name: str, size: int) -> list[dict]:
    """1 つの評価のセットを予測し、COCO の結果の形式にする。

    Args:
        sess: ONNX Runtime のセッション。
        arch: 方式("gtr_seg" / "yolo_seg")。
        set_name: 評価のセット名。
        size: 入力の一辺 S。

    Returns:
        予測の並び。
    """
    loader = DataLoader(InstanceDataset(set_path(set_name), eval_transforms(size)), batch_size=1, num_workers=4,
                        collate_fn=collate)
    preds = []
    for batch in tqdm(loader, desc=f"predict {set_name}", mininterval=30):
        outs = [torch.from_numpy(o) for o in sess.run(None, {"images": batch["image"].float().numpy()})]
        if arch == "gtr_seg":  # logits, boxes, masks
            preds += rfdetr_to_predictions(outs[0], outs[1], outs[2], batch["meta"], pad_size=size)
        else:  # yolo_seg: logits, boxes, coef, protos
            preds += yolo_to_predictions(outs[0], outs[1], outs[2], outs[3], batch["meta"], size)
    return preds


def main() -> None:
    """評価して、results/{名前}.json に保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="学習の出力のフォルダ(config.json を読む)")
    p.add_argument("--onnx", required=True)
    p.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--tf32", type=int, choices=[0, 1], default=0)
    p.add_argument("--name", required=True, help="結果の名前(results/{名前}.json)")
    p.add_argument("--eval-sets", nargs="+", default=["coco_val_dev_cls8", "coco_val_test_cls8", "voc_val_cls8"])
    a = p.parse_args()
    cfg = json.loads((Path(a.run) / "config.json").read_text())
    sess = make_session(a.onnx, a.provider, bool(a.tf32))
    results, threshold = {}, None
    for s in a.eval_sets:  # 最初のセットで PQ などの閾値を選び、ほかのセットにはその値を使う(学習のときと同じ)
        m = evaluate(s, predict(sess, cfg["arch"], s, cfg["size"]), score_threshold=threshold)
        threshold = threshold if threshold is not None else m["pq"]["score_threshold"]
        results[s] = m
        print(f"[final] {s}: mask AP {m['segm']['AP']:.4f} AP50 {m['segm']['AP50']:.4f} | box AP {m['bbox']['AP']:.4f} "
              f"| PQ {m['pq']['PQ']:.4f} SQ {m['pq']['SQ']:.4f}", flush=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(common.RESULTS_DIR / f"{a.name}.json",
                 {"name": a.name, "run": Path(a.run).name, "onnx": Path(a.onnx).name, "provider": a.provider,
                  "tf32": bool(a.tf32), "eval": results})


if __name__ == "__main__":
    main()
