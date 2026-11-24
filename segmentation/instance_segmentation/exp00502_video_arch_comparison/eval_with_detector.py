"""exp00502: 画像のモデルの検出 + 学習した伝搬を組み合わせる方式を評価する。
    --method sam3: SAM 3 型(SAM 2 の伝搬 + 検出と追跡の照合。lib_common/models/video/sam2/tracker.py)
    --method deva: DEVA(DEVA の伝搬 + 数フレームの投票と統合。lib_common/models/video/deva/inference.py)

この組み立てには学習がない(学習した伝搬と、学習した画像のモデルを組み合わせるだけ)ので、評価だけを行い、動画のモデルと同じ評価
(動画の AP・画像 AP)を results/{名前}.json に保存する。

使い方(リポジトリ直下で):
    .venv/bin/python -u segmentation/instance_segmentation/exp00502_video_arch_comparison/eval_with_detector.py --method sam3 \
        --name sam3style_sam2vitb_minvisvitb --propagator <SAM 2 の学習の runs/…/best.pt> --detector <画像のモデルの runs/…/best.pt> \
        --sets vipseg_vis_cls8_dev vipseg_vis_cls8_val --max-vram-gb 11
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import common
import torch

from lib_common import REPO_ROOT  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter  # noqa: E402
from segmentation.instance_segmentation.lib_common.models.video.sam2.model import SAM2  # noqa: E402
from segmentation.instance_segmentation.lib_common.models.video.sam2.tracker import SAM3StyleVIS  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_train import evaluate_video_set  # noqa: E402


def load_sam2(path: str) -> SAM2:
    """SAM 2 の学習(train.py --arch sam2)の重みを、同じ場所の config.json の backbone・入力の大きさで作って読む。

    Args:
        path: 学習の重み(runs/{名前}/best.pt)。

    Returns:
        重みを読んだ SAM2。
    """
    cfg = json.loads((Path(path).parent / "config.json").read_text())
    model = SAM2(cfg["backbone"], cfg["size"], pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def load_deva(path: str):
    """DEVA の学習(train.py --arch deva)の重みを、同じ場所の config.json の backbone・入力の大きさで作って読む。

    Args:
        path: 学習の重み(runs/{名前}/best.pt)。

    Returns:
        重みを読んだ DEVA の伝搬のネットワーク。
    """
    from segmentation.instance_segmentation.lib_common.models.video.deva.network import DEVA
    cfg = json.loads((Path(path).parent / "config.json").read_text())
    model = DEVA(cfg["backbone"], cfg["size"], pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def main() -> None:
    """伝搬と画像のモデルを読み、選んだ組み立て(SAM 3 型か DEVA)で各セットを評価して保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--method", choices=["sam3", "deva"], required=True)
    p.add_argument("--propagator", required=True, help="伝搬の学習の重み(SAM 2 か DEVA の best.pt)")
    p.add_argument("--detector", required=True, help="画像のモデルの学習の重み(best.pt。Mask2Former・GTR 系)")
    p.add_argument("--sets", nargs="+", default=["vipseg_vis_cls8_dev", "vipseg_vis_cls8_val"])
    p.add_argument("--max-vram-gb", type=float, default=11.0)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    detector = ImageSegmenter.from_run(a.detector, a.sets[0])  # 評価の動画のセットはどれも同じクラスの並び
    if a.method == "sam3":
        model = SAM3StyleVIS(load_sam2(a.propagator), detector).cuda().eval()
    else:
        from segmentation.instance_segmentation.lib_common.models.video.deva.inference import DEVAVideo
        model = DEVAVideo(load_deva(a.propagator), detector).cuda().eval()
    size = json.loads((Path(a.propagator).parent / "config.json").read_text())["size"]
    ns = SimpleNamespace(size=size, top_k=100, workers=a.workers)
    rel = lambda x: os.path.relpath(x, REPO_ROOT) if str(x).startswith(str(REPO_ROOT)) else x  # 手元のパスを残さない
    result = {"name": a.name, "config": {"method": a.method, "propagator": rel(str(Path(a.propagator).resolve())),
                                         "detector": rel(str(Path(a.detector).resolve())), "size": size}, "eval": {}}
    for s in a.sets:
        result["eval"][s] = evaluate_video_set(model, s, ns)
        print(s, {k: v for k, v in result["eval"][s].items() if not isinstance(v, dict)}, flush=True)
    save_metrics(common.RESULTS_DIR / f"{a.name}.json", result)


if __name__ == "__main__":
    main()
