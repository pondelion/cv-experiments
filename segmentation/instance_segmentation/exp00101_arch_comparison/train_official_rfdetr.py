"""exp00101: RF-DETR-Seg(公式のコード。backbone だけ timm の ResNet-50 に差し替え)を、自前のモデルと同じ条件で学習して評価する。

backbone を自前の Mask2Former 系(train.py)と同じ timm の resnet50.a1_in1k(ImageNet で事前学習)にそろえ、backbone 以外の
方式(projector・decoder・セグメンテーションの head・損失・学習の仕方)の差を見る。コードは vendor/rf-detr(公式の 1.12.0 に
timm の backbone を足したもの。rfdetr.RFDETRSegTimm)。RF-DETR の事前学習の重み(COCO などで学習したもの)は使わない。

自前のモデルとそろえる条件: 学習データとクラス(既定は coco_train_cls8_2k・8 クラス。最初の 1 本は coco_train_10pct・20 クラス)・
入力 512・50 エポック・1 回の更新あたり 16 枚・評価(既定は coco_val_dev_cls8・voc_val_cls8、同じ評価のコード・見本の画像)。
RF-DETR の既定のまま使うもの(方式の一部とみなす): 学習率(1e-4、backbone 1.5e-4 × 0.7²)・データ拡張・複数の解像度での学習・
EMA・学習率の変化・損失。変えるもの: 学習中の評価の間隔(既定の毎エポックを 10 エポックごとに。時間を抑えるため。
評価には coco_val_dev の先頭 500 枚を使う)。

段:
    data:  RF-DETR の形式のデータ(train/・valid/ に、画像へのリンクと _annotations.coco.json)を datasets/rfdetr/ に作る
    train: 学習する(runs/{名前}/ に RF-DETR の重み・記録)
    eval:  学習した重みで coco_val_dev・voc_val を予測し、自前のモデルと同じ評価(results/{名前}.json、見本の画像、クラスごとの AP)

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00101_arch_comparison/train_official_rfdetr.py --name rfdetr_seg_r50_cls8_2k
    (最初の 1 本: --name rfdetr_seg_r50_coco10pct --train-set coco_train_10pct --eval-sets coco_val_dev voc_val)
    (動作確認: --train-set coco_train_1pct --epochs 1 --name SMOKETEST)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import common

REPO_ROOT = common.EXP_DIR.parents[2]  # exp00101 → instance_segmentation → segmentation → リポジトリ直下
sys.path.insert(0, str(REPO_ROOT / "vendor" / "rf-detr" / "src"))  # vendor の rfdetr を先に読む

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lib_common import DATASETS_DIR  # noqa: E402
from lib_common.datasets import load_set, set_path  # noqa: E402
from lib_common.masks import mask_to_rle  # noqa: E402
from lib_common.plots import plot_per_class_ap  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.samples import SAMPLE_COUNTS, render_compare  # noqa: E402
from segmentation.instance_segmentation.lib_common.train import class_image_counts, subset_set  # noqa: E402

ENCODER = "timm/resnet50.a1_in1k"


def prepare_split(set_name: str, split_dir: Path) -> None:
    """評価・学習のセットを、RF-DETR の形式(画像と _annotations.coco.json が同じフォルダ)にする(画像はリンク)。

    Args:
        set_name: lib_common/datasets.py のセット名。
        split_dir: 作るフォルダ(.../train または .../valid)。
    """
    if (split_dir / "_annotations.coco.json").exists():
        return
    split_dir.mkdir(parents=True, exist_ok=True)
    data = load_set(set_name)
    images = []
    for im in data["images"]:
        name = Path(im["file_name"]).name  # COCO の画像の名前は重ならない
        link = split_dir / name
        if not link.exists():
            os.symlink(os.path.relpath(DATASETS_DIR / im["file_name"], split_dir), link)
        images.append(dict(im, file_name=name))
    anns = [a for a in data["annotations"] if not a.get("iscrowd", 0)]  # 群衆の領域は学習に使わない(自前と同じ)
    (split_dir / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns,
                                                                 "categories": data["categories"]}))


@torch.no_grad()
def predict_set(model, set_name: str) -> list[dict]:
    """学習した RF-DETR で評価のセットを予測し、COCO の結果の形式にする(スコアの閾値なし。上位 num_select 個)。

    Args:
        model: rfdetr.RFDETRSegTimm(学習した重みを読んだもの)。
        set_name: 評価のセット名。

    Returns:
        予測のリスト。各要素は {"image_id", "category_id", "segmentation"(RLE), "score", "bbox"}。
    """
    data = load_set(set_name)
    ours = {c["name"]: c["id"] for c in data["categories"]}
    names = model.class_names  # RF-DETR の出すクラスの番号の順のクラス名
    preds = []
    for im in tqdm(data["images"], desc=f"predict {set_name}"):
        det = model.predict(Image.open(DATASETS_DIR / im["file_name"]).convert("RGB"), threshold=0.0)
        for m, s, c, b in zip(det.mask, det.confidence, det.class_id, det.xyxy):
            preds.append({"image_id": im["id"], "category_id": ours[names[int(c)]], "segmentation": mask_to_rle(m),
                          "score": float(s), "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    return preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train-set", default="coco_train_cls8_2k")
    p.add_argument("--eval-sets", nargs="+", default=["coco_val_dev_cls8", "coco_val_test_cls8", "voc_val_cls8"])
    p.add_argument("--epochs", type=int, default=50)
    # 16 × 1 は 8 × 2 より 1.23 倍速い(1 回の更新ごとの決まった処理が半分になる)。GPU のメモリは約 22GB
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1, help="batch-size × grad-accum = 1 回の更新あたりの枚数(16)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--stage", choices=["data", "train", "eval", "all"], default="all")
    p.add_argument("--resume", action="store_true",
                   help="runs/{名前}/last.ckpt(学習が途中で止まったときの重み・最適化の状態)から続きを学習する")
    p.add_argument("--encoder", default=ENCODER, help="backbone(timm のモデル名に timm/ を付けたもの。既定は R50)")
    p.add_argument("--out-feature-indexes", type=int, nargs="+", default=None,
                   help="encoder から使う段・層の番号(省略すると R50 の [2, 3, 4]。ViT は層の番号(0 始まり)。公式の Small の "
                        "DINOv2 の 3・6・9・12 層目は [2, 5, 8, 11])")
    a = p.parse_args()
    enc_kw = {"encoder": a.encoder} | ({"out_feature_indexes": a.out_feature_indexes} if a.out_feature_indexes else {})

    data_dir = DATASETS_DIR / "rfdetr" / a.train_set
    out_dir = common.RUNS_DIR / a.name
    from rfdetr import RFDETRSegTimm

    num_classes = len(load_set(a.train_set)["categories"])
    if a.stage in ("data", "all"):
        prepare_split(a.train_set, data_dir / "train")
        prepare_split(subset_set(a.eval_sets[0], 500), data_dir / "valid")
    if a.stage in ("train", "all"):
        model = RFDETRSegTimm(**enc_kw, resolution=a.size, positional_encoding_size=a.size // 16,
                              num_classes=num_classes)
        model.train(dataset_dir=str(data_dir), output_dir=str(out_dir), epochs=a.epochs, batch_size=a.batch_size,
                    grad_accum_steps=a.grad_accum, eval_interval=a.eval_interval, num_workers=8, tensorboard=False,
                    progress_bar=None, **({"resume": str(out_dir / "last.ckpt")} if a.resume else {}))
    if a.stage in ("eval", "all"):
        ckpt = out_dir / "checkpoint_best_total.pth"
        # 学習したエポック数は、評価だけを回すとき(--epochs の既定の値になる)でもずれないように、学習の設定のファイルから読む
        epochs = json.loads((out_dir / "training_config.json").read_text())["train_config"]["epochs"]
        model = RFDETRSegTimm(**enc_kw, resolution=a.size, positional_encoding_size=a.size // 16,
                              num_classes=num_classes, pretrain_weights=str(ckpt))
        result = {"name": a.name, "model": f"RF-DETR-Seg (vendor rfdetr 1.12.0 + timm {a.encoder})", "checkpoint": ckpt.name,
                  "train_set": a.train_set, "epochs": epochs, "train_class_images": class_image_counts(str(set_path(a.train_set))),
                  "eval": {}}
        threshold = None
        for s in a.eval_sets:  # 最初のセットで PQ などの閾値を選び、ほかのセットにはその値を使う(自前と同じ)
            preds = predict_set(model, s)
            m = evaluate(s, preds, score_threshold=threshold)
            threshold = threshold if threshold is not None else m["pq"]["score_threshold"]
            result["eval"][s] = m
            print(f"[final] {s}: mask AP {m['segm']['AP']:.4f} AP50 {m['segm']['AP50']:.4f} AP75 {m['segm']['AP75']:.4f} "
                  f"| box AP {m['bbox']['AP']:.4f} | PQ {m['pq']['PQ']:.4f} SQ {m['pq']['SQ']:.4f}", flush=True)
            if s in SAMPLE_COUNTS:
                render_compare(common.RESULTS_DIR / "samples", s, a.name, preds, threshold)
        save_metrics(common.RESULTS_DIR / f"{a.name}.json", result)
        # 評価に使った重み(checkpoint_best_total.pth)だけを残し、ほかの重み(途中の再開用・EMA の別の保存など)は消す
        for f in out_dir.iterdir():
            if f.suffix in (".pth", ".ckpt") and f.name != ckpt.name:
                f.unlink()
        plot_per_class_ap({s: m["per_class_segm_AP"] for s, m in result["eval"].items()}, result["train_class_images"],
                          common.RESULTS_DIR / "curves" / f"{a.name}_per_class.png", a.name)


if __name__ == "__main__":
    main()
