"""exp00101: YOLO26-seg(公式のコード。backbone だけ timm の ResNet-50 に差し替え)を、自作のモデルと同じ条件で学習して評価する。

公式の RF-DETR-Seg(train_official_rfdetr.py)と同じく、比べる相手の公式の実装。コードは vendor/ultralytics(公式の 8.4.211 に、timm の
モデルを backbone にする部品 TimmBackbone を足したもの)。モデルの設計図は yolo26-seg-timm-r50.yaml(公式の yolo26-seg.yaml の
backbone の段だけを timm の resnet50.a1_in1k に差し替え、SPPF・C2PSA・neck・head は公式のまま。大きさの版は l)。
YOLO26 の事前学習の重み(COCO で学習したもの)は使わない。

自作のモデルとそろえる条件: 学習データ(coco_train_cls8_2k)・入力 512・100 エポック・1 回の更新あたり 16 枚・
val(coco_val_dev_cls8 の先頭 500 枚。最良の重みを選ぶ)・test(coco_val_test_cls8・voc_val_cls8。同じ評価のコード・見本の画像)。
ultralytics の既定のまま使うもの(方式の一部とみなす): データ拡張(Mosaic など)・最適化の手法と学習率・学習率の変化・EMA・
損失・学習中の評価(毎エポック)と最良の重みの選び方(val のマスクの AP と枠の AP の和)。

段:
    data:  ultralytics の形式のデータ(images/・labels/ の train・val と data.yaml)を datasets/yolo/ に作る
    train: 学習する(runs/{名前}/ に ultralytics の重み・記録)
    eval:  最良の重み(weights/best.pt)で dev・test・voc を予測し、自作のモデルと同じ評価(results/{名前}.json、見本の画像、
           クラスごとの AP)

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00101_arch_comparison/train_official_yolo26.py --name yolo26_official_r50_cls8_2k_ep100
    (動作確認: --epochs 1 --fraction 0.02 --eval-max-images 20 --name SMOKETEST)
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import common

# ultralytics が足りないパッケージを勝手に入れたり、外部と通信したりしないようにする(import より前に決める)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lib_common import DATASETS_DIR  # noqa: E402
from lib_common.datasets import load_set, set_path  # noqa: E402
from lib_common.masks import mask_to_rle  # noqa: E402
from lib_common.plots import plot_per_class_ap  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics  # noqa: E402
from segmentation.instance_segmentation.lib_common.samples import SAMPLE_COUNTS, render_compare  # noqa: E402
from segmentation.instance_segmentation.lib_common.train import class_image_counts, subset_set  # noqa: E402

MODEL_YAML = common.EXP_DIR / "yolo26-seg-timm-r50.yaml"


def prepare_data(train_set: str, val_set: str, root: Path) -> Path:
    """学習・val のセットを ultralytics の形式にし、data.yaml のパスを返す(作ってあればそのまま使う)。

    注釈の変換(多角形の正規化、複数の部分からなる物体の多角形のつなぎ方、群衆の領域を除くこと)は公式の
    ultralytics.data.converter.convert_coco に任せる。画像はリンクにする。

    Args:
        train_set: 学習のセット名(lib_common/datasets.py)。
        val_set: val のセット名。
        root: 作るフォルダ(datasets/yolo/{学習のセット名})。

    Returns:
        data.yaml のパス。
    """
    from ultralytics.data.converter import convert_coco

    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        return data_yaml
    categories = load_set(train_set)["categories"]
    index = {c["id"]: i + 1 for i, c in enumerate(categories)}  # convert_coco はクラスの番号を category_id - 1 にする
    with tempfile.TemporaryDirectory() as tmp:
        for split, name in (("train", train_set), ("val", val_set)):
            data = load_set(name)
            assert [c["name"] for c in data["categories"]] == [c["name"] for c in categories]
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            images = []
            for im in data["images"]:
                link = root / "images" / split / Path(im["file_name"]).name  # COCO の画像の名前は重ならない
                if not link.exists():
                    os.symlink(os.path.relpath(DATASETS_DIR / im["file_name"], link.parent), link)
                images.append(dict(im, file_name=link.name))
            anns = [dict(a, category_id=index[a["category_id"]]) for a in data["annotations"]]
            (Path(tmp) / f"{split}.json").write_text(json.dumps({"images": images, "annotations": anns,
                                                                 "categories": categories}))
        conv = root / "converted"  # convert_coco は同じ名前のフォルダがあると別名にするので、まだないフォルダに書かせる
        convert_coco(labels_dir=tmp, save_dir=str(conv), use_segments=True, cls91to80=False)
    (conv / "labels").rename(root / "labels")  # labels/train・labels/val(images/ と並べる)
    conv.rmdir() if not any(conv.iterdir()) else None
    names = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(categories))
    data_yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return data_yaml


@torch.no_grad()
def predict_set(model, set_name: str, size: int, max_images: int | None = None) -> list[dict]:
    """学習した YOLO26-seg で評価のセットを予測し、COCO の結果の形式にする(スコアの閾値なし。上位 100 個)。

    Args:
        model: ultralytics.YOLO(学習した重みを読んだもの)。
        set_name: 評価のセット名。
        size: 推論の入力の大きさ(長い辺)。
        max_images: 動作確認用に、先頭の何枚だけ予測するか(None なら全部)。

    Returns:
        予測のリスト。各要素は {"image_id", "category_id", "segmentation"(RLE), "score", "bbox"}。
    """
    data = load_set(set_name)
    ours = {c["name"]: c["id"] for c in data["categories"]}
    preds = []
    for im in tqdm(data["images"][:max_images], desc=f"predict {set_name}"):
        # retina_masks=True: マスクを元の画像の大きさで返す。conf=0.0: 閾値で落とさず上位 max_det 個を出す
        r = model.predict(str(DATASETS_DIR / im["file_name"]), imgsz=size, conf=0.0, max_det=100, retina_masks=True,
                          verbose=False)[0]
        if r.masks is None:
            continue
        masks = r.masks.data.cpu().numpy() > 0.5  # (N, H, W)。N = 予測の数、H・W = 元の画像の高さ・幅
        for m, s, c, b in zip(masks, r.boxes.conf.tolist(), r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            preds.append({"image_id": im["id"], "category_id": ours[r.names[int(c)]], "segmentation": mask_to_rle(m),
                          "score": float(s), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
    return preds


def main() -> None:
    """引数を読み、data・train・eval の段を順に行う。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train-set", default="coco_train_cls8_2k")
    # 最初のセット(dev)の先頭 500 枚が val(最良の重みの選び方)。PQ の閾値は dev の全部で選び、精度は test・voc で見る
    p.add_argument("--eval-sets", nargs="+", default=["coco_val_dev_cls8", "coco_val_test_cls8", "voc_val_cls8"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max-vram-gb", type=float, default=14.0, help="この学習が使う GPU のメモリの上限")
    p.add_argument("--fraction", type=float, default=1.0, help="動作確認用: 学習データのうち使う割合")
    p.add_argument("--eval-max-images", type=int, default=None, help="動作確認用: 評価で予測する枚数")
    p.add_argument("--stage", choices=["data", "train", "eval", "all"], default="all")
    a = p.parse_args()

    from ultralytics import YOLO

    root = DATASETS_DIR / "yolo" / a.train_set
    out_dir = common.RUNS_DIR / a.name
    if a.stage in ("data", "train", "all"):
        data_yaml = prepare_data(a.train_set, subset_set(a.eval_sets[0], 500), root)
    if a.stage in ("train", "all"):
        torch.cuda.set_per_process_memory_fraction(a.max_vram_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        model = YOLO(str(MODEL_YAML), task="segment")
        # plots=False: 学習中の図は使わない(図の文字のフォントを取りに行く通信も起きない)。ほかは ultralytics の既定のまま
        model.train(data=str(data_yaml), epochs=a.epochs, batch=a.batch_size, imgsz=a.size, device=0, workers=8,
                    project=str(common.RUNS_DIR), name=a.name, exist_ok=True, fraction=a.fraction, plots=False)
    if a.stage in ("eval", "all"):
        ckpt = out_dir / "weights" / "best.pt"
        model = YOLO(str(ckpt), task="segment")
        epochs = int(torch.load(ckpt, map_location="cpu", weights_only=False)["train_args"]["epochs"])
        result = {"name": a.name, "model": "YOLO26-seg l (vendor ultralytics 8.4.211 + timm resnet50.a1_in1k)",
                  "checkpoint": "weights/best.pt", "train_set": a.train_set, "epochs": epochs,
                  "train_class_images": class_image_counts(str(set_path(a.train_set))), "eval": {}}
        threshold = None
        for s in a.eval_sets:  # 最初のセットで PQ などの閾値を選び、ほかのセットにはその値を使う(自作と同じ)
            preds = predict_set(model, s, a.size, a.eval_max_images)
            m = evaluate(s, preds, score_threshold=threshold)
            threshold = threshold if threshold is not None else m["pq"]["score_threshold"]
            result["eval"][s] = m
            print(f"[final] {s}: mask AP {m['segm']['AP']:.4f} AP50 {m['segm']['AP50']:.4f} AP75 {m['segm']['AP75']:.4f} "
                  f"| box AP {m['bbox']['AP']:.4f} | PQ {m['pq']['PQ']:.4f} SQ {m['pq']['SQ']:.4f}", flush=True)
            if s in SAMPLE_COUNTS:
                render_compare(common.RESULTS_DIR / "samples", s, a.name, preds, threshold)
        save_metrics(common.RESULTS_DIR / f"{a.name}.json", result)
        plot_per_class_ap({s: m["per_class_segm_AP"] for s, m in result["eval"].items()}, result["train_class_images"],
                          common.RESULTS_DIR / "curves" / f"{a.name}_per_class.png", a.name)


if __name__ == "__main__":
    main()
