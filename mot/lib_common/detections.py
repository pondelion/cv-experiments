"""画像のモデルの検出を動画ごとに作って保存し、読み出す(追跡の手法は、この保存した検出だけを入力にする)。

検出の作り方は、画像のインスタンスセグメンテーションの評価(segmentation/instance_segmentation/lib_common/train.py の evaluate)と
同じ: 評価の変換(長い辺を S にして右と下を埋める)→ モデル → 後処理。後処理は、GTR 系は postprocess.rfdetr_to_predictions
((クエリ, クラス) の確信度の上位 100 個、マスクを元の画像の大きさにして値 > 0 を物体、枠は枠の head の値)、Mask2Former は
postprocess.to_predictions(同じく上位 100 個。確信度 = クラスの確率 × マスクの内側の確率の平均。枠の head がないので、枠はマスクを
囲む最小の枠)。違いは 2 つ:
    - 確信度 min_score(既定 0.05)より低い予測は保存しない(ByteTrack 系は確信度 0.1 までの検出を使うので、それより少し下まで残す)
    - 保存するクラスを選べる(MOT17・DanceTrack・KITTI MOTS で使うのは人・車だけ)
保存の形: {out_dir}/{動画}.json.gz = {"model", "min_score", "labels", "frames": [フレームごとの [{"bbox": [左, 上, 幅, 高さ](元の
画像の画素), "score", "label"(0 始まりの画像のモデルのクラス), "segmentation"(元の大きさの RLE)}]]}(フレームは動画の最初から
全部。評価のセットのフレームの範囲は、読み出すときに切る)。
"""
import gzip
import json
from pathlib import Path

import cv2
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

from segmentation.instance_segmentation.lib_common.models.rfdetr_seg import masks_from_parts
from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter
from segmentation.instance_segmentation.lib_common.postprocess import rfdetr_to_predictions, to_predictions
from segmentation.instance_segmentation.lib_common.transforms import eval_transforms


class _Frames(Dataset):
    """画像のパスの並びを、評価の変換をかけた入力にする。

    Args:
        paths: 画像のパスの並び。
        size: 入力の一辺 S。
    """

    def __init__(self, paths: list[Path], size: int):
        self.paths, self.tf = paths, eval_transforms(size)

    def __len__(self) -> int:
        """画像の数を返す。

        Returns:
            画像の数。
        """
        return len(self.paths)

    def __getitem__(self, i: int):
        """i 番目の画像を読んで変換する。

        Args:
            i: 番号。

        Returns:
            (入力 (3, S, S), 元の大きさ (縦, 横))。
        """
        img = cv2.cvtColor(cv2.imread(str(self.paths[i])), cv2.COLOR_BGR2RGB)
        return self.tf(image=img)["image"], img.shape[:2]


def load_detector(run: str) -> tuple[ImageSegmenter, dict]:
    """画像のモデルを学習の重みから読む(GTR 系・Mask2Former)。

    Args:
        run: 学習の重み(runs/{名前}/best.pt)。

    Returns:
        (ImageSegmenter(cuda・eval), 学習の設定の dict)。クラスの番号は学習のデータ(train_json)の categories の並びのまま。
    """
    seg = ImageSegmenter.from_run(run)
    cfg = json.loads((Path(run).parent / "config.json").read_text())
    return seg.cuda().eval(), cfg


def _mask2former_predictions(logits: torch.Tensor, masks: torch.Tensor, out_hw: tuple[int, int], size: int) -> list[dict]:
    """Mask2Former の 1 フレームの予測を、画像の評価と同じ後処理(postprocess.to_predictions)で作り、枠をマスクから作る。

    Mask2Former には枠の head がないので、枠は予測のマスクを囲む最小の枠(COCO の枠の AP の計算と同じ。pycocotools の toBbox)。

    Args:
        logits: クラスの値 (1, Q, K+1)。
        masks: マスクの値 (1, Q, h, w)。
        out_hw: マスクを作る大きさ(縦, 横)。
        size: 入力の一辺 S。

    Returns:
        {"category_id"(1 始まり), "segmentation"(RLE), "score", "bbox"(out_hw の画素の左・上・幅・高さ)} の並び。
    """
    preds = to_predictions(logits, masks, [{"image_id": 0, "orig_size": out_hw}], size)
    for p in preds:
        rle = {"size": p["segmentation"]["size"], "counts": p["segmentation"]["counts"].encode("ascii")}
        p["bbox"] = mask_utils.toBbox(rle).tolist()
    return preds


@torch.no_grad()
def detect_video(seg: ImageSegmenter, cfg: dict, paths: list[Path], labels: list[int], min_score: float = 0.05,
                 batch_size: int = 4, workers: int = 4, mask_res: str = "orig") -> list[list[dict]]:
    """1 本の動画の全フレームの検出を作る。

    Args:
        seg: load_detector の 1 つ目。
        cfg: load_detector の 2 つ目(入力の大きさ・評価の変換・マスクの学習の仕方を使う)。
        paths: フレームの順の画像のパス。
        labels: 保存するクラス(0 始まりの画像のモデルのクラス)。
        min_score: これより低い確信度の予測は保存しない。
        batch_size: まとめて推論するフレームの数。
        workers: 画像の読み込みの worker の数。
        mask_res: マスクの解像度。"orig" = 元の画像の大きさ(画像の評価と同じ)/ "input" = 入力の中の画像の部分の大きさ
            (長い辺 S。1920×1080 なら 640×360)。マスクを評価しないデータ(枠の追跡)で、マスクを元の大きさにして RLE にする
            時間(1080p で 1 枚 約 4 ms × 1 フレーム約 85 個)を省くため。枠はどちらでも元の画像の画素で保存する。

    Returns:
        フレームごとの検出の並び(保存の形の "frames")。
    """
    size = cfg["size"]
    pad = size if cfg.get("eval_resize") == "pad" else None
    crop = cfg.get("gtr_mask_loss") == "box_crop"
    loader = DataLoader(_Frames(paths, size), batch_size=batch_size, num_workers=workers)
    frames = []
    for x, (hs, ws) in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = seg.model(x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
            if seg.arch == "gtr_seg":
                last = out["layers"][-1]
                masks = masks_from_parts(last["mask_parts"])
            else:  # Mask2Former: 最後の層の予測
                last, masks = out[-1], out[-1]["masks"]
        for i, (h, w) in enumerate(zip(hs.tolist(), ws.tolist())):  # フレームごとに後処理する(返り値をフレームで分けるため)
            # mask_res "input": 後処理に「元の大きさ」として入力の中の画像の部分の大きさを渡す(マスクはその大きさになる)。
            # 枠もその大きさの画素で返るので、元の画像の画素に戻す
            scale = 1.0 if mask_res == "orig" else size / max(h, w)
            out_hw = (h, w) if mask_res == "orig" else (round(h * scale), round(w * scale))
            if seg.arch == "gtr_seg":
                preds = rfdetr_to_predictions(last["logits"][i:i + 1], last["boxes"][i:i + 1], masks[i:i + 1],
                                              [{"image_id": 0, "orig_size": out_hw}], pad_size=pad, crop_to_box=crop,
                                              min_score=min_score)
            else:
                preds = _mask2former_predictions(last["logits"][i:i + 1], masks[i:i + 1], out_hw, size)
            frames.append([{"bbox": [round(v / scale, 2) for v in p["bbox"]], "score": round(p["score"], 4),
                            "label": p["category_id"] - 1, "segmentation": p["segmentation"]}
                           for p in preds if p["category_id"] - 1 in labels and p["score"] >= min_score])
    return frames


def save_detections(path: Path, frames: list[list[dict]], run: str, min_score: float, labels: list[int],
                    mask_res: str = "orig") -> None:
    """検出を保存する。

    Args:
        path: 保存するファイル({動画}.json.gz)。
        frames: detect_video の返り値。
        run: 画像のモデルの学習の重み(記録に書く)。
        min_score: 保存した確信度の下限(記録に書く)。
        labels: 保存したクラス(記録に書く)。
        mask_res: マスクの解像度(detect_video の mask_res。記録に書く)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({"model": run, "min_score": min_score, "labels": labels, "mask_res": mask_res, "frames": frames}, f)


def load_detections(path: Path) -> dict:
    """保存した検出を読む。

    Args:
        path: 保存したファイル。

    Returns:
        保存の形の dict。
    """
    with gzip.open(path, "rt") as f:
        return json.load(f)
