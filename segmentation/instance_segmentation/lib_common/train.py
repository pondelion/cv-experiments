"""インスタンスセグメンテーションのモデルの学習と評価(exp ごとのスクリプトから、設定を渡して呼ぶ)。

流れ(train):
    build_model → build_criterion → build_optimizer → build_ema → build_scheduler → 学習のデータ(transforms で拡張)
    → エポックごとに train_one_epoch → eval_every エポックごとに evaluate_quick → 最後に evaluate_final

モデルの方式(cfg.arch): "mask2former"(自作の Mask2Former。下の既定の値は公式の Mask2Former の設定)/
"rfdetr_seg"(自作の RF-DETR-Seg。公式の RF-DETR-Seg の学習の設定は RFDETR_RECIPE)/
"yolo_seg"(自作の YOLO26-seg 系。学習の設定は YOLO_RECIPE)。

- 最適化: AdamW、backbone の学習率は backbone_lr_mult 倍。正規化の層と埋め込みには weight decay をかけない(公式の Mask2Former と同じ)。
  学習率は最初の WARMUP 回で上げ、その後 poly(0.9 乗)で 0 に向けて下げる(LambdaLR)。
  勾配の大きさの上限 clip(Mask2Former の設定と同じ 0.01、モデル全体で)
- bf16 の自動の型変換で計算する
- 速さ: torch.compile(学習の forward だけ。評価は compile しない元のモデルで)、channels_last、fused の AdamW、
  cudnn.benchmark、pin_memory。生成を止めた GPU で、batch 16 の 1 回の GPU の計算が 54.7 → 78.5 枚/秒、VRAM 9.7 → 8.3 GiB
- VRAM の上限 max_vram_gb: 生成などと GPU を共有して動かすとき、上限を超えたら共有メモリにあふれて遅くなる代わりに
  エラーで止める(WSL では VRAM があふれると共有メモリに逃げて極端に遅くなるため)
- 途中の評価(evaluate_quick): 評価のセットの先頭 eval_n 枚で mask と box の AP(AP@[0.5:0.95]・AP50・AP75・
  APs/APm/APl・AR100)を測り(時間を抑えるため)、見本の画像(samples.py。どの exp でも同じ画像)の予測を描いて保存する
  (スコア SAMPLE_SCORE 以上の予測。途中では PQ の閾値を計算しないので決まった値にする)
- 最後の評価(evaluate_final): 全部の評価のセットで測り、見本の画像を dev で選んだスコアの閾値で描く
- 最後の評価は、途中の評価の mask AP が最良だった重み(best.pt)で行う
- 保存: runs/{名前}/best.pt・last.pt(最良と最後の重み)と train_log.jsonl(損失・学習率・速さ・途中の評価)、results/{名前}.json(評価の結果)、
  results/samples/{名前}/epoch{エポック}/{セット}/{画像の ID}.jpg(途中の見本)・results/samples/{名前}/{セット}/(最後の見本)。
  見本は「元の画像 | 正解(物体の数) | 予測(物体の数、スコアの閾値)」を横に並べた 1 枚。
  results/curves/{名前}.png(学習曲線: 損失・学習率・途中の評価の AP。途中の評価のたびに描き直す)、
  results/samples/{名前}/train_inputs.jpg(学習の初めに、データ拡張の後の学習の画像 8 枚に正解を描いたもの)
"""
import json
import os
import random
import time
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, RandomSampler

from lib_common import REPO_ROOT
from lib_common.datasets import load_set, set_path
from segmentation.instance_segmentation.lib_common.data import InstanceDataset, MosaicInstanceDataset, collate
from segmentation.instance_segmentation.lib_common.losses.gtr_criterion import GTRCriterion
from segmentation.instance_segmentation.lib_common.losses.rfdetr_criterion import targets_from_masks
from segmentation.instance_segmentation.lib_common.losses.yolo_criterion import YOLOCriterion
from segmentation.instance_segmentation.lib_common.models.eomt import EoMT
from segmentation.instance_segmentation.lib_common.optim.musgd import MuSGD
from segmentation.instance_segmentation.lib_common.models.pmt import PMT
from segmentation.instance_segmentation.lib_common.models.gtr_seg import GTRSeg
from segmentation.instance_segmentation.lib_common.models.yolo_seg import YOLOSeg, decode_boxes
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics
from segmentation.instance_segmentation.lib_common.losses.ifan_criterion import IFANCriterion
from segmentation.instance_segmentation.lib_common.losses.mask2former_criterion import Mask2FormerCriterion
from segmentation.instance_segmentation.lib_common.losses.rfdetr_criterion import RFDETRCriterion
from segmentation.instance_segmentation.lib_common.models.mask2former import Mask2Former
from segmentation.instance_segmentation.lib_common.models.rfdetr_seg import RFDETRSeg, masks_from_parts
from segmentation.instance_segmentation.lib_common.postprocess import (maskrcnn_to_predictions, rfdetr_to_predictions,
                                                                       to_predictions, yolo_to_predictions)
from segmentation.instance_segmentation.lib_common.models.maskrcnn import MaskRCNNLoss, MaskRCNNSeg
from lib_common.plots import plot_per_class_ap, plot_training_curves
from segmentation.instance_segmentation.lib_common.samples import (SAMPLE_COUNTS, render_compare, render_train_inputs,
                                                                   sample_set)
from segmentation.instance_segmentation.lib_common.transforms import (batch_scales, build_train_transforms,
                                                                      eval_transforms, square_eval_transforms)

WARMUP = 500
DEVA_STEP_FRACS = (120000 / 150000, 140000 / 150000)  # schedule="deva_step" で 0.1 倍にする所(公式の DEVA の s3_steps / s3_iterations)
MULTISTEP_WARMUP_FACTOR = 0.067  # schedule="multistep" の warmup の始めの倍率(detectron2 の new baselines)
MULTISTEP_FRACS = (163889 / 184375, 177546 / 184375)  # schedule="multistep" で 1/10 にする所(全部の回数に対する割合。同上)
LOG_EVERY = 50  # 損失などを記録する回数の間隔
_LAST_LOG: dict = {}  # 前の記録の (回数, 時刻)。今の速さの計算に使う(train の初めに空にする)
SAMPLE_SCORE = 0.3  # 途中の見本に描く予測のスコアの下限


@dataclass
class TrainConfig:
    """学習の設定(各項目の意味は右のコメント。学習率などは公式の Mask2Former の設定)。"""
    name: str  # 学習の名前(runs/{名前}/・results/{名前}.json に保存する)
    train_json: str  # COCO 形式の学習データの JSON(datasets/ からの相対の画像のパスを持つ)
    out_dir: str  # exp の runs/
    results_dir: str  # exp の results/
    # 評価のセット: 最初(dev)は途中の評価・最良の重み・PQ の閾値を選ぶのに使い、精度は test(dev と画像が重ならない)と voc で見る
    eval_sets: list = field(default_factory=lambda: ["coco_val_dev_cls8", "coco_val_test_cls8", "voc_val_cls8"])
    eval_n: int = 500  # 途中の評価に使う枚数(最初の評価のセットの先頭)
    test_sets: list = field(default_factory=list)  # 学習の途中に全部の画像で評価して記録する test の役のセット(選択には使わない)
    test_every: int = 10  # test_sets の評価の間隔(エポック。eval_every の倍数にする)
    # 動画の評価(exp00502 の MinVIS): 画像のモデルに MinVIS のフレーム間の照合をかけて、動画のモデルと同じ動画の AP・画像 AP を測る。
    # video_dev_set があれば、途中の評価ごとに dev の動画で測り、最良の重みは dev の動画の AP で選ぶ。video_test_set は test_every ごと
    video_dev_set: str | None = None
    video_test_set: str | None = None
    video_top_k: int = 100  # 動画ごとに残す (クエリ, クラス) の組の数(動画のモデルと同じ)
    size: int = 512
    batch_size: int = 16
    workers: int = 8
    epochs: int = 50
    # 1 エポックの画像の枚数(None なら学習のデータを 1 周。指定すると、毎エポックそれだけの枚数を学習のデータから無作為に選ぶ。
    # 動画のフレームのように画像が多いとき、エポックを細かく区切って途中の評価の間隔を動画のモデルと揃えるため)
    epoch_size: int | None = None
    eval_every: int = 10
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    clip: float = 0.01
    backbone: str = "resnet50.a1_in1k"
    num_queries: int = 100
    pixel_decoder: str = "fpn"  # mask2former の pixel decoder("fpn": 簡略版 / "fpn_m2f": 公式の FPN と同じ構成 / "msdeform": 公式と同じ deformable な注意の encoder / "unet")
    mask_pos: str = "none"  # mask2former のマスクを作る画素の特徴に位置の情報を足すか("none" / "sine")
    gtr_encoder: str = "c2f"  # gtr_seg の encoder("c2f": 公式の GTR と同じ / "hybrid": RT-DETR の hybrid encoder。ECInsSeg と同じ)
    # モデルを作るときに足す引数(gtr_seg・yolo_seg・rfdetr_seg。例: {"dim": 128, "mask_stride": 8}。exp00104 の軽い構成)。空なら今までと同じ
    model_kwargs: dict = field(default_factory=dict)
    gtr_mask_loss: str = "points"  # gtr_seg のマスクの損失("points": 公式の GTR と同じ / "box_crop": D-FINE-seg と同じ)
    ifan: bool = False  # eomt: iFAN(マスクの質の head と、質の BCE・APMR・CLSD の損失、質を使う推論の点数)を足す
    grad_accum: int = 1  # gradient accumulation steps(micro-batch size = batch_size / grad_accum。optimizer step 1 回あたりは batch_size 枚)
    grad_checkpoint: bool = False  # gtr_seg: backbone とマスクの head で gradient checkpointing を使う(結果は同じ、メモリが減り少し遅い)
    seed: int = 0
    amp: bool = True  # AMP(autocast)を使うか。学習の forward・損失と推論を amp_dtype で計算する(重み・optimizer は float32)
    amp_dtype: str = "bfloat16"  # AMP の精度。"bfloat16" だけ(float16 は損失のスケーリング(GradScaler)が要り、まだ対応していない)
    max_steps: int | None = None  # 動作確認用(この回数で止める)
    compile: bool = True
    compile_backbone: bool = False  # compile が False のときでも、backbone だけを torch.compile する(入力の形が変わらない部分だけ。exp00105)
    max_vram_gb: float | None = 12.0  # この学習が使える VRAM の上限(None なら制限しない)
    augmentation: str = "lsj"  # 学習のデータ拡張(transforms.build_train_transforms の名前)
    warmup: int = WARMUP  # 学習率を上げていく回数
    arch: str = "mask2former"  # モデルの方式("mask2former" / "rfdetr_seg")。下の項目は RFDETR_RECIPE も参照
    groups: int = 13  # rfdetr_seg: 学習のときのクエリの組の数(Group DETR)
    decoder_lr_mult: float = 1.0  # rfdetr_seg: decoder の学習率の倍率
    schedule: str = "poly"  # 学習率の変化("poly": warmup の後 poly で 0 へ / "constant": warmup の後は一定)
    ema_decay: float | None = None  # 重みの指数移動平均(EMA)の減衰。None なら使わない。使うときは評価も EMA の重みで行う
    ema_tau: float = 100.0  # EMA の減衰を最初の更新で小さくする時間の定数(減衰 = ema_decay × (1 − exp(−更新の回数 / tau)))
    multi_scale: bool = False  # まとめ(バッチ)ごとに入力の一辺を batch_scales(size) から選んで拡大縮小する
    eval_resize: str = "pad"  # 評価の変換("pad": transforms.eval_transforms / "square": transforms.square_eval_transforms)
    optimizer: str = "adamw"  # "adamw" / "sgd"(momentum・nesterov。YOLO 系)/ "musgd"(YOLO26 の MuSGD。yolo_seg)
    momentum: float = 0.937  # sgd の momentum(yolo_seg で adamw のときは AdamW の 1 次の moment の係数)
    lrf: float = 0.01  # schedule="linear" の最後の学習率の倍率(warmup の後、1 から lrf へ直線で下げる)
    close_mosaic: int = 0  # 最後のこのエポック数は Mosaic を止める(augmentation="yolo_mosaic" のとき、"yolo" に切り替える)
    flat_frac: float = 0.2  # schedule="flatcosine": warmup の後、学習率を一定にする所までの割合(全部の回数に対して)
    tail_frac: float = 0.0  # schedule="flatcosine": 最後のこの割合は最小の学習率(lr × lrf)のまま
    backbone_warmup: int = 0  # 0 より大きいと 2 段の warmup(eomt): backbone は最初の warmup 回は 0、その後この回数で上げる
    llrd: float = 1.0  # eomt: ViT の層ごとの学習率の減衰(最後の層 1、1 つ前の層ほど llrd 倍)


# 自作の RF-DETR-Seg を、公式の RF-DETR-Seg の学習の設定(1.12.0 の既定)で学習するときの値(exp のスクリプトで TrainConfig に渡す)。
# 学習率: backbone 以外 1e-4、backbone 1.5e-4 × 0.7² = 7.35e-5、decoder 1e-4 × 0.7 = 7e-5。weight decay 1e-4(backbone の
# 1 次元の重みだけ 0)。勾配の大きさの上限 0.1。学習率は一定(公式の lr_scheduler="step" は lr_drop=100 エポックで 1/10 にするが、
# 100 エポック以下では下がらない)。warmup なし。EMA 0.993・tau 100。データ拡張は transforms.rfdetr_train_transforms と、
# まとめごとの一辺の切り替え(432〜592)。評価は縦横の比を保たない 512 × 512。
# torch.compile は使わない(一辺が 11 通りに変わるので最初の compile に 6 分かかり、その後の 1 回の更新は 298 → 276 ms と 7% しか
# 速くならない(batch 8)。50 エポックでは compile しない方が早く終わる)。
RFDETR_RECIPE = dict(arch="rfdetr_seg", lr=1e-4, backbone_lr_mult=0.735, decoder_lr_mult=0.7, weight_decay=1e-4, clip=0.1,
                     schedule="constant", warmup=0, ema_decay=0.993, ema_tau=100.0, augmentation="rfdetr", multi_scale=True,
                     eval_resize="square", compile=False)

# 自作の EoMT の学習の設定(exp00103 の C)。公式の lightning_module.py・mask_classification_instance.py の既定と
# configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml の値: AdamW 1e-4・weight decay 0.05(全部の重み)、ViT の層ごとの学習率の
# 減衰 0.8、2 段の warmup(最初の 500 回は backbone を止めて他を上げ、次の 1,000 回で backbone を上げる)の後 poly(0.9 乗)、
# 勾配の大きさの上限 0.01、クエリ 200 個、損失は Mask2Former と同じ。backbone は R50 と同じ程度の大きさの ViT-S/16(ImageNet-1K)。
# データ拡張・評価の変換は統一の条件(lsj・pad)。
EOMT_RECIPE = dict(arch="eomt", backbone="deit3_small_patch16_224.fb_in1k", num_queries=200, lr=1e-4, weight_decay=0.05,
                   llrd=0.8, warmup=500, backbone_warmup=1000, clip=0.01, schedule="poly")

# 自作の PMT の学習の設定(exp00103 の E)。公式の lightning_module.py・mask_classification_instance.py の既定と
# configs/coco/panoptic/pmt_s_640.yaml の値: AdamW 2e-4・weight decay 0.05(学習する全部の重み。encoder は固定)、warmup の後
# cosine で 0 へ(公式の warmup 6,000 回 / 全部 73,910 回 = 8.1% を、この学習の 12,500 回に割合で合わせて 1,015 回)、
# 勾配の大きさの上限 0.01、クエリ 200 個、decoder 6 層・MLP 384、損失は Mask2Former と同じ。データ拡張・評価の変換は統一の条件。
PMT_RECIPE = dict(arch="pmt", backbone="deit3_small_patch16_224.fb_in1k", num_queries=200, lr=2e-4, weight_decay=0.05,
                  warmup=1015, clip=0.01, schedule="cosine")

# 自作の YOLO26-seg 系の学習の設定。論文(arXiv 2606.03748)の最適化 MuSGD(Muon + SGD)は混ぜ方の値が論文になく、torch の Muon は
# 2 次元の重みだけ(畳み込みの 4 次元の重みに使えない)なので、YOLO 系の標準の SGD にする(違いとして README に書く)。
# 値は Ultralytics の公開の既定の値: SGD lr 0.01・momentum 0.937・nesterov・weight decay 5e-4(BN と bias は 0)、warmup 3 エポック
# (2,000 枚・batch 16 で 375 回)、その後 lrf 0.01 まで直線で下げる、勾配の大きさの上限 10、EMA 0.9999・tau 2000、Mosaic を
# 最後の 10 エポックで止める。評価は transforms.eval_transforms(長い辺を 512 にして埋める)。
YOLO_RECIPE = dict(arch="yolo_seg", optimizer="sgd", lr=0.01, momentum=0.937, weight_decay=5e-4, schedule="linear", lrf=0.01,
                   warmup=375, clip=10.0, ema_decay=0.9999, ema_tau=2000.0, augmentation="yolo_mosaic", close_mosaic=10,
                   eval_resize="pad")

# 自作の GTR-Seg 系の学習の設定。公式の configs/seg(gtrseg_base.yml・gtrseg_m.yml)と configs/det/gtr_base.yml の値:
# AdamW 5e-4(backbone 1.5e-5 = 0.03 倍)、weight decay 1e-4(正規化の層と bias は 0)、勾配の上限 0.1、EMA 0.9999・warmup 1000 回、
# 学習率は flat-cosine(warmup は 2 乗で上げる min(2000 回, 3 エポック) = 375 回、Mosaic を始める 6/30 エポックまで一定、
# 0.5 倍まで cosine で下げ、データ拡張を止める最後の 2/30 は 0.5 倍のまま。ここでは割合で 100 エポックに合わせる)。
# データ拡張と評価の入力は、方式どうしを公平に比べるため、基本の lsj と eval_transforms にする(公式は Mosaic・IoU の切り抜き・
# mixup など。データ拡張の差は exp00107 で見る)。公式は fp32 で学習するが、ほかの方式と同じ bf16 にする。torch.compile は使わない
# (denoising のクエリの数が画像ごとの物体の数で変わり、形が毎回変わるため)。
# Mask R-CNN(古典の方式の参考の値。exp00103)の学習の設定。detectron2 の configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py
# (backbone と head に BN を入れた Mask R-CNN を LSJ で 100 エポック。この比較と同じ条件)の値: SGD momentum 0.9(nesterov なし)、
# 学習率 0.1(1 回の更新あたり 64 枚)を 16 枚に合わせて 0.025、weight decay 4e-5(正規化の層は 0)、warmup 500 回(0.067 倍から)を
# 全部の回数 184,375 回に対する割合でこの学習の 12,500 回に合わせて 34 回、全部の 88.9%・96.3% で 1/10、勾配の上限なし。
# 公式との違い: backbone は公式(ゼロから学習)と違い、ほかの方式と同じ timm の ImageNet の R50 で初期化する。bias の weight decay は
# 0(公式は 4e-5。YOLO 系と同じ分け方を使うため)。torch.compile は使わない(候補の数などで形が毎回変わるため)。
MASKRCNN_RECIPE = dict(arch="maskrcnn", optimizer="sgd", lr=0.025, momentum=0.9, weight_decay=4e-5, schedule="multistep",
                       warmup=34, clip=float("inf"), augmentation="lsj", eval_resize="pad", compile=False)

GTR_RECIPE = dict(arch="gtr_seg", lr=5e-4, backbone_lr_mult=0.03, weight_decay=1e-4, clip=0.1, schedule="flatcosine",
                  warmup=375, lrf=0.5, flat_frac=6 / 30, tail_frac=2 / 30, ema_decay=0.9999, ema_tau=1000.0,
                  augmentation="lsj", eval_resize="pad", compile=False)


def subset_set(name: str, n: int) -> str:
    """評価のセットの先頭 n 枚だけのセットを作り(なければ)、その名前を返す。

    Args:
        name: 元の評価のセット名。
        n: 先頭から使う枚数。

    Returns:
        作ったセットの名前("{name}_first{n}")。
    """
    sub = f"{name}_first{n}"
    if not set_path(sub).exists():
        data = load_set(name)
        images = data["images"][:n]
        ids = {im["id"] for im in images}
        set_path(sub).write_text(json.dumps({"images": images, "categories": data["categories"],
                                             "annotations": [a for a in data["annotations"] if a["image_id"] in ids]}))
    return sub


def amp_autocast(cfg, cache_enabled: bool = True) -> torch.autocast:
    """学習の設定の amp・amp_dtype で autocast を作る(画像・動画の学習で共通。設定にない古い記録は bf16 の AMP)。

    Args:
        cfg: amp・amp_dtype を持つ設定(TrainConfig・VideoTrainConfig。持たない設定は既定の値)。
        cache_enabled: autocast の重みの変換の使い回し(CUDA graph で記録するときは False にする。PyTorch の決まり)。

    Returns:
        with で使う torch.autocast。

    Raises:
        ValueError: amp_dtype が bfloat16 でないとき(float16 は損失のスケーリングが要り、学習のコードがまだ対応していない)。
    """
    dtype = getattr(cfg, "amp_dtype", "bfloat16")
    if dtype != "bfloat16":
        raise ValueError(f"amp_dtype {dtype} には対応していない(bfloat16 だけ)")
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(cfg, "amp", True), cache_enabled=cache_enabled)


def build_model(cfg: TrainConfig, num_classes: int) -> torch.nn.Module:
    """モデルを作り、GPU に置く(畳み込みが速い channels_last の並びにする)。

    Args:
        cfg: 学習の設定。
        num_classes: クラスの数(物体なしを含めない)。

    Returns:
        Mask2Former / RFDETRSeg(cfg.arch)。
    """
    if cfg.arch == "mask2former":
        model = Mask2Former(num_classes, cfg.backbone, cfg.num_queries, pixel_decoder=cfg.pixel_decoder,
                            mask_pos=cfg.mask_pos)
    elif cfg.arch == "rfdetr_seg":
        model = RFDETRSeg(num_classes, cfg.backbone, cfg.num_queries, cfg.groups, **cfg.model_kwargs)
    elif cfg.arch == "yolo_seg":
        model = YOLOSeg(num_classes, cfg.backbone, **cfg.model_kwargs)
    elif cfg.arch == "gtr_seg":
        model = GTRSeg(num_classes, cfg.backbone, encoder=cfg.gtr_encoder, grad_checkpoint=cfg.grad_checkpoint,
                       **cfg.model_kwargs)
    elif cfg.arch == "eomt":
        model = EoMT(num_classes, cfg.backbone, cfg.num_queries, img_size=cfg.size, quality_head=cfg.ifan)
    elif cfg.arch == "pmt":
        model = PMT(num_classes, cfg.backbone, cfg.num_queries, img_size=cfg.size)
    elif cfg.arch == "maskrcnn":
        model = MaskRCNNSeg(num_classes, cfg.backbone, size=cfg.size)
    else:
        raise ValueError(f"知らないモデルの方式: {cfg.arch}")
    model = model.cuda()
    for t in list(model.parameters()) + list(model.buffers()):  # 4 次元の重み(2 次元の畳み込み)だけ。5 次元(Qwen3-VL の 3 次元の畳み込み)は変えない
        if t.dim() == 4:
            t.data = t.data.contiguous(memory_format=torch.channels_last)
    return model


def build_criterion(cfg: TrainConfig, num_classes: int) -> torch.nn.Module:
    """損失を作る(cfg.arch に合わせる)。

    Args:
        cfg: 学習の設定。
        num_classes: クラスの数(物体なしを含めない)。

    Returns:
        Mask2FormerCriterion / RFDETRCriterion。呼び方はどちらも criterion(モデルの出力, 正解) → (損失, 内訳)。
    """
    if cfg.arch == "rfdetr_seg":
        return RFDETRCriterion(cfg.groups).cuda()
    if cfg.arch == "yolo_seg":
        return YOLOCriterion(num_classes).cuda()
    if cfg.arch == "gtr_seg":
        return GTRCriterion(groups=3, mask_loss=cfg.gtr_mask_loss, checkpoint=cfg.grad_checkpoint).cuda()
    if cfg.arch == "maskrcnn":  # 損失はモデルの中で計算する(models/maskrcnn.py)。内訳を足すだけ
        return MaskRCNNLoss()
    if cfg.ifan:
        return IFANCriterion(num_classes).cuda()
    return Mask2FormerCriterion(num_classes).cuda()


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW を作る。学習率と weight decay の分け方は方式ごとに公式に合わせる。

    - mask2former: backbone(lr × backbone_lr_mult)とそれ以外。正規化の層と埋め込みには weight decay をかけない
    - rfdetr_seg: backbone(lr × backbone_lr_mult)・decoder(層・最後の LayerNorm・ref_point_head。lr × decoder_lr_mult)・
      それ以外(lr)。weight decay は backbone の 1 次元の重み(BN・bias)だけ 0 で、ほかは全部にかける(公式の get_param_dict)

    Args:
        model: モデル。
        cfg: 学習の設定。

    Returns:
        AdamW(学習率の記録に使う「backbone 以外の、元の学習率 lr の組」は param_groups[2])。
    """
    if cfg.optimizer == "sgd" or cfg.arch == "yolo_seg":  # YOLO 系: 重み(2 次元以上)だけ weight decay、BN の重みと bias は 0
        named = list(model.named_parameters())
        groups = [{"params": [p for n, p in named if p.ndim > 1], "weight_decay": cfg.weight_decay},
                  {"params": [p for n, p in named if p.ndim == 1 and not n.endswith(".bias")], "weight_decay": 0.0},
                  {"params": [p for n, p in named if p.ndim == 1 and n.endswith(".bias")], "weight_decay": 0.0}]
        if cfg.optimizer == "sgd":  # maskrcnn は detectron2 と同じ nesterov なし
            return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.arch != "maskrcnn")
        if cfg.optimizer == "musgd":  # exp00102 の Y1: YOLO26 の論文の MuSGD(値は ultralytics の公式の実装と同じ)
            # 2 次元・4 次元の重みは Muon + SGD(weight decay あり)、BN の重み・bias は SGD だけ(weight decay なし)。
            # クラスの枝と意味のマスクの枝(公式の cv3・one2one_cv3・semseg)は学習率を 3 倍にする
            boosted = lambda n: ".cls." in n or n.startswith("proto.semantic")
            muon = [(n, p) for n, p in named if p.ndim in (2, 4)]
            rest = [(n, p) for n, p in named if p.ndim not in (2, 4)]
            groups = []
            for ps, use_muon, wd in [(muon, True, cfg.weight_decay), (rest, False, 0.0)]:
                for boost in (False, True):
                    sel = [p for n, p in ps if boosted(n) == boost]
                    if sel:
                        groups.append({"params": sel, "use_muon": use_muon, "weight_decay": wd,
                                       "lr": cfg.lr * (3 if boost else 1)})
            return MuSGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
        # YOLO 系の AdamW(exp00102 の Y1): 公式の ultralytics の AdamW と同じく、1 次の moment の係数に momentum を使う
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.momentum, 0.999), fused=True)
    if cfg.arch == "gtr_seg":  # 公式の正規表現と同じ分け方: 名前に norm・bn・bias を含む重みは weight decay 0
        named = list(model.named_parameters())
        nd = lambda n: any(s in n for s in ("norm", "bn", "bias"))
        lr_bb = cfg.lr * cfg.backbone_lr_mult
        # ViT の adapter(backbones/vit_adapter.py)の STA と融合の層は学習済みの重みがないので、backbone の倍率をかけない(DEIMv2 と同じ)
        is_bb = lambda n: n.startswith("backbone.") and not n.startswith(("backbone.sta.", "backbone.fuse."))
        bb = [(n, p) for n, p in named if is_bb(n)]
        other = [(n, p) for n, p in named if not is_bb(n)]
        groups = [{"params": [p for n, p in bb if not nd(n)], "lr": lr_bb, "weight_decay": cfg.weight_decay},
                  {"params": [p for n, p in bb if nd(n)], "lr": lr_bb, "weight_decay": 0.0},
                  {"params": [p for n, p in other if not nd(n)], "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                  {"params": [p for n, p in other if nd(n)], "lr": cfg.lr, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, fused=True)
    if cfg.arch == "pmt":  # 公式と同じ: 学習する全部の重み(encoder は固定なので除く)に同じ学習率・weight decay
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                                 weight_decay=cfg.weight_decay, fused=True)
    if cfg.arch == "eomt":  # 公式と同じ: ViT の i 番目の層は lr × llrd^(層の数 − 1 − i)、層より前(埋め込み)は 1 番目の層と同じ、
        # ViT の最後の LayerNorm とほかの部分は lr。weight decay は全部の重みにかける。backbone の組は 2 段の warmup の対象
        vit = model.backbone
        n_blocks = len(vit.blocks)
        groups = []
        for n, p in model.named_parameters():
            mult, is_bb = 1.0, n.startswith("backbone.")
            if n.startswith("backbone.blocks."):
                mult = cfg.llrd ** (n_blocks - 1 - int(n.split(".")[2]))
            elif is_bb and not n.startswith("backbone.norm."):
                mult = cfg.llrd ** (n_blocks - 1)
            groups.append({"params": [p], "lr": cfg.lr * mult, "weight_decay": cfg.weight_decay, "backbone": is_bb})
        groups.sort(key=lambda g: g["backbone"])  # backbone 以外を先に(記録に使う param_groups[2] が元の学習率 lr の組になる)
        return torch.optim.AdamW(groups, fused=True)
    if cfg.arch == "rfdetr_seg":
        named = list(model.named_parameters())
        bb = [(n, p) for n, p in named if n.startswith("features.backbone.")]
        dec = [p for n, p in named if n.startswith(("layers.", "decoder_norm.", "ref_point_head."))]
        bb_ids, dec_ids = {id(p) for _, p in bb}, {id(p) for p in dec}
        other = [p for _, p in named if id(p) not in bb_ids | dec_ids]
        lr_bb = cfg.lr * cfg.backbone_lr_mult
        groups = [{"params": [p for _, p in bb if p.ndim > 1], "lr": lr_bb, "weight_decay": cfg.weight_decay},
                  {"params": [p for _, p in bb if p.ndim == 1], "lr": lr_bb, "weight_decay": 0.0},
                  {"params": other, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                  {"params": dec, "lr": cfg.lr * cfg.decoder_lr_mult, "weight_decay": cfg.weight_decay}]
        return torch.optim.AdamW(groups, fused=True)
    norm_types = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.Embedding)
    no_decay = {id(p) for mod in model.modules() if isinstance(mod, norm_types) for p in mod.parameters(recurse=False)}
    groups = []
    # ViT の adapter の STA と融合の層は学習済みの重みがないので、backbone の倍率をかけない(gtr_seg と同じ。ほかの backbone は今までと同じ)
    in_bb = lambda n: n.startswith("backbone.") and not n.startswith(("backbone.sta.", "backbone.fuse."))
    for is_bb, lr in [(True, cfg.lr * cfg.backbone_lr_mult), (False, cfg.lr)]:
        ps = [p for n, p in model.named_parameters() if in_bb(n) == is_bb]
        groups += [{"params": [p for p in ps if id(p) not in no_decay], "lr": lr, "weight_decay": cfg.weight_decay},
                   {"params": [p for p in ps if id(p) in no_decay], "lr": lr, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, fused=True)


def build_ema(model: torch.nn.Module, cfg: TrainConfig) -> AveragedModel | None:
    """重みの指数移動平均(EMA)を作る(公式の RF-DETR の ModelEma と同じ式。BN の統計などの buffer も平均する)。

    t 回目の更新の後: EMA ← d_t × EMA + (1 − d_t) × 重み、d_t = ema_decay × (1 − exp(−t / ema_tau))。
    作るときに今の重みを写す(公式も最初は今のモデルの複製)。

    Args:
        model: 元のモデル(compile しないもの)。
        cfg: 学習の設定。

    Returns:
        AveragedModel(.module が EMA の重みのモデル)。cfg.ema_decay が None なら None。
    """
    if cfg.ema_decay is None:
        return None

    def ema_fn(avg: list[torch.Tensor], cur: list[torch.Tensor], n: torch.Tensor) -> None:
        """n 回目の更新の平均をその場で書き換える(同じ型の重みをまとめて受け取る)。

        整数の buffer(BN の更新の回数)は平均せずにそのまま写す。

        Args:
            avg: EMA の重みの並び(書き換える)。
            cur: 今の重みの並び(avg と同じ順)。
            n: 何回目の更新か(1 始まり)。
        """
        d = cfg.ema_decay * (1 - math.exp(-float(n) / cfg.ema_tau))
        if avg[0].is_floating_point():
            torch._foreach_lerp_(avg, cur, 1 - d)  # avg + (1 − d)(cur − avg) = d·avg + (1 − d)·cur
        else:
            torch._foreach_copy_(avg, cur)

    ema = AveragedModel(model, multi_avg_fn=ema_fn, use_buffers=True)
    ema.update_parameters(model)  # 1 回目の呼び出しは写すだけ(以後の n が 1, 2, ... になる)
    return ema


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup: int = WARMUP,
                    schedule: str = "poly", lrf: float = 0.01, flat_frac: float = 0.2,
                    tail_frac: float = 0.0, backbone_warmup: int = 0) -> torch.optim.lr_scheduler.LambdaLR:
    """学習率の変化: 最初の warmup 回で 0 から上げ、その後 poly(0.9 乗)で 0 に向けて下げる(schedule="constant" なら一定)。

    Args:
        optimizer: 最適化。
        total_steps: 学習の全部の回数。
        warmup: 学習率を上げていく回数(0 なら最初から元の学習率)。
        schedule: "poly" / "constant" / "linear"(1 から lrf へ直線で下げる)/ "flatcosine"(一定の後 cosine で lrf へ)/
            "cosine"(warmup の後 cosine で 0 へ)/ "multistep"(0.067 倍からの warmup の後、MULTISTEP_FRACS の所で 1/10。
            detectron2)。
        lrf: schedule="linear" の最後の倍率 / "flatcosine" の最小の倍率。
        flat_frac: schedule="flatcosine" の一定の所までの割合。
        tail_frac: schedule="flatcosine" の最後の最小のままの割合。
        backbone_warmup: 0 より大きいと 2 段の warmup(公式の EoMT の TwoStageWarmupPolySchedule。schedule は poly か constant)。
            param_groups の "backbone" が True の組は、最初の warmup 回は 0、次の backbone_warmup 回で上げ、その後 poly で 0 へ。
            ほかの組は warmup 回で上げ、その後 poly で 0 へ。

    Returns:
        LambdaLR(1 回の更新ごとに step() を呼ぶ)。
    """
    def factor(step: int) -> float:
        """更新の回数ごとに、元の学習率に掛ける倍率を返す。

        Args:
            step: 今までの更新の回数(0 始まり)。

        Returns:
            倍率。
        """
        up = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
        if schedule == "constant":
            return up
        if schedule == "flatcosine":  # 公式の GTR / DEIM の flat_cosine_schedule(warmup は 2 乗)
            if step < warmup:
                return ((step + 1) / warmup) ** 2
            flat, tail = round(flat_frac * total_steps), round(tail_frac * total_steps)
            if step <= flat:
                return 1.0
            if step >= total_steps - tail:
                return lrf
            return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * (step - flat) / (total_steps - flat - tail)))
        if schedule == "linear":
            return up * ((1 - step / total_steps) * (1 - lrf) + lrf)
        if schedule == "deva_step":  # 公式の DEVA の stage 3(MultiStepLR。120k・140k / 150k 回で 0.1 倍。warmup の後に同じ割合で下げる)
            return up * 0.1 ** sum(step >= round(f * total_steps) for f in DEVA_STEP_FRACS)
        if schedule == "multistep":  # detectron2 の WarmupParamScheduler + MultiStepParamScheduler(new baselines の値)
            if step < warmup:  # 0.067 倍から直線で 1 へ
                return MULTISTEP_WARMUP_FACTOR + (1 - MULTISTEP_WARMUP_FACTOR) * step / warmup
            return 0.1 ** sum(step >= round(f * total_steps) for f in MULTISTEP_FRACS)
        if schedule == "cosine":  # 公式の PMT の WarmupCosineSchedule(warmup は直線で 0 から、その後 cosine で 0 へ)
            if step < warmup:
                return step / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        return up * max(0.0, 1 - step / total_steps) ** 0.9

    def two_stage(start: int, length: int):
        """2 段の warmup の 1 つの組の倍率の関数: start 回まで 0、その後 length 回で 0 から上げ、残りを poly(0.9 乗)で 0 へ。

        Args:
            start: 0 のままの回数。
            length: 上げていく回数。

        Returns:
            更新の回数(0 始まり)から倍率を返す関数。
        """
        def f(step: int) -> float:
            """更新の回数ごとの、元の学習率に掛ける倍率。

            Args:
                step: 今までの更新の回数(0 始まり)。

            Returns:
                倍率。
            """
            if step < start:
                return 0.0
            if step < start + length:
                return (step - start) / length
            if schedule == "constant":  # warmup の後は一定(exp00502 の学習の量を見る学習)
                return 1.0
            return max(0.0, 1 - (step - start - length) / max(1, total_steps - start - length)) ** 0.9
        return f

    if backbone_warmup > 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, [two_stage(warmup, backbone_warmup) if g.get("backbone")
                                                             else two_stage(0, warmup) for g in optimizer.param_groups])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(forward, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, loader: DataLoader, cfg: TrainConfig, epoch: int, step: int, total_steps: int,
                    t0: float, log, ema: AveragedModel | None = None) -> tuple[int, dict]:
    """1 エポック分学習する(全体で total_steps 回に達したら途中で止める)。

    cfg.multi_scale なら、まとめごとに入力の画像だけを batch_scales(cfg.size) のどれかの一辺に拡大縮小する(公式の RF-DETR の
    multi_scale="per-batch"。選び方も公式と同じく更新の回数を種にした乱数)。正解のマスクはそのままの大きさで使う(損失は 0〜1 の
    位置で扱うため)。

    Args:
        forward: 学習で呼ぶモデル(torch.compile したもの。重みは model と共有)。
        model: 元のモデル(勾配の上限の計算に使う)。
        criterion: 損失。
        optimizer: 最適化。
        scheduler: 学習率の変化(build_scheduler)。
        loader: 学習のデータ。
        cfg: 学習の設定。
        epoch: 今のエポック(1 始まり)。
        step: このエポックの前までの更新の回数。
        total_steps: 学習の全部の回数。
        t0: 学習を始めた時刻(速さの計算に使う)。
        log: 記録を書くファイル(train_log.jsonl)。
        ema: 重みの指数移動平均(build_ema)。None なら使わない。

    Returns:
        (このエポックの後の更新の回数, このエポックの損失の平均 {"loss", <損失の内訳>...})。
    """
    model.train()
    sums, n = {}, 0
    scales = batch_scales(cfg.size)
    accum = cfg.grad_accum  # gradient accumulation steps(micro-batch size は batch_size / accum)
    for i, batch in enumerate(loader):
        images = batch["image"].cuda(non_blocking=True)  # (B, 3, S, S)
        if cfg.multi_scale:
            s = random.Random(step).choice(scales)
            images = F.interpolate(images, size=(s, s), mode="bilinear", align_corners=False)  # (B, 3, S, S) -> (B, 3, s, s)
        images = images.contiguous(memory_format=torch.channels_last)
        targets = [{"masks": t["masks"].cuda(non_blocking=True), "labels": t["labels"].cuda(non_blocking=True)}
                   for t in batch["targets"]]
        with amp_autocast(cfg):
            if cfg.arch == "gtr_seg":  # denoising のクエリを作るために正解(枠・クラス)も渡す
                outs = forward(images, targets_from_masks(targets, tuple(targets[0]["masks"].shape[-2:])))
            elif cfg.arch == "maskrcnn":  # 損失をモデルの中で計算するので正解も渡す
                outs = forward(images, targets)
            else:
                outs = forward(images)
        loss, parts = criterion(outs, targets)
        if i % accum == 0:
            optimizer.zero_grad(set_to_none=True)
        (loss / accum).backward()  # accum 回分の勾配をためる(まとめごとの損失の平均の勾配になる)
        rec = {"loss": float(loss.detach()), **parts}
        for k, v in rec.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
        if (i + 1) % accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        optimizer.step()
        scheduler.step()
        for obj in (model, criterion):  # eomt・pmt: 層ごとの注意のマスクを使う確率 / iFAN: CLSD の重みを、学習の進み方に合わせる
            if hasattr(obj, "set_progress"):
                obj.set_progress((step + 1) / total_steps)
        if ema is not None:
            ema.update_parameters(model)
        step += 1
        if step % LOG_EVERY == 0 or step == 1:
            now = time.perf_counter()
            prev_step, prev_t = _LAST_LOG.get("step", 0), _LAST_LOG.get("t", t0)
            # img_per_s は学習を始めてからの平均、recent_img_per_s は前の記録からの速さ(今の速さ)
            rec |= {"step": step, "epoch": epoch, "lr": optimizer.param_groups[min(2, len(optimizer.param_groups) - 1)]["lr"],
                    "img_per_s": step * cfg.batch_size / (now - t0),
                    "recent_img_per_s": (step - prev_step) * cfg.batch_size / max(now - prev_t, 1e-9), **timestamp(t0)}
            _LAST_LOG.update(step=step, t=now)
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps(rec), flush=True)
        if step >= total_steps:
            break
    return step, {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def predict(model: torch.nn.Module, set_name: str, cfg: TrainConfig, workers: int | None = None) -> tuple[list[dict], float]:
    """評価のセットの全部の画像を予測する(cfg.eval_resize の変換。cfg.arch の後処理)。

    Args:
        model: 評価するモデル(compile しない元のモデル、または EMA の重みのモデル)。
        set_name: 評価のセット名。
        cfg: 学習の設定(入力の大きさ・まとめる数)。
        workers: データを読む並列の数(None なら cfg.workers)。

    Returns:
        (COCO の結果の形式の予測のリスト, 1 枚あたりの時間(ミリ秒。読み込み・後処理を含む))。
    """
    model.eval()
    torch.cuda.empty_cache()
    transforms = square_eval_transforms(cfg.size) if cfg.eval_resize == "square" else eval_transforms(cfg.size)
    ds = InstanceDataset(set_path(set_name), transforms)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.workers if workers is None else workers,
                        collate_fn=collate)
    preds, t0 = [], time.perf_counter()
    for batch in loader:
        with amp_autocast(cfg):
            out = model(batch["image"].cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
            if cfg.arch in ("rfdetr_seg", "gtr_seg"):
                last = out["layers"][-1]
                masks = masks_from_parts(last["mask_parts"])  # (B, Q, mh, mw)
        if cfg.arch in ("rfdetr_seg", "gtr_seg"):
            preds += rfdetr_to_predictions(last["logits"], last["boxes"], masks, batch["meta"],
                                           pad_size=cfg.size if cfg.eval_resize == "pad" else None,
                                           crop_to_box=cfg.arch == "gtr_seg" and cfg.gtr_mask_loss == "box_crop")
        elif cfg.arch == "maskrcnn":
            preds += maskrcnn_to_predictions(out, batch["meta"], cfg.size)
        elif cfg.arch == "yolo_seg":
            o = out["o2o"]
            preds += yolo_to_predictions(o["logits"], decode_boxes(o["ltrb"], out["anchors"], out["strides"]), o["coef"],
                                         out["protos"], batch["meta"], cfg.size)
        else:
            preds += to_predictions(out[-1]["logits"], out[-1]["masks"], batch["meta"], cfg.size, out[-1].get("quality"))
    ms = (time.perf_counter() - t0) / len(ds) * 1000
    model.train()
    torch.cuda.empty_cache()
    return preds, ms


def render_samples(model: torch.nn.Module, cfg: TrainConfig, condition: str, score: float, sets: list[str]) -> None:
    """見本の画像(samples.py。どの exp でも同じ画像)だけを予測し、「元の画像 | 正解 | 予測」を並べて results/samples に保存する。

    Args:
        model: Mask2Former(compile しない元のモデル)。
        cfg: 学習の設定。
        condition: 保存する条件の名前(例: "{名前}/epoch010")。
        score: このスコア以上の予測だけを描く。
        sets: 見本を描く評価のセット(SAMPLE_COUNTS にあるものだけ)。
    """
    for s in [s for s in sets if s in SAMPLE_COUNTS]:
        preds, _ = predict(model, sample_set(s), cfg, workers=0)
        render_compare(Path(cfg.results_dir) / "samples", s, condition, preds, score_threshold=score)


def timestamp(t0: float) -> dict:
    """記録に付ける時刻(あとで速さや、ほかの学習と並べて動かしたときの遅れを見返すため)。

    Args:
        t0: 学習を始めた時刻(time.perf_counter)。

    Returns:
        {"time": 今の時刻(秒まで), "elapsed_s": 学習を始めてからの秒数}。
    """
    return {"time": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(time.perf_counter() - t0, 1)}


def evaluate_quick(model: torch.nn.Module, cfg: TrainConfig, quick_set: str, epoch: int, step: int, log) -> dict:
    """途中の評価: quick_set で mask と box の AP を測り、見本の画像の予測を描く。

    Args:
        model: Mask2Former(compile しない元のモデル)。
        cfg: 学習の設定。
        quick_set: 途中の評価に使うセット(評価のセットの先頭 eval_n 枚)。
        epoch: 今のエポック。
        step: 今までの更新の回数。
        log: 記録を書くファイル(train_log.jsonl)。

    Returns:
        {"epoch", "step", "set", "segm": {"AP", "AP50", ...}, "bbox": {...}}。
    """
    te = time.perf_counter()
    preds, _ = predict(model, quick_set, cfg)
    m = evaluate(quick_set, preds, boundary=False, pixel=False) if preds else {"segm": {"AP": 0.0}, "bbox": {"AP": 0.0}}
    rec = {"epoch": epoch, "step": step, "set": quick_set, "segm": m["segm"], "bbox": m["bbox"],
           "per_class_segm_AP": m.get("per_class_segm_AP", {}), "time": datetime.now().isoformat(timespec="seconds"),
           "eval_s": round(time.perf_counter() - te, 1)}  # 予測と AP の計算の秒数(見本の画像の描画は含めない)
    log.write(json.dumps({"eval": rec}) + "\n"), log.flush()
    print(f"[eval] epoch {epoch} step {step} {quick_set}: " + " ".join(f"mask {k} {v:.4f}" for k, v in m["segm"].items())
          + f" | box AP {m['bbox']['AP']:.4f}", flush=True)
    render_samples(model, cfg, f"{cfg.name}/epoch{epoch:03d}", SAMPLE_SCORE, cfg.eval_sets[:1])
    return rec


def evaluate_video(model: torch.nn.Module, cfg: TrainConfig, set_name: str, epoch: int, log) -> dict:
    """画像のモデルに MinVIS の照合をかけ、動画のセットで動画の AP と画像 AP を測って記録する(動画のモデルと同じ評価)。

    Args:
        model: 画像のモデル。
        cfg: 学習の設定(入力の大きさ・video_top_k などを使う)。
        set_name: 動画のセットの名前。
        epoch: 今のエポック(記録に書く)。
        log: 記録を書くファイル。

    Returns:
        video_train.evaluate_video_set の結果(AP・image_AP など)+ {"set", "epoch"}。
    """
    from types import SimpleNamespace
    from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter
    from segmentation.instance_segmentation.lib_common.models.video.minvis import MinVIS
    from segmentation.instance_segmentation.lib_common.video_train import evaluate_video_set
    torch.cuda.empty_cache()
    # まとめて処理するフレームの数は学習の micro-batch と同じ(学習で入る大きさなので、推論も GPU のメモリの上限に収まる)。
    # 画像のモデル(Mask2Former・GTR 系)は ImageSegmenter で同じ形の出力にそろえる
    seg = ImageSegmenter(model, cfg.arch, crop_to_box=cfg.arch == "gtr_seg" and cfg.gtr_mask_loss == "box_crop")
    tracker = MinVIS(seg, chunk=max(cfg.batch_size // cfg.grad_accum, 1))
    # 画像のセットと動画のセットではクラスの並びが違うことがある(vipseg_frames_cls8 は person, bicycle, car, …、vipseg_vis_cls8 は
    # car, bicycle, person, …)ので、学習のデータのクラスの名前でカテゴリーの id に対応させる
    class_names = [c["name"] for c in json.loads(Path(cfg.train_json).read_text())["categories"]]
    m = evaluate_video_set(tracker, set_name, SimpleNamespace(size=cfg.size, top_k=cfg.video_top_k, workers=cfg.workers),
                           class_names=class_names)
    torch.cuda.empty_cache()
    rec = {"set": set_name, "epoch": epoch, **m, "time": datetime.now().isoformat(timespec="seconds")}
    log.write(json.dumps({"video_eval": rec}) + "\n"), log.flush()
    print(f"[video eval] epoch {epoch} {set_name}: AP {m['AP']:.4f} AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} "
          f"image_AP {m['image_AP']:.4f} s_per_video {m['s_per_video']:.2f}", flush=True)
    return rec


def evaluate_test(model: torch.nn.Module, cfg: TrainConfig, epoch: int, log) -> dict:
    """学習の途中に、test の役のセット(cfg.test_sets)の全部の画像で mask・box の AP を測って記録する(重みの選択には使わない)。

    Args:
        model: 画像のモデル。
        cfg: 学習の設定。
        epoch: 今のエポック(記録に書く)。
        log: 記録を書くファイル。

    Returns:
        {セット名: {"segm", "bbox", "per_class_segm_AP"}}。
    """
    out = {}
    for s in cfg.test_sets:
        te = time.perf_counter()
        preds, _ = predict(model, s, cfg)
        m = evaluate(s, preds, boundary=False, pixel=False) if preds else {"segm": {"AP": 0.0}, "bbox": {"AP": 0.0}}
        out[s] = {"segm": m["segm"], "bbox": m["bbox"], "per_class_segm_AP": m.get("per_class_segm_AP", {}),
                  "eval_s": round(time.perf_counter() - te, 1)}
        log.write(json.dumps({"test": {"epoch": epoch, "set": s, **out[s]}}) + "\n"), log.flush()
        print(f"[test] epoch {epoch} {s}: mask AP {m['segm']['AP']:.4f} | box AP {m['bbox']['AP']:.4f}", flush=True)
    return out


def evaluate_final(model: torch.nn.Module, cfg: TrainConfig) -> dict:
    """最後の評価: 全部の評価のセットで測る。最初のセットで PQ などのスコアの閾値を選び、ほかのセットにはその値を使う。

    Args:
        model: Mask2Former(compile しない元のモデル)。
        cfg: 学習の設定。

    Returns:
        {セット名: evaluate の結果 + "ms_per_image_batch"}。
    """
    results, threshold = {}, None
    for s in cfg.eval_sets:
        preds, ms = predict(model, s, cfg)
        if not preds:  # 学習の初めなどで予測が 1 つもないときは 0 として記録する
            results[s] = {"segm": {"AP": 0.0}, "pq": {"SQ": 0.0}, "n_predictions": 0, "ms_per_image_batch": ms}
            continue
        m = evaluate(s, preds, score_threshold=threshold)
        threshold = threshold if threshold is not None else m["pq"]["score_threshold"]
        results[s] = m | {"ms_per_image_batch": ms}
        print(f"[final] {s}: mask AP {m['segm']['AP']:.4f} AP50 {m['segm']['AP50']:.4f} AP75 {m['segm']['AP75']:.4f} "
              f"| box AP {m['bbox']['AP']:.4f} | boundary AP {m['boundary']['AP']:.4f} | SQ {m['pq']['SQ']:.4f} "
              f"PQ {m['pq']['PQ']:.4f}", flush=True)
        print(f"[final] {s} per-class mask AP: " + " ".join(f"{n} {100 * v:.1f}" for n, v in m["per_class_segm_AP"].items()),
              flush=True)
    render_samples(model, cfg, cfg.name, threshold or 0.0, cfg.eval_sets)
    return results


def train(cfg: TrainConfig) -> dict:
    """モデルを学習し、途中と最後に評価する。

    Args:
        cfg: 学習の設定。

    Returns:
        {"name", "config", "history"(途中の評価), "train_img_per_s", "eval"(評価のセットごとの結果)}
        (results/{名前}.json にも保存する)。
    """
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    out_dir, res_dir = Path(cfg.out_dir) / cfg.name, Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    if cfg.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(cfg.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark = True  # 入力の大きさが固定なので、畳み込みの速い計算の仕方を最初に選ばせる

    num_classes = len(load_set(cfg.eval_sets[0])["categories"])
    model = build_model(cfg, num_classes)
    forward = torch.compile(model) if cfg.compile else model  # 重みは model と共有
    criterion = build_criterion(cfg, num_classes)
    optimizer = build_optimizer(model, cfg)
    ema = build_ema(model, cfg)
    eval_model = ema.module if ema is not None else model  # 評価する重み(EMA を使うときは EMA だけ。公式の RF-DETR と同じ)
    if cfg.compile_backbone and not cfg.compile:  # その場で compile するので、保存する重みの名前は変わらない(EMA の複製の後に行う)
        getattr(model, "backbone", None).compile()
    def make_loader(augmentation: str) -> tuple[InstanceDataset, DataLoader]:
        """学習のデータと読み込みを作る(Mosaic のときは MosaicInstanceDataset)。

        Args:
            augmentation: データ拡張の名前(transforms.build_train_transforms の名前)。

        Returns:
            (学習のデータ, DataLoader)。
        """
        cls = MosaicInstanceDataset if augmentation == "yolo_mosaic" else InstanceDataset
        ds = cls(cfg.train_json, build_train_transforms(augmentation, cfg.size))
        sampler = RandomSampler(ds, num_samples=cfg.epoch_size) if cfg.epoch_size else None
        return ds, DataLoader(ds, batch_size=cfg.batch_size // cfg.grad_accum, shuffle=sampler is None, sampler=sampler,
                              num_workers=cfg.workers,
                              collate_fn=collate, drop_last=True, persistent_workers=True, pin_memory=True)

    train_ds, loader = make_loader(cfg.augmentation)
    total = cfg.max_steps or cfg.epochs * (len(loader) // cfg.grad_accum)  # 更新の回数
    scheduler = build_scheduler(optimizer, total, cfg.warmup, cfg.schedule, cfg.lrf, cfg.flat_frac, cfg.tail_frac,
                                cfg.backbone_warmup)
    quick = subset_set(cfg.eval_sets[0], cfg.eval_n)
    # データ拡張の後の学習の画像と正解の確認用の画像(変換とマスクがずれていないかを、学習の前に目で確かめるため)
    names = [c["name"] for c in load_set(cfg.eval_sets[0])["categories"]]
    render_train_inputs(train_ds, names, res_dir / "samples" / cfg.name / "train_inputs.jpg")
    curves = res_dir / "curves" / f"{cfg.name}.png"

    history, step, t0, best_epoch = [], 0, time.perf_counter(), None
    _LAST_LOG.clear()
    with open(out_dir / "train_log.jsonl", "a") as log:
        for epoch in range(1, cfg.epochs + 1):
            if cfg.close_mosaic and epoch == cfg.epochs - cfg.close_mosaic + 1 and cfg.augmentation == "yolo_mosaic":
                _, loader = make_loader("yolo")  # 最後の close_mosaic エポックは Mosaic を止める
            if hasattr(criterion, "progress"):  # ProgLoss の重み(YOLO 系)
                criterion.progress = (epoch - 1) / max(cfg.epochs - 1, 1)
            step, losses = train_one_epoch(forward, model, criterion, optimizer, scheduler, loader, cfg, epoch, step,
                                           total, t0, log, ema)
            log.write(json.dumps({"epoch_end": epoch, "step": step, **losses, **timestamp(t0)}) + "\n"), log.flush()
            if step >= total or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                history.append(evaluate_quick(eval_model, cfg, quick, epoch, step, log))
                is_test = epoch % cfg.test_every == 0 or step >= total or epoch == cfg.epochs
                if cfg.test_sets and is_test:
                    history[-1]["test"] = evaluate_test(eval_model, cfg, epoch, log)  # 記録だけ(最良の重みの選択には使わない)
                if cfg.video_dev_set:
                    history[-1]["video"] = evaluate_video(eval_model, cfg, cfg.video_dev_set, epoch, log)
                    if cfg.video_test_set and is_test:  # 記録だけ
                        history[-1]["video_test"] = evaluate_video(eval_model, cfg, cfg.video_test_set, epoch, log)
                torch.save(eval_model.state_dict(), out_dir / "last.pt")
                # 最良の重み: 動画の評価があれば dev の動画の AP(動画のモデルと同じ選び方)、なければ途中の評価の mask AP
                score = (lambda h: h["video"]["AP"]) if cfg.video_dev_set else (lambda h: h["segm"]["AP"])
                if score(history[-1]) >= max(score(h) for h in history):
                    torch.save(eval_model.state_dict(), out_dir / "best.pt")
                    best_epoch = epoch
                plot_training_curves(out_dir / "train_log.jsonl", curves, cfg.name)
            if step >= total:
                break
    img_per_s = step * cfg.batch_size / (time.perf_counter() - t0)
    # 最後の評価は、途中の評価の mask AP が最良だった重みで行う(過学習などで最後の重みが最良とは限らないため)
    eval_model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cuda"))
    # 結果のファイルは GitHub に上げるので、設定の中のパスはリポジトリ直下からの相対パスにする(手元の環境のパスを残さない)
    config = {k: os.path.relpath(v, REPO_ROOT) if isinstance(v, str) and v.startswith(str(REPO_ROOT)) else v
              for k, v in asdict(cfg).items()}
    result = {"name": cfg.name, "config": config, "history": history, "best_epoch": best_epoch,
              "train_img_per_s": img_per_s, "eval": evaluate_final(eval_model, cfg),
              "train_class_images": class_image_counts(cfg.train_json)}
    if cfg.video_test_set:  # 最良の重みでの動画の評価(MinVIS)
        with open(out_dir / "train_log.jsonl", "a") as log:
            result["video_eval"] = evaluate_video(eval_model, cfg, cfg.video_test_set, best_epoch, log)
    save_metrics(res_dir / f"{cfg.name}.json", result)
    plot_per_class_ap({s: m["per_class_segm_AP"] for s, m in result["eval"].items() if "per_class_segm_AP" in m},
                      result["train_class_images"], res_dir / "curves" / f"{cfg.name}_per_class.png", cfg.name)
    return result


def evaluate_saved(cfg: TrainConfig) -> dict:
    """学習済みの重み(runs/{名前}/best.pt、なければ last.pt)を読み、cfg.eval_sets で最後の評価をし直して results/{名前}.json を更新する。

    評価のセットを足したとき(テストのセットなど)に、学習し直さずに測るため。途中の評価の記録などはそのまま残す。

    Args:
        cfg: 学習の設定(学習のときと同じ方式・名前。eval_sets は測りたいセット。最初のセットで PQ などの閾値を選ぶ)。

    Returns:
        更新した結果。
    """
    out_dir, res_dir = Path(cfg.out_dir) / cfg.name, Path(cfg.results_dir)
    weights = out_dir / "best.pt" if (out_dir / "best.pt").exists() else out_dir / "last.pt"
    model = build_model(cfg, len(load_set(cfg.eval_sets[0])["categories"]))
    model.load_state_dict(torch.load(weights, map_location="cuda"))
    result = json.loads((res_dir / f"{cfg.name}.json").read_text())
    result["eval"] = evaluate_final(model, cfg)
    result["eval_weights"] = weights.name
    save_metrics(res_dir / f"{cfg.name}.json", result)
    plot_per_class_ap({s: m["per_class_segm_AP"] for s, m in result["eval"].items() if "per_class_segm_AP" in m},
                      result["train_class_images"], res_dir / "curves" / f"{cfg.name}_per_class.png", cfg.name)
    return result


def class_image_counts(coco_json: str) -> dict[str, int]:
    """学習データで、クラスごとにそのクラスの物体を含む画像の数を数える(群衆の領域は数えない)。

    Args:
        coco_json: COCO 形式の JSON のパス。

    Returns:
        {クラス名: 画像の数}。
    """
    data = json.loads(Path(coco_json).read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    pairs = {(a["image_id"], a["category_id"]) for a in data["annotations"] if not a.get("iscrowd", 0)}
    counts = {n: 0 for n in names.values()}
    for _, c in pairs:
        counts[names[c]] += 1
    return counts
