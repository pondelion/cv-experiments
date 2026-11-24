"""動画インスタンスセグメンテーションの学習と評価のループ(方式によらず共通。exp005xx)。

モデルに求めること:
    forward(clips (B, T, 3, S, S)) → (層ごとの {"logits": (B, T, Q, K+1), "masks": (B, Q, T, h, w)}(フレームごと)の並び, 状態)
        (損失は losses/video_consistent_criterion.py の VideoConsistentCriterion。公式の videomt_online・DVIS と同じ割り当てと
        フレームごとの損失)
    predict_video(frames (T, 3, S, S)) → (フレームごとのクラスの値 (T, Q, K+1), マスクの値 (Q, T, h, w))
    backbone(学習率の層ごとの減衰と 2 段の warmup の対象)を `vit` という名前で持つ(timm の ViT)

学習: AdamW、ViT の層ごとの学習率の減衰(llrd)、2 段の warmup(backbone 以外 → backbone)+ poly(0.9 乗)、bf16、勾配の大きさの上限、
gradient accumulation。学習の途中の評価は dev のセットの動画の AP で、最良の重みを best.pt に保存する。最後に eval_sets で評価する。
"""
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib_common import REPO_ROOT
from segmentation.instance_segmentation.lib_common.evaluate import save_metrics
from segmentation.instance_segmentation.lib_common.losses.video_consistent_criterion import VideoConsistentCriterion
from segmentation.instance_segmentation.lib_common.train import build_optimizer, build_scheduler
from segmentation.instance_segmentation.lib_common.video_data import (VideoClipDataset, VideoEvalDataset, collate_clips,
                                                                       load_video_set)
from segmentation.instance_segmentation.lib_common.video_eval import evaluate_frames, evaluate_vis
from segmentation.instance_segmentation.lib_common.video_infer import video_predictions


@dataclass
class VideoTrainConfig:
    """動画の学習の設定(既定の値は VidEoMT の公式の ViT-S の設定)。"""
    name: str
    out_dir: str  # exp の runs/
    results_dir: str  # exp の results/
    arch: str = "videomt"
    backbone: str = "deit3_small_patch16_224.fb_in1k"
    init_weights: str | None = None  # ViT の初期値の重み(exp00106 の自己教師ありの学習の重み、または画像の段階の学習済みのモデルの
    # best.pt(`backbone.vit.` の部分だけを読む))。あれば backbone の学習済みの重みの代わりに読む
    init_model: str | None = None  # モデル全体の重み(前の学習の last.pt など)から続けて学習する(optimizer の状態は引き継がない)
    test_set: str | None = None  # 学習の途中に test の役のセットでも評価して記録する(最良の重みの選択には使わない)
    test_every_epochs: int = 10  # test_set の評価の間隔(エポック。epochs のとき)
    train_set: str = "vipseg_vis_train"
    dev_set: str = "vipseg_vis_dev"  # 学習の途中の評価・最良の重みの選択
    eval_sets: list = field(default_factory=lambda: ["vipseg_vis_val"])  # 最後の評価
    dev_n: int | None = None  # 途中の評価に使う dev の先頭の本数(None なら全部)
    size: int = 512
    clip_len: int = 5
    frame_range: int = 2
    frame_skip_p: float = 0.0  # 0 より大きいと、区間をフレームの間引きのデータ拡張つきで選ぶ(video_data.sample_clip_skip)
    frame_skip_random_p: float = 0.0  # 間引きのうち、等間隔の代わりに無作為に選ぶ確率
    batch_size: int = 8  # 1 回の optimizer step あたりの区間の数
    grad_accum: int = 1  # gradient accumulation steps(micro-batch size = batch_size / grad_accum)
    steps: int = 40000
    eval_every: int = 4000
    epochs: int | None = None  # 学習の量をエポックで決める(1 エポック = 各動画から 1 区間ずつ)。あれば steps・eval_every の代わりに使う
    eval_every_epochs: int = 5  # epochs のときの途中の評価の間隔(エポック)
    schedule: str = "poly"  # warmup の後の学習率: "poly"(公式。0 へ下げる)/ "constant"(下げない)
    log_every: int = 200  # ログを書く間隔(optimizer step)
    lr: float = 1e-4
    llrd: float = 0.6
    weight_decay: float = 0.05
    warmup: int = 500  # backbone 以外の warmup
    backbone_warmup: int = 1000  # その後の backbone の warmup
    clip: float = 0.01  # 勾配の大きさの上限(公式と同じ)
    num_queries: int = 200
    num_blocks: int = 3
    norm_queries: bool = True  # 補助の予測と次のフレームへ渡すクエリに LayerNorm をかける(公式の VIPSeg の DINOv3 の設定)
    # DVIS(arch="dvis"): 固定する segmenter(画像の Mask2Former の学習の best.pt。同じ場所の config.json で作り方を読む)、
    # 追跡の部の層の数(公式 6)、学習の初めのこの割合の step は割り当てを segmenter の予測で決める(公式は MAX_ITER // 2)
    segmenter: str | None = None
    tracker_layers: int = 6
    dvis_guide_frac: float = 0.5
    noise_ratio: float = 0.5  # DVIS++(arch="dvis_plus"): 学習中に noiser("wa")を使う確率(公式の VIPSeg の設定)
    # DVIS-DAQ(arch="dvis_daq"): 学習の最初のこの割合の step は 2 枚・段階 2(公式の VIPSeg は 10,000 / 90,000)
    daq_first_stage_frac: float = 10000 / 90000
    # CTVIS(arch="ctvis"): Mask2Former の pixel decoder と backbone の学習率の倍率(MinVIS の画像の学習と同じ値)
    pixel_decoder: str = "msdeform"
    backbone_lr_mult: float = 0.1
    # 推論で動画ごとに残す (クエリ, クラス) の組の数。公式の MAX_NUM = 20 は YouTube-VIS(物体が少ない)の値で、VIPSeg の 8 クラスは
    # 1 本に最大 69 物体(val)あり 20 では出し切れないため、AP の最大の検出の数と同じ 100 にする
    top_k: int = 100
    # DEVA(arch="deva"): 公式の学習の段階。0 = 静止画からの疑似の動画(3 フレーム・物体 1 個・参照 2 枚)、3 = 動画(8 フレーム・
    # 物体最大 3 個・参照 3 枚)。公式は 0 の重みから 3 を始める(init_model に 0 の best.pt を与える)
    deva_stage: int = 3
    workers: int = 6
    seed: int = 0
    max_vram_gb: float | None = 15.0
    max_steps: int | None = None  # 動作確認用


def build_video_model(cfg: VideoTrainConfig, num_classes: int) -> torch.nn.Module:
    """方式に合わせてモデルを作る。

    Args:
        cfg: 学習の設定(arch で方式を選ぶ)。
        num_classes: クラスの数。

    Returns:
        モデル。
    """
    if cfg.arch == "videomt":
        from segmentation.instance_segmentation.lib_common.models.video.videomt import VidEoMT
        return VidEoMT(num_classes, cfg.backbone, cfg.num_queries, cfg.num_blocks, cfg.size, init_weights=cfg.init_weights,
                       norm_queries=cfg.norm_queries)
    if cfg.arch == "ctvis":  # reid の head を持つ Mask2Former を動画の区間で学習する(backbone は timm の学習済みの重み)
        from segmentation.instance_segmentation.lib_common.models.mask2former import Mask2Former
        from segmentation.instance_segmentation.lib_common.models.video.ctvis import CTVIS
        return CTVIS(Mask2Former(num_classes, cfg.backbone, cfg.num_queries, pixel_decoder=cfg.pixel_decoder, reid_head=True))
    if cfg.arch == "cavis_seg":  # CAVIS の segmenter: 文脈の特徴 + reid の head を持つ Mask2Former を動画の区間で学習する
        from segmentation.instance_segmentation.lib_common.models.mask2former import Mask2Former
        from segmentation.instance_segmentation.lib_common.models.video.cavis import CAVISSeg
        return CAVISSeg(Mask2Former(num_classes, cfg.backbone, cfg.num_queries, pixel_decoder=cfg.pixel_decoder, reid_head=True,
                                    context_head=True))
    if cfg.arch == "masktrack":  # MaskTrack R-CNN(画像の Mask R-CNN + 追跡の head。backbone は timm の ResNet)
        from segmentation.instance_segmentation.lib_common.models.video.masktrack import MaskTrackRCNN
        return MaskTrackRCNN(num_classes, cfg.backbone, cfg.size)
    if cfg.arch == "cavis":  # 学習済みの CAVIS の segmenter(arch="cavis_seg" の best.pt)を固定し、CAVIS の追跡の部を学習する
        from segmentation.instance_segmentation.lib_common.models.video.cavis import CAVISOnline, load_cavis_segmenter
        return CAVISOnline(load_cavis_segmenter(cfg.segmenter), num_classes, cfg.tracker_layers)
    if cfg.arch == "dvis":  # 学習済みの画像のモデル(MinVIS の学習の Mask2Former、または GTR 系の画像の学習の best.pt)を固定し、
        # 追跡の部だけを学習する
        from segmentation.instance_segmentation.lib_common.models.video.dvis import DVISOnline, load_segmenter
        from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter
        seg_arch = json.loads((Path(cfg.segmenter).parent / "config.json").read_text()).get("arch")
        # どちらも出力のクラスの番号を、学習の動画のセットの並びにそろえる(画像のセットと並びが違うことがある)
        segmenter = ImageSegmenter.from_run(cfg.segmenter, cfg.train_set) if seg_arch == "gtr_seg" else \
            load_segmenter(cfg.segmenter, cfg.train_set)
        return DVISOnline(segmenter, num_classes, cfg.tracker_layers)
    if cfg.arch == "deva":  # DEVA の伝搬のネットワーク(クラスによらない。画素の encoder の幹は exp の backbone)
        from segmentation.instance_segmentation.lib_common.models.video.deva.network import DEVA
        return DEVA(cfg.backbone, cfg.size)
    if cfg.arch == "sam2":  # SAM 2(記憶つきの物体ごとの伝搬。backbone は timm の学習済みの重み)
        from segmentation.instance_segmentation.lib_common.models.video.sam2.model import SAM2
        return SAM2(cfg.backbone, cfg.size)
    if cfg.arch == "dvis_daq":  # 学習済みの segmenter(公式は DVIS++ の segmenter = CTVIS)を固定し、DAQ の追跡の部を学習する
        from segmentation.instance_segmentation.lib_common.models.video.dvis import load_segmenter
        from segmentation.instance_segmentation.lib_common.models.video.dvis_daq import DVISDAQOnline
        from segmentation.instance_segmentation.lib_common.models.video.dvis_plus import load_ctvis_segmenter
        seg_cfg = json.loads((Path(cfg.segmenter).parent / "config.json").read_text())
        segmenter = load_ctvis_segmenter(cfg.segmenter) if seg_cfg.get("arch") == "ctvis" else \
            load_segmenter(cfg.segmenter, cfg.train_set)
        return DVISDAQOnline(segmenter, num_classes, cfg.daq_first_stage_frac)
    if cfg.arch == "dvis_plus":  # 学習済みの CTVIS の Mask2Former(arch="ctvis" の学習の best.pt)を固定し、DVIS++ の追跡の部を学習する
        from segmentation.instance_segmentation.lib_common.models.video.dvis_plus import DVISPlusOnline, load_ctvis_segmenter
        return DVISPlusOnline(load_ctvis_segmenter(cfg.segmenter), num_classes, cfg.tracker_layers, cfg.noise_ratio)
    raise ValueError(f"知らない方式: {cfg.arch}")


def build_video_optimizer(model: torch.nn.Module, cfg: VideoTrainConfig) -> torch.optim.Optimizer:
    """方式に合わせて optimizer を作る。

    既定(VidEoMT)は AdamW。ViT の i 番目の層は lr × llrd^(層の数 − 1 − i)、層より前(埋め込み)は最初の層と同じ、ViT の最後の
    LayerNorm とほかの部分は lr(公式の EoMT・VidEoMT と同じ)。weight decay は全部の重みにかける。backbone の組は 2 段の warmup の
    対象。DVIS など(ViT を学習しない)は、学習する重み(追跡の部)を全部 lr・weight decay で 1 組にする(公式の DVIS と同じ AdamW)。
    SAM 2・CTVIS・CAVIS の segmenter・MaskTrack R-CNN は、下の分岐のそれぞれの公式の分け方。

    Args:
        model: モデル。
        cfg: 学習の設定。

    Returns:
        optimizer。
    """
    if cfg.arch == "sam2":  # 公式の SAM 2 の学習の分け方: image_encoder.*(幹と neck)は lr × backbone_lr_mult(公式の vision_lr)、
        # 幹の ViT の層は層ごとの減衰 0.9(i 番目の層は 0.9^(層の数 − i)、埋め込みは 0.9^(層の数 + 1)、pos_embed は減衰なし)、
        # weight decay は bias と LayerNorm の重みだけ 0
        n_blocks = len(model.image_encoder.trunk.vit.blocks)
        ln_ids = {id(p) for mod in model.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters(recurse=False)}
        groups = []
        for n, p in model.named_parameters():
            mult = cfg.backbone_lr_mult if n.startswith("image_encoder.") else 1.0
            if n.startswith("image_encoder.trunk.vit."):
                sub = n[len("image_encoder.trunk.vit."):]
                if sub.startswith("blocks."):
                    mult *= 0.9 ** (n_blocks - int(sub.split(".")[1]))
                elif sub.startswith(("patch_embed", "cls_token", "reg_token")):
                    mult *= 0.9 ** (n_blocks + 1)
            wd = 0.0 if (n.endswith(".bias") or id(p) in ln_ids) else cfg.weight_decay
            groups.append({"params": [p], "lr": cfg.lr * mult, "weight_decay": wd, "backbone": False})
        return torch.optim.AdamW(groups, fused=True)
    if cfg.arch == "masktrack":  # 公式と同じ SGD(momentum 0.9)。分け方は画像の Mask R-CNN と同じ(BN の重み・bias は wd 0)
        from types import SimpleNamespace
        return build_optimizer(SimpleNamespace(optimizer="sgd", arch="maskrcnn", lr=cfg.lr, weight_decay=cfg.weight_decay,
                                               momentum=0.9), model)
    if cfg.arch in ("ctvis", "cavis_seg"):  # 画像の Mask2Former と同じ分け方(backbone は lr × backbone_lr_mult、正規化・埋め込みは wd 0)
        from types import SimpleNamespace
        return build_optimizer(SimpleNamespace(optimizer="adamw", arch="mask2former", lr=cfg.lr, weight_decay=cfg.weight_decay,
                                               backbone_lr_mult=cfg.backbone_lr_mult), model.segmenter)
    if not hasattr(model, "vit"):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW([{"params": params, "lr": cfg.lr, "weight_decay": cfg.weight_decay, "backbone": False}],
                                 fused=True)
    n_blocks = len(model.vit.blocks)
    groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mult, is_bb = 1.0, n.startswith("vit.")
        if n.startswith("vit.blocks."):
            mult = cfg.llrd ** (n_blocks - 1 - int(n.split(".")[2]))
        elif is_bb and not n.startswith("vit.norm."):
            mult = cfg.llrd ** (n_blocks - 1)
        groups.append({"params": [p], "lr": cfg.lr * mult, "weight_decay": cfg.weight_decay, "backbone": is_bb})
    return torch.optim.AdamW(groups, fused=True)


@torch.no_grad()
def evaluate_video_set(model: torch.nn.Module, set_name: str, cfg: VideoTrainConfig, limit: int | None = None,
                       class_names: list[str] | None = None) -> dict:
    """セットの動画を推論して、動画の AP を計算する。

    Args:
        model: 学習したモデル(eval にして使う)。
        set_name: 共通の形式のセットの名前。
        cfg: 学習の設定(入力の大きさ・top_k)。
        limit: 先頭のこの本数だけ評価する(None なら全部)。
        class_names: モデルのクラスの番号の順のクラスの名前(None ならモデルのクラスの並び = このセットの categories の並び)。
            画像のセットで学習したモデル(学習中の MinVIS の評価)は並びが違うことがあるので、名前でカテゴリーの id に対応させる。

    Returns:
        evaluate_vis の結果(動画の AP)+ evaluate_frames の結果(画像 AP: image_AP など)+ {"n_videos", "s_per_video"}。
    """
    was_training = model.training
    model.eval()
    gt = load_video_set(set_name)
    if limit is not None:
        keep = {v["id"] for v in gt["videos"][:limit]}
        gt = {**gt, "videos": [v for v in gt["videos"] if v["id"] in keep],
              "annotations": [a for a in gt["annotations"] if a["video_id"] in keep]}
    cat_ids = [c["id"] for c in gt["categories"]]
    if class_names is not None:
        by_name = {c["name"]: c["id"] for c in gt["categories"]}
        cat_ids = [by_name[n] for n in class_names]
    ds = VideoEvalDataset(set_name, cfg.size)
    order = [i for i, v in enumerate(ds.videos) if limit is None or v["id"] in keep]
    loader = DataLoader(torch.utils.data.Subset(ds, order), batch_size=None, num_workers=min(cfg.workers, 4))
    preds, t0 = [], time.time()
    for item in loader:
        if hasattr(model, "track_video"):  # SAM 3 型(sam2/tracker.py)・DEVA(deva/inference.py): ID ごとの予測を直接作る
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tracks = model.track_video(item["images"].cuda(non_blocking=True), tuple(item["valid_size"]),
                                           tuple(item["orig_size"]))
            preds += model.to_predictions(tracks, item["video_id"], cat_ids, cfg.size, tuple(item["valid_size"]),
                                          tuple(item["orig_size"]), len(item["images"]))
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, masks = model.predict_video(item["images"].cuda(non_blocking=True))
        preds += video_predictions(logits, masks, item["video_id"], cat_ids, cfg.size, item["valid_size"], item["orig_size"],
                                   cfg.top_k, getattr(model, "activation", "softmax"))  # GTR 系の画像のモデルを使う方式は sigmoid
    torch.cuda.empty_cache()
    s_per_video = (time.time() - t0) / max(len(order), 1)  # 推論の時間(AP の計算を含めない)
    result = {**evaluate_vis(gt, preds), **evaluate_frames(gt, preds), "n_videos": len(order), "s_per_video": s_per_video}
    model.train(was_training)
    return result


@torch.no_grad()
def evaluate_vos_set(model: torch.nn.Module, set_name: str, cfg: VideoTrainConfig, limit: int | None = None) -> dict:
    """マスクを与えて追う方式(SAM 2・DEVA)の伝搬だけの出来を J&F で測る(物体が初めて写ったフレームで正解のマスクを与える)。

    Args:
        model: SAM2 か DEVA の伝搬のネットワーク。
        set_name: 動画のセットの名前。
        cfg: 学習の設定(入力の大きさなど)。
        limit: 先頭のこの本数だけ測る(None なら全部)。

    Returns:
        video_eval.evaluate_jf の結果 + {"n_videos", "s_per_video"}。
    """
    from pycocotools import mask as mask_util  # noqa: F401
    from segmentation.instance_segmentation.lib_common.models.video.sam2.vos import propagate, to_input_mask, to_orig_rles
    from segmentation.instance_segmentation.lib_common.video_eval import evaluate_jf
    was_training = model.training
    model.eval()
    gt = load_video_set(set_name)
    videos = gt["videos"][:limit] if limit is not None else gt["videos"]
    keep = {v["id"] for v in videos}
    gt = {**gt, "videos": videos, "annotations": [a for a in gt["annotations"] if a["video_id"] in keep]}
    anns = {}
    for a in gt["annotations"]:
        anns.setdefault(a["video_id"], []).append(a)
    ds = VideoEvalDataset(set_name, cfg.size)
    order = [i for i, v in enumerate(ds.videos) if v["id"] in keep]
    loader = DataLoader(torch.utils.data.Subset(ds, order), batch_size=None, num_workers=min(cfg.workers, 4))
    tracks, t0 = {}, time.time()
    for item in loader:
        frames = item["images"].cuda(non_blocking=True)
        valid, orig = tuple(item["valid_size"]), tuple(item["orig_size"])
        by_start = {}
        for a in anns.get(item["video_id"], []):
            first = next((t for t, s in enumerate(a["segmentations"]) if s), None)
            if first is not None:
                by_start.setdefault(first, []).append(a)
        prompts = {s: torch.stack([to_input_mask(a["segmentations"][s], valid, cfg.size, frames.device) for a in group])
                   for s, group in by_start.items()}
        if cfg.arch == "deva":  # DEVA: 全部の物体を 1 つの記憶で伝搬し、各画素を argmax の物体に割り当てる(公式と同じ)
            from segmentation.instance_segmentation.lib_common.models.video.deva.inference import propagate_vos
            first = {a["id"]: (s, to_input_mask(a["segmentations"][s], valid, cfg.size, frames.device))
                     for s, group in by_start.items() for a in group}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = propagate_vos(model, frames, first)
            for oid, m in outs.items():
                tracks[oid] = to_orig_rles(m * 2 - 1, valid, orig)  # 0/1 → 値 > 0 が物体
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = propagate(model, frames, prompts)
        for s, group in by_start.items():
            for k, a in enumerate(group):
                tracks[a["id"]] = to_orig_rles(outs[s][:, k], valid, orig)
    torch.cuda.empty_cache()
    s_per_video = (time.time() - t0) / max(len(order), 1)
    result = {**evaluate_jf(gt, tracks), "n_videos": len(order), "s_per_video": s_per_video}
    model.train(was_training)
    return result


def _timestamp(t0: float) -> dict:
    """記録に付ける時刻(画像の学習の train.timestamp と同じ形。速さや、ほかの学習と並べて動かしたときの遅れを見返すため)。

    Args:
        t0: 学習を始めた時刻(time.time)。

    Returns:
        {"time": 今の時刻(秒まで), "elapsed_s": 学習を始めてからの秒数}。
    """
    return {"time": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(time.time() - t0, 1)}


def train_video(cfg: VideoTrainConfig) -> dict:
    """学習して、最後に eval_sets で評価し、results/{名前}.json に保存する。

    Args:
        cfg: 学習の設定。

    Returns:
        保存した結果(設定・途中の評価の記録・最後の評価)。
    """
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    if cfg.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(cfg.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    out_dir, res_dir = Path(cfg.out_dir) / cfg.name, Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1, ensure_ascii=False))
    ds = VideoClipDataset(cfg.train_set, cfg.clip_len, cfg.frame_range, cfg.size, cfg.seed, cfg.frame_skip_p,
                          cfg.frame_skip_random_p)
    num_classes = len(ds.cat_ids)
    deva_s0 = cfg.arch == "deva" and cfg.deva_stage == 0
    if deva_s0:  # DEVA の静止画の段階: 学習のセットのフレーム 1 枚 + 物体 1 個から 3 フレームを作る(deva/static_data.py)
        from segmentation.instance_segmentation.lib_common.models.video.deva.static_data import StaticPseudoVideoDataset
        ds = StaticPseudoVideoDataset(cfg.train_set, cfg.size, cfg.clip_len, cfg.seed)
    model = build_video_model(cfg, num_classes).cuda()
    if cfg.init_model:  # 前の学習のモデル全体の重みから続ける
        model.load_state_dict(torch.load(cfg.init_model, map_location="cuda"))
    if cfg.arch == "ctvis":  # フレームごとの画像の損失 + reid の対照学習
        from segmentation.instance_segmentation.lib_common.losses.ctvis_criterion import CTVISCriterion
        criterion = CTVISCriterion(num_classes, cfg.clip_len).cuda()
    elif cfg.arch == "cavis_seg":  # フレームごとの画像の損失 + 文脈の対照学習 + 原型の対照学習
        from segmentation.instance_segmentation.lib_common.losses.cavis_criterion import CAVISSegCriterion
        criterion = CAVISSegCriterion(num_classes, cfg.clip_len).cuda()
    else:  # VidEoMT・DVIS: 物体が最初に写ったフレームでの割り当て + フレームごとの損失
        criterion = VideoConsistentCriterion(num_classes).cuda()
    optimizer = build_video_optimizer(model, cfg)
    micro = cfg.batch_size // cfg.grad_accum
    steps_per_epoch = len(ds) // micro // cfg.grad_accum
    total = cfg.max_steps or (cfg.epochs * steps_per_epoch if cfg.epochs else cfg.steps)
    eval_every = cfg.eval_every_epochs * steps_per_epoch if cfg.epochs else cfg.eval_every
    scheduler = build_scheduler(optimizer, total, cfg.warmup, cfg.schedule, backbone_warmup=cfg.backbone_warmup)
    loader_kw = dict(batch_size=micro, shuffle=True, num_workers=cfg.workers, collate_fn=collate_clips, drop_last=True,
                     persistent_workers=cfg.workers > 0)
    sam2_rng = np.random.default_rng(cfg.seed)  # SAM 2 のプロンプト・物体の選び方の乱数(公式も専用の rng を持つ)
    log = open(out_dir / "train_log.jsonl", "a")
    step, epoch, best, history, t0 = 0, 0, -1.0, [], time.time()
    model.train()
    while step < total:
        ds.set_epoch(epoch)
        if cfg.arch == "deva" and not deva_s0:  # 公式の max_jump の切り替え(公式は学習の途中で loader を作り直す。エポックの初めに決める)
            from segmentation.instance_segmentation.lib_common.models.video.deva.training import max_jump_at
            ds.deva_max_jump = max_jump_at(step / total)
        for i, batch in enumerate(DataLoader(ds, **loader_kw)):
            targets = [{"labels": t["labels"].cuda(non_blocking=True), "masks": t["masks"].cuda(non_blocking=True)}
                       for t in batch["targets"]]  # 正解のマスクは (N, T, S, S) のまま(損失の中でフレームごとに分ける)
            if cfg.arch == "deva":  # 最初のフレームに写る物体を最大 3 個選び、正解のマスクから伝搬する(公式の DEVA の do_pass)
                from segmentation.instance_segmentation.lib_common.models.video.deva import training as deva_tr
                picked = [(b, deva_tr.select_objects(tg["masks"], sam2_rng, 1 if deva_s0 else deva_tr.MAX_NUM_OBJ, deva_s0))
                          for b, tg in enumerate(targets)]
                picked = [(b, p) for b, p in picked if p is not None]  # 最初のフレームに物体がない区間は使わない
                if picked:
                    images = batch["images"].cuda(non_blocking=True)[[b for b, _ in picked]]
                    cls_gt = torch.stack([p[0] for _, p in picked]); first = torch.stack([p[1] for _, p in picked])
                    sel = torch.stack([p[2] for _, p in picked]); nobj = [p[3] for _, p in picked]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = deva_tr.train_forward(model, images, first, sel, 2 if deva_s0 else deva_tr.NUM_REF_FRAMES)
                        loss, parts = deva_tr.deva_loss(out, cls_gt, nobj, step)
                else:
                    loss, parts = batch["images"].cuda().sum() * 0, {}
            elif cfg.arch == "sam2":  # 区間ごとに物体を選び、プロンプトを与えて伝搬する(公式の SAM2Train)
                from segmentation.instance_segmentation.lib_common.models.video.sam2.training import (sam2_loss, select_objects,
                                                                                                       train_forward)
                loss, parts, n = 0.0, {}, 0
                images = batch["images"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for b, tg in enumerate(targets):
                        obj = select_objects(tg["masks"], sam2_rng)
                        if obj is None:  # 最初のフレームに物体がない区間は使わない
                            continue
                        gt = tg["masks"][obj.to(tg["masks"].device)].transpose(0, 1).bool()  # (N, T, S, S) -> (T, O, S, S)
                        l, parts = sam2_loss(train_forward(model, images[b], gt, sam2_rng), gt)
                        loss, n = loss + l, n + 1
                loss = loss / max(n, 1) if n else images.sum() * 0
            elif cfg.arch == "masktrack":  # 区間の 2 フレームの一方を今、もう一方を参照にする(損失はモデルの中で計算する)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts = model.train_loss(batch["images"].cuda(non_blocking=True), targets, sam2_rng)
            elif cfg.arch == "dvis_daq":  # 割り当てが追跡の部の中で順に決まるので、損失はモデルの中で計算する
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts = model.train_loss(batch["images"].cuda(non_blocking=True), targets, step / total)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs, extra = model(batch["images"].cuda(non_blocking=True))
                # DVIS: 学習の前半は、割り当てを固定した segmenter の予測(追跡の部の並びに並べ替えたもの)で決める(公式の DVIS_online)
                guide = cfg.arch in ("dvis", "dvis_plus", "cavis") and step < total * cfg.dvis_guide_frac
                loss, parts = criterion(outs, targets, match_outs=extra if guide else None)
            if i % cfg.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)
            (loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step(); scheduler.step(); step += 1
            if step % cfg.log_every == 0 or step == 1:
                rec = {"step": step, "epoch": epoch, "loss": float(loss.detach()), **parts,
                       "lr": optimizer.param_groups[-1]["lr"], **_timestamp(t0)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
            if step % eval_every == 0 or step == total:
                # SAM 2(マスクを与えて追う)は伝搬だけの出来を J&F で、ほかは動画の AP・画像 AP で測る
                vos = cfg.arch in ("sam2", "deva")
                evaluate = evaluate_vos_set if vos else evaluate_video_set
                keys = ("J&F", "J", "F") if vos else ("AP", "AP50", "AP75", "image_AP", "image_AP50", "image_AP75")
                m = evaluate(model, cfg.dev_set, cfg, cfg.dev_n)
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2), **{k: m[k] for k in keys},
                                "s_per_video": m["s_per_video"]})
                test_every = cfg.test_every_epochs * steps_per_epoch if cfg.epochs else cfg.eval_every
                if cfg.test_set and (step % test_every == 0 or step == total):  # test の役のセット(記録だけ。選択には使わない)
                    mt = evaluate(model, cfg.test_set, cfg)
                    history[-1].update({f"test_{k}": mt[k] for k in keys})
                    if not vos:
                        history[-1]["test_image_per_class_AP"] = mt["image_per_class_AP"]
                        history[-1]["test_per_class_AP"] = mt["per_class_AP"]
                history[-1].update(_timestamp(t0))  # 評価が終わった時刻(評価にかかった時間も、前後の記録の時刻の差でわかる)
                log.write(json.dumps({"eval": history[-1]}) + "\n"); log.flush()
                print("[eval]", history[-1], flush=True)
                torch.save(model.state_dict(), out_dir / "last.pt")
                if m[keys[0]] > best:  # 最良の重み: dev の動画の AP(SAM 2 は J&F)
                    best = m[keys[0]]
                    torch.save(model.state_dict(), out_dir / "best.pt")
            if step >= total:
                break
        epoch += 1
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cuda"))
    # 結果のファイルは GitHub に上げるので、設定の中のパスはリポジトリ直下からの相対パスにする(手元の環境のパスを残さない)
    config = {k: os.path.relpath(v, REPO_ROOT) if isinstance(v, str) and v.startswith(str(REPO_ROOT)) else v
              for k, v in asdict(cfg).items()}
    result = {"name": cfg.name, "config": config, "history": history,
              "eval": {s: (evaluate_vos_set if cfg.arch in ("sam2", "deva") else evaluate_video_set)(model, s, cfg) for s in cfg.eval_sets}}
    save_metrics(res_dir / f"{cfg.name}.json", result)
    return result
