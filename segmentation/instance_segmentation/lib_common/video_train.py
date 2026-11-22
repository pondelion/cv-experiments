"""動画インスタンスセグメンテーションの学習と評価のループ(方式によらず共通。exp005xx)。

モデルに求めること:
    forward(clips (B, T, 3, S, S)) → (層ごとの {"logits": (B, T, Q, K+1), "masks": (B, Q, T, h, w)}(フレームごと)の並び, 状態)
        (損失は losses/videomt_criterion.py の VideoConsistentCriterion。公式の videomt_online と同じ割り当てとフレームごとの損失)
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
from segmentation.instance_segmentation.lib_common.losses.videomt_criterion import VideoConsistentCriterion
from segmentation.instance_segmentation.lib_common.train import build_scheduler
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
    # 推論で動画ごとに残す (クエリ, クラス) の組の数。公式の MAX_NUM = 20 は YouTube-VIS(物体が少ない)の値で、VIPSeg の 8 クラスは
    # 1 本に最大 69 物体(val)あり 20 では出し切れないため、AP の最大の検出の数と同じ 100 にする
    top_k: int = 100
    workers: int = 6
    seed: int = 0
    max_vram_gb: float | None = 15.0
    max_steps: int | None = None  # 動作確認用


def build_video_model(cfg: VideoTrainConfig, num_classes: int) -> torch.nn.Module:
    """方式に合わせてモデルを作る。"""
    if cfg.arch == "videomt":
        from segmentation.instance_segmentation.lib_common.models.videomt import VidEoMT
        return VidEoMT(num_classes, cfg.backbone, cfg.num_queries, cfg.num_blocks, cfg.size, init_weights=cfg.init_weights,
                       norm_queries=cfg.norm_queries)
    raise ValueError(f"知らない方式: {cfg.arch}")


def build_video_optimizer(model: torch.nn.Module, cfg: VideoTrainConfig) -> torch.optim.Optimizer:
    """AdamW。ViT の i 番目の層は lr × llrd^(層の数 − 1 − i)、層より前(埋め込み)は最初の層と同じ、ViT の最後の LayerNorm と
    ほかの部分は lr(公式の EoMT・VidEoMT と同じ)。weight decay は全部の重みにかける。backbone の組は 2 段の warmup の対象。"""
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
def evaluate_video_set(model: torch.nn.Module, set_name: str, cfg: VideoTrainConfig, limit: int | None = None) -> dict:
    """セットの動画を推論して、動画の AP を計算する。

    Args:
        model: 学習したモデル(eval にして使う)。
        set_name: 共通の形式のセットの名前。
        cfg: 学習の設定(入力の大きさ・top_k)。
        limit: 先頭のこの本数だけ評価する(None なら全部)。

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
    ds = VideoEvalDataset(set_name, cfg.size)
    order = [i for i, v in enumerate(ds.videos) if limit is None or v["id"] in keep]
    loader = DataLoader(torch.utils.data.Subset(ds, order), batch_size=None, num_workers=min(cfg.workers, 4))
    preds, t0 = [], time.time()
    for item in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, masks = model.predict_video(item["images"].cuda(non_blocking=True))
        preds += video_predictions(logits, masks, item["video_id"], cat_ids, cfg.size, item["valid_size"], item["orig_size"],
                                   cfg.top_k)
    torch.cuda.empty_cache()
    s_per_video = (time.time() - t0) / max(len(order), 1)  # 推論の時間(AP の計算を含めない)
    result = {**evaluate_vis(gt, preds), **evaluate_frames(gt, preds), "n_videos": len(order), "s_per_video": s_per_video}
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
    """学習して、最後に eval_sets で評価し、results/{名前}.json に保存する。"""
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    if cfg.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(cfg.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    out_dir, res_dir = Path(cfg.out_dir) / cfg.name, Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1, ensure_ascii=False))
    ds = VideoClipDataset(cfg.train_set, cfg.clip_len, cfg.frame_range, cfg.size, cfg.seed)
    num_classes = len(ds.cat_ids)
    model = build_video_model(cfg, num_classes).cuda()
    if cfg.init_model:  # 前の学習のモデル全体の重みから続ける
        model.load_state_dict(torch.load(cfg.init_model, map_location="cuda"))
    criterion = VideoConsistentCriterion(num_classes).cuda()
    optimizer = build_video_optimizer(model, cfg)
    micro = cfg.batch_size // cfg.grad_accum
    steps_per_epoch = len(ds) // micro // cfg.grad_accum
    total = cfg.max_steps or (cfg.epochs * steps_per_epoch if cfg.epochs else cfg.steps)
    eval_every = cfg.eval_every_epochs * steps_per_epoch if cfg.epochs else cfg.eval_every
    scheduler = build_scheduler(optimizer, total, cfg.warmup, cfg.schedule, backbone_warmup=cfg.backbone_warmup)
    loader_kw = dict(batch_size=micro, shuffle=True, num_workers=cfg.workers, collate_fn=collate_clips, drop_last=True,
                     persistent_workers=cfg.workers > 0)
    log = open(out_dir / "train_log.jsonl", "a")
    step, epoch, best, history, t0 = 0, 0, -1.0, [], time.time()
    model.train()
    while step < total:
        ds.set_epoch(epoch)
        for i, batch in enumerate(DataLoader(ds, **loader_kw)):
            targets = [{"labels": t["labels"].cuda(non_blocking=True), "masks": t["masks"].cuda(non_blocking=True)}
                       for t in batch["targets"]]  # 正解のマスクは (N, T, S, S) のまま(損失の中でフレームごとに分ける)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs, _ = model(batch["images"].cuda(non_blocking=True))
            loss, parts = criterion(outs, targets)
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
                m = evaluate_video_set(model, cfg.dev_set, cfg, cfg.dev_n)
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2),
                                **{k: m[k] for k in ("AP", "AP50", "AP75", "image_AP", "image_AP50", "image_AP75")},
                                "s_per_video": m["s_per_video"]})
                test_every = cfg.test_every_epochs * steps_per_epoch if cfg.epochs else cfg.eval_every
                if cfg.test_set and (step % test_every == 0 or step == total):  # test の役のセット(記録だけ。選択には使わない)
                    mt = evaluate_video_set(model, cfg.test_set, cfg)
                    history[-1].update({f"test_{k}": mt[k] for k in ("AP", "AP50", "AP75", "image_AP", "image_AP50", "image_AP75")})
                    history[-1]["test_image_per_class_AP"] = mt["image_per_class_AP"]
                    history[-1]["test_per_class_AP"] = mt["per_class_AP"]
                history[-1].update(_timestamp(t0))  # 評価が終わった時刻(評価にかかった時間も、前後の記録の時刻の差でわかる)
                log.write(json.dumps({"eval": history[-1]}) + "\n"); log.flush()
                print("[eval]", history[-1], flush=True)
                torch.save(model.state_dict(), out_dir / "last.pt")
                if m["AP"] > best:
                    best = m["AP"]
                    torch.save(model.state_dict(), out_dir / "best.pt")
            if step >= total:
                break
        epoch += 1
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cuda"))
    # 結果のファイルは GitHub に上げるので、設定の中のパスはリポジトリ直下からの相対パスにする(手元の環境のパスを残さない)
    config = {k: os.path.relpath(v, REPO_ROOT) if isinstance(v, str) and v.startswith(str(REPO_ROOT)) else v
              for k, v in asdict(cfg).items()}
    result = {"name": cfg.name, "config": config, "history": history,
              "eval": {s: evaluate_video_set(model, s, cfg) for s in cfg.eval_sets}}
    save_metrics(res_dir / f"{cfg.name}.json", result)
    return result
