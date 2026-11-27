"""途中で止めた動画の学習(train.py)の、最後の評価だけを行って results/{名前}.json を作る。

学習の最後(lib_common/video_train.py の train_video の最後)と同じこと: 最良の重み(runs/{名前}/best.pt。dev の動画の AP が最良だった
重み)で eval_sets の動画の評価(evaluate_video_set)を行い、学習の途中の評価の記録(train_log.jsonl の eval)を history に並べて
保存する。止めたエポック("stopped_at_epoch")も書く。
加えて、README の表の「画像 AP(1 枚ずつ予測)」を測る(frame_eval): val のフレームのセット(vipseg_frames_cls8_val)の各フレームを
1 フレームの動画としてモデルに入れ、画像の評価(lib_common/evaluate.py の evaluate。mask AP)にかける。動画のモデルのクラスの番号は
動画のセットの並びなので、クラスの名前でフレームのセットの並びに直してから評価する。
    - predict_video を持つモデル(VidEoMT・Mask2Former × VidEoMT など): 各画像を 1 フレームの区間として forward に通し、最後の層の
      出力を画像の Mask2Former と同じ後処理(train.predict)にかける
    - track_video を持つモデル(SAM 3 型): 各画像を 1 フレームの動画として track_video・to_predictions に通し、その 1 フレームの
      マスクと確信度を画像の予測にする

使い方(リポジトリ直下で):
    .venv/bin/python -u segmentation/instance_segmentation/exp00502_video_arch_comparison/finalize_video_stopped.py \
        --name videomt_m2f_vitb_minvisseg_vipsegaug_cls8_ep100_const --max-vram-gb 10
"""
import argparse
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import common
import torch
from torch import nn

from lib_common import REPO_ROOT  # noqa: E402
from segmentation.instance_segmentation.lib_common.evaluate import evaluate, save_metrics, set_path  # noqa: E402
from segmentation.instance_segmentation.lib_common.train import amp_autocast, predict  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_data import load_video_set  # noqa: E402
from segmentation.instance_segmentation.lib_common.video_train import (VideoTrainConfig, build_video_model,  # noqa: E402
                                                                       evaluate_video_set)

FRAMES = "vipseg_frames_cls8_val"


class OneFrame(nn.Module):
    """動画のモデルに各画像を 1 フレームの区間として入れ、クラスの列をフレームのセットの並びにした画像の Mask2Former の形にする。

    Args:
        vm: forward(区間 (B, T, 3, S, S)) → (層ごとの {"logits": (B, T, Q, K+1), "masks": (B, Q, T, h, w)}, ·) の動画のモデル。
        order: フレームのセットの j 番目のクラス = 動画のモデルの order[j] 番目のクラス。
    """

    def __init__(self, vm: nn.Module, order: list[int]):
        super().__init__()
        self.vm, self.order = vm, order + [len(order)]  # 最後は「物体なし」

    def forward(self, x: torch.Tensor) -> list[dict]:
        """画像をまとめて 1 フレームの区間として通す。

        Args:
            x: 画像 (B, 3, S, S)。

        Returns:
            [{"logits": (B, Q, K+1)(フレームのセットの並び), "masks": (B, Q, h, w)}](画像の Mask2Former の最後の層と同じ形)。
        """
        outs, _ = self.vm(x[:, None])  # (B, 3, S, S) -> (B, 1, 3, S, S)
        last = outs[-1]
        return [{"logits": last["logits"][:, 0][..., self.order], "masks": last["masks"][:, :, 0]}]


@torch.no_grad()
def frame_eval_tracker(model: nn.Module, cfg: VideoTrainConfig, video_cat_ids: list[int]) -> dict:
    """track_video を持つモデルの「画像 AP(1 枚ずつ予測)」: 各画像を 1 フレームの動画として追跡し、その予測で画像の評価をする。

    Args:
        model: track_video・to_predictions を持つモデル(SAM 3 型)。
        cfg: 学習の設定(入力の大きさ・AMP)。
        video_cat_ids: 動画のモデルのクラスの番号 → 動画のセットのカテゴリーの id。

    Returns:
        evaluate の返り値。
    """
    from segmentation.instance_segmentation.lib_common.data import InstanceDataset
    from segmentation.instance_segmentation.lib_common.transforms import eval_transforms
    frame_cats = json.load(open(set_path(FRAMES)))["categories"]
    vid_cats = {c["id"]: c["name"] for c in load_video_set(cfg.train_set)["categories"]}
    to_frame_id = {cid: next(c["id"] for c in frame_cats if c["name"] == name) for cid, name in vid_cats.items()}
    ds = InstanceDataset(set_path(FRAMES), eval_transforms(cfg.size))
    preds = []
    for i in range(len(ds)):
        item = ds[i]
        h, w = item["orig_size"]
        scale = cfg.size / max(h, w)
        valid, orig = (round(h * scale), round(w * scale)), (h, w)
        img = item["image"].cuda()[None]  # (1, 3, S, S)
        with amp_autocast(cfg):
            tracks = model.track_video(img, valid, orig)
            vp = model.to_predictions(tracks, 0, video_cat_ids, cfg.size, valid, orig, 1)
        preds += [{"image_id": item["image_id"], "category_id": to_frame_id[p["category_id"]], "segmentation": p["segmentations"][0],
                   "score": float(p["score"])} for p in vp if p["segmentations"][0] is not None]
    return evaluate(FRAMES, preds, boundary=False, pixel=False)


def main() -> None:
    """止めた学習の最後の評価を行い、results/{名前}.json に保存する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--max-vram-gb", type=float, default=10.0)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    run_dir = common.RUNS_DIR / a.name
    d = json.loads((run_dir / "config.json").read_text())
    fields = set(VideoTrainConfig.__dataclass_fields__)
    cfg = VideoTrainConfig(**{k: v for k, v in d.items() if k in fields})
    for k in ("segmenter", "init_model", "init_weights"):  # 設定はリポジトリ直下からの相対パスで書いてある
        v = getattr(cfg, k, None)
        if isinstance(v, str) and not os.path.isabs(v):
            setattr(cfg, k, str(REPO_ROOT / v))
    cfg.init_model = None  # 最良の重みを読むので、学習の初めの重みは読まない
    gt_cats = load_video_set(cfg.train_set)["categories"]
    model = build_video_model(cfg, len(gt_cats)).cuda()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cuda"))
    model.eval()
    history = [json.loads(l)["eval"] for l in open(run_dir / "train_log.jsonl") if '"eval"' in l]
    # 結果のファイルは GitHub に上げるので、設定の中のパスはリポジトリ直下からの相対パスにする(train_video の最後と同じ)
    rel = {k: os.path.relpath(v, REPO_ROOT) if isinstance(v, str) and v.startswith(str(REPO_ROOT)) else v for k, v in d.items()}
    result = {"name": a.name, "config": rel, "history": history,
              "stopped_at_epoch": max((h["epoch"] for h in history), default=None),
              "eval": {s: evaluate_video_set(model, s, cfg) for s in cfg.eval_sets}}
    print({s: {k: v for k, v in r.items() if not isinstance(v, dict)} for s, r in result["eval"].items()}, flush=True)
    # 画像 AP(1 枚ずつ予測)
    video_names = [c["name"] for c in gt_cats]
    if hasattr(model, "track_video"):
        m = frame_eval_tracker(model, cfg, [c["id"] for c in gt_cats])
    else:
        frame_names = [c["name"] for c in sorted(json.load(open(set_path(FRAMES)))["categories"], key=lambda c: c["id"])]
        order = [video_names.index(n) for n in frame_names]
        pcfg = SimpleNamespace(size=cfg.size, eval_resize="pad", batch_size=8, workers=4, arch="mask2former",
                               amp=cfg.amp, amp_dtype=cfg.amp_dtype)
        preds, _ = predict(OneFrame(model, order), FRAMES, pcfg)
        m = evaluate(FRAMES, preds, boundary=False, pixel=False)
    result["frame_eval"] = {FRAMES: {"mask_AP": m["segm"]["AP"], "per_class_segm_AP": m["per_class_segm_AP"]}}
    print("画像 AP(1 枚ずつ予測)", round(m["segm"]["AP"] * 100, 2), flush=True)
    save_metrics(common.RESULTS_DIR / f"{a.name}.json", result)


if __name__ == "__main__":
    main()
