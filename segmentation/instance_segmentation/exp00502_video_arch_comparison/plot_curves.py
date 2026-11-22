"""exp00502 の全部の学習の、学習中の評価の推移を、同じ指標・同じ軸で重ねて描く(学習の途中でも何度でも描き直せる)。

指標(どのモデルも同じ dev・val の動画の全フレームで、同じ計算):
    - 動画の AP: 動画全体の物体ごとに予測と正解を照らす(YouTube-VIS と同じ)。MOT の仕組みを持つモデルだけ
      (VidEoMT: クエリの受け渡しで学習した ID、MinVIS: 推論の照合の ID)
    - 画像 AP: 各フレームを 1 枚の画像とみなした COCO の mask AP(ID は使わない)。全部のモデル
      (画像のモデル単体は、フレームごとの予測をそのまま評価したもの)
横軸はエポック。どのモデルも 1 エポック ≒ 1 万 500 フレーム(VidEoMT: 262 step × 8 区間 × 5 フレーム = 10,480、
MinVIS・画像のモデル: 10,608 枚)で、学習したフレームの数がそろう。

読むもの: runs/{学習の名前}/train_log.jsonl の評価の記録
    - 動画のモデル(train.py): {"eval": {"epoch", "AP", "image_AP", "test_AP", "test_image_AP", ...}}
    - MinVIS(train_image.py の --video-dev-set): {"video_eval": {"set", "epoch", "AP", "image_AP", ...}}
    - 画像のモデル(train_image.py): {"eval": {"epoch", "set", "segm": {"AP"}}}・{"test": {"epoch", "set", "segm": {"AP"}}}
出力: results/curves/compare_{dev,val}_{video_AP,image_AP}.png

使い方(リポジトリ直下で):
    .venv/bin/python segmentation/instance_segmentation/exp00502_video_arch_comparison/plot_curves.py
"""
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[2]))
from lib_common.plots import plot_eval_curves  # noqa: E402

DEV_VIDEO, VAL_VIDEO = "vipseg_vis_cls8_dev", "vipseg_vis_cls8_val"
DEV_FRAMES, VAL_FRAMES = "vipseg_frames_cls8_dev", "vipseg_frames_cls8_val"
SKIP = ("SMOKETEST", "_ft100")  # 動作確認と、止めた続きの学習
OLD = ("videomt_vitb_dinov3_cls8_ep100_const", "videomt_vitb_from00105_cls8_ep100_const")  # 公式と違う実装の 2 本(記録)


def read_run(log_path: Path) -> dict[str, list[tuple[float, float]]]:
    """1 つの学習の評価の記録から、指標ごとの [(エポック, 値 %)] を取り出す。"""
    out = {k: [] for k in ("dev_video_AP", "val_video_AP", "dev_image_AP", "val_image_AP")}
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        if "video_eval" in rec:  # MinVIS
            r = rec["video_eval"]
            split = "dev" if r["set"] == DEV_VIDEO else "val" if r["set"] == VAL_VIDEO else None
            if split:
                out[f"{split}_video_AP"].append((r["epoch"], 100 * r["AP"]))
                out[f"{split}_image_AP"].append((r["epoch"], 100 * r["image_AP"]))
        elif "eval" in rec and "AP" in rec["eval"]:  # 動画のモデル
            r = rec["eval"]
            out["dev_video_AP"].append((r["epoch"], 100 * r["AP"]))
            if "image_AP" in r:
                out["dev_image_AP"].append((r["epoch"], 100 * r["image_AP"]))
            if "test_AP" in r:
                out["val_video_AP"].append((r["epoch"], 100 * r["test_AP"]))
            if "test_image_AP" in r:
                out["val_image_AP"].append((r["epoch"], 100 * r["test_image_AP"]))
        elif "eval" in rec or "test" in rec:  # 画像のモデルのフレームごとの評価(MinVIS の学習にもある。動画の評価がある学習では使わない)
            r = rec.get("eval") or rec.get("test")
            split = "dev" if r["set"].startswith(DEV_FRAMES) else "val" if r["set"] == VAL_FRAMES else None
            if split:
                out[f"frame_{split}_image_AP"] = out.get(f"frame_{split}_image_AP", []) + [(r["epoch"], 100 * r["segm"]["AP"])]
    for split in ("dev", "val"):  # 動画の評価がない学習(画像のモデル単体)は、フレームごとの評価を画像 AP にする
        if not out[f"{split}_image_AP"] and out.get(f"frame_{split}_image_AP"):
            out[f"{split}_image_AP"] = out[f"frame_{split}_image_AP"]
    return out


def main() -> None:
    runs = {}
    for log in sorted((EXP_DIR / "runs").glob("*/train_log.jsonl")):
        name = log.parent.name
        if any(s in name for s in SKIP):
            continue
        runs[name + ("(公式と違う旧実装)" if name in OLD else "")] = read_run(log)
    titles = {"dev_video_AP": "dev(200 本)の動画の AP", "val_video_AP": "val(279 本、test の役)の動画の AP",
              "dev_image_AP": "dev の画像 AP(フレームごとの mask AP)", "val_image_AP": "val の画像 AP(フレームごとの mask AP)"}
    for key, title in titles.items():
        series = {name: sorted(r[key]) for name, r in runs.items() if r[key]}
        if series:
            out = EXP_DIR / "results" / "curves" / f"compare_{key}.png"
            plot_eval_curves(series, out, title, ylabel=key.split("_", 1)[1].replace("_", " ") + " (%)", legend_outside=True)
            print(out.relative_to(EXP_DIR), len(series), "本")


if __name__ == "__main__":
    main()
