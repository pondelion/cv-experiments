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
# 図に入れない学習: 動作確認、止めた続きの学習、メモリが足りずに止めた試し、README の結果の表に載せていない学習
# (GTR 系 × VidEoMT 型(崩れた学習・学習中)、学習中の DEVA、伝搬だけを J&F で測る SAM 2)
SKIP = ("SMOKETEST", "_ft100", "_oomtry", "videomt_gtr_", "deva_", "sam2_")
OLD = ("videomt_vitb_dinov3_cls8_ep100_const", "videomt_vitb_from00105_cls8_ep100_const")  # 公式と違う実装の 2 本(記録)


def read_run(log_path: Path, corrected: dict | None = None) -> dict[str, list[tuple[float, float]]]:
    """1 つの学習の評価の記録から、指標ごとの [(エポック, 値 %)] を取り出す。

    Args:
        log_path: 学習の記録(train_log.jsonl)。
        corrected: MinVIS の測り直した動画の AP(results/minvis_corrected_video_ap.json の 1 本分)か None。あれば、記録の
            動画の評価(video_eval。person と car が入れ替わった値)を使わず、8 クラスの動画の AP の曲線は描かない(途中の値がないため。
            推移は minvis_unaffected_classes の 6 クラスの図)。画像 AP はフレームのセットの評価にする。

    Returns:
        {指標の名前: [(エポック, 値 %)]}。
    """
    out = {k: [] for k in ("dev_video_AP", "val_video_AP", "dev_image_AP", "val_image_AP")}
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        if "video_eval" in rec and corrected:  # 直す前の値(使わない)
            continue
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


SWAPPED = ("person", "car")  # MinVIS の学習中の動画の評価で入れ替わっていたクラス


def minvis_unaffected_classes(log_path: Path) -> dict[str, list[tuple[float, float]]]:
    """MinVIS の学習中の動画の評価から、入れ替わりの影響を受けない 6 クラスの AP の平均と、クラスごとの AP を取り出す。

    学習中の評価は person と car が入れ替わっていた(person の予測を car の正解と、car の予測を person の正解と照らした)。AP はクラスごとに
    別々に計算するので、ほかの 6 クラスの値はそのまま正しい(直した後の評価と小数第 2 位まで一致することを、同じ重みで確かめた)。

    Args:
        log_path: 学習の記録(train_log.jsonl)。

    Returns:
        {"{split}_video_AP6": [(エポック, 6 クラスの平均 %)]}(split は dev・val)。
    """
    out, seen = {}, set()
    for line in log_path.read_text().splitlines():
        r = json.loads(line).get("video_eval")
        if not r:
            continue
        split = "dev" if r["set"] == DEV_VIDEO else "val" if r["set"] == VAL_VIDEO else None
        if split is None or (split, r["epoch"]) in seen:  # 最後の評価で同じエポックを測り直した記録は、最初の 1 つだけ使う(6 クラスは同じ値)
            continue
        seen.add((split, r["epoch"]))
        pc = {k: 100 * v for k, v in r["per_class_AP"].items() if k not in SWAPPED}
        out.setdefault(f"{split}_video_AP6", []).append((r["epoch"], sum(pc.values()) / len(pc)))
    return out


def main() -> None:
    """全部の学習の記録を読み、指標ごとの学習曲線の図を保存する。"""
    runs = {}
    corrected = json.loads((EXP_DIR / "results" / "minvis_corrected_video_ap.json").read_text())
    for log in sorted((EXP_DIR / "runs").glob("*/train_log.jsonl")):
        name = log.parent.name
        if any(s in name for s in SKIP):
            continue
        # 図の中の文字は英語にする(既定の字体に日本語の字がないため)
        label = name + (" (non-official impl.)" if name in OLD else "")
        runs[label] = read_run(log, corrected.get(name))
        if name in corrected:  # MinVIS の動画の AP は、入れ替わりの影響を受けない 6 クラスの平均を同じ図に重ねる
            six = minvis_unaffected_classes(log)
            runs[label]["dev_video_AP"], runs[label]["val_video_AP"] = six.get("dev_video_AP6", []), six.get("val_video_AP6", [])
            runs[label + " (6 classes)"] = runs.pop(label)
    titles = {"dev_video_AP": "dev (200 videos) video AP", "val_video_AP": "val (279 videos, test role) video AP",
              "dev_image_AP": "dev image AP (per-frame mask AP)", "val_image_AP": "val image AP (per-frame mask AP)"}
    for key, title in titles.items():
        series = {name: sorted(r[key]) for name, r in runs.items() if r[key]}
        if series:
            out = EXP_DIR / "results" / "curves" / f"compare_{key}.png"
            plot_eval_curves(series, out, title, ylabel=key.split("_", 1)[1].replace("_", " ") + " (%)", legend_outside=True)
            print(out.relative_to(EXP_DIR), len(series), "本")


if __name__ == "__main__":
    main()
