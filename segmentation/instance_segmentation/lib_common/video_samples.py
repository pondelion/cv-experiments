"""動画インスタンスセグメンテーションの予測の見本(正解と予測を左右に並べた動画と、数フレームを並べた画像)を描く。

色は動画を通した物体の ID ごとに固定し、1 本の動画の中では物体ごとに違う色にする(正解は物体の注釈ごと、予測は描くクエリごと)。
追跡が切れたり別の物体に移ったりすると、同じ物体の色が途中で変わるので、目で見てわかる。マスクの中心に、物体の色の帯の上に
クラス名(予測はスコアも)を書く(文字は縮めた後の画像に書く)。

保存するもの(1 本の動画につき):
    {out_dir}/{動画の名前}.mp4  正解 | 予測 を左右に並べた動画(5fps)
    {out_dir}/{動画の名前}.jpg  同じ並べ方で、等間隔に選んだ n_frames 枚を縦に並べた画像
"""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from PIL import Image

from lib_common import DATASETS_DIR

def _colors(n: int) -> np.ndarray:
    """互いに違う n 色を作る(色相を黄金比ずつずらすので、何色でも重ならず、隣どうしの番号の色も離れる)。

    Args:
        n: 色の数。

    Returns:
        (n, 3) の RGB の uint8。
    """
    hsv = np.array([[int(180 * ((i * 0.618034) % 1.0)), 200, 240] for i in range(max(n, 1))], np.uint8)[None]  # OpenCV の H は 0〜179
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def _draw(img: np.ndarray, objs: list[tuple[np.ndarray, dict, str]], t: int) -> tuple[np.ndarray, list]:
    """1 フレームに物体のマスクと輪郭を重ねる(文字は縮めた後に書くので、位置だけ返す)。

    Args:
        img: (H, W, 3) の uint8(RGB)。
        objs: [(色 (3,), 注釈か予測({"segmentations": [フレームごとの RLE か None]}), 書く文字)]。
        t: フレームの番号。

    Returns:
        ((H, W, 3) の uint8, [(文字, 色, マスクの中心の (y, x))])。
    """
    out = img.astype(np.float32)
    drawn = []
    for color, o, text in objs:
        rle = o["segmentations"][t]
        if rle is None:
            continue
        m = mask_util.decode(rle).astype(bool)  # (H, W)
        if not m.any():
            continue
        out[m] = 0.45 * out[m] + 0.55 * color.astype(np.float32)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        drawn.append((contours, color, text, np.argwhere(m).mean(0)))
    out = out.astype(np.uint8)
    labels = []
    for contours, color, text, center in drawn:
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 3)
        labels.append((text, color, center))
    return out, labels


def _put_labels(img: np.ndarray, labels: list, scale: float, dx: int) -> None:
    """縮めた後の画像に、物体の色の帯の上に白い文字で書く。

    Args:
        img: 縮めた後の画像 (h, w, 3)(その場で書く)。
        labels: (文字, 色, 元の画素での中心 (cy, cx)) の並び。
        scale: 元の画素 → 縮めた画素の倍率。
        dx: 横のずれ(画素)。
    """
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    for text, color, (cy, cx) in labels:
        x, y = int(cx * scale) + dx, int(cy * scale)
        (w, h), base = cv2.getTextSize(text, font, fs, th)
        x = min(max(x - w // 2, 0), img.shape[1] - w)
        cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + base), tuple(int(c) for c in color), -1)
        cv2.putText(img, text, (x, y), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """フレームを mp4 に書く(ffmpeg があれば H.264(ブラウザ・エディタで再生できる)、なければ cv2 の mp4v)。

    Args:
        frames: RGB のフレーム(同じ大きさ)の並び。
        path: 書くファイル。
        fps: フレームの速さ。
    """
    h, w = frames[0].shape[:2]
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
               str(path)]  # yuv420p は縦横が偶数でないといけないので、pad で偶数にする
        subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
        return
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def render_video_compare(video: dict, gt_anns: list[dict], preds: list[dict], names: dict[int, str], out_dir: Path,
                         score_thr: float = 0.3, width: int = 1600, n_frames: int = 6) -> None:
    """1 本の動画の、正解と予測を左右に並べた見本を保存する。

    Args:
        video: セットの動画の情報({"name", "file_names", ...})。
        gt_anns: その動画の正解の物体(セットの annotations)。
        preds: その動画の予測(video_infer.video_predictions の出力)。同じクエリ(物体)が別のクラスで何度か出るときは、
            スコアの最も高いものだけを描く。
        names: カテゴリーの id → クラス名。
        out_dir: 保存先のフォルダ。
        score_thr: 描く予測の最小のスコア。
        width: 並べた画像の横の画素の数(左右の合計)。
        n_frames: 画像に並べるフレームの数。
    """
    best = {}
    for p in sorted(preds, key=lambda p: -p["score"]):
        if p["score"] >= score_thr and p["query"] not in best:
            best[p["query"]] = p
    gt_colors, pr_colors = _colors(len(gt_anns)), _colors(len(best))  # 1 本の動画の中で、物体ごとに違う色(正解と予測は別々に)
    gt_objs = [(gt_colors[i], a, names[a["category_id"]]) for i, a in enumerate(gt_anns)]
    pr_objs = [(pr_colors[i], p, f"{names[p['category_id']]} {p['score']:.2f}") for i, p in enumerate(best.values())]
    frames = []
    for t, f in enumerate(video["file_names"]):
        img = np.array(Image.open(DATASETS_DIR / f).convert("RGB"))
        (left, gl), (right, pl) = _draw(img, gt_objs, t), _draw(img, pr_objs, t)
        pair = np.concatenate([left, right], axis=1)  # (H, 2W, 3) 左: 正解、右: 予測
        scale = width / pair.shape[1]
        out = cv2.resize(pair, (width, round(pair.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        _put_labels(out, gl, scale, 0)  # 文字は縮めた後に書く(縮めると読めなくなるため)
        _put_labels(out, pl, scale, width // 2)
        frames.append(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = video["name"]
    _write_mp4(frames, out_dir / f"{name}.mp4", fps=5)
    pick = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
    Image.fromarray(np.concatenate([frames[i] for i in pick], axis=0)).save(out_dir / f"{name}.jpg", quality=85)
