"""評価のセット(データセットの中の動画・フレームの範囲と、train・val・test の役)と、TrackEval が読む形の正解の書き出し。

役の決め方は mot/exp00001_data_and_eval/README.md の「データセットごとの役」:
    MOT17(FRCNN の版だけを使う。DPM・SDP の版は画像と正解が同じ):
        mot17_train      各動画の前半(1 〜 L // 2 フレーム)
        mot17_val        後半の前の半分(L // 2 + 1 〜 L // 2 + L // 4)
        mot17_test       後半の残り
        mot17_valhalf    後半全部(論文の MOT17 の val と同じ。公開の手法の値と比べるときだけ使う)
    DanceTrack:
        dancetrack_train  公式の train 40 本から、dancetrack_val の 8 本を除いた 32 本
        dancetrack_val    公式の train から seed 0 で選んだ 8 本
        dancetrack_test   公式の val 25 本
    KITTI MOTS(MOTS の論文の分け方。mots_tools の train.seqmap・val.seqmap):
        kitti_mots_train  公式の train 12 本から、kitti_mots_val の 3 本を除いた 9 本
        kitti_mots_val    公式の train 12 本から seed 0 で選んだ 3 本
        kitti_mots_test   公式の val 9 本

フレームの範囲を切ったセット(MOT17)は、TrackEval の決まりに合わせてフレームの番号を 1 から振り直した正解を書き出す
(ByteTrack の val_half と同じやり方)。追跡の結果も、セットの中のフレームの番号(1 始まり)で書く。

形の記号: L = 動画のフレームの数。
"""
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from lib_common import DATASETS_DIR

SETS_DIR = DATASETS_DIR / "mot_sets"  # TrackEval が読む形の正解の置き場(作り直せる)
MOT17_DIR = DATASETS_DIR / "mot17" / "MOT17" / "train"
DANCETRACK_DIR = DATASETS_DIR / "dancetrack"
KITTI_MOTS_DIR = DATASETS_DIR / "kitti_mots"
KITTI_MOTS_TRAIN = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
KITTI_MOTS_VAL = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
TUNE_SEED = 0


@dataclass
class Sequence:
    """評価のセットの中の 1 本の動画(の一部のフレーム)。

    Attributes:
        name: 動画の名前(セットの中で一意)。
        image_dir: 画像のフォルダ。
        image_pattern: 元の動画のフレームの番号 → 画像のファイル名の書式(例 "{:06d}.jpg")。
        first_frame: セットに入る最初のフレームの、元の動画での番号。
        num_frames: セットに入るフレームの数。
        width: 画像の横の画素数。
        height: 画像の縦の画素数。
        fps: フレームの速さ。
    """
    name: str
    image_dir: str
    image_pattern: str
    first_frame: int
    num_frames: int
    width: int
    height: int
    fps: float

    def image_paths(self) -> list[Path]:
        """セットの中のフレームの順の画像のパス。

        Returns:
            長さ num_frames のパスの並び。
        """
        return [Path(self.image_dir) / self.image_pattern.format(self.first_frame + i) for i in range(self.num_frames)]


def _seqinfo(seq_dir: Path) -> dict:
    """MOTChallenge の seqinfo.ini を読む。

    Args:
        seq_dir: 動画のフォルダ。

    Returns:
        {"seqLength", "imWidth", "imHeight", "frameRate", ...}。
    """
    out = {}
    for line in (seq_dir / "seqinfo.ini").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _mot_style_sequences(seq_dirs: list[Path], frame_range=None) -> list[tuple[Sequence, Path]]:
    """MOTChallenge の形の動画(MOT17・DanceTrack)のセットの動画を作る。

    Args:
        seq_dirs: 動画のフォルダの並び。
        frame_range: 動画のフレームの数 L → (最初のフレーム(1 始まり), フレームの数) を返す関数(None なら全部)。

    Returns:
        (Sequence, 正解のファイル gt.txt) の並び。
    """
    out = []
    for d in seq_dirs:
        info = _seqinfo(d)
        n = int(info["seqLength"])
        first, count = frame_range(n) if frame_range else (1, n)
        out.append((Sequence(d.name, str(d / "img1"), "{:06d}" + info["imExt"], first, count, int(info["imWidth"]),
                             int(info["imHeight"]), float(info["frameRate"])), d / "gt" / "gt.txt"))
    return out


def _split_tune(names: list[str], k: int) -> tuple[list[str], list[str]]:
    """名前の並びから、設定の値を選ぶ用の k 本を seed 0 で選び、(残り, 選んだもの) にする。

    Args:
        names: 動画の名前の並び。
        k: 選ぶ数。

    Returns:
        (残りの名前の並び, 選んだ名前の並び)(どちらも名前の順)。
    """
    tune = sorted(random.Random(TUNE_SEED).sample(sorted(names), k))
    return [n for n in sorted(names) if n not in tune], tune


def mot_sets() -> dict[str, list[tuple[Sequence, Path]]]:
    """枠の追跡のセット(MOT17・DanceTrack)。

    Returns:
        {セットの名前: [(Sequence, 元の正解のファイル)]}。
    """
    mot17 = sorted(MOT17_DIR.glob("*-FRCNN"))
    dt_train = {d.name: d for part in ("train1", "train2") for d in (DANCETRACK_DIR / part).glob("dancetrack*")}
    rest, tune = _split_tune(list(dt_train), 8)
    return {
        "mot17_train": _mot_style_sequences(mot17, lambda n: (1, n // 2)),
        "mot17_val": _mot_style_sequences(mot17, lambda n: (n // 2 + 1, n // 4)),
        "mot17_test": _mot_style_sequences(mot17, lambda n: (n // 2 + n // 4 + 1, n - n // 2 - n // 4)),
        "mot17_valhalf": _mot_style_sequences(mot17, lambda n: (n // 2 + 1, n - n // 2)),
        "dancetrack_train": _mot_style_sequences([dt_train[n] for n in rest]),
        "dancetrack_val": _mot_style_sequences([dt_train[n] for n in tune]),
        "dancetrack_test": _mot_style_sequences(sorted((DANCETRACK_DIR / "val").glob("dancetrack*"))),
    }


def kitti_mots_sets() -> dict[str, list[tuple[Sequence, Path]]]:
    """マスクの追跡のセット(KITTI MOTS)。

    Returns:
        {セットの名前: [(Sequence, 元の正解のファイル instances_txt/{動画}.txt)]}。
    """
    def seqs(names: list[str]) -> list[tuple[Sequence, Path]]:
        """KITTI MOTS の動画の名前の並びから、動画の情報と正解のファイルの組の並びを作る。

        Args:
            names: 動画の名前("0000" など)の並び。

        Returns:
            (動画の情報, その動画の正解のファイル(instances_txt の中)) の並び。
        """
        out = []
        for s in names:
            img_dir = KITTI_MOTS_DIR / "training" / "image_02" / s
            n = len(list(img_dir.glob("*.png")))
            h, w = cv2.imread(str(img_dir / "000000.png")).shape[:2]
            out.append((Sequence(s, str(img_dir), "{:06d}.png", 0, n, w, h, 10.0), KITTI_MOTS_DIR / "instances_txt" / f"{s}.txt"))
        return out
    rest, tune = _split_tune(KITTI_MOTS_TRAIN, 3)
    return {"kitti_mots_train": seqs(rest), "kitti_mots_val": seqs(tune), "kitti_mots_test": seqs(KITTI_MOTS_VAL)}


def write_mot_set(name: str, seqs: list[tuple[Sequence, Path]]) -> Path:
    """枠の追跡のセットの正解を、TrackEval の MotChallenge2DBox が読む形で書き出す(フレームの番号は 1 から振り直す)。

    書き出す形: {SETS_DIR}/{name}/{動画}/gt/gt.txt・seqinfo.ini、{SETS_DIR}/{name}/seqmap.txt、{SETS_DIR}/{name}/sequences.json。

    Args:
        name: セットの名前。
        seqs: mot_sets の値。

    Returns:
        書き出したフォルダ。
    """
    root = SETS_DIR / name
    for seq, gt_path in seqs:
        d = root / seq.name / "gt"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for line in gt_path.read_text().splitlines():
            f = line.split(",")
            t = int(f[0]) - seq.first_frame + 1  # セットの中の番号(1 始まり)
            if 1 <= t <= seq.num_frames:
                rows.append(",".join([str(t)] + f[1:]))
        (d / "gt.txt").write_text("\n".join(rows) + "\n")
        (root / seq.name / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq.name}\nimDir=img1\nframeRate={seq.fps:g}\nseqLength={seq.num_frames}\n"
            f"imWidth={seq.width}\nimHeight={seq.height}\nimExt=.jpg\n")
    (root / "seqmap.txt").write_text("name\n" + "\n".join(s.name for s, _ in seqs) + "\n")
    (root / "sequences.json").write_text(json.dumps([asdict(s) for s, _ in seqs], indent=1))
    return root


def write_kitti_mots_set(name: str, seqs: list[tuple[Sequence, Path]]) -> Path:
    """マスクの追跡のセットの正解を、TrackEval の KittiMOTS が読む形で書き出す(KITTI MOTS は動画全体なので、元の正解を写すだけ)。

    書き出す形: {SETS_DIR}/{name}/label_02/{動画}.txt、{SETS_DIR}/{name}/evaluate_mots.seqmap.val、{SETS_DIR}/{name}/sequences.json。

    Args:
        name: セットの名前。
        seqs: kitti_mots_sets の値。

    Returns:
        書き出したフォルダ。
    """
    root = SETS_DIR / name
    (root / "label_02").mkdir(parents=True, exist_ok=True)
    for seq, gt_path in seqs:
        (root / "label_02" / f"{seq.name}.txt").write_text(gt_path.read_text())
    (root / "evaluate_mots.seqmap.val").write_text(
        "".join(f"{s.name} empty {0:06d} {s.num_frames:06d}\n" for s, _ in seqs))
    (root / "sequences.json").write_text(json.dumps([asdict(s) for s, _ in seqs], indent=1))
    return root


def load_sequences(name: str) -> list[Sequence]:
    """書き出したセットの動画の並びを読む。

    Args:
        name: セットの名前。

    Returns:
        Sequence の並び。
    """
    return [Sequence(**d) for d in json.loads((SETS_DIR / name / "sequences.json").read_text())]


def build_all() -> None:
    """全部のセットを書き出す。"""
    for name, seqs in mot_sets().items():
        write_mot_set(name, seqs)
    for name, seqs in kitti_mots_sets().items():
        write_kitti_mots_set(name, seqs)
