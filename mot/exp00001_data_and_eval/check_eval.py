"""評価のセットを書き出し、評価(TrackEval)が正しく動くかを、答えのわかる追跡の結果で確かめる(CPU だけで動く)。

確かめること(セットごと):
    1. 正解をそのまま追跡の結果として渡すと、HOTA・IDF1・MOTA が 100、IDSW が 0 になる
    2. 動画の真ん中で全部の ID を付け替える(同じ物体が途中で別の ID になる)と、AssA・IDF1 が下がり、DetA は 100 のまま
    3. 1 フレームおきに結果を消す(見逃し)と、DetA が下がる

結果: results/check_eval.json。使い方(リポジトリ直下で): .venv/bin/python -u mot/exp00001_data_and_eval/check_eval.py
"""
import json

import common

from mot.lib_common.evaluate import evaluate_kitti_mots, evaluate_mot, write_kitti_mots_results, write_mot_results
from mot.lib_common.sets import SETS_DIR, build_all, load_sequences

ID_OFFSET = 100000  # 付け替えた ID(元の ID と重ならない値)


def mot_gt_tracks(set_name: str, seq_name: str) -> list[tuple]:
    """枠の追跡のセットの正解のうち、評価に使う歩行者(クラス 1・印 1)を、確信度 1 の追跡の結果の形にする。

    Args:
        set_name: セットの名前。
        seq_name: 動画の名前。

    Returns:
        (フレーム, ID, 左, 上, 幅, 高さ, 確信度) の並び。
    """
    out = []
    for line in (SETS_DIR / set_name / seq_name / "gt" / "gt.txt").read_text().splitlines():
        f = line.split(",")
        if int(f[6]) == 1 and int(f[7]) == 1:
            out.append((int(f[0]), int(f[1]), *map(float, f[2:6]), 1.0))
    return out


def kitti_gt_tracks(set_name: str, seq_name: str) -> list[tuple]:
    """マスクの追跡のセットの正解のうち、車・歩行者(無視する領域を除く)を追跡の結果の形にする。

    Args:
        set_name: セットの名前。
        seq_name: 動画の名前。

    Returns:
        (フレーム, ID, クラス, 縦, 横, RLE) の並び。
    """
    out = []
    for line in (SETS_DIR / set_name / "label_02" / f"{seq_name}.txt").read_text().splitlines():
        t, oid, c, h, w, rle = line.split(" ", 5)
        if int(c) in (1, 2):
            out.append((int(t), int(oid), int(c), int(h), int(w), rle))
    return out


def perturb(tracks: list[tuple], num_frames: int, kind: str, frame_index: int, id_index: int) -> list[tuple]:
    """正解の追跡の結果を崩す。

    Args:
        tracks: 追跡の結果の並び。
        num_frames: 動画のフレームの数。
        kind: "gt"(そのまま)/ "swap"(真ん中より後のフレームで ID を付け替える)/ "drop"(1 フレームおきに消す)。
        frame_index: 行の中のフレームの位置。
        id_index: 行の中の ID の位置。

    Returns:
        崩した追跡の結果の並び。
    """
    first = min(r[frame_index] for r in tracks) if tracks else 0
    if kind == "gt":
        return tracks
    if kind == "swap":
        mid = first + num_frames // 2
        return [r[:id_index] + (r[id_index] + ID_OFFSET,) + r[id_index + 1:] if r[frame_index] >= mid else r for r in tracks]
    return [r for r in tracks if (r[frame_index] - first) % 2 == 0]


def main() -> None:
    """全部のセットを書き出し、セットごとに 3 通りの追跡の結果を評価して保存する。"""
    build_all()
    results = {}
    mot_sets = ["mot17_train", "mot17_val", "mot17_test", "mot17_valhalf", "dancetrack_train", "dancetrack_val", "dancetrack_test"]
    for s in mot_sets + ["kitti_mots_train", "kitti_mots_val", "kitti_mots_test"]:
        is_kitti = s.startswith("kitti")
        results[s] = {}
        for kind in ("gt", "swap", "drop"):
            out_dir = common.RUNS_DIR / s
            for seq in load_sequences(s):
                if is_kitti:
                    tracks = perturb(kitti_gt_tracks(s, seq.name), seq.num_frames, kind, 0, 1)
                    write_kitti_mots_results(out_dir / kind / f"{seq.name}.txt", tracks)
                else:
                    tracks = perturb(mot_gt_tracks(s, seq.name), seq.num_frames, kind, 0, 1)
                    write_mot_results(out_dir / kind / f"{seq.name}.txt", tracks)
            r = (evaluate_kitti_mots if is_kitti else evaluate_mot)(s, out_dir, kind)
            results[s][kind] = r
            print(s, kind, {c: {k: round(v, 1) for k, v in m.items()} for c, m in r.items()}, flush=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (common.RESULTS_DIR / "check_eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
