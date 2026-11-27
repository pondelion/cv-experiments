"""results/ の追跡の結果から、README の結果の表(test の役のセットごと・val の役のセットでの閾値の選び方)を Markdown で出す。

追跡の手法 × 検出に使ったセグメンテーションのモデルの組ごとに、val の役のセットで HOTA(KITTI MOTS は car と pedestrian の平均)が
最良の閾値を選び、その値と公式の値の test の役のセットの結果を並べる。セットの違う結果は比べないので、表はセット(とクラス)ごとに
分ける。追跡の手法を足すときは TRACKERS に足す(同じ表に行が増える)。

使い方(リポジトリ直下で):
    python3 mot/exp00101_association_methods/make_tables.py
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
# (検出を作ったモデルの名前, セグメンテーションのモデル, 学習のデータ)
DETECTORS = [("gtr_cnxtb_dinov3_bighead640s4_cls8_2k_ep100", "GTR 系", "COCO の画像(8 クラス・2,000 枚)"),
             ("gtr_cnxtb_dinov3_bighead640s4_frames_cls8_ep100_const", "GTR 系", "VIPSeg のフレーム(8 クラス)"),
             ("minvis_cnxtb_dinov3_cls8_ep100_const", "Mask2Former(MinVIS の画像のモデル)", "VIPSeg のフレーム(8 クラス)")]
SETS = [("mot17", ["pedestrian"]), ("dancetrack", ["pedestrian"]), ("kitti_mots", ["car", "pedestrian"])]
# 見出しに書く、セットのクラスと、それに使った検出のクラス(学習のデータ(COCO・VIPSeg の 8 クラス)のクラスの名前)
CLASS_NOTE = {"pedestrian": "pedestrian(person で代用)", "car": "car"}
# (結果のファイルの名前の先頭, 表に書く追跡の手法の名前, 選ぶ閾値の名前, 公式の閾値の値)
TRACKERS = [("bytetrack", "ByteTrack", "track_thresh", 0.6),
            ("sort", "SORT", "det_thresh", 0.0),  # 公式は確信度で検出を選ばない(全部使う)ので、公式の値は 0
            ("ocsort", "OC-SORT", "det_thresh", 0.6)]


def load(prefix: str, det: str, set_name: str) -> dict:
    """1 つの追跡の手法・1 つの検出・1 つのセットの結果を読む。

    Args:
        prefix: 結果のファイルの名前の先頭(TRACKERS の 1 つ目)。
        det: 検出を作ったモデルの名前。
        set_name: セットの名前。

    Returns:
        {"{prefix}_th{値}": {"metrics": {クラス: {指標: 値}}}}。
    """
    return json.loads((RESULTS / det / f"{prefix}_{set_name}.json").read_text())


def thresh(key: str) -> float:
    """結果の名前から、その結果で使った閾値の値を取り出す。

    Args:
        key: 結果の名前("bytetrack_th0.2" など。"_th" の後ろが閾値)。

    Returns:
        閾値の値。
    """
    return float(key.split("_th")[-1])


def mean_hota(metrics: dict) -> float:
    """1 つの結果の HOTA を、評価したクラスで平均する(KITTI MOTS は car と pedestrian、ほかは pedestrian だけ)。

    Args:
        metrics: {クラス: {指標の名前: 値}}。

    Returns:
        クラスで平均した HOTA。
    """
    return sum(c["HOTA"] for c in metrics.values()) / len(metrics)


def chosen(prefix: str, det: str, ds: str) -> str:
    """val の役のセットで HOTA(クラスの平均)が最も高かった閾値の結果を選ぶ。

    Args:
        prefix: 結果のファイルの名前の先頭(追跡の手法)。
        det: 検出を作ったモデルの名前。
        ds: データセットの名前("mot17" など)。

    Returns:
        選んだ結果の名前("bytetrack_th0.2" など)。
    """
    v = load(prefix, det, f"{ds}_val")
    return max(v, key=lambda k: mean_hota(v[k]["metrics"]))


def main() -> None:
    """表を出力する。"""
    out = []
    for ds, classes in SETS:
        for cls in classes:
            # 見出しにセットの対象のクラスを書く(クラスが 2 つ以上のセットは、クラスごとの表なので、どのクラスの表かも書く)
            out += [f"#### {ds}_test({', '.join(CLASS_NOTE[c] for c in classes)})" + (f": {cls} の結果" if len(classes) > 1 else "")
                    + "\n",
                    "| 追跡の手法 | セグメンテーションのモデル | 学習のデータ | 閾値 | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
            for prefix, tracker, th_name, official in TRACKERS:
                for det, model, data in DETECTORS:
                    t = load(prefix, det, f"{ds}_test")
                    for k, tag in [(chosen(prefix, det, ds), "(val で選択)"), (f"{prefix}_th{official:g}", "(公式の値)")]:
                        m = t[k]["metrics"][cls]
                        out.append(f"| {tracker} | {model} | {data} | {th_name} {thresh(k):g}{tag} | {m['HOTA']:.1f} | "
                                   f"{m['DetA']:.1f} | {m['AssA']:.1f} | {m['LocA']:.1f} | {m['MOTA']:.1f} | {m['IDF1']:.1f} | "
                                   f"{int(m['IDSW']):,} |")
            out.append("")
    out.append("#### val の役のセットでの閾値の選び方(HOTA。KITTI MOTS は car と pedestrian の平均。太字が選んだ値)\n")
    for prefix, tracker, th_name, _ in TRACKERS:
        for ds, _ in SETS:
            # ByteTrack は新しい ID を始める確信度が track_thresh + 0.1 以上なので、0.9 以上は ID が 1 つもできない(候補から外す)
            keys = sorted({thresh(k) for det, _, _ in DETECTORS for k in load(prefix, det, f"{ds}_val")
                           if prefix != "bytetrack" or thresh(k) < 0.9})
            out += [f"{tracker}・{ds}_val({th_name}):\n",
                    "| 追跡の手法 | セグメンテーションのモデル | 学習のデータ | " + " | ".join(f"{x:g}" for x in keys) + " |",
                    "|---|---|---|" + "---|" * len(keys)]
            for det, model, data in DETECTORS:
                v, c = load(prefix, det, f"{ds}_val"), thresh(chosen(prefix, det, ds))
                by = {thresh(k): mean_hota(r["metrics"]) for k, r in v.items()}
                out.append(f"| {tracker} | {model} | {data} | " + " | ".join(
                    (("**%.1f**" if x == c else "%.1f") % by[x]) if x in by else "測っていない" for x in keys) + " |")
            out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
