"""学習の経過の図(学習曲線)。

train_log.jsonl(1 行に 1 つの記録)を読み、損失・学習率・途中の評価を回数やエポックに沿って描き、1 枚の PNG に保存する。
記録の形(segmentation/instance_segmentation/lib_common/train.py が書く):
    {"step", "epoch", "loss", <損失の内訳>..., "lr", "img_per_s", "time", "elapsed_s"}   LOG_EVERY 回ごと
    {"eval": {"epoch", "step", "segm": {"AP", "AP50", ...}, "bbox": {"AP", ...}, "time", "eval_s"}}   途中の評価ごと
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面のない環境で保存だけする
import matplotlib.pyplot as plt  # noqa: E402

LOSS_KEYS_SKIP = {"step", "epoch", "lr", "img_per_s", "recent_img_per_s", "time", "elapsed_s"}


def series_style(i: int, n: int) -> dict:
    """i 番目の系列の色・線の種類・印(系列が多くても見分けられるように)。

    10 本までは matplotlib の既定の色。それより多いときは tab20 の 20 色に、線の種類(実線・破線・点線)と印を組み合わせる。

    Args:
        i: 系列の番号。
        n: 系列の数。

    Returns:
        ax.plot・ax.bar に渡す引数の dict(color・linestyle・marker)。
    """
    if n <= 10:
        return {"color": f"C{i}", "linestyle": "-", "marker": "o"}
    return {"color": plt.get_cmap("tab20")(i % 20), "linestyle": ["-", "--", ":"][(i // 20) % 3],
            "marker": "osD^v<>p*h"[i % 10]}


def plot_eval_curves(series: dict[str, list[tuple[float, float]]], out_path: Path, title: str,
                     xlabel: str = "epoch", ylabel: str = "mask AP (%)", legend_outside: bool = False) -> None:
    """学習中の評価の推移を、学習ごとに 1 本の折れ線で重ねて描く(方式・条件どうしの学習の進み方の比較)。

    Args:
        series: {学習の名前: [(エポック, 値), ...]}。値は % のまま渡す。
        out_path: 保存先の PNG のパス。
        title: 図の題。
        xlabel, ylabel: 軸の名前。
        legend_outside: 凡例を図の右の外に出す(系列が多いとき)。
    """
    fig, ax = plt.subplots(figsize=(9, 4.5) if legend_outside else (6, 3.6))
    for i, (name, pts) in enumerate(series.items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, **({"markersize": 4} if legend_outside else {}),
                **series_style(i, len(series)))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if legend_outside:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    else:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80, bbox_inches="tight" if legend_outside else None)  # 外の凡例も含めて保存する
    plt.close(fig)


def plot_per_class_ap(per_class: dict[str, dict[str, float]], train_images: dict[str, int], out_path: Path,
                      title: str, legend_outside: bool = False) -> None:
    """クラスごとの mask AP(評価のセットごとの棒)と、学習データでそのクラスを含む画像の数(折れ線)を 1 枚に描く。

    クラスは学習の画像の数の多い順に並べる(学習の量と精度の関係を見るため)。

    Args:
        per_class: {評価のセット名: {クラス名: AP(0〜1)}}。
        train_images: {クラス名: 学習データでそのクラスを含む画像の数}。
        out_path: 保存先の PNG のパス。
        title: 図の題。
        legend_outside: 凡例を図の右の外に出す(棒の系列が多いとき)。
    """
    import numpy as np
    names = sorted(train_images, key=lambda n: -train_images[n])
    fig, ax = plt.subplots(figsize=(16, 4.5) if legend_outside else (12, 3.6))
    width = 0.8 / len(per_class)
    for i, (s, ap) in enumerate(per_class.items()):
        ax.bar(np.arange(len(names)) + (i - (len(per_class) - 1) / 2) * width, [100 * ap.get(n, float("nan")) for n in names],
               width, label=s if legend_outside else f"mask AP: {s}", color=series_style(i, len(per_class))["color"])
    ax.set(ylabel="mask AP (%)", title=title)
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(names)), [train_images[n] for n in names], color="gray", marker=".", label="train images")
    ax2.set(ylabel="train images (log)", yscale="log")
    if legend_outside:  # 棒と折れ線の凡例を 1 つにまとめて、右の外(右の軸の目盛りより外)に出す
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", bbox_to_anchor=(1.06, 1.0), borderaxespad=0)
    else:
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80, bbox_inches="tight" if legend_outside else None)  # 外の凡例も含めて保存する
    plt.close(fig)


def plot_training_curves(log_path: Path, out_path: Path, title: str, evals: list[dict] | None = None) -> None:
    """学習曲線(損失の合計・内訳、学習率、途中の評価の AP)を 1 枚に描いて保存する。

    Args:
        log_path: train_log.jsonl のパス。
        out_path: 保存先の PNG のパス。
        title: 図の題(学習の名前など)。
        evals: 途中の評価の記録(None なら train_log.jsonl の "eval" の記録を使う。記録が入っていない古い学習のとき渡す)。
    """
    steps, found = [], []
    for line in Path(log_path).read_text().splitlines():
        r = json.loads(line)
        if "eval" in r:
            found.append(r["eval"])
        elif "step" in r and "loss" in r and "epoch_end" not in r:  # エポックの終わりの平均の記録は除く
            steps.append(r)
    evals = found if evals is None else evals
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    x = [r["step"] for r in steps]
    axes[0].plot(x, [r["loss"] for r in steps])
    axes[0].set(title="total loss", xlabel="step", yscale="log")
    for k in [k for k in steps[0] if k not in LOSS_KEYS_SKIP | {"loss"}] if steps else []:
        axes[1].plot(x, [r[k] for r in steps], label=k)
    axes[1].set(title="loss parts (last layer)", xlabel="step")
    axes[1].legend(fontsize=8)
    axes[2].plot(x, [r["lr"] for r in steps])
    axes[2].set(title="learning rate (decoder)", xlabel="step")
    if evals:
        ex = [e["epoch"] for e in evals]
        for name, kind, key in [("mask AP", "segm", "AP"), ("mask AP50", "segm", "AP50"), ("mask AP75", "segm", "AP75"),
                                ("box AP", "bbox", "AP")]:
            ys = [e.get(kind, {}).get(key) for e in evals]
            if any(y is not None for y in ys):  # 記録のない指標は描かない
                axes[3].plot(ex, [100 * y if y is not None else float("nan") for y in ys], marker="o", label=name)
        axes[3].legend(fontsize=8)
    axes[3].set(title=f"quick eval ({evals[0]['set'] if evals else '-'})", xlabel="epoch", ylabel="%")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
