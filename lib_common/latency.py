"""推論の遅延を測る。

測り方は instance_segmentation/README.md の「速度の指標」のとおり: batch 1、最初の warmup 回は捨て、
GPU の処理の終わりを待ってから時刻を取り、中央値と p90(遅い方から 10% の所)を出す。
"""
import time
from typing import Callable

import numpy as np
import torch


class Timer:
    """GPU の処理の終わりを待ってから、経過時間(ミリ秒)を測る。

    使い方:
        with Timer() as t:
            model(x)
        print(t.ms)
    """

    def __enter__(self):
        """GPU の処理の終わりを待ってから、測り始める。

        Returns:
            self。
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        """GPU の処理の終わりを待ってから、経過時間を self.ms に入れる。

        Args:
            *exc: with の中で起きた例外の情報(使わない)。
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self._start) * 1000


def summarize(times_ms: list[float], warmup: int = 0) -> dict:
    """測った時間の並びから、最初の warmup 個を捨てて中央値・p90・平均を出す。

    Args:
        times_ms: 測った時間(ミリ秒)の並び。
        warmup: 先頭から捨てる個数(最初の数回は準備で遅いため)。

    Returns:
        {"median_ms", "p90_ms", "mean_ms", "n"(使った個数)}。
    """
    t = np.asarray(times_ms[warmup:])
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)), "mean_ms": float(t.mean()),
            "n": int(len(t))}


def measure(fn: Callable[[], object], warmup: int = 10, repeat: int = 50) -> dict:
    """引数なしの関数 fn を warmup + repeat 回呼び、後ろの repeat 回の時間をまとめる。

    Args:
        fn: 測る処理(引数なし)。
        warmup: 先に呼んで捨てる回数。
        repeat: 測る回数。

    Returns:
        summarize と同じ辞書。
    """
    times = []
    for _ in range(warmup + repeat):
        with Timer() as t:
            fn()
        times.append(t.ms)
    return summarize(times, warmup)
