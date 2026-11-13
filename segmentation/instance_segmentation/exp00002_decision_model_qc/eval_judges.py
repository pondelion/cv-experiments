"""判定モデルに、検品の問題(build_checks.py で作ったもの)を答えさせ、精度・較正・速さを測る。

質問:
    (1) 物体のクラス: 「画像の主な物体はどれか」。選択肢は 20 クラス + none(どれでもない・はっきりしない)
    (2) 背景の物体: 「次のどれかの物体が 1 つでも写っているか(灰色の四角は空白で、物体ではない)」。はい / いいえ

測る値:
    (1) 種類ごとの正解率。検品としての使い方(頼んだクラスの確率で、正しい組と違う組を分けられるか)の AUC と、閾値ごとの残す割合:
        正しい組 = COCO の切り出し・生成した物体と、その正解のクラス
        違う組   = 同じ画像と、正解以外のクラスを 1 つ seed 0 で選んだもの + 20 クラス以外の物体と、20 クラスのどれか
    (2) 種類ごとの「はい」と答えた割合、物体あり と なし・隠した の AUC。
        見逃さない割合(物体のある画像を「ある」と答える割合)を 90〜100% にそろえたときの閾値と、そのときに物体のない画像・隠した画像を
        捨てる割合(背景のフィルターは見逃さないことを優先するので、こちらで閾値を決める)
    どちらも、一番確率の高い答えの確率と正解率のずれ(ECE、10 個の区間)と、1 枚あたりの時間

出力: runs/{判定モデル}_answers.json(問題ごとの確率)、results/{判定モデル}.json(まとめ)

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00002_decision_model_qc/eval_judges.py --judge intern
        (--summary-only: 判定モデルを動かさず、保存した runs/{判定モデル}_answers.json からまとめだけを計算し直す)
"""
import argparse
import json
import random
import time

import common
import numpy as np
from PIL import Image

from lib_common.classes import VOC20
from lib_common.synthetic_data.judges import InternDecisionJudge, OllamaVlmJudge, background_question, class_question

CHECKS = common.RUNS_DIR / "checks"
NAMES = [n for n, _ in VOC20]
CLASS_INSTRUCTION, CLASS_OPTIONS = class_question(NAMES)  # 合成データのフィルターと同じ文面(lib_common.synthetic_data.judges)
BG_INSTRUCTION, YES_NO = background_question(NAMES)
JUDGES = {"intern": InternDecisionJudge, "ollama": OllamaVlmJudge}


def auc(pos: list[float], neg: list[float]) -> float:
    """正例の値が負例の値より大きい確率(同じ値は 0.5)。"""
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=list(JUDGES), default="intern")
    parser.add_argument("--summary-only", action="store_true", help="判定モデルを動かさず、保存した答えからまとめだけを計算し直す")
    args = parser.parse_args()
    answers_path = common.RUNS_DIR / f"{args.judge}_answers.json"
    result_path = common.RESULTS_DIR / f"{args.judge}.json"
    if args.summary_only:
        answers = json.loads(answers_path.read_text())
        times = json.loads(result_path.read_text())["ms_per_image"]  # 時間は、判定モデルを動かしたときの値を引き継ぐ
    else:
        labels = json.loads((CHECKS / "labels.json").read_text())
        judge = JUDGES[args.judge]()
        answers, times = [], {}
        for task, instruction, options in [("t1", CLASS_INSTRUCTION, CLASS_OPTIONS), ("t2", BG_INSTRUCTION, YES_NO)]:
            items = [l for l in labels if l["task"] == task]
            images = [Image.open(CHECKS / l["file"]).convert("RGB") for l in items]
            judge.ask(images[:4], instruction, options)  # 最初の呼び出し(読み込み・コンパイル)は時間に入れない
            t = time.perf_counter()
            probs = judge.ask(images, instruction, options)
            times[task] = (time.perf_counter() - t) / len(items) * 1000
            answers += [l | {"probs": p} for l, p in zip(items, probs)]
            print(task, f"{len(items)} 枚 {times[task]:.1f}ms/枚", flush=True)
        answers_path.write_text(json.dumps(answers, ensure_ascii=False))

    res = {"judge": args.judge, "ms_per_image": times}
    t1 = [a for a in answers if a["task"] == "t1"]
    top = lambda p: max(p, key=p.get)  # noqa: E731
    res["t1_accuracy"] = {k: float(np.mean([top(a["probs"]) == a["label"] for a in t1 if a["kind"] == k]))
                          for k in ["coco", "coco_none", "gen"]}
    res["t1_ece"] = ece([max(a["probs"].values()) for a in t1], [top(a["probs"]) == a["label"] for a in t1])
    rng = random.Random(0)
    right = [a["probs"][a["label"]] for a in t1 if a["kind"] in ("coco", "gen")]
    wrong_cls = [a["probs"][rng.choice([n for n in NAMES if n != a["label"]])] for a in t1 if a["kind"] in ("coco", "gen")]
    wrong_none = [a["probs"][rng.choice(NAMES)] for a in t1 if a["kind"] == "coco_none"]  # 20 クラスのどれでもない物体
    wrong = wrong_cls + wrong_none
    res["t1_qc_auc"] = auc(right, wrong)
    res["t1_qc_keep"] = {f"{th:.2f}": {"keep_right": float(np.mean(np.asarray(right) >= th)),
                                       "keep_wrong": float(np.mean(np.asarray(wrong) >= th)),
                                       "keep_wrong_other_class": float(np.mean(np.asarray(wrong_cls) >= th)),
                                       "keep_wrong_not_in_classes": float(np.mean(np.asarray(wrong_none) >= th))}
                         for th in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    t2 = [a for a in answers if a["task"] == "t2"]
    yes = {k: [a["probs"]["yes"] for a in t2 if a["kind"] == k] for k in ["pos", "neg", "boxed"]}
    res["t2_yes_rate@0.5"] = {k: float(np.mean(np.asarray(v) > 0.5)) for k, v in yes.items()}
    res["t2_auc"] = {"pos_vs_neg": auc(yes["pos"], yes["neg"]), "pos_vs_boxed": auc(yes["pos"], yes["boxed"])}
    # 見逃さない割合をそろえたときの閾値と、捨てる割合(閾値は、物体のある画像のうち rec の割合が閾値より上になる値)
    res["t2_at_recall"] = {}
    for rec in [0.90, 0.95, 0.98, 0.99, 1.00]:
        th = float(np.quantile(yes["pos"], 1 - rec)) - 1e-9
        res["t2_at_recall"][f"{rec:.2f}"] = {"threshold": th,
                                             "discard_neg": float(np.mean(np.asarray(yes["neg"]) > th)),
                                             "discard_boxed": float(np.mean(np.asarray(yes["boxed"]) > th))}
    res["t2_ece"] = ece([max(a["probs"].values()) for a in t2], [top(a["probs"]) == a["label"] for a in t2])
    common.RESULTS_DIR.mkdir(exist_ok=True)
    result_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
