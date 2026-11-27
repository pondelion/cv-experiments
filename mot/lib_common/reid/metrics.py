"""ReID の評価の指標(Rank-k・mAP・mINP)を、追跡の中の ReID の問い・探す先の決まり(mot/lib_common/reid/data.py)で計算する。

定義(ReID の論文の標準。mINP は Ye ら、ReID のサーベイ(TPAMI 2021)):
    - 問いごとに、探す先(同じ動画で、問いのフレームから gap フレーム以上離れた切り出し)を特徴の cos の類似度の高い順に並べる
    - Rank-k: 上位 k 個に正解(同じ物体)があれば 1。問いで平均する
    - AP: 正解の位置ごとの適合率(その位置までの正解の数 ÷ その位置)の平均。mAP は問いで平均する
    - INP: 正解の数 ÷ 最後の正解の順位。mINP は問いで平均する
    - 探す先に正解がない問いは数えない
形の記号: Nq = 問いの数、Ng = 探す先の数、C = 特徴の次元。
"""
import numpy as np


def retrieval_metrics(feats: np.ndarray, seqs: list[str], frames: np.ndarray, ids: np.ndarray, gap: int,
                      ranks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """全部の切り出しを、問いと探す先の両方に使って指標を計算する(問い 1 つにつき、gap の決まりで探す先を選ぶ)。

    Args:
        feats: 切り出しごとの特徴 (N, C)(正規化していなくてよい。cos の類似度で比べる)。
        seqs: 切り出しごとの動画の名前(長さ N)。
        frames: 切り出しごとのフレームの番号 (N,)。
        ids: 切り出しごとの物体の ID (N,)(動画の中で一意)。
        gap: 問いと探す先のフレームの番号の差の下限。
        ranks: Rank-k の k。

    Returns:
        {"rank1"・"rank5"・…, "mAP", "mINP"(どれも %), "n_queries"(数えた問いの数),
         "pos_sim_p10_50_90"・"neg_sim_p10_50_90"(同じ物体の組・違う物体の組の cos の類似度の 10・50・90% 点),
         "pos_frac_above_0.5"・"neg_frac_above_0.5"(類似度が 0.5 を超える組の割合。BoT-SORT の公式の閾値(cos の距離 ÷ 2 が
         0.25 以下 = 類似度 0.5 以上)で見た目を使う組に当たる)}。
    """
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)
    seqs_a = np.asarray(seqs)
    hits = {k: [] for k in ranks}
    aps, inps = [], []
    for s in np.unique(seqs_a):
        idx = np.nonzero(seqs_a == s)[0]
        sim = f[idx] @ f[idx].T  # (n, n)
        fr, tid = frames[idx], ids[idx]
        for i in range(len(idx)):
            cand = np.abs(fr - fr[i]) >= gap
            if not cand.any():
                continue
            match = tid[cand] == tid[i]  # (Ng,)
            if not match.any():
                continue
            order = np.argsort(-sim[i, cand], kind="stable")
            m = match[order]
            pos = np.nonzero(m)[0]  # 正解の位置(0 始まり)
            for k in ranks:
                hits[k].append(float(pos[0] < k))
            aps.append(float(np.mean((np.arange(len(pos)) + 1) / (pos + 1))))
            inps.append(len(pos) / (pos[-1] + 1))
    # 同じ物体の組・違う物体の組の cos の類似度の分布(問いと探す先の決まりの組。追跡で見た目を使う閾値を決める材料)
    pos_s, neg_s = [], []
    for s in np.unique(seqs_a):
        idx = np.nonzero(seqs_a == s)[0]
        sim = f[idx] @ f[idx].T
        far = np.abs(frames[idx][:, None] - frames[idx][None, :]) >= gap
        same = ids[idx][:, None] == ids[idx][None, :]
        pos_s.append(sim[far & same])
        neg_s.append(sim[far & ~same])
    pos_s, neg_s = np.concatenate(pos_s), np.concatenate(neg_s)
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [10, 50, 90])] if len(x) else None
    sims = {"pos_sim_p10_50_90": q(pos_s), "neg_sim_p10_50_90": q(neg_s),
            "neg_frac_above_0.5": round(float((neg_s > 0.5).mean()), 4) if len(neg_s) else None,
            "pos_frac_above_0.5": round(float((pos_s > 0.5).mean()), 4) if len(pos_s) else None}
    out = {f"rank{k}": 100 * float(np.mean(v)) if v else float("nan") for k, v in hits.items()}
    out |= sims
    out |= {"mAP": 100 * float(np.mean(aps)) if aps else float("nan"),
            "mINP": 100 * float(np.mean(inps)) if inps else float("nan"), "n_queries": len(aps)}
    return out
