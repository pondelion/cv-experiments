"""MuSGD(Muon + SGD。YOLO26 の論文 arXiv 2606.03748 の最適化の手法)の自作の実装。exp00102 の Y1。

論文に混ぜ方の値がないので、ultralytics 8.4.211 の公式の実装(`ultralytics/optim/muon.py` の MuSGD・muon_update・
zeropower_via_newtonschulz5 と、`engine/trainer.py` の build_optimizer の MuSGD の部分)から仕組みと値を読み取り、写さずに
自分で書いた(公式の速さのための工夫(同じ形の重みをまとめて直交化する処理)は入れていない。計算は同じ)。

更新(1 回の step):
    Muon を使う組(2 次元・4 次元の重み)は、次の 2 つを両方足す。
        Muon: 勾配の指数移動平均 m ← β·m + (1 − β)·g(nesterov なら u = β·m + (1 − β)·g、でなければ u = m)を 2 次元
              (出力 × 残り)に並べ、Newton-Schulz の 5 回の反復でおおよそ直交化し、√max(1, 勾配の最後の 2 つの次元の比)を
              掛ける。重み ← 重み − 学習率 × muon_weight(0.2)× その値
        SGD:  weight decay を足した勾配 g' = g + λ·重み で、別の momentum の値 b ← μ·b + g'(nesterov なら g' + μ·b)。
              重み ← 重み − 学習率 × sgd_weight(1.0)× その値
    ほかの組は SGD だけ(weight decay は組ごと)。

形の記号: O = 重みの出力の次元、I = 残りの次元(4 次元の重みは 入力 × kh × kw)。
"""
import torch

NS_COEFS = (3.4445, -4.7750, 2.0315)  # Newton-Schulz の 5 次の反復の係数(公式と同じ)


@torch.no_grad()
def orthogonalize(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """行列をおおよそ直交化する(Newton-Schulz の反復。特異値をおおよそ 1 にそろえる)。bf16 で計算する(公式と同じ)。

    Args:
        g: (O, I)。
        steps: 反復の回数。
        eps: 0 で割らないための小さい値。

    Returns:
        (O, I) の bf16。
    """
    a, b, c = NS_COEFS
    x = g.bfloat16()
    x = x / (x.norm() + eps)  # 最大の特異値を 1 以下にする
    tall = x.shape[0] > x.shape[1]
    if tall:  # 小さい方の次元で X·Xᵀ を作る
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    return x.T if tall else x


class MuSGD(torch.optim.Optimizer):
    """Muon と SGD を混ぜた最適化。

    Args:
        params: 重みの組の並び。組ごとに "use_muon"(Muon も使うか)・"lr"・"weight_decay" を持てる。
        lr: 学習率。
        momentum: momentum の係数(Muon の β と SGD の μ に同じ値を使う。公式と同じ)。
        weight_decay: weight decay(SGD の側にだけ掛ける)。
        nesterov: nesterov の momentum にするか。
        muon_weight: Muon の更新の倍率(公式は 0.2)。
        sgd_weight: Muon を使う組での SGD の更新の倍率(公式は 1.0)。
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0, nesterov: bool = True,
                 muon_weight: float = 0.2, sgd_weight: float = 1.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                                      use_muon=False))
        self.muon_weight, self.sgd_weight = muon_weight, sgd_weight

    @torch.no_grad()
    def step(self, closure=None):
        """1 回更新する。

        Args:
            closure: 損失を計算し直す関数(使わないときは None)。

        Returns:
            closure の損失(closure がなければ None)。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, nesterov, wd = group["lr"], group["momentum"], group["nesterov"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                sgd_lr = lr
                if group["use_muon"]:
                    m = state.setdefault("muon_buffer", torch.zeros_like(p))
                    m.mul_(mu).add_(g, alpha=1 - mu)  # m ← β·m + (1 − β)·g
                    u = m * mu + g * (1 - mu) if nesterov else m
                    o = orthogonalize(u.reshape(u.shape[0], -1)).to(p.dtype).reshape(p.shape)  # (O, I) で直交化して戻す
                    scale = max(1.0, g.shape[-2] / g.shape[-1]) ** 0.5  # 公式と同じく、最後の 2 つの次元の比
                    p.add_(o, alpha=-lr * self.muon_weight * scale)
                    sgd_lr = lr * self.sgd_weight
                buf = state.setdefault("sgd_buffer", torch.zeros_like(p))
                g = g + wd * p if wd else g
                buf.mul_(mu).add_(g)  # b ← μ·b + g'
                p.add_(g + mu * buf if nesterov else buf, alpha=-sgd_lr)
        return loss
