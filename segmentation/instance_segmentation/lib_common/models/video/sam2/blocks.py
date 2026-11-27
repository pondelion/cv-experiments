"""SAM 2 の部品(位置の埋め込み・注意・two-way transformer・MLP など)。exp00502。

参考: 公式のコード(facebookresearch/sam2 の `sam2/modeling/position_encoding.py`・`sam/transformer.py`・`sam2_utils.py`)。
コードは写さず、計算を読み取って書いた(式は公式と同じ)。

形の記号: B = まとめの数、N = トークンの数、C = 次元、H・W = 特徴の縦・横、h = 注意の頭の数。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """num_layers 層の MLP(層の間は activation。最後に sigmoid をかけることもできる)。公式の sam2_utils.MLP と同じ。

    Args:
        d_in: 入力の次元。
        d_hidden: 中間の次元。
        d_out: 出力の次元。
        num_layers: 層の数。
        activation: 層の間の活性化の nn.Module のクラス。
        sigmoid_output: 最後に sigmoid をかけるか。
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, num_layers: int, activation=nn.ReLU, sigmoid_output: bool = False):
        super().__init__()
        dims = [d_in] + [d_hidden] * (num_layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = activation()
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP をかける。

        Args:
            x: (..., d_in)。

        Returns:
            (..., d_out)。
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x.sigmoid() if self.sigmoid_output else x


class LayerNorm2d(nn.Module):
    """チャンネルの軸の LayerNorm((B, C, H, W) のまま)。

    Args:
        c: チャンネル。
        eps: 分散に足す小さい値(公式は 1e-6)。
    """

    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """チャンネルの軸で正規化する。

        Args:
            x: (B, C, H, W)。

        Returns:
            (B, C, H, W)。
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


class PositionEmbeddingSine(nn.Module):
    """画像の格子の sine の位置の埋め込み(DETR と同じ)。

    Args:
        num_pos_feats: 出力のチャンネルの数(縦・横で半分ずつ)。
        temperature: 周波数の底。
        normalize: 位置を 0〜2π にそろえるか。
    """

    def __init__(self, num_pos_feats: int, temperature: float = 10000, normalize: bool = True):
        super().__init__()
        self.half, self.temperature, self.normalize, self.scale = num_pos_feats // 2, temperature, normalize, 2 * math.pi

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x と同じ縦・横の格子の位置の埋め込みを作る(値は x の形だけで決まる)。

        Args:
            x: (B, *, H, W)。

        Returns:
            (B, num_pos_feats, H, W)。
        """
        b, h, w, dev = x.shape[0], x.shape[-2], x.shape[-1], x.device
        y = torch.arange(1, h + 1, dtype=torch.float32, device=dev)[:, None].expand(h, w)
        xx = torch.arange(1, w + 1, dtype=torch.float32, device=dev)[None, :].expand(h, w)
        if self.normalize:
            y = y / (h + 1e-6) * self.scale
            xx = xx / (w + 1e-6) * self.scale
        dim_t = self.temperature ** (2 * (torch.arange(self.half, dtype=torch.float32, device=dev) // 2) / self.half)
        px, py = xx[..., None] / dim_t, y[..., None] / dim_t  # (H, W, half)
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=-1).flatten(-2)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((py, px), dim=-1).permute(2, 0, 1)  # (C, H, W)
        return pos[None].expand(b, -1, -1, -1)


class PositionEmbeddingRandom(nn.Module):
    """ランダムな周波数の位置の埋め込み(SAM の prompt encoder。2 × num_pos_feats チャンネル)。

    Args:
        num_pos_feats: 周波数の数。
        scale: 周波数の正規分布の標準偏差。
    """

    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0):
        super().__init__()
        self.register_buffer("gaussian", scale * torch.randn(2, num_pos_feats))

    def _encode(self, coords: torch.Tensor) -> torch.Tensor:
        """座標を sin・cos の埋め込みにする。

        Args:
            coords: (..., 2) の 0〜1 の (x, y)。

        Returns:
            (..., 2 × num_pos_feats)。
        """
        c = 2 * math.pi * ((2 * coords - 1) @ self.gaussian)
        return torch.cat([c.sin(), c.cos()], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        """格子の埋め込み(各画素の中心の座標)。

        Args:
            size: 格子の (h, w)。

        Returns:
            (C, h, w)。
        """
        h, w = size
        dev = self.gaussian.device
        y = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) / h
        x = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) / w
        grid = torch.stack(torch.broadcast_tensors(x[None, :], y[:, None]), dim=-1)  # (h, w, 2) の (x, y)
        return self._encode(grid).permute(2, 0, 1)

    def forward_with_coords(self, coords: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """画素の座標の点の埋め込み。

        Args:
            coords: (B, N, 2) の画素の (x, y)。
            image_size: 画像の (縦, 横)(座標を 0〜1 にするのに使う)。

        Returns:
            (B, N, C)。
        """
        c = coords.clone().float()
        c[..., 0] = c[..., 0] / image_size[1]
        c[..., 1] = c[..., 1] / image_size[0]
        return self._encode(c)


def get_1d_sine_pe(pos: torch.Tensor, dim: int, temperature: float = 10000) -> torch.Tensor:
    """1 次元の sine の位置の埋め込み(時刻の埋め込みに使う)。

    Args:
        pos: 位置 (N,)。
        dim: 埋め込みの次元。
        temperature: 周波数の底。

    Returns:
        (N, dim)。
    """
    half = dim // 2
    dim_t = temperature ** (2 * (torch.arange(half, dtype=torch.float32, device=pos.device) // 2) / half)
    p = pos[:, None] / dim_t
    return torch.cat([p.sin(), p.cos()], dim=-1)


def axial_rope(dim: int, w: int, h: int, theta: float = 10000.0) -> torch.Tensor:
    """2 次元の軸ごとの RoPE の回転(複素数)。x と y にそれぞれ dim / 4 個の周波数。

    Args:
        dim: 注意の頭の次元。
        w: 格子の横。
        h: 格子の縦。
        theta: 周波数の底。

    Returns:
        (w·h, dim / 2) の複素数。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    t = torch.arange(w * h, dtype=torch.float32)
    tx, ty = t % w, torch.div(t, w, rounding_mode="floor")
    fx, fy = torch.outer(tx, freqs), torch.outer(ty, freqs)
    return torch.cat([torch.polar(torch.ones_like(fx), fx), torch.polar(torch.ones_like(fy), fy)], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, repeat: int = 1) -> torch.Tensor:
    """RoPE をかける。

    Args:
        x: (B, h, N, d)。
        freqs: 回転の (cos, sin)、どちらも (N / repeat, d / 2)(axial_rope の複素数の実部と虚部)。
        repeat: 同じ格子が並んだ回数(記憶のフレームのキーは、同じ格子が repeat 回並ぶ)。

    Returns:
        (B, h, N, d)。
    """
    # 複素数の掛け算 (x0 + i x1)(cos + i sin) を、同じ式の実数の計算で行う(torch.compile が複素数で切れないように):
    # (x0 cos − x1 sin, x0 sin + x1 cos)。freqs は (cos, sin) の実数の組
    cos, sin = freqs
    if repeat > 1:
        cos, sin = cos.repeat(repeat, 1), sin.repeat(repeat, 1)  # (N, d / 2)
    xr = x.float().reshape(*x.shape[:-1], -1, 2)  # (B, h, N, d / 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(3).type_as(x)


class Attention(nn.Module):
    """q・k・v を射影してから downsample_rate で次元を減らす注意(SAM の Attention)。

    Args:
        dim: 入力・出力の次元。
        num_heads: 頭の数。
        downsample_rate: 内部の次元 = dim / downsample_rate。
        dropout: 注意の dropout。
        kv_in_dim: k・v の入力の次元(省略すると dim)。
    """

    def __init__(self, dim: int, num_heads: int, downsample_rate: int = 1, dropout: float = 0.0, kv_in_dim: int | None = None):
        super().__init__()
        kv = kv_in_dim or dim
        self.inner, self.num_heads, self.dropout = dim // downsample_rate, num_heads, dropout
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(dim, self.inner), nn.Linear(kv, self.inner), nn.Linear(kv, self.inner)
        self.out_proj = nn.Linear(self.inner, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """頭の軸に分ける。

        Args:
            x: (B, N, C)。

        Returns:
            (B, h, N, C / h)。
        """
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)  # (B, N, C) -> (B, h, N, C/h)

    def _attend(self, q, k, v) -> torch.Tensor:
        """頭ごとの注意をとり、頭をまとめて出力の射影をかける。

        Args:
            q: (B, h, N, d)。
            k: (B, h, M, d)。
            v: (B, h, M, d)。

        Returns:
            (B, N, dim)。
        """
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        b, h, n, d = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, n, h * d))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """注意をとる。

        Args:
            q: 問い (B, N, dim)。
            k: キー (B, M, kv_in_dim)。
            v: 値 (B, M, kv_in_dim)。

        Returns:
            (B, N, dim)。
        """
        return self._attend(self._heads(self.q_proj(q)), self._heads(self.k_proj(k)), self._heads(self.v_proj(v)))


class RoPEAttention(Attention):
    """2 次元の RoPE をかける注意(SAM 2 の記憶の注意)。問いは H × W の格子のトークン。

    Args:
        *args: Attention の引数。
        rope_theta: RoPE の周波数の底。
        rope_k_repeat: キーが問いと同じ格子を何回か並べたもの(記憶のフレーム)のとき True(同じ回転をくり返してかける)。
        **kw: Attention の引数。
    """

    def __init__(self, *args, rope_theta: float = 10000.0, rope_k_repeat: bool = False, **kw):
        super().__init__(*args, **kw)
        self.theta, self.rope_k_repeat = rope_theta, rope_k_repeat
        self._cache = {}

    def forward(self, q, k, v, num_k_exclude_rope: int = 0):
        """問いとキーに RoPE をかけて注意をとる。

        Args:
            q: 問い (B, H·W, dim)(正方形の格子)。
            k: キー (B, M, kv_in_dim)。
            v: 値 (B, M, kv_in_dim)。
            num_k_exclude_rope: キーの最後のこの数のトークン(物体のポインタ)には RoPE をかけない。

        Returns:
            (B, H·W, dim)。
        """
        q, k, v = self._heads(self.q_proj(q)), self._heads(self.k_proj(k)), self._heads(self.v_proj(v))
        side = int(math.sqrt(q.shape[-2]))  # 問いは正方形の格子(公式と同じ前提)
        key = (side, q.shape[-1], q.device)
        if key not in self._cache:  # GPU の上に置いて使い回す(呼ぶたびに CPU で並べて GPU へ送らない)。実数の (cos, sin) の組
            f = axial_rope(q.shape[-1], side, side, self.theta)
            self._cache[key] = (f.real.contiguous().to(q.device), f.imag.contiguous().to(q.device))
        freqs = self._cache[key]
        n_rope = k.shape[-2] - num_k_exclude_rope
        if q.shape[-2] != n_rope:
            assert self.rope_k_repeat
        q = apply_rope(q, freqs)
        if n_rope > 0:
            k = torch.cat([apply_rope(k[:, :, :n_rope], freqs, n_rope // q.shape[-2]), k[:, :, n_rope:]], dim=2)
        return self._attend(q, k, v)


class TwoWayBlock(nn.Module):
    """two-way transformer の 1 層: トークンの self-attention → トークンから画像への注意 → MLP → 画像からトークンへの注意。

    Args:
        dim: 次元。
        num_heads: 頭の数。
        mlp_dim: MLP の中間の次元。
        downsample: トークンと画像の間の注意の次元の減らし方。
        skip_first_pe: 最初の層なら True(self-attention に位置の埋め込みを足さず、残差も足さない。公式と同じ)。
    """

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, downsample: int, skip_first_pe: bool):
        super().__init__()
        self.self_attn = Attention(dim, num_heads)
        self.cross_t2i = Attention(dim, num_heads, downsample)
        self.mlp = MLP(dim, mlp_dim, dim, 2)
        self.cross_i2t = Attention(dim, num_heads, downsample)
        self.norm1, self.norm2, self.norm3, self.norm4 = (nn.LayerNorm(dim) for _ in range(4))
        self.skip_first_pe = skip_first_pe

    def forward(self, queries, keys, query_pe, key_pe):
        """1 層進める。

        Args:
            queries: トークン (B, N, dim)。
            keys: 画像のトークン (B, H·W, dim)。
            query_pe: トークンの位置の埋め込み (B, N, dim)(最初のトークンそのもの)。
            key_pe: 画像の位置の埋め込み (B, H·W, dim)。

        Returns:
            (トークン (B, N, dim), 画像のトークン (B, H·W, dim))。
        """
        if self.skip_first_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.norm1(queries)
        queries = self.norm2(queries + self.cross_t2i(queries + query_pe, keys + key_pe, keys))
        queries = self.norm3(queries + self.mlp(queries))
        keys = self.norm4(keys + self.cross_i2t(keys + key_pe, queries + query_pe, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    """SAM のマスクの decoder の transformer(depth 層 + 最後のトークンから画像への注意)。

    Args:
        depth: 層の数。
        dim: 次元。
        num_heads: 頭の数。
        mlp_dim: MLP の中間の次元。
        downsample: トークンと画像の間の注意の次元の減らし方。
    """

    def __init__(self, depth: int = 2, dim: int = 256, num_heads: int = 8, mlp_dim: int = 2048, downsample: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(TwoWayBlock(dim, num_heads, mlp_dim, downsample, i == 0) for i in range(depth))
        self.final_t2i = Attention(dim, num_heads, downsample)
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor, image_pe: torch.Tensor, tokens: torch.Tensor):
        """トークンと画像を相互に注意で混ぜる。

        Args:
            image: 画像の特徴 (B, C, H, W)。
            image_pe: 画像の位置の埋め込み (B, C, H, W)。
            tokens: トークン (B, N, C)。

        Returns:
            (トークン (B, N, C), 画像のトークン (B, H·W, C))。
        """
        keys, key_pe = image.flatten(2).transpose(1, 2), image_pe.flatten(2).transpose(1, 2)
        queries = tokens
        for layer in self.layers:
            queries, keys = layer(queries, keys, tokens, key_pe)
        queries = self.norm_final(queries + self.final_t2i(queries + tokens, keys + key_pe, keys))
        return queries, keys


def select_closest_cond_frames(t: int, cond: dict, max_num: int) -> tuple[dict, dict]:
    """条件のフレーム(マスクを与えたフレーム)から、時刻 t に近いものを max_num 個選ぶ(前の最も近い 1 つ・後の最も近い 1 つ・残りは近い順)。

    Args:
        t: 今の時刻。
        cond: {時刻: 条件のフレームの出力}。
        max_num: 選ぶ数(−1 なら全部)。

    Returns:
        (選んだもの, 選ばなかったもの)(どちらも cond と同じ形の dict)。
    """
    if max_num == -1 or len(cond) <= max_num:
        return cond, {}
    sel = {}
    before = max((i for i in cond if i < t), default=None)
    after = min((i for i in cond if i >= t), default=None)
    for i in (before, after):
        if i is not None:
            sel[i] = cond[i]
    rest = sorted((i for i in cond if i not in sel), key=lambda i: abs(i - t))[: max_num - len(sel)]
    sel.update((i, cond[i]) for i in rest)
    return sel, {i: v for i, v in cond.items() if i not in sel}
