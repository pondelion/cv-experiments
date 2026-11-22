"""RT-DETR の hybrid encoder(ECInsSeg・D-FINE 系が使う版)の自作の実装。exp00103 の A で、自作の GTR 系の encoder と差し替える。

参考: RT-DETR(Zhao ら CVPR 2024)の論文と、EdgeCrafter(arXiv 2603.18739)の公式のコードの構成
(`ecdetseg/engine/edgecrafter/hybrid_encoder.py`、設定は `configs/ecdet/ecdet.yml`・`ecdet_l.yml`)。コードは写さず、構成と設定の
値だけを読み取って書いた。推論用に畳み込みと BN を 1 つにまとめる処理(convert_to_deploy)は入れていない。

構成:
    入力の射影:   backbone の stride 8・16・32 の段を、1×1 の畳み込み + BN で D_MODEL チャンネルにする
    AIFI:         最も粗い段(stride 32)だけに、自己注意の層を 1 つ通す(位置の埋め込みは 2 次元の sin・cos。後置きの LayerNorm)
    CCFF:         上から下(stride 32 → 8)、下から上(stride 8 → 32)の 2 回、段を足し合わせて融合のブロックを通す
                  (上から下: 横からの 1×1 + BN → 2 倍に拡大して足す、下から上: 1×1 + BN → 3×3 の depthwise の stride 2 で縮めて足す)
    融合のブロック: ELAN 型(1×1 で 2 つに分け、片方に CSP のブロックを 2 回重ね、途中の出力を全部つなげて 1×1 で混ぜる)。
                  CSP のブロックの中身は RepVGG 型(3×3 + BN と 1×1 + BN の和)を n 回

形の記号: B = 画像の数、C = D_MODEL(256)、h・w = 段の縦・横の画素数、N = h × w。
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

D_MODEL = 256


def conv_bn(cin: int, cout: int, k: int = 1, stride: int = 1, groups: int = 1, act: bool = True) -> nn.Sequential:
    """畳み込み(bias なし)+ BN(+ SiLU)。

    Args:
        cin: 入力のチャンネル。
        cout: 出力のチャンネル。
        k: カーネルの大きさ。
        stride: stride。
        groups: 畳み込みの組の数(cin と同じにすると depthwise)。
        act: 最後に SiLU を掛けるか。

    Returns:
        (B, cin, h, w) -> (B, cout, h / stride, w / stride) の層。
    """
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, padding=k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout),
                         *([nn.SiLU(inplace=True)] if act else []))


class RepBlock(nn.Module):
    """RepVGG 型のブロック: 3×3 の畳み込み + BN と 1×1 の畳み込み + BN の和に SiLU を掛ける(残差はない)。

    Args:
        c: 入力・出力のチャンネル。
    """

    def __init__(self, c: int):
        super().__init__()
        self.conv3, self.conv1 = conv_bn(c, c, 3, act=False), conv_bn(c, c, 1, act=False)
        self.fused = None  # fuse_() の後は、2 つの枝と BN をまとめた 1 つの 3×3 の畳み込み

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, c, h, w) -> (B, c, h, w)。"""
        if self.fused is not None:
            return F.silu(self.fused(x))
        return F.silu(self.conv3(x) + self.conv1(x))

    @torch.no_grad()
    def fuse_(self) -> None:
        """推論用に、3×3 + BN と 1×1 + BN の 2 つの枝を 1 つの 3×3 の畳み込み(bias あり)にまとめる(RepVGG の
        re-parameterization。計算は同じ。exp00104 の E4)。BN は推論のときの統計(running_mean・running_var)を使う。"""
        def conv_bn_weights(seq: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
            """畳み込み + BN を、1 つの畳み込みの (重み, bias) にする。"""
            conv, bn = seq[0], seq[1]
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # (c,)
            return conv.weight * scale[:, None, None, None], bn.bias - bn.running_mean * scale
        w3, b3 = conv_bn_weights(self.conv3)
        w1, b1 = conv_bn_weights(self.conv1)
        conv = nn.Conv2d(w3.shape[1], w3.shape[0], 3, padding=1).to(w3.device, w3.dtype)
        conv.weight.copy_(w3 + F.pad(w1, (1, 1, 1, 1)))  # 1×1 の重みを 3×3 の真ん中に置いて足す
        conv.bias.copy_(b3 + b1)
        self.fused = conv


class DWBlock(nn.Module):
    """RepBlock の代わりの、depthwise 3×3 + BN → pointwise 1×1 + BN → SiLU(MobileNet の depthwise separable な畳み込み。
    幅はそのままで、3×3 の普通の畳み込みの計算を約 1/9 にする。exp00104 の E3)。

    Args:
        c: 入力・出力のチャンネル。
    """

    def __init__(self, c: int):
        super().__init__()
        self.dw, self.pw = conv_bn(c, c, 3, groups=c, act=False), conv_bn(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, c, h, w) -> (B, c, h, w)。"""
        return self.pw(self.dw(x))


class CSPBlock(nn.Module):
    """CSP のブロック: 1×1 で 2 倍の幅にして 2 つに分け、片方に RepBlock を n 回通して、もう片方と足す。

    Args:
        cin: 入力のチャンネル。
        cout: 出力のチャンネル(途中の幅も cout)。
        n: RepBlock の数。
        block: 中のブロック("rep": RepBlock / "dw": DWBlock)。
    """

    def __init__(self, cin: int, cout: int, n: int, block: str = "rep"):
        super().__init__()
        self.split = conv_bn(cin, 2 * cout)
        self.blocks = nn.Sequential(*[(RepBlock if block == "rep" else DWBlock)(cout) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, cin, h, w) -> (B, cout, h, w)。"""
        a, b = self.split(x).chunk(2, dim=1)  # (B, 2·cout, h, w) -> 2 × (B, cout, h, w)
        return a + self.blocks(b)


class FuseBlock(nn.Module):
    """ELAN 型の融合のブロック: 1×1 で wide チャンネルにして半分に分け、後ろ半分に CSPBlock を 2 回続けて通し、
    分けた 2 つと 2 回の出力を全部つなげて 1×1 で cout にする。

    Args:
        cin: 入力のチャンネル。
        cout: 出力のチャンネル。
        wide: 最初の 1×1 の出力のチャンネル(公式の c3 = 2 × D_MODEL)。
        mid: CSPBlock の幅(公式の c4 = expansion × D_MODEL / 2)。
        n: CSPBlock の中の RepBlock の数。
        block: CSPBlock の中のブロック("rep" / "dw")。
    """

    def __init__(self, cin: int, cout: int, wide: int, mid: int, n: int, block: str = "rep"):
        super().__init__()
        self.inp = conv_bn(cin, wide)
        self.csp1, self.csp2 = CSPBlock(wide // 2, mid, n, block), CSPBlock(mid, mid, n, block)
        self.out = conv_bn(wide + 2 * mid, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, cin, h, w) -> (B, cout, h, w)。"""
        a, b = self.inp(x).chunk(2, dim=1)  # (B, wide, h, w) -> 2 × (B, wide/2, h, w)
        c = self.csp1(b)  # (B, mid, h, w)
        d = self.csp2(c)  # (B, mid, h, w)
        return self.out(torch.cat([a, b, c, d], dim=1))  # (B, wide + 2·mid, h, w) -> (B, cout, h, w)


def sincos_pos_2d(h: int, w: int, dim: int, device, temperature: float = 10000.0) -> torch.Tensor:
    """2 次元の位置の埋め込み(sin・cos)。公式の build_2d_sincos_position_embedding と同じ並べ方にする。

    公式は位置の格子を (横, 縦) の順に作って横が外側になるように 1 列に並べ、チャンネルを「横の sin・cos、縦の sin・cos」に
    する。特徴のトークンは縦が外側の順に並ぶので、正方形の入力では、トークン (縦 y, 横 x) に「横の位置 = y、縦の位置 = x」の
    埋め込みが足される(前半のチャンネルが縦の位置を表す)。公式と同じ計算にするため、この並べ方をそのまま使う。

    Args:
        h: 縦の画素数。
        w: 横の画素数。
        dim: 埋め込みの次元(4 の倍数)。
        device: 作る場所。
        temperature: 周期の基準。

    Returns:
        (N, dim)。N = h × w。
    """
    q = dim // 4
    omega = 1.0 / temperature ** (torch.arange(q, device=device, dtype=torch.float32) / q)  # (q,) 周波数
    gw, gh = torch.meshgrid(torch.arange(w, device=device, dtype=torch.float32),
                            torch.arange(h, device=device, dtype=torch.float32), indexing="ij")  # 2 × (w, h)
    pw = rearrange(gw, "w h -> (w h) 1") * omega  # (N, q) 横が外側の順
    ph = rearrange(gh, "w h -> (w h) 1") * omega  # (N, q)
    return torch.cat([pw.sin(), pw.cos(), ph.sin(), ph.cos()], dim=1)  # (N, dim)


class AIFILayer(nn.Module):
    """AIFI: 最も粗い段の特徴に掛ける自己注意の層(後置きの LayerNorm、FFN の活性は SiLU、dropout なし)。

    Args:
        heads: 注意の頭の数。
        ffn: FFN の中間の次元。
        dim: 特徴の次元(既定は D_MODEL)。
    """

    def __init__(self, heads: int = 8, ffn: int = 1024, dim: int = D_MODEL):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.SiLU(inplace=True), nn.Linear(ffn, dim))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) -> (B, C, h, w)。位置の埋め込みは注意の問いと鍵にだけ足す。"""
        b, c, h, w = x.shape
        tokens = rearrange(x, "b c h w -> b (h w) c")  # (B, N, C)
        pos = sincos_pos_2d(h, w, c, x.device).to(x.dtype)  # (N, C)
        qk = tokens + pos
        tokens = self.norm1(tokens + self.attn(qk, qk, tokens, need_weights=False)[0])
        tokens = self.norm2(tokens + self.ffn(tokens))
        return rearrange(tokens, "b (h w) c -> b c h w", h=h, w=w)


class HybridEncoder(nn.Module):
    """RT-DETR の hybrid encoder(AIFI + CCFF)。出力は自作の GTR 系の `Encoder` と同じ形(stride 8・16・32 の 3 段、C チャンネル)。

    「hybrid」は、注意(Transformer)の処理と畳み込み(CNN)の処理を、役割で分けて組み合わせていること:
    - AIFI(Attention-based Intra-scale Feature Interaction): 最も粗い段(stride 32)だけに自己注意の層を 1 つ通し、同じ段の中の
      離れた場所どうしの関係を混ぜる。画素の数が少ない段だけに注意を使うので、計算が軽い(`self.aifi`)
    - CCFF(CNN-based Cross-scale Feature Fusion): 3 つの段どうしを畳み込みで混ぜる。上から下(FPN の経路。粗い段を拡大して
      細かい段に足す: `self.lateral`・`self.fpn`)と、下から上(PAN の経路。細かい段を縮めて粗い段に足す: `self.down`・
      `self.pan`)の 2 回

    Args:
        cins: backbone の stride 8・16・32 の段のチャンネル。
        expansion: CSPBlock の幅の倍率(公式の L の版は 0.75)。
        n: CSPBlock の中の RepBlock の数(公式は round(3 × depth_mult)。L の版は 3)。
        heads: AIFI の注意の頭の数。
        ffn: AIFI の FFN の中間の次元。
        dim: 出力の特徴の次元 C(既定は D_MODEL。公式の hidden_dim)。
        block: 融合のブロックの中の 3×3 のブロック("rep": 公式と同じ RepVGG 型 / "dw": depthwise separable。exp00104 の E3)。
    """

    def __init__(self, cins: list[int], expansion: float = 0.75, n: int = 3, heads: int = 8, ffn: int = 1024,
                 dim: int = D_MODEL, block: str = "rep"):
        super().__init__()
        self.proj = nn.ModuleList(conv_bn(c, dim, act=False) for c in cins)
        self.aifi = AIFILayer(heads, ffn, dim)  # AIFI: 最も粗い段の中の注意
        # ---- ここから CCFF(段どうしの畳み込みの融合)の部品: lateral・fpn(上から下)、down・pan(下から上) ----
        # 段の数 L(公式は stride 8・16・32 の 3 段。exp00105 では stride 4 を足した 4 段も使う)に対して、どれも L − 1 個
        mid = round(expansion * dim // 2)
        fuse = lambda: FuseBlock(dim, dim, 2 * dim, mid, n, block)  # 足し合わせるので入力も C チャンネル
        k = len(cins) - 1
        self.lateral = nn.ModuleList(conv_bn(dim, dim, act=False) for _ in range(k))  # 粗い段から順に、横からの 1×1
        self.fpn = nn.ModuleList(fuse() for _ in range(k))  # 上から下: 粗い方から 2 番目の段 … 最も細かい段を作る
        self.down = nn.ModuleList(nn.Sequential(conv_bn(dim, dim, act=False),
                                                conv_bn(dim, dim, 3, 2, groups=dim, act=False))
                                  for _ in range(k))  # 下から上: 細かい段から順に、1 つ粗い段へ縮める
        self.pan = nn.ModuleList(fuse() for _ in range(k))  # 下から上: 細かい方から 2 番目の段 … 最も粗い段を作る
        # ---- CCFF の部品はここまで ----

    def forward(self, feats: list[torch.Tensor], image_size: tuple[int, int] | None = None) -> list[torch.Tensor]:
        """段ごとの特徴を作る(段の数は入力と同じ L。3 段なら公式と同じ計算)。

        Args:
            feats: backbone の段の特徴を細かい順に [(B, C_i, S/s_i, S/s_i)](s は 8・16・32、または 4・8・16・32)。
            image_size: 使わない(自作の GTR 系の `Encoder` と呼び方をそろえるための引数)。

        Returns:
            細かい順の [(B, C, S/s_i, S/s_i)]。
        """
        p = [proj(f) for proj, f in zip(self.proj, feats)]  # 段ごとに (B, C_i, h, w) -> (B, C, h, w)
        p[-1] = self.aifi(p[-1])  # AIFI: 最も粗い段(stride 32)だけに自己注意
        # ---- ここから CCFF(段どうしの畳み込みの融合。FPN の経路 + PAN の経路)----
        # 上から下(FPN の経路): 粗い段を横からの 1×1 に通し、2 倍に拡大して 1 つ細かい段と足し、融合のブロックを通す
        n_lv = len(p)
        tops = [None] * n_lv  # tops[i] = 段 i の横からの出力(下から上で足す)
        cur = p[-1]
        for j in range(n_lv - 1):
            i = n_lv - 1 - j  # 今の段(粗い方から)
            tops[i] = self.lateral[j](cur)  # (B, C, h_i, w_i)
            cur = self.fpn[j](p[i - 1] + F.interpolate(tops[i], size=p[i - 1].shape[-2:], mode="nearest"))  # 段 i − 1
        # 下から上(PAN の経路): 細かい段を stride 2 で縮め、上から下のときの横からの出力と足し、融合のブロックを通す
        outs = [cur]  # 最も細かい段
        for j in range(n_lv - 1):
            outs.append(self.pan[j](self.down[j](outs[-1]) + tops[j + 1]))  # 段 j + 1
        # ---- CCFF はここまで ----
        return outs
