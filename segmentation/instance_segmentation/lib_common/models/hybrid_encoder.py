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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, c, h, w) -> (B, c, h, w)。"""
        return F.silu(self.conv3(x) + self.conv1(x))


class CSPBlock(nn.Module):
    """CSP のブロック: 1×1 で 2 倍の幅にして 2 つに分け、片方に RepBlock を n 回通して、もう片方と足す。

    Args:
        cin: 入力のチャンネル。
        cout: 出力のチャンネル(途中の幅も cout)。
        n: RepBlock の数。
    """

    def __init__(self, cin: int, cout: int, n: int):
        super().__init__()
        self.split = conv_bn(cin, 2 * cout)
        self.blocks = nn.Sequential(*[RepBlock(cout) for _ in range(n)])

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
    """

    def __init__(self, cin: int, cout: int, wide: int, mid: int, n: int):
        super().__init__()
        self.inp = conv_bn(cin, wide)
        self.csp1, self.csp2 = CSPBlock(wide // 2, mid, n), CSPBlock(mid, mid, n)
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
    """

    def __init__(self, heads: int = 8, ffn: int = 1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_MODEL, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, ffn), nn.SiLU(inplace=True), nn.Linear(ffn, D_MODEL))
        self.norm1, self.norm2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)

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

    Args:
        cins: backbone の stride 8・16・32 の段のチャンネル。
        expansion: CSPBlock の幅の倍率(公式の L の版は 0.75)。
        n: CSPBlock の中の RepBlock の数(公式は round(3 × depth_mult)。L の版は 3)。
        heads: AIFI の注意の頭の数。
        ffn: AIFI の FFN の中間の次元。
    """

    def __init__(self, cins: list[int], expansion: float = 0.75, n: int = 3, heads: int = 8, ffn: int = 1024):
        super().__init__()
        self.proj = nn.ModuleList(conv_bn(c, D_MODEL, act=False) for c in cins)
        self.aifi = AIFILayer(heads, ffn)
        mid = round(expansion * D_MODEL // 2)
        fuse = lambda: FuseBlock(D_MODEL, D_MODEL, 2 * D_MODEL, mid, n)  # 足し合わせるので入力も C チャンネル
        self.lateral = nn.ModuleList(conv_bn(D_MODEL, D_MODEL, act=False) for _ in range(2))  # stride 32・16 の段の横から
        self.fpn = nn.ModuleList(fuse() for _ in range(2))  # stride 16・8 を作る
        self.down = nn.ModuleList(nn.Sequential(conv_bn(D_MODEL, D_MODEL, act=False),
                                                conv_bn(D_MODEL, D_MODEL, 3, 2, groups=D_MODEL, act=False))
                                  for _ in range(2))  # stride 8 → 16、16 → 32
        self.pan = nn.ModuleList(fuse() for _ in range(2))  # stride 16・32 を作る

    def forward(self, feats: list[torch.Tensor], image_size: tuple[int, int] | None = None) -> list[torch.Tensor]:
        """3 段の特徴を作る。

        Args:
            feats: backbone の stride 8・16・32 の段の特徴 [(B, C_i, S/s_i, S/s_i)]。
            image_size: 使わない(自作の GTR 系の `Encoder` と呼び方をそろえるための引数)。

        Returns:
            [(B, C, S/8, S/8), (B, C, S/16, S/16), (B, C, S/32, S/32)]。
        """
        p3, p4, p5 = [proj(f) for proj, f in zip(self.proj, feats)]  # 段ごとに (B, C_i, h, w) -> (B, C, h, w)
        p5 = self.aifi(p5)
        # 上から下: 粗い段を横からの 1×1 に通し、2 倍に拡大して細かい段と足し、融合のブロックを通す
        top5 = self.lateral[0](p5)  # (B, C, S/32, S/32)
        mid4 = self.fpn[0](p4 + F.interpolate(top5, size=p4.shape[-2:], mode="nearest"))  # (B, C, S/16, S/16)
        top4 = self.lateral[1](mid4)
        out3 = self.fpn[1](p3 + F.interpolate(top4, size=p3.shape[-2:], mode="nearest"))  # (B, C, S/8, S/8)
        # 下から上: 細かい段を stride 2 で縮め、上から下のときの横からの出力と足し、融合のブロックを通す
        out4 = self.pan[0](self.down[0](out3) + top4)  # (B, C, S/16, S/16)
        out5 = self.pan[1](self.down[1](out4) + top5)  # (B, C, S/32, S/32)
        return [out3, out4, out5]
