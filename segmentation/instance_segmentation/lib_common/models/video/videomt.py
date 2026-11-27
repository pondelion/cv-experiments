"""VidEoMT(Norouzi ら CVPR 2026、"VidEoMT: Your ViT is Secretly Also a Video Segmentation Model")の自作の実装。exp00502。

参考: 論文と公式のコード(NielsRogge/videomt。`videomt/modeling/backbone/videomt.py`・`configs/ytvis19/videomt/vit-small`)。
コードは写さず、構成と設定の値を読み取って書いた。土台は画像の段階の自作の EoMT(`eomt.py`)。

MOT(動画全体での物体の ID の管理・識別)の形: **学習して身に付ける形**(TrackFormer・MOTR と同じ種類)。クエリを次のフレームに渡し、
クエリの番号 = ID。学習の正解の割り当て(losses/video_consistent_criterion.py)で、物体は最初に写ったフレームのクエリにずっと割り当て、
隠れているフレームもクラスの正解をその物体にし、物体を持つクエリは後から出た物体に使わない。これで「再出現は同じクエリ・新しい物体は
空いたクエリ」を学習する。推論には規則はなく学習した振る舞いだけ。クエリを空きに戻す仕組みはなく、動画 1 本で区別できる物体は
クエリの数まで。学習の区間(5 フレーム、毎回初期のクエリから始める)より長い隠れ・長い受け渡しは学習に出てこない。

構成(1 フレームずつ、時刻の順に処理する online の方式):
    ViT(timm)の最初の L1 層は、フレームごとに画像のトークンだけを処理する(フレームどうしは独立)。
    最後の L2 層(公式の ViT-S は 3 層)の前で、クエリ Q 個(公式は 200)を画像のトークンの前につなげ、クエリと画像のトークンを
    ViT の自己注意でまとめて処理する(EoMT と同じ)。
    フレーム間の情報のやり取り(追跡)はクエリだけで行う:
        最初のフレーム: 学習するクエリ q をそのまま使う
        2 フレーム目から: 前のフレームの最後の層の出力のクエリを線形の層(query_updater)に通し、学習するクエリ q に足したもの
    クエリの番号が物体の ID になる(前のフレームと同じ番号のクエリが同じ物体を追う)。新しく出た物体は、足している学習するクエリ q
    から拾う。別の追跡の部・照合はない。
    予測: クエリから、クラス(Linear。K + 1)とマスクの埋め込み(3 層の MLP)を出し、画像のトークンを ScaleBlock で 1/4 まで上げた
    特徴との内積をマスクにする(EoMT と同じ)。学習では、クエリを入れた各層の前でも予測を出す(補助の損失)。
    公式と同じく、EoMT の学習中の注意のマスク(mask annealing)は使わない。

出力(学習): 層ごとの {"logits": (B, T, Q, K+1), "masks": (B, Q, T, 4h, 4w)}(フレームごとの予測)の並び。損失は
`losses/video_consistent_criterion.py`(公式の videomt_online と同じ、物体が最初に写ったフレームでの割り当てと、フレームごとの損失)。

形の記号: B = 区間(動画)の数、T = フレームの数、Q = クエリの数、P = ViT の先頭のトークン(クラスのトークンなど)の数、
h・w = パッチの縦・横の数(マスクは 4h・4w)、N = h × w、C = ViT の次元、K = クラスの数、S = 入力の一辺。
"""
import math

import timm
import torch
from einops import rearrange, repeat
from torch import nn

from segmentation.instance_segmentation.lib_common.models.eomt import ScaleBlock


class VidEoMT(nn.Module):
    """VidEoMT。

    Args:
        num_classes: クラスの数(物体なしを含めない)。
        backbone: timm の ViT の名前(全部の層が同じ解像度のもの。パッチ 16)。
        num_queries: クエリの数 Q(公式の ViT-S は 200)。
        num_blocks: クエリを入れる最後の層の数 L2(公式の ViT-S は 3)。
        img_size: 入力の一辺 S(位置の埋め込みをこの大きさで作る)。
        pretrained: ViT の学習済みの重みを読むか。
        init_weights: 自己教師ありの学習(exp00106 の `lib_common/ssl/`)で保存した ViT の重み。あれば pretrained の代わりに読む。
        norm_queries: 補助の予測と次のフレームへ渡すクエリに ViT の最後の LayerNorm をかけるか(公式の NORM_QUERIES。
            既定の False は、この設定を入れる前の学習と同じ動き)。
    """

    def __init__(self, num_classes: int, backbone: str = "deit3_small_patch16_224.fb_in1k", num_queries: int = 200,
                 num_blocks: int = 3, img_size: int = 512, pretrained: bool = True, init_weights: str | None = None,
                 norm_queries: bool = False):
        super().__init__()
        self.norm_queries = norm_queries
        self.vit = timm.create_model(backbone, pretrained=pretrained and not init_weights, num_classes=0, img_size=img_size)
        if init_weights:
            from lib_common.ssl.load import load_ssl_vit
            load_ssl_vit(self.vit, init_weights)
        c, patch = self.vit.embed_dim, self.vit.patch_embed.patch_size[0]
        self.num_queries, self.num_blocks = num_queries, num_blocks
        # RoPE の ViT(timm の Eva。DINOv3 の ViT): 位置の情報は、各 block の attention で、先頭の num_prefix_tokens 個([CLS]・register)
        # 以外のトークン(パッチ)の q・k に RoPE をかけて入れる。クエリを先頭に足す最後の num_blocks 層では、先頭のクエリにも
        # RoPE をかけないように、その数をクエリの分だけ増やす
        self.rope = getattr(self.vit, "rope", None) is not None
        if self.rope:
            for block in self.vit.blocks[len(self.vit.blocks) - num_blocks:]:
                block.attn.num_prefix_tokens = num_queries + self.vit.num_prefix_tokens
        self.q = nn.Embedding(num_queries, c)
        self.query_updater = nn.Linear(c, c)  # 前のフレームのクエリ → 今のフレームに足す分
        self.class_head = nn.Linear(c, num_classes + 1)
        self.mask_head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c), nn.GELU(), nn.Linear(c, c))
        self.upscale = nn.Sequential(*[ScaleBlock(c) for _ in range(max(1, int(math.log2(patch)) - 2))])  # 1/16 → 1/4

    def _predict(self, x: torch.Tensor, hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """ViT の層の出力から、クエリごとのクラスの予測とマスクの予測を作る。

        入力の並びの先頭 Q 個がクエリ、その後ろに ViT の先頭のトークン(クラスのトークンなど)P 個、画像のパッチのトークン N 個が続く。
        クラスはクエリを全結合の層に通して出す。マスクは、クエリを MLP に通したベクトルと、パッチのトークンを入力の 1/4 の大きさまで
        拡大した特徴との内積で出す(EoMT と同じ)。

        Args:
            x: ViT の層の出力 (B, Q + P + N, C)。
            hw: パッチの縦・横の数 (h, w)。

        Returns:
            (クラスの値 (B, Q, K+1)(最後が「物体なし」), マスクの値 (B, Q, 4h, 4w)(sigmoid の前))。
        """
        q = x[:, : self.num_queries]  # (B, Q, C)
        img = rearrange(x[:, self.num_queries + self.vit.num_prefix_tokens:], "b (h w) c -> b c h w", h=hw[0])  # (B, C, h, w)
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_head(q), self.upscale(img))  # (B, Q, 4h, 4w)
        return self.class_head(q), masks

    def frame_tokens(self, frames: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """フレームを ViT の前半の層に通し、画像のトークンにする。

        VidEoMT は ViT の層を 2 つに分けて使う。前半の層(全体の層の数 − num_blocks 個。論文の L1)は、クエリを入れずに、フレームを
        1 枚ずつ独立に処理する。この関数はその前半を受け持つ。フレームどうしのやり取りがないので、複数のフレームをまとめて処理できる。
        後半の層は step が受け持つ。

        Args:
            frames: フレームの画像 (M(フレームの数), 3, S, S)。

        Returns:
            (前半の層の出力のトークン (M, P + N, C)(先頭のトークン P 個 + パッチのトークン N 個), パッチの縦・横の数 (h, w))。
        """
        v = self.vit
        x = v.patch_embed(frames)  # (M, N, C) か (M, h, w, C)(dynamic_img_size のとき)
        hw = (frames.shape[-2] // v.patch_embed.patch_size[0], frames.shape[-1] // v.patch_embed.patch_size[1])
        pe = v._pos_embed(x)  # 位置の埋め込みと先頭のトークン → (M, P + N, C)(Eva は RoPE の値 (N, d) も返す)
        x, rope = pe if isinstance(pe, tuple) else (pe, None)
        if getattr(v, "patch_drop", None) is not None:  # Eva は None のことがある
            x = v.patch_drop(x)
        x = v.norm_pre(x)
        for block in v.blocks[: len(v.blocks) - self.num_blocks]:
            x = block(x, rope=rope) if rope is not None else block(x)
        return x, hw

    def _rope(self, hw: tuple[int, int]) -> torch.Tensor | None:
        """後半の層に渡す、パッチの位置の情報(RoPE)を作る。

        DINOv3 の ViT(timm の Eva)は、位置の情報を足し算の埋め込みではなく、各層の注意の中でパッチのトークンに回転(RoPE)をかけて
        入れる。後半の層を自分で呼ぶときにも、その回転の値を層に渡す必要があるので、パッチの格子の大きさから作る。

        Args:
            hw: パッチの縦・横の数 (h, w)。

        Returns:
            RoPE の値 (h·w, d)。RoPE を使わない ViT なら None。
        """
        return self.vit.rope.get_embed(shape=list(hw)) if self.rope else None

    def step(self, tokens: torch.Tensor, hw: tuple[int, int], prev: torch.Tensor | None) -> tuple[list, torch.Tensor]:
        """1 フレーム分の、ViT の後半の層の処理と予測を行い、次のフレームに渡すクエリを返す。

        ViT の後半の層(最後の num_blocks 個。論文の L2)では、クエリ(物体を表すベクトル Q 個)を、前半の層の出力のトークンの前に
        つなげて一緒に処理する。注意の中で、クエリが画像のトークンから物体の情報を集める。クエリの作り方は次の通り(公式と同じ):
            - 動画の最初のフレーム(prev が None): 学習するクエリ q をそのまま使う
            - 2 フレーム目から: 前のフレームの最後の層の出力のクエリを全結合の層(query_updater)に通し、学習するクエリ q に足す
        前のフレームと同じ番号のクエリが、同じ物体を追う(クエリの番号が物体の ID になる)。最後の層の出力のクエリを、次のフレームに
        渡す。学習のときは、各層に入る前の値からも予測を作る(途中の層にもかける補助の損失のため)。

        Args:
            tokens: このフレームの、frame_tokens の出力 (B, P + N, C)。
            hw: パッチの縦・横の数 (h, w)。
            prev: 前のフレームから渡すクエリ (B, Q, C)。動画の最初のフレームでは None。

        Returns:
            (予測の並び [(クラスの値 (B, Q, K+1), マスクの値 (B, Q, 4h, 4w))]。学習のときは後半の層の前の予測(最後の層の前を除く L2 − 1 個。公式の損失と同じ)と最後の層の予測、
            推論のときは最後の層の予測だけ,
             次のフレームに渡すクエリ (B, Q, C))。
        """
        b = tokens.shape[0]
        q = repeat(self.q.weight, "q c -> b q c", b=b)  # (B, Q, C)
        if prev is not None:
            q = self.query_updater(prev) + q  # 前のフレームのクエリを足す(公式と同じ)
        x = torch.cat([q, tokens], dim=1)  # (B, Q + P + N, C)
        preds, rope = [], self._rope(hw)
        # norm_queries(公式の NORM_QUERIES。VIPSeg の DINOv3 の設定は True): 補助の予測と次のフレームへ渡すクエリに、
        # ViT の最後の LayerNorm をかける(False なら LayerNorm の前の値を使う)
        norm = self.vit.norm if self.norm_queries else (lambda y: y)
        for j, block in enumerate(self.vit.blocks[len(self.vit.blocks) - self.num_blocks:]):
            # 補助の損失のための、層の前の予測。公式は L2 個の層それぞれの前で予測を作るが、損失の重みの表(weight_dict)に補助の
            # 損失が L2 − 1 個(_0 … _{L2-2})しかなく、最後の層の前の予測(_{L2-1})は損失から外れる(公式の videomt.py の
            # aux_weight_dict)。損失に入らない予測は作らない(勾配は同じ)
            if self.training and j < self.num_blocks - 1:
                preds.append(self._predict(norm(x), hw))
            x = block(x, rope=rope) if rope is not None else block(x)
        preds.append(self._predict(self.vit.norm(x), hw))
        return preds, norm(x)[:, : self.num_queries]  # 次のフレームへは、最後の層の出力のクエリを渡す(公式と同じ)

    def forward(self, clips: torch.Tensor, prev: torch.Tensor | None = None) -> tuple[list[dict], torch.Tensor]:
        """学習の区間(連続した数フレーム)を処理し、各フレームの予測を作る。

        前半の層(frame_tokens)はフレームが独立なので、区間の全フレームをまとめて処理する。後半の層(step)は前のフレームのクエリを
        使うので、フレームを時刻の順に 1 枚ずつ処理する。学習では、区間の最初のフレームは前のフレームのクエリなし(動画の最初と同じ)で
        始める。

        Args:
            clips: 区間の画像 (B(区間の数), T(フレームの数), 3, S, S)。
            prev: 区間の最初のフレームに渡すクエリ (B, Q, C)。動画の最初から始めるときは None。

        Returns:
            (予測の並び(step の予測の並びと同じ順。学習は後半の層の前(最後の層の前を除く)と最後の層、推論は最後の層だけ)。各要素は
            {"logits": クラスの値 (B, T, Q, K+1), "masks": マスクの値 (B, Q, T, 4h, 4w)},
             区間の最後のフレームの、次に渡すクエリ (B, Q, C))。
        """
        b, t = clips.shape[:2]
        tokens, hw = self.frame_tokens(rearrange(clips, "b t c h w -> (b t) c h w"))  # 最初の L1 層はフレームをまとめて処理
        tokens = rearrange(tokens, "(b t) n c -> b t n c", b=b)
        per_frame = []
        for i in range(t):  # 最後の L2 層は時刻の順に(前のフレームのクエリを使うため)
            preds, prev = self.step(tokens[:, i], hw, prev)
            per_frame.append(preds)
        outs = []
        for layer in range(len(per_frame[0])):
            logits = torch.stack([f[layer][0] for f in per_frame], dim=1)  # (B, T, Q, K+1)
            masks = torch.stack([f[layer][1] for f in per_frame], dim=2)  # (B, Q, T, 4h, 4w)
            outs.append({"logits": logits, "masks": masks})
        return outs, prev

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor, chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        """推論: 1 本の動画の全フレームの予測を作る。

        動画の最初のフレームから順に、前のフレームのクエリを次のフレームに渡しながら処理する(online。未来のフレームは使わない)。
        前半の層(frame_tokens)はフレームが独立なので、GPU のメモリに収まるように chunk 枚ずつまとめて処理し、後半の層(step)は
        1 枚ずつ処理する。クエリの番号が動画を通した物体の ID になる。

        Args:
            frames: 動画の全フレームの画像 (T, 3, S, S)。
            chunk: 前半の層でまとめて処理するフレームの数。

        Returns:
            (各フレームのクラスの値 (T, Q, K+1), 各フレームのマスクの値 (Q, T, 4h, 4w)(sigmoid の前))。
        """
        logits, masks, prev = [], [], None
        for s in range(0, len(frames), chunk):
            tokens, hw = self.frame_tokens(frames[s: s + chunk])  # (chunk, P + N, C)
            for i in range(len(tokens)):
                preds, prev = self.step(tokens[i: i + 1], hw, prev)
                logits.append(preds[-1][0][0])  # (Q, K+1)
                masks.append(preds[-1][1][0])  # (Q, 4h, 4w)
        return torch.stack(logits), torch.stack(masks, dim=1)  # (T, Q, K+1), (Q, T, 4h, 4w)
