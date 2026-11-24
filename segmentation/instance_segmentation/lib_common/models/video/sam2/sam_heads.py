"""SAM 2 の prompt encoder とマスクの decoder。exp00502。

参考: 公式のコード(facebookresearch/sam2 の `sam2/modeling/sam/prompt_encoder.py`・`mask_decoder.py`、`sam2_base.py` の
_build_sam_heads)。コードは写さず、計算を読み取って書いた。設定は公式の SAM 2.1 と同じ(マスクの出力 3 + 1、two-way transformer 2 層、
IoU の head 3 層 sigmoid、物体らしさの head(MLP 3 層)、高解像度の特徴(stride 4・8)を使う、物体のポインタに複数マスクの token を使う)。

形の記号: B = まとめの数、N = 点の数、C = 次元(256)、H・W = 画像の特徴の縦・横(入力の 1/16)、S = 入力の一辺。
"""
import torch
from torch import nn

from segmentation.instance_segmentation.lib_common.models.video.sam2.blocks import (MLP, LayerNorm2d, PositionEmbeddingRandom,
                                                                                    TwoWayTransformer)


class PromptEncoder(nn.Module):
    """点・箱・マスクのプロンプトを埋め込む。

    Args:
        dim: 埋め込みの次元。
        embed_size: 画像の特徴の (H, W)。
        input_size: 入力の (S, S)。
        mask_in_chans: マスクを縮める畳み込みの幅(公式は 16)。
    """

    def __init__(self, dim: int, embed_size: tuple[int, int], input_size: tuple[int, int], mask_in_chans: int = 16):
        super().__init__()
        self.dim, self.embed_size, self.input_size = dim, embed_size, input_size
        self.pe_layer = PositionEmbeddingRandom(dim // 2)
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, dim) for _ in range(4))  # 負の点・正の点・箱の左上・右下
        self.not_a_point_embed = nn.Embedding(1, dim)
        self.mask_input_size = (4 * embed_size[0], 4 * embed_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, 2, stride=2), LayerNorm2d(mask_in_chans // 4), nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, 2, stride=2), LayerNorm2d(mask_in_chans), nn.GELU(),
            nn.Conv2d(mask_in_chans, dim, 1))
        self.no_mask_embed = nn.Embedding(1, dim)

    def get_dense_pe(self) -> torch.Tensor:
        """画像の特徴の格子の位置の埋め込み。

        Returns:
            (1, C, H, W)。
        """
        return self.pe_layer(self.embed_size)[None]

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """点を位置の埋め込み + ラベルの埋め込みにする。

        Args:
            points: (B, N, 2) の (x, y)(入力の画素)。
            labels: (B, N)(0 負・1 正・2 左上・3 右下・−1 点でない)。
            pad: 「点でない」点を 1 つ足すか(箱がないとき。公式と同じ)。

        Returns:
            (B, N(+1), C)。
        """
        points = points + 0.5  # 画素の中心
        if pad:  # 箱がないときは「点でない」点を 1 つ足す(公式と同じ)
            points = torch.cat([points, points.new_zeros(points.shape[0], 1, 2)], dim=1)
            labels = torch.cat([labels, -labels.new_ones(labels.shape[0], 1)], dim=1)
        emb = self.pe_layer.forward_with_coords(points, self.input_size)
        emb = torch.where((labels == -1)[..., None], torch.zeros_like(emb) + self.not_a_point_embed.weight, emb)
        for k in range(4):
            emb = torch.where((labels == k)[..., None], emb + self.point_embeddings[k].weight, emb)
        return emb

    def forward(self, points: tuple[torch.Tensor, torch.Tensor] | None, masks: torch.Tensor | None, batch: int):
        """プロンプトを埋め込む(箱は点(ラベル 2・3)として渡す。SAM 2 の学習と同じ)。

        Args:
            points: (座標 (B, N, 2), ラベル (B, N)) か None。
            masks: マスクのプロンプト (B, 1, 4H, 4W)(マスクの値)か None。
            batch: まとめの数 B。

        Returns:
            (疎な埋め込み (B, N', C), 密な埋め込み (B, C, H, W))。
        """
        dev = self.no_mask_embed.weight.device
        sparse = torch.empty(batch, 0, self.dim, device=dev)
        if points is not None:
            sparse = torch.cat([sparse, self._embed_points(points[0], points[1], pad=True)], dim=1)
        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(batch, -1, *self.embed_size)
        return sparse, dense


class MaskDecoder(nn.Module):
    """マスクの decoder(SAM 2 の MaskDecoder。高解像度の特徴・物体らしさの head つき)。

    Args:
        dim: 次元(256)。
        num_multimask: 複数マスクの数(公式は 3)。
        dynamic_multimask: 単一のマスクが不安定なとき複数マスクの最良に替える(公式の video predictor の推論の設定)。
        stability_delta: 安定さを測る値の幅 δ。
        stability_thresh: 安定とみなす面積の比の閾値。
    """

    def __init__(self, dim: int = 256, num_multimask: int = 3, dynamic_multimask: bool = False,
                 stability_delta: float = 0.05, stability_thresh: float = 0.98):
        super().__init__()
        self.transformer = TwoWayTransformer(2, dim, 8, 2048, 2)
        self.iou_token = nn.Embedding(1, dim)
        self.num_mask_tokens = num_multimask + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, dim)
        self.obj_score_token = nn.Embedding(1, dim)
        self.output_upscaling = nn.ModuleList([nn.ConvTranspose2d(dim, dim // 4, 2, stride=2), LayerNorm2d(dim // 4), nn.GELU(),
                                               nn.ConvTranspose2d(dim // 4, dim // 8, 2, stride=2), nn.GELU()])
        self.conv_s0 = nn.Conv2d(dim, dim // 8, 1)  # stride 4 の特徴 → 32
        self.conv_s1 = nn.Conv2d(dim, dim // 4, 1)  # stride 8 の特徴 → 64
        self.output_hypernetworks_mlps = nn.ModuleList(MLP(dim, dim, dim // 8, 3) for _ in range(self.num_mask_tokens))
        self.iou_prediction_head = MLP(dim, 256, self.num_mask_tokens, 3, sigmoid_output=True)
        self.pred_obj_score_head = MLP(dim, dim, 1, 3)
        self.dynamic_multimask, self.delta, self.thresh = dynamic_multimask, stability_delta, stability_thresh

    def forward(self, image: torch.Tensor, image_pe: torch.Tensor, sparse: torch.Tensor, dense: torch.Tensor,
                multimask: bool, high_res: list[torch.Tensor]):
        """マスク・IoU の予測・物体らしさを出す。

        Args:
            image: 画像の特徴 (B, C, H, W)。
            image_pe: その位置の埋め込み (1, C, H, W)。
            sparse: 疎なプロンプトの埋め込み (B, N, C)。
            dense: 密なプロンプトの埋め込み (B, C, H, W)。
            multimask: 3 つのマスクを出すか。
            high_res: [stride 4 の特徴 (B, C/8, 4H, 4W), stride 8 の特徴 (B, C/4, 2H, 2W)](conv_s0・conv_s1 の後)。

        Returns:
            (マスクの値 (B, M, 4H, 4W), IoU の予測 (B, M), 物体のポインタの元の token (B, M, C), 物体らしさの値 (B, 1))
            (M = 3(multimask)か 1)。
        """
        b = sparse.shape[0]
        out_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight])[None].expand(b, -1, -1)
        tokens = torch.cat([out_tokens, sparse], dim=1)
        src = image + dense
        pos = image_pe.expand(b, -1, -1, -1)
        _, c, h, w = src.shape
        hs, src = self.transformer(src, pos, tokens)
        iou_out, mask_out = hs[:, 1], hs[:, 2: 2 + self.num_mask_tokens]
        src = src.transpose(1, 2).reshape(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        s0, s1 = high_res
        up = act1(ln1(dc1(src) + s1))
        up = act2(dc2(up) + s0)  # (B, C/8, 4H, 4W)
        hyper = torch.stack([mlp(mask_out[:, i]) for i, mlp in enumerate(self.output_hypernetworks_mlps)], dim=1)  # (B, 4, C/8)
        masks = (hyper @ up.flatten(2)).reshape(b, -1, *up.shape[-2:])
        iou = self.iou_prediction_head(iou_out)
        obj = self.pred_obj_score_head(hs[:, 0])
        if multimask:
            masks, iou, tok = masks[:, 1:], iou[:, 1:], mask_out[:, 1:]
        elif self.dynamic_multimask and not self.training:
            masks, iou = self._dynamic(masks, iou)
            tok = mask_out[:, 0:1]
        else:
            masks, iou, tok = masks[:, :1], iou[:, :1], mask_out[:, 0:1]
        return masks, iou, tok, obj

    def _dynamic(self, masks: torch.Tensor, iou: torch.Tensor):
        """単一のマスクの安定さ(値 > δ と > −δ の面積の比)が閾値未満なら、複数マスクのうち IoU の予測が最良のものに替える。

        Args:
            masks: 全部のマスクの値 (B, 4, h, w)(先頭が単一のマスク)。
            iou: 全部の IoU の予測 (B, 4)。

        Returns:
            (選んだマスクの値 (B, 1, h, w), その IoU の予測 (B, 1))。
        """
        multi, multi_iou = masks[:, 1:], iou[:, 1:]
        best = multi_iou.argmax(-1)
        ar = torch.arange(len(best), device=best.device)
        best_m, best_iou = multi[ar, best][:, None], multi_iou[ar, best][:, None]
        single = masks[:, :1].flatten(-2)
        area_i, area_u = (single > self.delta).sum(-1).float(), (single > -self.delta).sum(-1).float()
        stable = torch.where(area_u > 0, area_i / area_u, torch.ones_like(area_u)) >= self.thresh  # (B, 1)
        return (torch.where(stable[..., None, None], masks[:, :1], best_m),
                torch.where(stable, iou[:, :1], best_iou))
