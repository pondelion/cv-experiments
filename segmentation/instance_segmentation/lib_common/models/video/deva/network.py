"""DEVA の伝搬のネットワーク(Cheng ら ICCV 2023、"Tracking Anything with Decoupled Video Segmentation")。exp00502。

参考: 公式のコード(hkchengrex/Tracking-Anything-with-DEVA の `deva/model/network.py`・`big_modules.py`・`modules.py`・
`group_modules.py`・`cbam.py`・`memory_utils.py`・`resnet.py`)。コードは写さず、計算を読み取って書いた。設定は公式と同じ
(key 64 次元・value 512 次元・画素の特徴 512 次元、感覚の記憶(sensory)512 次元、CBAM、マスクの encoder は ResNet-18)。

作り(XMem 系。クラスによらず、与えたマスクの物体を記憶で次のフレームに運ぶ):
    - 画素の encoder: 画像から stride 16・8・4 の特徴。stride 16 から key(64)と、マスクの decoder の入力(512)を作る
    - key の射影: key(3×3)、shrinkage(d² + 1)、selection(sigmoid)
    - マスクの encoder: 画像 + 物体のマスク(4 チャンネル)を ResNet-18(stride 16 まで)に通し、画素の特徴と CBAM つきの block で
      混ぜて value(512)にする。感覚の記憶を GRU 型の門(forget・update)で更新する(deep update)
    - 記憶の読み出し: 記憶の key と今の key の類似度(L2 を selection と shrinkage で重みづけた負の距離 ÷ √64)の softmax で value を平均
    - マスクの decoder: 読み出し + 感覚の記憶(と前のマスク)を画素の特徴と混ぜ、stride 8・4 で上げて 1 チャンネルの値を出す。
      感覚の記憶を 3 段の特徴から更新する。物体ごとの確率から、背景つきの soft aggregation(公式の aggregate)で全体の確率にする

公式との違い(exp の条件): 画素の encoder の幹は ResNet-50 の代わりに exp の backbone(DINOv3 ViT-B の ViT-Adapter など。
lib_common/backbones。stride 4・8・16 の特徴の次元は backbone による)。マスクの encoder の ResNet-18 は公式と同じ torchvision の重み
(timm の resnet18.tv_in1k)で、マスクの入力チャンネルの重みは公式の load_weights_add_extra_dim と同じく orthogonal で初期化する。

形の記号: B = まとめの数、N = 物体の数、C = チャンネル、T = 記憶のフレームの数、H・W = stride 16 の特徴の縦・横。
"""
import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

from segmentation.instance_segmentation.lib_common.backbones import create_feature_backbone

KEY_DIM, VALUE_DIM, PIX_DIM = 64, 512, 512


# ---- 物体の軸を持つ特徴の部品(公式の group_modules) ----
def interp_groups(g: torch.Tensor, ratio: float, mode: str, align_corners) -> torch.Tensor:
    """(B, N, C, H, W) を物体ごとに拡大縮小する。"""
    b, n = g.shape[:2]
    g = F.interpolate(g.flatten(0, 1), scale_factor=ratio, mode=mode, align_corners=align_corners)
    return g.view(b, n, *g.shape[1:])


def upsample_groups(g, ratio=2.0):
    return interp_groups(g, ratio, "bilinear", False)


def downsample_groups(g, ratio=0.5):
    return interp_groups(g, ratio, "area", None)


class GConv2D(nn.Conv2d):
    """(B, N, C, H, W) に物体ごとに同じ畳み込み。"""

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        b, n = g.shape[:2]
        g = super().forward(g.flatten(0, 1))
        return g.view(b, n, *g.shape[1:])


class GroupResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.downsample = None if cin == cout else GConv2D(cin, cout, 1)
        self.conv1, self.conv2 = GConv2D(cin, cout, 3, padding=1), GConv2D(cout, cout, 3, padding=1)

    def forward(self, g):
        out = self.conv2(F.relu(self.conv1(F.relu(g))))
        return out + (g if self.downsample is None else self.downsample(g))


def distribute(x: torch.Tensor, g: torch.Tensor, method: str = "cat") -> torch.Tensor:
    """画像の特徴 x (B, C, H, W) を物体の軸に広げて g (B, N, C', H, W) と合わせる(公式の MainToGroupDistributor)。"""
    x = x.unsqueeze(1).expand(-1, g.shape[1], -1, -1, -1)
    return torch.cat([x, g], 2) if method == "cat" else x + g


class ChannelGate(nn.Module):
    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(c, c // reduction), nn.ReLU(), nn.Linear(c // reduction, c))

    def forward(self, x):
        att = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(att)[..., None, None]


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        comp = torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.spatial(comp))


class CBAM(nn.Module):
    """CBAM(チャンネルの門 → 空間の門)。"""

    def __init__(self, c: int):
        super().__init__()
        self.channel, self.spatial = ChannelGate(c), SpatialGate()

    def forward(self, x):
        return self.spatial(self.channel(x))


class GroupFeatureFusionBlock(nn.Module):
    """画像の特徴と物体ごとの特徴を合わせる(連結 → ResBlock → CBAM を足す → ResBlock)。"""

    def __init__(self, x_in: int, g_in: int, g_mid: int, g_out: int):
        super().__init__()
        self.block1, self.attention, self.block2 = GroupResBlock(x_in + g_in, g_mid), CBAM(g_mid), GroupResBlock(g_mid, g_out)

    def forward(self, x, g):
        b, n = g.shape[:2]
        g = self.block1(distribute(x, g))
        r = self.attention(g.flatten(0, 1)).view(b, n, *g.shape[2:])
        return self.block2(g + r)


class KeyProjection(nn.Module):
    """key・shrinkage・selection(key の重みは公式と同じく orthogonal、bias 0)。"""

    def __init__(self, cin: int, key_dim: int = KEY_DIM):
        super().__init__()
        self.key_proj, self.d_proj, self.e_proj = (nn.Conv2d(cin, key_dim, 3, padding=1), nn.Conv2d(cin, 1, 3, padding=1),
                                                   nn.Conv2d(cin, key_dim, 3, padding=1))
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)

    def forward(self, x, need_s: bool = True, need_e: bool = True):
        s = self.d_proj(x) ** 2 + 1 if need_s else None
        e = torch.sigmoid(self.e_proj(x)) if need_e else None
        return self.key_proj(x), s, e


class SensoryUpdater(nn.Module):
    """decoder の 3 段の特徴から感覚の記憶を GRU 型の門で更新する。"""

    def __init__(self, g_dims: list[int], mid: int, sensory: int):
        super().__init__()
        self.sensory = sensory
        self.g16, self.g8, self.g4 = GConv2D(g_dims[0], mid, 1), GConv2D(g_dims[1], mid, 1), GConv2D(g_dims[2], mid, 1)
        self.transform = GConv2D(mid + sensory, sensory * 3, 3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16(g[0]) + self.g8(downsample_groups(g[1], 0.5)) + self.g4(downsample_groups(g[2], 0.25))
        return _gru(self.transform(torch.cat([g, h], 2)), h, self.sensory)


class SensoryDeepUpdater(nn.Module):
    """マスクの encoder の特徴から感覚の記憶を更新する(deep update)。"""

    def __init__(self, f_dim: int, sensory: int):
        super().__init__()
        self.sensory = sensory
        self.transform = GConv2D(f_dim + sensory, sensory * 3, 3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, f, h):
        return _gru(self.transform(torch.cat([f, h], 2)), h, self.sensory)


def _gru(values: torch.Tensor, h: torch.Tensor, d: int) -> torch.Tensor:
    forget, update = torch.sigmoid(values[:, :, :d]), torch.sigmoid(values[:, :, d: 2 * d])
    new = torch.tanh(values[:, :, 2 * d:])
    return forget * h * (1 - update) + update * new


class MaskUpsampleBlock(nn.Module):
    def __init__(self, up_dim: int, out_dim: int):
        super().__init__()
        self.out_conv = GroupResBlock(up_dim, out_dim)

    def forward(self, skip, up_g):
        return self.out_conv(distribute(skip, upsample_groups(up_g, 2.0), "add"))


class LinearPredictor(nn.Module):
    """感覚の記憶から、画像の特徴との線形の組み合わせで補助の予測を出す。"""

    def __init__(self, in_dim: int, pred_dim: int):
        super().__init__()
        self.projection = GConv2D(in_dim, pred_dim + 1, 1)

    def forward(self, im_feat, pred_feat):
        p = self.projection(pred_feat)
        im = im_feat.unsqueeze(1).expand(-1, pred_feat.shape[1], -1, -1, -1)
        return (im * p[:, :, :-1]).sum(2, keepdim=True) + p[:, :, -1:]


# ---- 大きな部品(公式の big_modules) ----
class PixelEncoder(nn.Module):
    """画像の encoder(exp の backbone。stride 4・8・16)。Returns: ((f16 の射影 (B, 512, H, W), f8, f4), key 用の f16 の射影)。

    階層型の backbone(ResNet など)は stride 16 までの段だけを作る(公式の ResNet-50 も layer3 まで)。素の ViT の adapter は
    stride 4〜32 の 4 段か 8〜32 の 3 段しか作れないので、4 段を作って stride 32 の段は使わない。
    """

    def __init__(self, backbone: str, img_size: int, pretrained: bool = True):
        super().__init__()
        try:
            self.backbone, self.drop_last = create_feature_backbone(backbone, pretrained, img_size=img_size, strides=(4, 8, 16)), False
        except ValueError:  # 素の ViT(vit_adapter.py)
            self.backbone, self.drop_last = create_feature_backbone(backbone, pretrained, img_size=img_size, strides=(4, 8, 16, 32)), True
        c4, c8, c16 = self.backbone.feature_info.channels()[:3]
        self.channels = (c16, c8, c4)
        self.proj1, self.proj2 = nn.Conv2d(c16, PIX_DIM, 1), nn.Conv2d(c16, PIX_DIM, 1)

    def forward(self, x):
        f4, f8, f16 = self.backbone(x)[:3]
        return (self.proj1(f16), f8, f4), self.proj2(f16)


class MaskEncoder(nn.Module):
    """マスクの encoder(ResNet-18、入力は画像 + マスク)+ 画素の特徴と混ぜる block + 感覚の記憶の deep update。"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        r = timm.create_model("resnet18.tv_in1k", pretrained=pretrained)
        w = r.conv1.weight.data  # (64, 3, 7, 7)
        self.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        extra = torch.zeros(64, 1, 7, 7)
        nn.init.orthogonal_(extra)  # 公式の load_weights_add_extra_dim
        self.conv1.weight.data = torch.cat([w, extra], 1)
        self.bn1, self.relu, self.maxpool = r.bn1, r.act1, r.maxpool
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3
        self.fuser = GroupFeatureFusionBlock(PIX_DIM, 256, VALUE_DIM, VALUE_DIM)
        self.sensory_update = SensoryDeepUpdater(VALUE_DIM, VALUE_DIM)

    def forward(self, image, ms_features, sensory, masks, is_deep_update: bool = True):
        """Args: image (B, 3, 16H, 16W)、masks (B, N, 16H, 16W)、sensory (B, N, 512, H, W)。Returns: (value (B, N, 512, H, W), 感覚の記憶)。"""
        g = distribute(image, masks.unsqueeze(2))  # (B, N, 4, ·, ·)
        b, n = g.shape[:2]
        g = g.flatten(0, 1)
        g = self.relu(self.maxpool(self.bn1(self.conv1(g))))  # 公式と同じ順(maxpool の後に ReLU)
        g = self.layer3(self.layer2(self.layer1(g)))
        g = self.fuser(ms_features[0], g.view(b, n, *g.shape[1:]))
        new_sensory = self.sensory_update(g, sensory) if is_deep_update else sensory
        return g, new_sensory


class MaskDecoder(nn.Module):
    """マスクの decoder。Args の dims は画素の encoder の (c16, c8, c4)。"""

    def __init__(self, dims: tuple[int, int, int]):
        super().__init__()
        _, c8, c4 = dims
        self.fuser = GroupFeatureFusionBlock(PIX_DIM, VALUE_DIM, VALUE_DIM, VALUE_DIM)
        self.sensory_compress = GConv2D(VALUE_DIM + 1, VALUE_DIM, 1)
        self.sensory_update = SensoryUpdater([VALUE_DIM, 256, 256 + 1], 512, 512)
        self.decoder_feat_proc = nn.ModuleList([nn.Conv2d(c8, VALUE_DIM, 1), nn.Conv2d(c4, 256, 1)])  # 公式: [512, 256] → [val, 256]
        self.up_16_8, self.up_8_4 = MaskUpsampleBlock(VALUE_DIM, 256), MaskUpsampleBlock(256, 256)
        self.pred = nn.Conv2d(256, 1, 3, padding=1)
        self.sensory_linear_pred = LinearPredictor(VALUE_DIM, PIX_DIM)

    def forward(self, ms_features, readout, sensory, last_mask, need_aux: bool = False, update_sensory: bool = True):
        f16, f8, f4 = ms_features
        b, n = readout.shape[:2]
        aux = self.sensory_linear_pred(f16, sensory) if need_aux else None  # (B, N, 1, H, W)
        d8, d4 = self.decoder_feat_proc[0](f8), self.decoder_feat_proc[1](f4)
        p16 = self.fuser(f16, readout + self.sensory_compress(torch.cat([sensory, last_mask], 2)))
        p8 = self.up_16_8(d8, p16)
        p4 = self.up_8_4(d4, p8)
        with torch.autocast("cuda", enabled=False):
            logits = self.pred(F.relu(p4.flatten(0, 1).float()))  # (B·N, 1, 4H, 4W)
        if update_sensory:
            p4 = torch.cat([p4, logits.view(b, n, 1, *logits.shape[-2:]).to(p4.dtype)], 2)
            sensory = self.sensory_update([p16, p8, p4], sensory)
        return sensory, logits.view(b, n, *logits.shape[-2:]), aux


# ---- 記憶の読み出し(公式の memory_utils) ----
def get_similarity(mk, ms, qk, qe):
    """記憶の key と今の key の類似度。Args: mk (B, CK, N)、ms (B, 1, N) か None、qk・qe (B, CK, HW)。Returns: (B, N, HW)。"""
    ck = mk.shape[1]
    if qe is not None:
        mk_t = mk.transpose(1, 2)
        sim = -(mk_t.pow(2) @ qe) + 2 * (mk_t @ (qk * qe)) - (qe * qk.pow(2)).sum(1, keepdim=True)
    else:
        sim = -mk.pow(2).sum(1).unsqueeze(2) + 2 * (mk.transpose(1, 2) @ qk)
    if ms is not None:
        sim = sim * ms.transpose(1, 2)
    return sim / math.sqrt(ck)


def do_softmax(sim, top_k: int | None = None, return_usage: bool = False):
    """記憶の軸(dim 1)の softmax(top_k があれば上位 k 個だけ)。return_usage なら記憶ごとの重みの和も返す。"""
    if top_k is not None:
        values, idx = sim.topk(k=min(top_k, sim.shape[1]), dim=1)
        x = values.exp()
        x = x / x.sum(1, keepdim=True)
        aff = torch.zeros_like(sim).scatter_(1, idx, x)
    else:
        x = torch.exp(sim - sim.max(1, keepdim=True)[0])
        aff = x / x.sum(1, keepdim=True)
    return (aff, aff.sum(2)) if return_usage else aff


class DEVA(nn.Module):
    """DEVA の伝搬のネットワーク。

    Args:
        backbone: 画素の encoder の幹(timm の名前)。img_size: 入力の一辺。pretrained: 事前学習の重みを読むか。
    """

    def __init__(self, backbone: str, img_size: int, pretrained: bool = True):
        super().__init__()
        self.pixel_encoder = PixelEncoder(backbone, img_size, pretrained)
        self.mask_encoder = MaskEncoder(pretrained)
        self.key_proj = KeyProjection(PIX_DIM, KEY_DIM)
        self.mask_decoder = MaskDecoder(self.pixel_encoder.channels)

    @staticmethod
    def aggregate(prob: torch.Tensor, dim: int) -> torch.Tensor:
        """物体ごとの確率に背景(全物体の (1 − p) の積)を足し、logit にする(公式の soft aggregation)。"""
        prob = prob.float()
        p = torch.cat([torch.prod(1 - prob, dim=dim, keepdim=True), prob], dim).clamp(1e-7, 1 - 1e-7)
        return torch.log(p / (1 - p))

    def encode_image(self, image):
        return self.pixel_encoder(image)

    def transform_key(self, feat, need_sk: bool = True, need_ek: bool = True):
        return self.key_proj(feat, need_sk, need_ek)

    def encode_mask(self, image, ms_features, sensory, masks, is_deep_update: bool = True):
        return self.mask_encoder(image, ms_features, sensory, masks, is_deep_update)

    def read_memory(self, qk, qe, mk, ms, mv):
        """Args: qk・qe (B, CK, H, W)、mk (B, CK, T, H, W)、ms (B, 1, T, H, W)、mv (B, N, CV, T, H, W)。Returns: (B, N, CV, H, W)。"""
        b, n, cv = mv.shape[:3]
        aff = do_softmax(get_similarity(mk.flatten(2), ms.flatten(2) if ms is not None else None, qk.flatten(2),
                                        qe.flatten(2) if qe is not None else None))
        mem = torch.bmm(mv.flatten(1, 2).flatten(2), aff.to(mv.dtype))  # (B, N·CV, T·HW) × (B, T·HW, HW)
        return mem.view(b, n, cv, *qk.shape[-2:])

    def segment(self, ms_features, readout, sensory, last_mask, selector=None, need_aux: bool = False,
                update_sensory: bool = True):
        """マスクを出す。Args: last_mask (B, N, 16H, 16W)(前のマスク。stride 16 に area で縮めて使う)。

        Returns: (感覚の記憶, 背景つきの logit (B, N+1, 4·4H, ·), 背景つきの確率, [補助の logit, 補助の確率])。
        """
        last = F.interpolate(last_mask, size=readout.shape[-2:], mode="area").unsqueeze(2)
        sensory, logits, aux = self.mask_decoder(ms_features, readout, sensory, last, need_aux, update_sensory)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
        logits = F.interpolate(self.aggregate(prob, 1), scale_factor=4, mode="bilinear", align_corners=False)
        prob = F.softmax(logits, dim=1)
        if not need_aux:
            return sensory, logits, prob
        aux_prob = torch.sigmoid(aux)
        if selector is not None:
            aux_prob = aux_prob * selector.unsqueeze(2)
        aux_logits = upsample_groups(self.aggregate(aux_prob, 1), 16.0)
        return sensory, logits, prob, aux_logits, F.softmax(aux_logits, dim=1)
