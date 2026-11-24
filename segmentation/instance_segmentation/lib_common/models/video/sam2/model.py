"""SAM 2(Ravi ら ICLR 2025、"SAM 2: Segment Anything in Images and Videos")のモデル本体。exp00502。

参考: 公式のコード(facebookresearch/sam2 の `sam2/modeling/sam2_base.py`・`backbones/image_encoder.py`・`training/model/sam2.py`・
`sam2/utils/misc.py`、設定 `sam2.1_hiera_b+.yaml`・`sam2.1_training/..._MOSE_finetune.yaml`、`build_sam.py` の推論の設定)。
コードは写さず、規則を読み取って書いた。設定は公式の SAM 2.1(b+)と同じ(下の SAM2.__init__ の値)。

公式との違い(exp の条件):
    - 画像の encoder の幹: Hiera の代わりに exp の backbone(DINOv3 ViT-B の ViT-Adapter など。lib_common/backbones)。neck は公式と同じ
      FPN(stride 4・8・16・32 の 4 段を 256 次元に。上から足すのは stride 16・32 の段だけ、最も粗い段は捨てる)
    - 入力の一辺: 1,024 の代わりに exp の 512(画像の特徴は 32 × 32)

MOT(動画全体での物体の ID の管理・識別)の形: **物体ごとの記憶で伝搬する形**。物体 1 つ = 1 本の追跡(与えたマスク・点の物体を、
過去のフレームの記憶(マスクの記憶の特徴 7 フレーム分 + 物体のポインタ 16 個)への注意で次のフレームに運ぶ)。ID は追跡の本ごとに固定。
物体が隠れると物体らしさの値が負になり(マスクを出さない・記憶に「物体なし」の埋め込みを足す)、記憶から再び出たときに戻る。
新しい物体の追加・追跡の終了は、このモデルの外の組み立て(sam2/tracker.py。SAM 3 の照合の規則)が行う。

形の記号: B = 物体の数(追跡の本の数)、C = 256、M = 記憶の次元(64)、H・W = 画像の特徴の縦・横(入力の 1/16)、S = 入力の一辺。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from segmentation.instance_segmentation.lib_common.backbones import create_feature_backbone
from segmentation.instance_segmentation.lib_common.models.video.sam2.blocks import (MLP, PositionEmbeddingSine, get_1d_sine_pe,
                                                                                    select_closest_cond_frames)
from segmentation.instance_segmentation.lib_common.models.video.sam2.memory import MemoryAttention, MemoryEncoder
from segmentation.instance_segmentation.lib_common.models.video.sam2.sam_heads import MaskDecoder, PromptEncoder

NO_OBJ_SCORE = -1024.0  # 物体がないときのマスクの値(公式と同じ)


class ImageEncoder(nn.Module):
    """画像の encoder(exp の backbone + 公式の FpnNeck。scalp 1 で最も粗い段を捨てる)。

    Args:
        backbone: timm のモデルの名前(stride 4・8・16・32 の 4 段を作る)。
        image_size: 入力の一辺。
        pretrained: 事前学習の重みを読むか。
        dim: FPN の出力のチャンネル(公式は 256)。
    """

    def __init__(self, backbone: str, image_size: int, pretrained: bool = True, dim: int = 256):
        super().__init__()
        self.trunk = create_feature_backbone(backbone, pretrained, img_size=image_size, strides=(4, 8, 16, 32))
        chans = self.trunk.feature_info.channels()  # stride 4・8・16・32 の順
        self.convs = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in chans)
        self.top_down = (2, 3)  # 公式の fpn_top_down_levels(stride 16・32 の段)
        self.pos = PositionEmbeddingSine(dim)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """backbone の 4 段を FPN で混ぜ(上から足すのは stride 16・32 の段だけ。公式の fpn_top_down_levels)、最も粗い段を捨てる。

        Args:
            x: (B, 3, S, S)。

        Returns:
            ([stride 4, 8, 16 の特徴 (B, C, ·, ·)], [同じ段の位置の埋め込み])。
        """
        xs = self.trunk(x)
        out, prev = [None] * len(xs), None
        for i in range(len(xs) - 1, -1, -1):
            lat = self.convs[i](xs[i])
            if i in self.top_down and prev is not None:
                prev = lat + F.interpolate(prev.float(), scale_factor=2.0, mode="nearest").to(lat.dtype)
            else:
                prev = lat
            out[i] = prev
        out = out[:-1]  # scalp 1
        return out, [self.pos(f).to(f.dtype) for f in out]


def fill_holes(mask: torch.Tensor, max_area: int = 8) -> torch.Tensor:
    """マスクの値の小さな穴(値 ≤ 0 の連結成分のうち面積 max_area 以下)を 0.1 で埋める(公式の fill_holes_in_mask_scores。推論だけ)。

    Args:
        mask: マスクの値 (B, 1, h, w)。
        max_area: 埋める穴の面積の上限(画素)。

    Returns:
        穴を埋めたマスクの値 (B, 1, h, w)。
    """
    import cv2
    out = mask.clone()
    m = (mask <= 0).cpu().numpy().astype(np.uint8)
    for b in range(m.shape[0]):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m[b, 0], connectivity=8)
        small = np.isin(labels, [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] <= max_area])
        if small.any():
            out[b, 0][torch.as_tensor(small, device=mask.device)] = 0.1
    return out


class SAM2(nn.Module):
    """SAM 2(記憶つきの、物体ごとのマスクの伝搬)。

    Args:
        backbone: exp の backbone の名前(timm)。
        image_size: 入力の一辺 S。
        pretrained: backbone に事前学習の重みを読むか。
    """

    # 公式の SAM 2.1 b+ の設定
    num_maskmem = 7
    sigmoid_scale_for_mem_enc, sigmoid_bias_for_mem_enc = 20.0, -10.0
    max_obj_ptrs_in_encoder = 16
    max_cond_frames_in_attn = -1
    multimask_min_pt_num, multimask_max_pt_num = 0, 1

    def __init__(self, backbone: str, image_size: int = 512, pretrained: bool = True):
        super().__init__()
        c, m = 256, 64
        self.hidden_dim, self.mem_dim, self.image_size = c, m, image_size
        self.image_encoder = ImageEncoder(backbone, image_size, pretrained, c)
        self.memory_attention = MemoryAttention(c, m, 4)
        self.memory_encoder = MemoryEncoder(c, m)
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(self.num_maskmem, 1, 1, m))
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, c))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, c))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, c))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, m))
        for p in (self.maskmem_tpos_enc, self.no_mem_embed, self.no_mem_pos_enc, self.no_obj_ptr, self.no_obj_embed_spatial):
            nn.init.trunc_normal_(p, std=0.02)
        self.mask_downsample = nn.Conv2d(1, 1, 4, stride=4)  # 与えたマスクから物体のポインタを作るとき
        e = image_size // 16
        self.sam_prompt_encoder = PromptEncoder(c, (e, e), (image_size, image_size), 16)
        self.sam_mask_decoder = MaskDecoder(c)
        self.obj_ptr_proj = MLP(c, c, c, 3)
        self.obj_ptr_tpos_proj = nn.Linear(c, m)
        # 推論だけの設定(公式の build_sam2_video_predictor): 単一のマスクが不安定なら複数マスクの最良、点から作ったマスクを
        # 記憶に入れるときは 0/1 にする、小さな穴を埋める
        self.binarize_mask_from_pts_for_mem_enc, self.fill_hole_area = True, 8

    # ---- 画像 ----
    def forward_image(self, frames: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """画像の特徴を作る(高解像度の 2 段は、マスクの decoder の conv_s0・conv_s1 を先にかける。公式と同じ)。

        Args:
            frames: (B, 3, S, S)。

        Returns:
            ([stride 4(conv_s0 の後), 8(conv_s1 の後), 16 の特徴], [位置の埋め込み])。
        """
        fpn, pos = self.image_encoder(frames)
        fpn[0] = self.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.sam_mask_decoder.conv_s1(fpn[1])
        return fpn, pos

    # ---- SAM の head ----
    def _sam_heads(self, feat, points=None, masks=None, high_res=None, multimask=False):
        """プロンプトとマスクの decoder でマスクを出す(公式の _forward_sam_heads)。

        Args:
            feat: 記憶で直した画像の特徴 (B, C, H, W)(stride 16)。
            points: {"point_coords": (B, P, 2), "point_labels": (B, P)} か None(点なし)。
            masks: マスクのプロンプト (B, 1, ·, ·) か None。
            high_res: 高解像度の 2 段の特徴 [stride 4, stride 8]。
            multimask: 複数のマスクを出して IoU の最良を選ぶか。

        Returns:
            (低解像度の複数マスク, 高解像度の複数マスク, IoU, 低解像度の最良のマスク, 高解像度の最良のマスク,
             物体のポインタ (B, C), 物体らしさの値 (B, 1))。
        """
        b, dev = feat.shape[0], feat.device
        if points is not None:
            pts = (points["point_coords"], points["point_labels"])
        else:
            pts = (torch.zeros(b, 1, 2, device=dev), -torch.ones(b, 1, dtype=torch.int32, device=dev))
        prompt = None
        if masks is not None:
            prompt = masks.float()
            if prompt.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                prompt = F.interpolate(prompt, size=self.sam_prompt_encoder.mask_input_size, mode="bilinear",
                                       align_corners=False, antialias=True)
        sparse, dense = self.sam_prompt_encoder(pts, prompt, b)
        low_multi, ious, tokens, obj = self.sam_mask_decoder(feat, self.sam_prompt_encoder.get_dense_pe(), sparse, dense,
                                                             multimask, high_res)
        appearing = obj > 0
        low_multi = torch.where(appearing[:, None, None], low_multi, NO_OBJ_SCORE).float()
        high_multi = F.interpolate(low_multi, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        token = tokens[:, 0]
        if multimask:
            best = ious.argmax(-1)
            ar = torch.arange(b, device=dev)
            low, high = low_multi[ar, best][:, None], high_multi[ar, best][:, None]
            if tokens.shape[1] > 1:
                token = tokens[ar, best]
        else:
            low, high = low_multi, high_multi
        ptr = self.obj_ptr_proj(token)
        lam = appearing.float()  # 公式の fixed_no_obj_ptr(soft_no_obj_ptr は False)
        ptr = lam * ptr + (1 - lam) * self.no_obj_ptr
        return low_multi, high_multi, ious, low, high, ptr, obj

    def _mask_as_output(self, feat, high_res, mask_inputs):
        """与えたマスクを、SAM を通さずにそのまま出力にする(公式の use_mask_input_as_output_without_sam)。

        物体のポインタは、マスクを 1/4 に縮めたものをプロンプトにして SAM の head から作る。

        Args:
            feat: 画像の特徴 (B, C, H, W)(stride 16)。
            high_res: 高解像度の 2 段の特徴。
            mask_inputs: 与えたマスク (B, 1, S, S)。

        Returns:
            _sam_heads と同じ 7 つ組(マスクは与えたマスクを ±10 にしたもの、IoU は 1)。
        """
        mf = mask_inputs.float()
        high = mf * 20.0 - 10.0
        low = F.interpolate(high, size=(high.shape[-2] // 4, high.shape[-1] // 4), mode="bilinear", align_corners=False,
                            antialias=True)
        ious = mf.new_ones(mf.shape[0], 1)
        ptr = self._sam_heads(feat, masks=self.mask_downsample(mf), high_res=high_res)[5]
        lam = (mf.flatten(1) > 0).any(1, keepdim=True).float()
        obj = 20.0 * lam - 10.0
        ptr = lam * ptr + (1 - lam) * self.no_obj_ptr
        return low, high, ious, low, high, ptr, obj

    # ---- 記憶 ----
    def _memory_conditioned(self, t: int, is_init_cond: bool, feat, pos, out_dict: dict, num_frames: int):
        """今のフレームの画像の特徴を記憶で直す(公式の _prepare_memory_conditioned_features。時刻は順方向だけ)。

        Args:
            t: 今の時刻。
            is_init_cond: 最初の条件のフレームか(記憶がないので no_mem_embed を足すだけ)。
            feat: 画像の特徴 (B, C, H, W)(stride 16)。
            pos: その位置の埋め込み (B, C, H, W)。
            out_dict: 過去の出力 {"cond": {t: 出力}, "non_cond": {t: 出力}}。
            num_frames: 動画(区間)のフレームの数(物体のポインタの数の上限に使う)。

        Returns:
            記憶で直した画像の特徴 (B, C, H, W)。
        """
        b, c, h, w = feat.shape
        if is_init_cond:  # 公式の directly_add_no_mem_embed
            return (feat.flatten(2).transpose(1, 2) + self.no_mem_embed).transpose(1, 2).reshape(b, c, h, w)
        mems, mem_pos = [], []
        cond, uncond = select_closest_cond_frames(t, out_dict["cond"], self.max_cond_frames_in_attn)
        t_prevs = [(0, o) for o in cond.values()]
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            prev = t - t_rel  # 学習・推論とも stride 1(公式の memory_temporal_stride_for_eval = 1)
            o = out_dict["non_cond"].get(prev, uncond.get(prev))
            t_prevs.append((t_pos, o))
        for t_pos, o in t_prevs:
            if o is None:
                continue
            mems.append(o["maskmem_features"].flatten(2).transpose(1, 2))  # (B, H·W, M)
            mem_pos.append(o["maskmem_pos_enc"].flatten(2).transpose(1, 2) + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1, 0])
        # 物体のポインタ(過去の最大 16 フレーム。推論は条件のフレームも過去のものだけ)
        max_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
        ptr_cond = cond if self.training else {s: o for s, o in cond.items() if s <= t}
        pos_ptrs = [(t - s, o["obj_ptr"]) for s, o in ptr_cond.items()]  # 符号つきの時刻の差
        for d in range(1, max_ptrs):
            s = t - d
            if s < 0 or s >= num_frames:
                break
            o = out_dict["non_cond"].get(s, uncond.get(s))
            if o is not None:
                pos_ptrs.append((d, o["obj_ptr"]))
        n_ptr_tokens = 0
        if pos_ptrs:
            tpos, ptrs = zip(*pos_ptrs)
            ptrs = torch.stack(ptrs)  # (P, B, C)
            tp = get_1d_sine_pe(torch.tensor(tpos, device=feat.device, dtype=torch.float32) / max(max_ptrs - 1, 1), c)
            tp = self.obj_ptr_tpos_proj(tp)[:, None].expand(-1, b, -1)  # (P, B, M)
            # 256 次元のポインタを 64 次元の 4 トークンに分ける(公式と同じ並べ方)
            ptrs = ptrs.reshape(-1, b, c // self.mem_dim, self.mem_dim).permute(0, 2, 1, 3).flatten(0, 1)  # (P·4, B, M)
            tp = tp.repeat_interleave(c // self.mem_dim, dim=0)
            mems.append(ptrs.transpose(0, 1)); mem_pos.append(tp.transpose(0, 1))
            n_ptr_tokens = ptrs.shape[0]
        memory, memory_pos = torch.cat(mems, dim=1), torch.cat(mem_pos, dim=1)
        curr, curr_pos = feat.flatten(2).transpose(1, 2), pos.flatten(2).transpose(1, 2)
        out = self.memory_attention(curr, memory, curr_pos, memory_pos, n_ptr_tokens)
        return out.transpose(1, 2).reshape(b, c, h, w)

    def _encode_memory(self, feat, high_res_masks, obj_logits, from_pts: bool):
        """予測のマスクから記憶を作る(公式の _encode_new_memory)。

        Args:
            feat: 画像の特徴 (B, C, H, W)(stride 16)。
            high_res_masks: 高解像度のマスクの値 (B, 1, S, S)。
            obj_logits: 物体らしさの値 (B, 1)(物体がなければ no_obj_embed_spatial を足す)。
            from_pts: 点のプロンプトから出したマスクか(推論では 0/1 にしてから記憶にする)。

        Returns:
            (記憶の特徴 (B, M, H, W), 位置の埋め込み)。
        """
        if self.binarize_mask_from_pts_for_mem_enc and from_pts and not self.training:
            m = (high_res_masks > 0).float()
        else:
            m = torch.sigmoid(high_res_masks)
        m = m * self.sigmoid_scale_for_mem_enc + self.sigmoid_bias_for_mem_enc
        mem, mem_pos = self.memory_encoder(feat, m, skip_mask_sigmoid=True)
        appearing = (obj_logits > 0).float()
        mem = mem + (1 - appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None]
        return mem, mem_pos

    def _use_multimask(self, is_init_cond: bool, points) -> bool:
        """複数のマスクを出すか(点の数が multimask_min_pt_num〜multimask_max_pt_num のとき。公式の _use_multimask)。

        Args:
            is_init_cond: 最初の条件のフレームか(公式の multimask_output_in_sam・for_tracking が True なので結果に関係しない)。
            points: 点のプロンプトか None。

        Returns:
            複数のマスクを出すなら True。
        """
        n = 0 if points is None else points["point_labels"].shape[1]
        return self.multimask_min_pt_num <= n <= self.multimask_max_pt_num  # multimask_output_in_sam・for_tracking は True

    def track_step(self, t: int, is_init_cond: bool, feats: list[torch.Tensor], pos: list[torch.Tensor], points, mask_inputs,
                   out_dict: dict, num_frames: int, gt_masks=None, correct: bool = False, run_mem: bool = True,
                   rng: np.random.Generator | None = None, prob_gt_pt: float = 0.0) -> dict:
        """1 フレーム分(公式の SAM2Train.track_step。correct なら修正の点を 7 回足す)。

        Args:
            t: 今の時刻。
            is_init_cond: 条件のフレーム(プロンプトを与えるフレーム)か。
            feats: この物体のまとめの [stride 4, 8, 16] の特徴 (B, ·, ·, ·)。
            pos: 同じ段の位置の埋め込み。
            points: 点のプロンプトか None。
            mask_inputs: マスクのプロンプト (B, 1, S, S) か None。
            out_dict: 過去の出力 {"cond": {t: 出力}, "non_cond": {t: 出力}}。
            num_frames: 動画(区間)のフレームの数。
            gt_masks: 修正の点を選ぶ元の正解 (B, 1, S, S)(correct のとき)。
            correct: 修正の点を 7 回足すか(公式の num_correction_pt_per_frame)。
            run_mem: このフレームの記憶を作るか。
            rng: 修正の点を正解だけから選ぶかの乱数。
            prob_gt_pt: 修正の点を予測の誤りでなく正解だけから選ぶ確率(公式の prob_to_sample_from_gt_for_train)。

        Returns:
            {"pred_masks", "pred_masks_high_res", "obj_ptr", "object_score_logits", "maskmem_features", "maskmem_pos_enc",
             "multistep_*"(損失用)}。
        """
        high_res = feats[:-1]
        if mask_inputs is not None:
            outs = self._mask_as_output(feats[-1], high_res, mask_inputs)
            pix = feats[-1]
        else:
            pix = self._memory_conditioned(t, is_init_cond, feats[-1], pos[-1], out_dict, num_frames)
            outs = self._sam_heads(pix, points, None, high_res, self._use_multimask(is_init_cond, points))
        cur = {"multistep_pred_multimasks_high_res": [outs[1]], "multistep_pred_ious": [outs[2]],
               "multistep_object_score_logits": [outs[6]]}
        if correct:
            for _ in range(7):  # 公式の num_correction_pt_per_frame
                from_gt = self.training and prob_gt_pt > 0 and rng.random() < prob_gt_pt
                pred = None if from_gt else (outs[4] > 0)
                p, l = sample_points_from_errors(gt_masks.bool(), pred)
                points = concat_points(points, p, l)
                outs = self._sam_heads(pix, points, outs[3], high_res, self._use_multimask(is_init_cond, points))
                cur["multistep_pred_multimasks_high_res"].append(outs[1])
                cur["multistep_pred_ious"].append(outs[2])
                cur["multistep_object_score_logits"].append(outs[6])
        low, high, ptr, obj = outs[3], outs[4], outs[5], outs[6]
        cur.update({"pred_masks": low, "pred_masks_high_res": high, "obj_ptr": ptr, "object_score_logits": obj,
                    "point_inputs": points})
        if run_mem:  # 記憶は穴を埋める前のマスクから作る(公式の video predictor と同じ順)
            cur["maskmem_features"], cur["maskmem_pos_enc"] = self._encode_memory(feats[-1], high, obj, points is not None)
        if not self.training and self.fill_hole_area > 0:  # 出力のマスクだけ小さな穴を埋める
            cur["pred_masks"] = fill_holes(low, self.fill_hole_area)
            cur["pred_masks_high_res"] = F.interpolate(cur["pred_masks"], size=(self.image_size, self.image_size),
                                                       mode="bilinear", align_corners=False)
        return cur


def concat_points(old, pts, labels):
    """点のプロンプトに点を足す(公式の concat_points)。

    Args:
        old: 今までの点のプロンプト {"point_coords", "point_labels"} か None。
        pts: 足す点 (B, n, 2)。
        labels: 足す点のラベル (B, n)。

    Returns:
        {"point_coords": (B, P + n, 2), "point_labels": (B, P + n)}。
    """
    if old is None:
        return {"point_coords": pts, "point_labels": labels}
    return {"point_coords": torch.cat([old["point_coords"], pts], 1), "point_labels": torch.cat([old["point_labels"], labels], 1)}


def sample_points_from_errors(gt: torch.Tensor, pred: torch.Tensor | None, num_pt: int = 1):
    """誤りの領域から点を一様に選ぶ(公式の sample_random_points_from_errors。学習の "uniform")。

    Args:
        gt: 正解 (B, 1, S, S) の bool。
        pred: 予測 (B, 1, S, S) の bool か None(空とみなす)。
        num_pt: 選ぶ点の数。

    Returns:
        (点 (B, num_pt, 2) の (x, y), ラベル (B, num_pt)(1 = 正 = 見逃し、0 = 負 = 誤検出))。
    """
    if pred is None:
        pred = torch.zeros_like(gt)
    b, _, h, w = gt.shape
    fp, fn = ~gt & pred, gt & ~pred
    all_ok = (gt == pred).flatten(2).all(2)[..., None, None]  # 全部正しいときは、正解の外から負の点を選ぶ
    noise = torch.rand(b, num_pt, h, w, 2, device=gt.device)
    noise[..., 0] *= (fp | (all_ok & ~gt)).float()
    noise[..., 1] *= fn.float()
    idx = noise.flatten(2).argmax(2)
    labels = (idx % 2).to(torch.int32)
    idx = idx // 2
    return torch.stack([idx % w, idx // w], dim=2).float(), labels


def sample_box_points(masks: torch.Tensor, noise: float = 0.1, noise_bound: int = 20):
    """マスクの箱の左上・右下を、箱の大きさの 10%(最大 20 画素)まで揺らした 2 点にする(公式の sample_box_points)。

    Args:
        masks: (B, 1, S, S)。
        noise: 箱の大きさに対する揺らしの割合。
        noise_bound: 揺らしの上限(画素)。

    Returns:
        (点 (B, 2, 2), ラベル (B, 2)(2 = 左上、3 = 右下))。
    """
    b, _, h, w = masks.shape
    dev = masks.device
    boxes = []
    for m in masks[:, 0]:
        ys, xs = torch.nonzero(m, as_tuple=True)
        boxes.append(torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).float() if len(xs) else torch.zeros(4, device=dev))
    box = torch.stack(boxes)[:, None]  # (B, 1, 4)
    bw, bh = box[..., 2] - box[..., 0], box[..., 3] - box[..., 1]
    bound = torch.tensor(float(noise_bound), device=dev)
    mdx, mdy = torch.minimum(bw * noise, bound), torch.minimum(bh * noise, bound)
    box = box + (2 * torch.rand(b, 1, 4, device=dev) - 1) * torch.stack([mdx, mdy, mdx, mdy], dim=-1)
    lim = torch.tensor([w, h, w, h], device=dev, dtype=torch.float32) - 1
    box = torch.minimum(torch.maximum(box, torch.zeros_like(lim)), lim)
    labels = torch.tensor([2, 3], dtype=torch.int32, device=dev).repeat(b, 1)
    return box.reshape(b, 2, 2), labels
