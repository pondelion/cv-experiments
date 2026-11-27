"""SAM 3 型の動画インスタンスセグメンテーション: 画像のモデルの検出 + SAM 2 の物体ごとの伝搬 + 検出と追跡の照合。exp00502。

参考: 公式のコード(facebookresearch/sam3 の `sam3/model/sam3_video_base.py`(_det_track_one_frame・_associate_det_trk_compilable・
_process_hotstart・_suppress_overlapping_based_on_recent_occlusion・_recondition_masklets・_tracker_update_memories・
_tracker_add_new_objects・build_outputs)、`sam3_video_inference.py`(propagate_in_video・_postprocess_output)、
`sam3_tracking_predictor.py`(_suppress_object_pw_area_shrinkage・_apply_object_wise_non_overlapping_constraints)、
`sam3_tracker_utils.py`(fill_holes_in_mask_scores)、`perflib/nms.py`(nms_masks)、`model_builder.py` の動画の設定)。
コードは写さず、規則を読み取って書いた。設定は公式の動画のモデルの既定(apply_temporal_disambiguation=True)と同じ。

MOT(動画全体での物体の ID の管理・識別)の形: **推論の規則(検出と追跡の照合)+ 物体ごとの記憶の伝搬**。1 本の追跡 = 1 つの ID。
フレームごとに:
    1. 画像のモデルで検出(フレームごとの点数の上位 TOP_K 個、マスクの IoU 0.1 の NMS。公式の点数 > 0.5 の閾値は使わない)
    2. 追跡している物体を SAM 2 で今のフレームに伝搬(記憶はまだ作らない。全部の物体をまとめて計算する)。マスクの小さな穴を埋め、
       小さな点を消す(面積 16)
    3. 照合(マスクの IoU): 追跡は IoU ≥ 0.5 の検出があれば「照合あり」。検出は、どの追跡とも IoU < 0.1 なら新しい物体(公式の
       点数 ≥ 0.7 の閾値は使わない)
    4. 生存のカウンタ(初期 30): 照合ありで +1(上限 30)、照合なし(マスクが空でないのに検出がない)で −1(下限 −1)。
       ホットスタート: 初めて写ってから 15 フレームの間に、照合なしが 8 回、または同じ検出に先に出た追跡と一緒に照合されたことが 8 回あれば、
       その ID を除く(出力からも消す)
    5. 16 フレームごとに、点数 0.8 以上の検出と IoU 0.8 以上で一意に組になった追跡(物体らしさの値 > 0.8)は、検出のマスクを与え直す
    6. 最近隠れた方の物体のマスクを消す(IoU ≥ 0.7 の 2 つの追跡のうち、隠れたのが最近の方)
    7. 記憶を作る(画素ごとに最も値の大きい物体だけを残したとき面積が 0.3 未満に縮む物体は消してから。物体らしさは面積の有無で ±10)
    8. 新しい物体を、検出のマスクを与えた新しい追跡として足す(新しい ID。点数はその検出の点数、クラスはその検出のクラス)。
       同時に追跡する物体の数の上限 MAX_TRACKS(100)まで、点数の高い順(確信度の閾値ではない)
    出力: 除いた ID を隠し、空のマスクを除き、重なる画素は追跡の点数の高い物体に割り当てる。動画ごとに点数の上位 100 個
    (ほかの方式の評価と同じ。AP の評価の前に予測を確信度の閾値で切らない)
クラス: SAM 3 は 1 つの概念(文章)ごとに上を行う。ここでは画像のモデルの決まったクラスごとに同じことを行う(クラスをまたいだ照合はしない)。
生存のカウンタは、公式の既定(suppress_unmatched_only_within_hotstart=True)では ID を除く・隠すことに使われない(数えるだけ)。
ホットスタートの 15 フレームを過ぎた追跡は除かれず、隠れている間は物体らしさの値が負になって空のマスクを出し(出力では空として除く)、
記憶から伝搬が続くので、再び写れば同じ ID に戻る。

公式との違い(exp の条件・私の判断):
    - 検出器: SAM 3 の検出の head の代わりに exp の画像のモデル(ImageSegmenter。Mask2Former・GTR 系)。画像の特徴も SAM 2 と別
    - 追跡の部(SAM 2)は物体ごとに 1 本ずつ track_step を呼ぶ(公式は同じフレームに足した物体をまとめるが、物体ごとの記憶は独立なので
      結果は同じ)
    - 物体の数の上限(max_num_objects)は設けない(公式の既定も、評価で上限にかからない大きさ)

形の記号: T = フレームの数、R = 検出の数、M = 追跡の数、h・w = マスクの縦・横(入力の 1/4)、S = 入力の一辺、H・W = 元の大きさ。
"""
import numpy as np
import torch
import torch.nn.functional as F

from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter
from segmentation.instance_segmentation.lib_common.models.video.sam2.model import SAM2

# 公式の動画のモデルの既定(model_builder.py の apply_temporal_disambiguation=True)。ただし、AP の評価の前に予測を確信度の閾値で
# 切らないため、検出の確信度の閾値(公式 0.5)と新しい物体の確信度の閾値(公式 0.7)は使わない(私の判断。ほかの方式の評価と同じ条件)
NO_SCORE_THRESHOLD, DET_NMS_THRESH = float("-inf"), 0.1
ASSOC_IOU_THRESH, TRK_ASSOC_IOU_THRESH = 0.1, 0.5
MAX_TRACKS = 100  # 同時に追跡する物体の数の上限(出力の上位 100 個と同じ数。確信度の閾値ではない)
TOP_K_OUT = 100  # 動画ごとに出す予測の数(ほかの方式の評価と同じ)
HOTSTART_DELAY, HOTSTART_UNMATCH_THRESH, HOTSTART_DUP_THRESH = 15, 8, 8
INIT_TRK_KEEP_ALIVE, MAX_TRK_KEEP_ALIVE, MIN_TRK_KEEP_ALIVE = 30, 30, -1
SUPPRESS_OVERLAP_OCCLUSION_THRESH, FILL_HOLE_AREA, RECONDITION_EVERY = 0.7, 16, 16
HIGH_CONF_THRESH, IOU_THRESH_RECONDITION, SHRINK_THRESH = 0.8, 0.8, 0.3


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2 組の 0/1 のマスクの IoU。

    Args:
        a: (N, h, w) の bool。
        b: (M, h, w) の bool。

    Returns:
        (N, M)(和が 0 の組は 0)。
    """
    a, b = a.flatten(1).float(), b.flatten(1).float()
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter))


def nms_masks(scores: torch.Tensor, masks: torch.Tensor, prob_thresh: float, iou_thresh: float) -> torch.Tensor:
    """点数 > prob_thresh の検出の、マスクの IoU での NMS(公式の nms_masks・generic_nms。IoU > 閾値を消す)。

    Args:
        scores: 検出の点数 (R,)。
        masks: 検出のマスクの値 (R, h, w)。
        prob_thresh: 使う検出の点数の閾値。
        iou_thresh: 消す IoU の閾値。

    Returns:
        残す検出 (R,) の bool。
    """
    valid = scores > prob_thresh
    keep = torch.zeros_like(valid)
    idx = torch.nonzero(valid).squeeze(-1)
    if len(idx) == 0:
        return keep
    iou = mask_iou(masks[idx] > 0, masks[idx] > 0)
    order = scores[idx].argsort(descending=True)
    removed = torch.zeros(len(idx), dtype=torch.bool, device=scores.device)
    for i in order.tolist():
        if removed[i]:
            continue
        keep[idx[i]] = True
        removed |= iou[i] > iou_thresh
    return keep


def fill_holes_and_sprinkles(mask: torch.Tensor, max_area: int = FILL_HOLE_AREA) -> torch.Tensor:
    """マスクの小さな穴を埋め、小さな点を消す(公式の SAM 3 の fill_holes_in_mask_scores)。

    値 ≤ 0 の連結成分で面積 ≤ max_area を 0.1 に(穴埋め)、値 > 0 の連結成分で面積 ≤ min(全体の面積 // 2, max_area) を
    −0.1 に(小さな点を消す)。連結は 8 近傍。

    Args:
        mask: マスクの値 (M, h, w)。
        max_area: 対象の面積の上限(画素。0 以下なら何もしない)。

    Returns:
        (M, h, w)。
    """
    import cv2
    if max_area <= 0 or mask.numel() == 0:
        return mask
    out = mask.clone()

    def small(m: np.ndarray, thresh: int) -> np.ndarray:
        """面積が thresh 以下の連結成分の画素。

        Args:
            m: (h, w) の bool。
            thresh: 面積の上限。

        Returns:
            (h, w) の bool。
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        return m & (areas[labels] <= thresh)

    # 全部の物体のマスクを 1 回で CPU に移して調べ、穴・点の画素を 1 回で GPU に戻す(物体ごとに転送しない。結果は同じ:
    # 穴を 0.1 にした後の前景(値 > 0)で点を調べる)
    bg_all = (mask <= 0).cpu().numpy()  # (M, h, w)
    holes = np.zeros_like(bg_all)
    sprinkles = np.zeros_like(bg_all)
    for i in range(mask.shape[0]):
        holes[i] = small(bg_all[i], max_area)
        fg = ~bg_all[i] | holes[i]
        sprinkles[i] = small(fg, min(int(fg.sum()) // 2, max_area))
    out = torch.where(torch.as_tensor(holes, device=mask.device), torch.full_like(out, 0.1), out)
    return torch.where(torch.as_tensor(sprinkles, device=mask.device), torch.full_like(out, -0.1), out)


_KEEP = ("maskmem_features", "maskmem_pos_enc", "obj_ptr", "_mem_tok", "_mem_pos_tok")  # 記憶の取り出し(_gather_memory)が使う値


def _slim(out_dict: dict, t: int, window: int) -> None:
    """1 本の追跡の過去の出力から、記憶の取り出しに使わない値と、使わない古いフレームを消す(結果は変わらない)。

    記憶の取り出しは、条件のフレーム(全部)と、直近の条件でないフレーム(記憶は 6 フレーム、物体のポインタは window − 1 フレーム)
    だけを使う。記憶を作った後の出力の、マスク・損失のための値も使わない。

    Args:
        out_dict: {"cond": {t: 出力}, "non_cond": {t: 出力}}(その場で書き換える)。
        t: 今のフレームの番号(記憶を作り終えたフレーム)。
        window: 物体のポインタを使う過去のフレームの数(SAM2.max_obj_ptrs_in_encoder)。
    """
    for store in (out_dict["cond"], out_dict["non_cond"]):
        for s in list(store):
            o = store[s]
            if "maskmem_features" in o:  # 記憶を作り終えた出力は、記憶の取り出しに使う値だけ残す
                for k in [k for k in o if k not in _KEEP]:
                    del o[k]
    for s in [s for s in out_dict["non_cond"] if s <= t - window]:
        del out_dict["non_cond"][s]


def _split_out(cur: dict, j: int, n: int) -> dict:
    """物体をまとめて計算した track_step の出力から、j 番目の物体の分(batch 1)を取り出す。"""
    out = {}
    for k, v in cur.items():
        if torch.is_tensor(v):
            out[k] = v[j: j + 1] if v.shape[0] == n else v
        elif isinstance(v, list):
            out[k] = [x[j: j + 1] if torch.is_tensor(x) and x.shape[0] == n else x for x in v]
        else:
            out[k] = v
    return out


class _Track:
    """1 本の追跡(1 つの ID)。

    Args:
        oid: ID。
        label: クラス(0 始まり)。
        score: 始めた検出の点数。
        first: 始めたフレームの番号。
    """

    def __init__(self, oid: int, label: int, score: float, first: int):
        self.oid, self.label, self.score, self.first = oid, label, score, first
        self.out_dict = {"cond": {}, "non_cond": {}}
        self.keep_alive, self.unmatched_frames, self.last_occluded = INIT_TRK_KEEP_ALIVE, [], -1
        self.removed = False


class SAM3StyleVIS(torch.nn.Module):
    """SAM 3 型の組み立て(上の規則)。

    Args:
        sam2: 学習した SAM2(伝搬の部)。
        detector: 学習した画像のモデル(ImageSegmenter)。
    """

    def __init__(self, sam2: SAM2, detector: ImageSegmenter):
        super().__init__()
        self.sam2, self.detector = sam2, detector
        self.sam2.fill_hole_area = 0  # 公式の SAM 3 の追跡の部の設定(穴埋めは上の fill_holes_and_sprinkles で行う)
        self.activation = "sigmoid"  # (後処理には渡さない。predict_video は ID ごとの予測をそのまま返す)

    @torch.no_grad()
    def track_video(self, frames: torch.Tensor, valid=None, orig=None, chunk: int = 4) -> list[dict]:
        """1 本の動画を処理する。

        Args:
            frames: (T, 3, S, S)(評価の変換)。
            valid: 使わない(DEVA と呼び方をそろえるため。元の大きさへは to_predictions で直す)。
            orig: 使わない(同上)。
            chunk: 画像の特徴をまとめて作るフレームの数。

        Returns:
            ID ごとの {"label", "score", "masks": {t: マスクの値 (h, w)}}(除いた ID を除く)。出力のときの重なりの処理は
            to_predictions で行う(元の大きさで)。
        """
        t_all, s = len(frames), frames.shape[-1]
        # 物体のポインタの位置の埋め込みの使い回し(SAM2._gather_memory)は、動画ごとに作り直す(学習で重みが変わった後に古い値を使わない)
        self.sam2.__dict__["_tp_cache"] = {}
        tracks: list[_Track] = []
        next_id = 0
        results = {}  # oid -> {t: マスクの値 (h, w)}
        frame_scores = {}  # t -> {oid: 追跡の点数(確率)}
        overlap_pairs = {}  # (先に出た oid, oid) -> [フレーム]
        for c0 in range(0, t_all, chunk):
            fpn, pos = self.sam2.forward_image(frames[c0: c0 + chunk])
            det_out = self.detector(frames[c0: c0 + chunk])
            dets_chunk = self.detector.detections(det_out)
            for j in range(len(dets_chunk)):
                t = c0 + j
                feats = [f[j: j + 1] for f in fpn]
                ps = [p[j: j + 1] for p in pos]
                det = dets_chunk[j]
                frame_scores[t] = {}
                # 2. 伝搬(記憶はまだ作らない)
                alive = [tr for tr in tracks if not tr.removed]
                trk_low, trk_obj = [], []
                # 追跡している全部の物体をまとめて伝搬する(公式の動画の推論と同じく物体を batch の向きに並べる。物体ごとの記憶は独立)
                curs = self.sam2.propagate_batched(t, feats, ps, [tr.out_dict for tr in alive], t_all) if alive else []
                for tr, cur in zip(alive, curs):
                    tr.out_dict["non_cond"][t] = cur
                    trk_low.append(cur["pred_masks"][0, 0].float())
                if curs:
                    trk_obj = torch.cat([cur["object_score_logits"] for cur in curs]).flatten().tolist()
                trk_low = (fill_holes_and_sprinkles(torch.stack(trk_low)) if trk_low
                           else torch.zeros(0, *det["masks"].shape[-2:], device=frames.device))
                low_h, low_w = trk_low.shape[-2:]
                new_tracks = []
                for c in det["labels"].unique().tolist() + sorted({tr.label for tr in alive} - set(det["labels"].tolist())):
                    # 1. このクラスの検出(点数 > 0.5、NMS 0.1)
                    in_c = det["labels"] == c
                    d_masks, d_scores = det["masks"][in_c], det["scores"][in_c]
                    if d_masks.shape[-2:] != (low_h, low_w) and len(d_masks):
                        d_masks = F.interpolate(d_masks[:, None], size=(low_h, low_w), mode="bilinear", align_corners=False)[:, 0]
                    keep = nms_masks(d_scores, d_masks, NO_SCORE_THRESHOLD, DET_NMS_THRESH)  # 確信度で切らない(NMS だけ)
                    d_masks, d_scores = d_masks[keep], d_scores[keep]
                    ti = [k for k, tr in enumerate(alive) if tr.label == c]
                    t_masks = trk_low[ti]
                    # 3. 照合
                    new_det, unmatched, det_to_trk, recond = self._associate(d_masks, d_scores, t_masks)
                    # 4. 生存のカウンタとホットスタート
                    matched = set()
                    for ks in det_to_trk.values():
                        matched.update(ti[k] for k in ks)
                    for k in matched:
                        alive[k].keep_alive = min(MAX_TRK_KEEP_ALIVE, alive[k].keep_alive + 1)
                    for k in unmatched:
                        tr = alive[ti[k]]
                        tr.unmatched_frames.append(t)
                        tr.keep_alive = max(MIN_TRK_KEEP_ALIVE, tr.keep_alive - 1)
                    for ks in det_to_trk.values():  # 同じ検出に照合された追跡: 先に出たもの以外に重なりを記録する
                        if len(ks) < 2:
                            continue
                        trs = [alive[ti[k]] for k in ks]
                        first = min(trs, key=lambda x: x.first)
                        for tr in trs:
                            if tr is not first:
                                overlap_pairs.setdefault((first.oid, tr.oid), []).append(t)
                    hot_from = t - HOTSTART_DELAY
                    for k in ti:
                        tr = alive[k]
                        within = tr.first > hot_from
                        if within and len(tr.unmatched_frames) >= HOTSTART_UNMATCH_THRESH:
                            tr.removed = True
                    for (fo, oid), fr in overlap_pairs.items():
                        tr = next((x for x in alive if x.oid == oid and x.label == c), None)
                        if tr is not None and not tr.removed and tr.first > hot_from and len(fr) >= HOTSTART_DUP_THRESH:
                            tr.removed = True
                    # 5. 再条件付け(16 フレームごと)
                    if RECONDITION_EVERY > 0 and t % RECONDITION_EVERY == 0:
                        for k, d in recond.items():
                            tr = alive[ti[k]]
                            if trk_obj[ti[k]] > HIGH_CONF_THRESH and not tr.removed:
                                m = F.interpolate(d_masks[d][None, None], size=(s, s), mode="bilinear", align_corners=False) > 0
                                tr.out_dict["non_cond"].pop(t, None)
                                tr.out_dict["cond"][t] = self.sam2.track_step(t, True, feats, ps, None, m, tr.out_dict, t_all)
                                tr.reconditioned_mask = d_masks[d].clone()
                    # 8.(準備)新しい物体の候補(クラス, 確信度, 検出のマスク)。追跡の数の上限は、全部のクラスを集めてから確信度の順に決める
                    for d in new_det:
                        new_tracks.append((c, float(d_scores[d]), d_masks[d]))
                    # 6. 最近隠れた方の物体のマスクを消す(このクラスの中)
                    self._suppress_recent_occlusion(trk_low, [alive[k] for k in ti], ti, t)
                # 7. 記憶を作る(クラスごとに面積の縮みの抑制をしてから)
                if alive:
                    high = F.interpolate(trk_low[:, None], size=(s, s), mode="bilinear", align_corners=False)  # (M, 1, S, S)
                    for c in sorted({tr.label for tr in alive}):
                        ks = [k for k, tr in enumerate(alive) if tr.label == c]
                        high[ks] = self._suppress_shrinkage(high[ks])
                    obj = torch.where((high > 0).flatten(1).any(-1, keepdim=True), 10.0, -10.0)  # (M, 1)
                    # 全部の物体の記憶をまとめて 1 回で作る(物体ごとに独立な計算)
                    mem, mem_pos = self.sam2._encode_memory(feats[-1].expand(len(alive), -1, -1, -1), high, obj, False)
                    for k, tr in enumerate(alive):
                        store = tr.out_dict["cond"] if t in tr.out_dict["cond"] else tr.out_dict["non_cond"]
                        store[t]["maskmem_features"] = mem[k: k + 1]
                        store[t]["maskmem_pos_enc"] = mem_pos[k: k + 1] if mem_pos.shape[0] == len(alive) else mem_pos
                # 出力(既存の追跡 → 再条件付けは検出のマスク)
                for k, tr in enumerate(alive):
                    if tr.removed:
                        continue
                    m = getattr(tr, "reconditioned_mask", None)
                    results.setdefault(tr.oid, {})[t] = (m if m is not None else trk_low[k]).clone()
                    tr.reconditioned_mask = None
                    frame_scores[t][tr.oid] = float(torch.sigmoid(torch.tensor(trk_obj[k])))
                # 8. 新しい物体を足す(検出のマスクを条件のフレームとして与え、記憶を作る)。同時に追跡する物体の数の上限 MAX_TRACKS まで、
                # 確信度の高い順(確信度の閾値は使わない)。全部の新しい物体をまとめて 1 回で始める(記憶を使わない計算なので独立)
                room = MAX_TRACKS - sum(not tr.removed for tr in tracks)
                new_tracks = sorted(new_tracks, key=lambda x: -x[1])[:max(room, 0)]
                if new_tracks:
                    born = []
                    for c, sc, dm in new_tracks:
                        tr = _Track(next_id, c, sc, t)
                        next_id += 1
                        tr.new_mask = fill_holes_and_sprinkles(dm[None])[0]
                        born.append(tr)
                    m = F.interpolate(torch.stack([tr.new_mask for tr in born])[:, None], size=(s, s), mode="bilinear",
                                      align_corners=False) > 0  # (N, 1, S, S)
                    n = len(born)
                    cur = self.sam2.track_step(t, True, [f.expand(n, -1, -1, -1) for f in feats],
                                               [p.expand(n, -1, -1, -1) for p in ps], None, m, {"cond": {}, "non_cond": {}}, t_all)
                    for j, tr in enumerate(born):
                        tr.out_dict["cond"][t] = _split_out(cur, j, n)
                        results.setdefault(tr.oid, {})[t] = tr.new_mask
                        frame_scores[t][tr.oid] = tr.score  # 新しい物体の追跡の点数は検出の点数(公式と同じ)
                        tracks.append(tr)
                # とっておく過去の出力を、記憶の取り出しに使う値だけにする(GPU のメモリを物体の数 × フレームの数で増やさない)
                for tr in tracks:
                    if not tr.removed:
                        _slim(tr.out_dict, t, self.sam2.max_obj_ptrs_in_encoder)
        self._frame_scores = frame_scores
        return [{"oid": tr.oid, "label": tr.label, "score": tr.score, "masks": results.get(tr.oid, {})}
                for tr in tracks if not tr.removed]

    @staticmethod
    def _associate(d_masks, d_scores, t_masks):
        """検出と追跡をマスクの IoU で照合する(公式の _associate_det_trk_compilable。o2o_matching なし)。

        Args:
            d_masks: 検出のマスクの値 (N, h, w)。
            d_scores: 検出の点数 (N,)。
            t_masks: 追跡の伝搬したマスクの値 (M, h, w)。

        Returns:
            (新しい物体にする検出の番号, 照合のない(空でない)追跡の番号, {検出: [照合した追跡]}, {追跡: 再条件付けの検出})。
        """
        n, m = len(d_masks), len(t_masks)
        if m == 0:  # 追跡がなければ全部の検出が新しい物体(確信度で切らない)
            return list(range(n)), [], {}, {}
        t_bin = t_masks > 0
        nonempty = t_bin.flatten(1).any(-1)
        if n == 0:
            return [], torch.nonzero(nonempty).squeeze(-1).tolist(), {}, {}
        d_bin = d_masks > 0
        iou = mask_iou(d_bin, t_bin)  # (N, M)
        trk_matched = (iou >= TRK_ASSOC_IOU_THRESH).any(0)
        unmatched = torch.nonzero(nonempty & ~trk_matched).squeeze(-1).tolist()
        is_new = ~(iou >= ASSOC_IOU_THRESH).any(1)  # どの追跡とも重ならない検出は新しい物体(確信度で切らない)
        many_trk = (iou >= IOU_THRESH_RECONDITION).sum(1) > 1  # 1 つの検出が複数の追跡と高い IoU(曖昧)
        many_det = (iou >= IOU_THRESH_RECONDITION).sum(0) > 1
        iou_u = torch.where(many_det[None, :], torch.zeros_like(iou), iou)
        iou_u = torch.where(many_trk[:, None], torch.zeros_like(iou_u), iou_u)
        best = iou_u.argmax(1)
        high_conf = (d_scores >= HIGH_CONF_THRESH) & ~is_new
        high_iou = iou_u.amax(1) >= IOU_THRESH_RECONDITION
        im = iou >= ASSOC_IOU_THRESH
        det_to_trk = {d: torch.nonzero(im[d]).squeeze(-1).tolist() for d in range(n)}
        recond = {}
        for d in torch.nonzero(high_conf & high_iou).squeeze(-1).tolist():
            recond[int(best[d])] = d
        return torch.nonzero(is_new).squeeze(-1).tolist(), unmatched, det_to_trk, recond

    @staticmethod
    def _suppress_recent_occlusion(trk_low: torch.Tensor, trs: list[_Track], idx: list[int], t: int) -> None:
        """重なる 2 つの追跡のうち、最近隠れた方のマスクを消す(公式の _suppress_overlapping_based_on_recent_occlusion)。

        Args:
            trk_low: 追跡の低解像度のマスクの値 (M, 1, h, w)(その場で書き換える)。
            trs: このクラスの追跡。
            idx: trs の各追跡の trk_low の中の番号。
            t: 今のフレームの番号。
        """
        if not idx:
            return
        b = trk_low[idx] > 0
        last = torch.tensor([tr.last_occluded for tr in trs], device=trk_low.device)
        sup = torch.zeros(len(idx), dtype=torch.bool, device=trk_low.device)
        if len(idx) > 1:
            pairs = torch.triu(mask_iou(b, b) >= SUPPRESS_OVERLAP_OCCLUSION_THRESH, diagonal=1)
            li, lj = last[:, None], last[None, :]
            si = pairs & (li > lj) & (lj > -1)
            sj = pairs & (lj > li) & (li > -1)
            sup = si.any(1) | sj.any(0)
        occluded = ~b.flatten(1).any(-1)
        for k, tr in enumerate(trs):
            if occluded[k] or sup[k]:
                tr.last_occluded = t
        for k in torch.nonzero(sup).squeeze(-1).tolist():
            trk_low[idx[k]] = -10.0

    @staticmethod
    def _suppress_shrinkage(high: torch.Tensor) -> torch.Tensor:
        """画素ごとに最も値の大きい物体だけを残したとき、面積が 0.3 未満に縮む物体を消す(公式の _suppress_object_pw_area_shrinkage)。

        Args:
            high: 物体ごとのマスクの値 (M, 1, H, W)。

        Returns:
            (M, 1, H, W)(消した物体は −10 以下)。
        """
        if high.shape[0] <= 1:
            return high
        arg = high.argmax(0, keepdim=True)
        keep_px = arg == torch.arange(high.shape[0], device=high.device)[:, None, None, None]
        non_overlap = torch.where(keep_px, high, high.clamp(max=-10.0))
        before = (high > 0).flatten(1).sum(-1).clamp(min=1).float()
        after = (non_overlap > 0).flatten(1).sum(-1).float()
        keep = (after / before) >= SHRINK_THRESH
        return torch.where(keep[:, None, None, None], high, high.clamp(max=-10.0))

    @torch.no_grad()
    def to_predictions(self, tracks: list[dict], video_id: int, cat_ids: list[int], size: int, valid: tuple[int, int],
                       orig: tuple[int, int], t_all: int) -> list[dict]:
        """ID ごとの予測を評価の形にする(公式の _postprocess_output: 空を除き、重なる画素は追跡の点数の高い物体へ。クラスごと)。

        Args:
            tracks: track_video の返り値。
            video_id: 動画の ID。
            cat_ids: クラスの番号 → カテゴリーの id。
            size: 入力の一辺 S。
            valid: 入力の中の画像の部分 (縦, 横)。
            orig: 元の大きさ (縦, 横)。
            t_all: 動画のフレームの数。

        Returns:
            評価の予測 [{"video_id", "category_id", "score", "segmentations", "query"}]。
        """
        from pycocotools import mask as mask_util
        # 動画ごとに点数の上位 TOP_K_OUT 個を出す(重なりの処理も、出す予測どうしで行う)
        tracks = sorted(tracks, key=lambda tr: -tr["score"])[:TOP_K_OUT]
        segs = {tr["oid"]: [None] * t_all for tr in tracks}
        for t in range(t_all):
            for c in sorted({tr["label"] for tr in tracks}):
                cur = [tr for tr in tracks if tr["label"] == c and t in tr["masks"]]
                if not cur:
                    continue
                m = torch.stack([tr["masks"][t] for tr in cur])[:, None]  # (n, 1, h, w)
                m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)[:, :, : valid[0], : valid[1]]
                b = F.interpolate(m, size=orig, mode="bilinear", align_corners=False)[:, 0] > 0  # (n, H, W)
                if len(cur) > 1:  # 公式の _apply_object_wise_non_overlapping_constraints(背景 0)
                    sc = torch.tensor([self._frame_scores[t].get(tr["oid"], 0.0) for tr in cur], device=b.device)
                    single = torch.where(b, sc[:, None, None], torch.zeros_like(b, dtype=sc.dtype))
                    owner = single.argmax(0)
                    b = b & (owner[None] == torch.arange(len(cur), device=b.device)[:, None, None])
                for tr, f in zip(cur, b.cpu().numpy()):
                    if f.any():
                        r = mask_util.encode(np.asfortranarray(f))
                        r["counts"] = r["counts"].decode("ascii")
                        segs[tr["oid"]][t] = r
        return [{"video_id": video_id, "category_id": cat_ids[tr["label"]], "score": tr["score"], "segmentations": segs[tr["oid"]],
                 "query": tr["oid"]} for tr in tracks if any(s is not None for s in segs[tr["oid"]])]
