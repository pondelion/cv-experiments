"""MaskTrack R-CNN(Yang ら ICCV 2019、"Video Instance Segmentation")。exp00502。

参考: 公式のコード(youtubevos/MaskTrackRCNN の `mmdet/models/track_heads/track_head.py`、`mmdet/models/detectors/two_stage.py` の
forward_train・simple_test_bboxes、`mmdet/datasets/ytvos.py` の sample_ref・prepare_train_img、`mmdet/core/bbox/bbox_target.py`、
`mmdet/core/evaluation/coco_utils.py` の results2json_videoseg、`configs/masktrack_rcnn_r50_fpn_1x_youtubevos.py`)。
コードは写さず、規則を読み取って書いた。

作り: 画像の Mask R-CNN(models/maskrcnn.py。torchvision の Mask R-CNN)に追跡の head を足す。
    - 追跡の head(公式の TrackHead): RoIAlign の 7 × 7 の特徴(256 チャネル。box の head と同じ RoI の特徴)を 2 層の全結合
      (12,544 → 1,024 → ReLU → 1,024。重みは正規分布 0.01、bias 0)で埋め込みにし、今のフレームの物体と参照の物体の内積を
      照合の値にする。先頭に「新しい物体」の列(値 0)を足す
    - 学習(公式と同じ): 動画の 1 フレーム(今)と、同じ動画の別のフレーム 1 枚(参照。無作為)の組。検出の損失は今のフレームだけ。
      追跡の損失: 今のフレームの RoI の正例(box の head の学習で選んだ候補のうち正例。正解の枠も候補に入る)ごとに、割り当てた
      正解の物体が参照のフレームにいればその物体の列、いなければ「新しい物体」の列を正解にした交差エントロピー(正例の数で平均、
      画像の数で平均、重み 1)。参照の物体の特徴は、参照のフレームの正解の枠の RoI の特徴(参照のフレームにも勾配が流れる)
    - 推論(公式の simple_test_bboxes): フレームごとの検出(score 0.05 以上・NMS 0.5)の RoI の特徴を埋め込みにし、今までの全部の
      物体の記憶と、log softmax(照合の値) + 1.0 × log(検出の点数) + 2.0 × 枠の IoU + 10 × (クラスが同じなら 1) の和で比べ、
      最大の列を選ぶ(新しい物体の列は IoU 0・クラスの項 1)。新しい物体の列なら新しい ID を足す。同じ物体を複数の検出が選んだら、
      和の大きい検出が ID を取り、負けた検出はそのフレームの出力から外す(公式と同じ。後から来た大きい検出が勝つと、先の検出の
      ID は −1 にならずに残る点も公式のまま)。選ばれた物体の記憶(特徴・枠)は今の検出で上書きする(クラスは最初のまま)
    - 動画の予測(公式の results2json_videoseg): ID ごとに、点数 = 写ったフレームの検出の点数の平均、クラス = フレームのクラスの多数決
MOT(動画全体での物体の ID の管理・識別)の形: **学習した埋め込み + 規則の照合(記憶のバンク)**。動画の初めからの全部の物体の記憶
(最後に照合したときの特徴・枠)を持ち、物体を除くことはない(隠れた物体も記憶に残り、再出現で同じ ID に戻れる)。区別できる物体の
数の上限はない。

公式との違い(私の判断。理由つき):
    - 1 画像の検出の数の上限: 公式の YouTube-VIS の設定は 10。VIPSeg の 8 クラスは 1 本に最大 69 物体あるので、画像の Mask R-CNN と
      同じ torchvision の既定(100)のまま(動画のモデルの top_k 100 と同じ理由)
    - box の head の学習の候補の数・正例の割合(公式 128・0.25)・RPN・入力の大きさなどは、この exp の画像の Mask R-CNN の設定のまま
      (追跡の head だけを足す。画像の基準と同じ検出の部にするため)

形の記号: B = 区間の数、S = 入力の一辺、n = 今のフレームの物体(RoI)の数、m = 参照の物体の数。
"""
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from torch import nn
from torchvision.ops import box_iou, masks_to_boxes

from segmentation.instance_segmentation.lib_common.models.maskrcnn import MaskRCNNSeg

MATCH_COEFF = (1.0, 2.0, 10.0)  # 公式の match_coeff(検出の点数の log・IoU・クラスが同じ)


class TrackHead(nn.Module):
    """追跡の head(公式の TrackHead。dynamic、平均プーリングなし)。

    Args:
        in_channels: RoI の特徴のチャンネル。
        roi_size: RoI の特徴の一辺。
        hidden: 全結合の次元(公式は 1024)。
    """

    def __init__(self, in_channels: int = 256, roi_size: int = 7, hidden: int = 1024):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(in_channels * roi_size * roi_size, hidden), nn.Linear(hidden, hidden)])
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.zeros_(fc.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """RoI の特徴を埋め込みにする。

        Args:
            x: (n, 256, 7, 7)。

        Returns:
            (n, 1024)(最後の層の後に ReLU はない)。
        """
        x = F.relu(self.fcs[0](x.flatten(1)))
        return self.fcs[1](x)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """今の物体と参照の物体の照合の値(埋め込みの内積。先頭に新しい物体の列 0)。

        Args:
            x: 今の物体の RoI の特徴 (n, 256, 7, 7)。
            ref: 参照の物体の RoI の特徴 (m, 256, 7, 7) か、埋め込み済みの (m, 1024)。

        Returns:
            照合の値 (n, m + 1)(列 0 = 新しい物体)。
        """
        e = self.embed(x)
        r = ref if ref.dim() == 2 else self.embed(ref)
        return torch.cat([e.new_zeros(len(e), 1), e @ r.T], 1)


def gt_boxes(masks: torch.Tensor) -> torch.Tensor:
    """正解のマスクから枠を作る(MaskRCNNSeg と同じ: 端の画素の外の端まで)。

    Args:
        masks: (N, S, S)。

    Returns:
        (N, 4) の (x0, y0, x1, y1)。
    """
    if not len(masks):
        return masks.new_zeros((0, 4), dtype=torch.float32)
    b = masks_to_boxes(masks).float()
    b[:, 2:] += 1
    return b


class MaskTrackRCNN(nn.Module):
    """MaskTrack R-CNN(画像の Mask R-CNN + 追跡の head)。

    Args:
        num_classes: クラスの数(背景を含めない)。
        backbone: timm の ResNet のモデル名(models/maskrcnn.py と同じ)。
        size: 入力の一辺 S。
        pretrained: backbone に事前学習の重みを読むか。
    """

    def __init__(self, num_classes: int, backbone: str = "resnet50.a1_in1k", size: int = 512, pretrained: bool = True):
        super().__init__()
        self.seg = MaskRCNNSeg(num_classes, backbone, size, pretrained)
        self.track_head = TrackHead()
        rh = self.seg.model.roi_heads
        select = rh.select_training_samples

        def select_and_keep(proposals, targets):
            """box の head の学習の候補を選び(torchvision の select_training_samples)、結果を追跡の損失のために残す。

            Args:
                proposals: 画像ごとの候補の枠。
                targets: 画像ごとの正解。

            Returns:
                select_training_samples の返り値(候補, 割り当てた正解の番号, クラス(0 = 背景), 回帰の正解)。
            """
            out = select(proposals, targets)
            self._sampled = out  # (候補, 割り当てた正解の番号, クラス(0 = 背景), 回帰の正解)
            return out
        rh.select_training_samples = select_and_keep  # torchvision の RoIHeads の中の 1 か所だけを包む

    def train_loss(self, clips: torch.Tensor, targets: list[dict], rng: np.random.Generator) -> tuple[torch.Tensor, dict]:
        """学習の 1 step の損失(公式の forward_train)。

        Args:
            clips: (B, 2, 3, S, S)(同じ動画の 2 フレーム。VideoClipDataset で clip_len 2 にしたもの)。
            targets: 区間ごとの {"labels": (N,), "masks": (N, 2, S, S)}。
            rng: 2 フレームのどちらを今のフレームにするかの乱数(公式は今のフレームを全部のフレームから、参照を同じ動画の別の
                フレームから無作為に選ぶ。2 フレームを無作為に選んで役を無作為に決めるのと同じ分布)。

        Returns:
            (損失の合計, 内訳)。
        """
        model = self.seg.model
        cur_i = rng.integers(2, size=len(clips))
        cur = torch.stack([c[i] for c, i in zip(clips, cur_i)])
        ref = torch.stack([c[1 - i] for c, i in zip(clips, cur_i)])
        tv_targets, ref_boxes = [], []
        for tg, i in zip(targets, cur_i):
            vis_c, vis_r = tg["masks"][:, i].flatten(1).any(-1), tg["masks"][:, 1 - i].flatten(1).any(-1)  # 写っている物体
            c_idx, r_idx = torch.nonzero(vis_c).squeeze(-1), torch.nonzero(vis_r).squeeze(-1)
            # 今のフレームの各物体の、参照のフレームでの番号 + 1(いなければ 0。公式の gt_pids)
            pos_in_ref = {int(o): k + 1 for k, o in enumerate(r_idx.tolist())}
            pids = torch.tensor([pos_in_ref.get(int(o), 0) for o in c_idx.tolist()], device=clips.device, dtype=torch.long)
            m = tg["masks"][c_idx, i]
            tv_targets.append({"boxes": gt_boxes(m), "labels": tg["labels"][c_idx] + 1, "masks": m, "pids": pids})
            ref_boxes.append(gt_boxes(tg["masks"][r_idx, 1 - i]))
        images, tv_targets = model.transform(list(cur), tv_targets)
        feats = model.backbone(images.tensors)
        proposals, rpn_losses = model.rpn(images, feats, tv_targets)
        _, det_losses = model.roi_heads(feats, proposals, images.image_sizes, tv_targets)
        # 追跡の損失
        ref_images, _ = model.transform(list(ref))
        ref_feats = model.backbone(ref_images.tensors)
        pool = model.roi_heads.box_roi_pool
        props, matched, labels, _ = self._sampled
        loss_match = 0.0
        for k, (p, mi, lb, tg) in enumerate(zip(props, matched, labels, tv_targets)):
            pos = lb > 0
            if not pos.any():
                continue
            x = pool({n: f[k: k + 1] for n, f in feats.items()}, [p[pos]], images.image_sizes[k: k + 1])
            r = pool({n: f[k: k + 1] for n, f in ref_feats.items()}, [ref_boxes[k]], ref_images.image_sizes[k: k + 1])
            score = self.track_head(x, r)  # (n, m + 1)
            loss_match = loss_match + F.cross_entropy(score.float(), tg["pids"][mi[pos]])
        loss_match = loss_match / len(clips) if torch.is_tensor(loss_match) else feats["0"].sum() * 0
        losses = {**rpn_losses, **det_losses, "loss_match": loss_match}
        return sum(losses.values()), {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def track_video(self, frames: torch.Tensor, valid: tuple[int, int], orig: tuple[int, int], chunk: int = 4) -> list[dict]:
        """1 本の動画を時刻の順に追跡する(公式の simple_test_bboxes の記憶と照合)。

        Args:
            frames: (T, 3, S, S)(評価の変換)。
            valid: 入力の中の画像の部分 (縦, 横)。
            orig: 元の大きさ (縦, 横)。
            chunk: 検出をまとめて処理するフレームの数。

        Returns:
            ID ごとの {"oid", "label", "score", "masks": {t: 元の大きさの RLE}}(点数 = 平均、クラス = 多数決)。
        """
        model = self.seg.model
        mem_feat = mem_box = mem_label = None
        tracks: dict[int, dict] = {}
        for s in range(0, len(frames), chunk):
            images, _ = model.transform(list(frames[s: s + chunk]))
            feats = model.backbone(images.tensors)
            proposals, _ = model.rpn(images, feats)
            dets, _ = model.roi_heads(feats, proposals, images.image_sizes)
            # マスク(28 × 28)を枠の位置に貼って入力の大きさにする(torchvision の GeneralizedRCNN と同じ後処理。枠は大きさが同じなので不変)
            dets = model.transform.postprocess(dets, images.image_sizes, [tuple(frames.shape[-2:])] * len(dets))
            for k, det in enumerate(dets):
                t = s + k
                if not len(det["boxes"]):
                    continue
                x = self.track_head.embed(model.roi_heads.box_roi_pool({n: f[k: k + 1] for n, f in feats.items()},
                                                                      [det["boxes"]], images.image_sizes[k: k + 1]).float())
                labels, scores, boxes = det["labels"] - 1, det["scores"].float(), det["boxes"].float()
                if mem_feat is None:  # 最初に検出のあったフレーム: 全部を新しい物体にする
                    ids = list(range(len(boxes)))
                    mem_feat, mem_box, mem_label = x, boxes, labels
                else:
                    match = torch.cat([x.new_zeros(len(x), 1), x @ mem_feat.T], 1).log_softmax(1)
                    iou = torch.cat([x.new_zeros(len(x), 1), box_iou(boxes, mem_box)], 1)
                    same = torch.cat([x.new_ones(len(x), 1), (mem_label[None] == labels[:, None]).float()], 1)
                    comp = match + MATCH_COEFF[0] * scores.log()[:, None] + MATCH_COEFF[1] * iou + MATCH_COEFF[2] * same
                    best_val, best = comp.max(1)
                    ids, best_score = [-1] * len(boxes), {}
                    for d, j in enumerate(best.tolist()):
                        if j == 0:  # 新しい物体
                            ids[d] = len(mem_feat)
                            mem_feat, mem_box = torch.cat([mem_feat, x[d: d + 1]]), torch.cat([mem_box, boxes[d: d + 1]])
                            mem_label = torch.cat([mem_label, labels[d: d + 1]])
                        elif float(comp[d, j]) > best_score.get(j - 1, -100.0):  # 同じ物体を選んだ検出のうち大きい方
                            ids[d], best_score[j - 1] = j - 1, float(comp[d, j])
                            mem_feat[j - 1], mem_box[j - 1] = x[d], boxes[d]
                prob = det["masks"][:, :, : valid[0], : valid[1]].float()
                prob = F.interpolate(prob, size=orig, mode="bilinear", align_corners=False)[:, 0] > 0.5  # (N, H, W)
                taken = {}
                for d, oid in enumerate(ids):  # 公式の bbox2result_with_id: ID −1 は外し、同じ ID は後の検出で上書き
                    if oid >= 0:
                        taken[oid] = d
                for oid, d in taken.items():
                    r = mask_util.encode(np.asfortranarray(prob[d].cpu().numpy()))
                    r["counts"] = r["counts"].decode("ascii")
                    tr = tracks.setdefault(oid, {"scores": [], "labels": [], "masks": {}})
                    tr["scores"].append(float(scores[d])); tr["labels"].append(int(labels[d])); tr["masks"][t] = r
        return [{"oid": oid, "label": int(np.bincount(tr["labels"]).argmax()), "score": float(np.mean(tr["scores"])),
                 "masks": tr["masks"]} for oid, tr in tracks.items()]

    def to_predictions(self, tracks: list[dict], video_id: int, cat_ids: list[int], size: int, valid: tuple[int, int],
                       orig: tuple[int, int], t_all: int) -> list[dict]:
        """ID ごとの予測を評価の形にする(マスクは track_video で元の大きさの RLE 済み)。

        Args:
            tracks: track_video の返り値。
            video_id: 動画の ID。
            cat_ids: クラスの番号 → カテゴリーの id。
            size: 使わない(ほかの方式と呼び方をそろえるため)。
            valid: 使わない(同上)。
            orig: 使わない(同上)。
            t_all: 動画のフレームの数。

        Returns:
            評価の予測 [{"video_id", "category_id", "score", "segmentations", "query"}]。
        """
        return [{"video_id": video_id, "category_id": cat_ids[tr["label"]], "score": tr["score"],
                 "segmentations": [tr["masks"].get(t) for t in range(t_all)], "query": tr["oid"]} for tr in tracks]
