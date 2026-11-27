"""自作の GTR 系の動画版(VidEoMT 型: クエリを次のフレームに引き継ぐ。segmenter ごと動画の区間で学習する)。exp00502。

土台は画像の段階の GTR 系(models/gtr_seg.py。VIPSeg のフレームで学習した重みから始める)。フレーム間の情報のやり取りは、
VidEoMT(公式 tue-mps/videomt の videomt_online)と同じくクエリだけで、規則も VidEoMT と同じにする:
    - 最初のフレーム: 画像の GTR 系と同じく、encoder の出力の上位 Q 個(組ごと)をクエリにする
    - 2 フレーム目から: **Q 個のクエリ全部**を次のフレームへ渡す。内容は「前のフレームの最後の層の出力のクエリを線形の層
      (query_updater)に通したもの + q」(VidEoMT の q = query_updater(prev) + q と同じ形)。クエリの番号が動画を通した ID になる
VidEoMT の q は学習するクエリ(どのフレームでも同じもの)だが、GTR 系には学習するクエリがなく、最初のフレームの encoder の上位から
選んだ内容がその役をする。そこで q には、最初のフレームで選んだ内容(番号ごとに動画を通して同じもの)を使う(私の判断)。
GTR 系の decoder は参照の枠の周りだけを見る(deformable な注意)ので、2 フレーム目からの参照の枠は前のフレームの予測の枠にする
(勾配は止める。VidEoMT にはない部分で、GTR 系に組み合わせるために要るもの)。

MOT(動画全体での物体の ID の管理・識別)の形: **学習して身に付ける形**(VidEoMT と同じ)。クエリの番号 = ID で、推論に規則はない。
    学習の割り当て(VidEoMT と同じ規則): 物体を、区間の中で最初に写ったフレームでだけ、まだ物体を持たないクエリとハンガリアン法で
    割り当て、その組を区間の最後まで使う。出力(decoder の層ごと + 最初の層の前の枠)ごとに別々に決める。隠れている間の正解は
    その物体(クラスはその物体、マスクは空)、割り当てのないクエリは「物体なし」。GTR 系の損失(MAL・枠・マスク・encoder・
    denoising)の中身は画像の GTR 系と同じ(losses/gtr_track_batch_criterion.py)。
    推論(VidEoMT と同じ): 全部のフレームの Q 個のクエリの予測を出し、クエリごとにフレームで平均したクラスの値から、動画ごとに
    上位 100 個を出す(video_infer.video_predictions)。

学習の手順(1 本の区間): フレームを時刻の順に 1 枚ずつ処理する(物体がまだ写っていないフレームも通す)。クエリの内容は前のフレームの
出力から作るので、勾配は区間の最初のフレームまで流れる(VidEoMT と同じ)。損失は、各フレームの損失(そのフレームの物体の数 ×
組の数で割ったもの)に、そのフレームの物体の数を掛けて足し、区間の物体の数の合計で割る。

形の記号: T = フレームの数、G = 組の数(学習のとき。推論は 1)、Q = 1 組のクエリの数、C = クエリの次元、S = 入力の一辺、
h・w = マスクの縦・横(入力の 1/4)、N = 区間の正解の物体の数、Kc = クラスの数。
"""
import torch
from torch import nn

from segmentation.instance_segmentation.lib_common.losses.gtr_track_batch_criterion import GTRTrackBatchCriterion
from segmentation.instance_segmentation.lib_common.losses.rfdetr_criterion import targets_from_masks
from segmentation.instance_segmentation.lib_common.models.gtr_seg import inverse_sigmoid
from segmentation.instance_segmentation.lib_common.models.rfdetr_seg import masks_from_parts


class _EncodeModule(nn.Module):
    """GTR 系の encode(backbone + encoder)だけを持つモジュール(CUDA graph で記録するため。重みは GTR 系と共有)。

    Args:
        gtr: 重みを共有する GTRSeg。
    """

    def __init__(self, gtr: nn.Module):
        super().__init__()
        self.backbone, self.encoder = gtr.backbone, gtr.encoder  # 重みを持つ部分(記録の対象の重み)
        object.__setattr__(self, "_gtr", gtr)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """GTRSeg.encode を通す。

        Args:
            x: 画像 (T, 3, S, S)。

        Returns:
            段ごとの特徴 3 × (T, C, h_l, w_l)。
        """
        return tuple(self._gtr.encode(x))


class GTRTrack(nn.Module):
    """GTR 系の動画版。

    Args:
        gtr: 画像の段階の GTRSeg(学習した重みを読み、クラスの並びを動画のセットにそろえたもの)。
        groups: 学習のときのクエリの組の数(画像の GTR 系と同じ)。
        mask_loss: GTR 系のマスクの損失("points" / "box_crop"。画像の学習と同じもの)。
    """

    activation = "sigmoid"  # 評価(video_infer.video_predictions)でのクラスの値の付け方

    def __init__(self, gtr: nn.Module, groups: int = 3, mask_loss: str = "points"):
        super().__init__()
        self.gtr = gtr
        c = gtr.score_head[0].in_features  # クエリの次元 C
        self.query_updater = nn.Linear(c, c)  # 前のフレームの出力のクエリ → 次のフレームのクエリに足す分(VidEoMT と同じ)
        self.batch_criterion = GTRTrackBatchCriterion(groups, mask_loss)  # 区間をまとめて計算する損失
        self.num_det = gtr.num_queries

    @classmethod
    def from_run(cls, path: str, video_set: str) -> "GTRTrack":
        """画像の GTR 系の学習の重みから作る。

        Args:
            path: 画像の学習の重み(runs/{名前}/best.pt。同じ場所の config.json で作り方を読む)。
            video_set: 学習の動画のセット(出力のクラスの番号をこのセットの並びにする)。

        Returns:
            GTRTrack(GTR 系の部分は学習した重み、query_updater は初期の値)。
        """
        import json
        from pathlib import Path

        from segmentation.instance_segmentation.lib_common.models.video.image_adapter import ImageSegmenter
        cfg = json.loads((Path(path).parent / "config.json").read_text())
        seg = ImageSegmenter.from_run(path, video_set)
        return cls(seg.model, seg.model.groups, cfg.get("gtr_mask_loss", "points"))

    def enable_graphed_encode(self, n_clips: int, frames: int, size: int, autocast) -> None:
        """学習の encode(backbone + encoder)を CUDA graph で記録し、以後は記録した処理の並びを再生する(Python・PyTorch の
        演算の処理・GPU の処理の起動を通らないので、CPU の時間が減る。計算は同じ)。

        記録した処理は出力と backward の途中の値を決まったメモリに置くので、backward の前に同じ記録を 2 回使えない。micro-batch の
        区間の位置ごとに別の記録を作る(重みは共有)。記録の予熱で BatchNorm の統計が変わるので、記録の前の値に戻す。
        形(区間の数・フレームの数・入力の大きさ)が記録と違うとき、学習中でないとき(評価)は、いつもの encode を使う。

        Args:
            n_clips: micro-batch の区間の数。
            frames: 区間のフレームの数。
            size: 入力の一辺。
            autocast: 記録するときの autocast(学習と同じ精度。cache_enabled=False のもの。train.amp_autocast で作る)。
        """
        # 区間の位置ごとに別のモジュール(重みは共有)。make_graphed_callables はモジュールの forward を記録したものに差し替えるので、
        # 同じモジュールを 2 回渡すと、どちらの位置も最後に記録した位置の記録を使ってしまう
        encs = tuple(_EncodeModule(self.gtr) for _ in range(n_clips))
        saved = {k: v.clone() for k, v in self.gtr.state_dict().items() if "running_" in k or "num_batches" in k}
        samples = tuple((torch.randn(frames, 3, size, size, device="cuda"),) for _ in range(n_clips))
        with autocast:
            graphed = torch.cuda.make_graphed_callables(encs, samples)
        sd = self.gtr.state_dict()
        with torch.no_grad():
            for k, v in saved.items():
                sd[k].copy_(v)
        # 子のモジュールとして登録しない(重みの保存に記録が入らないように)
        object.__setattr__(self, "_graphed_encode", (tuple(graphed), (frames, size)))

    def _encode_clip(self, frames: torch.Tensor, slot: int) -> list[torch.Tensor]:
        """区間 1 本の encode(記録があり、学習中で、形が同じなら記録を再生する)。

        Args:
            frames: 区間の画像 (T, 3, S, S)。
            slot: micro-batch の中の区間の位置(使う記録の番号)。

        Returns:
            段ごとの特徴 3 × (T, C, h_l, w_l)。
        """
        g = getattr(self, "_graphed_encode", None)
        if g is not None and self.training and slot < len(g[0]) and (frames.shape[0], frames.shape[-1]) == g[1]:
            return list(g[0][slot](frames))
        return self.gtr.encode(frames)

    @classmethod
    def from_scratch(cls, backbone: str, num_classes: int, encoder: str = "hybrid", model_kwargs: dict | None = None) -> "GTRTrack":
        """backbone だけ学習済みの重み(timm)で、backbone より後ろは初期の値の GTR 系から作る(VidEoMT A と同じ始め方)。

        モデルは画像の学習と同じ関数(train.build_model)で作る。既定の構成は exp00105 の後ろを大きくする前の構成
        (encoder "hybrid"、sdpa の自己注意。次元 256・decoder 4 層は GTRSeg の既定)。

        Args:
            backbone: timm の backbone の名前。
            num_classes: クラスの数(動画のセットの並び)。
            encoder: GTR 系の encoder("hybrid" / "c2f")。
            model_kwargs: GTRSeg に足す引数(None なら {"sdpa_self_attn": True})。

        Returns:
            GTRTrack。
        """
        from segmentation.instance_segmentation.lib_common.train import TrainConfig, build_model
        tc = TrainConfig(name="gtr_track", train_json="", out_dir="", results_dir="", arch="gtr_seg", backbone=backbone,
                         gtr_encoder=encoder, model_kwargs=model_kwargs if model_kwargs is not None else {"sdpa_self_attn": True})
        gtr = build_model(tc, num_classes)
        return cls(gtr, gtr.groups, tc.gtr_mask_loss)

    def _frame(self, image: torch.Tensor, dn_targets: dict | None, feats: list[torch.Tensor], prev: tuple | None,
               q0: torch.Tensor | None) -> dict:
        """1 フレームを GTR 系の decoder に通す(backbone と encoder は、区間・まとまりのフレームをまとめて先に通しておく)。

        Args:
            image: (B, 3, S, S)。
            dn_targets: denoising の正解(学習のとき)。
            feats: このフレームの段ごとの特徴(GTRSeg.encode の出力の 1 フレーム分)。
            prev: 前のフレームの最後の層の (出力のクエリ (B, G·Q, C), 予測の枠 (B, G·Q, 4))。最初のフレームは None。
            q0: 最初のフレームで decoder に入れたクエリの内容 (B, G·Q, C)(VidEoMT の q の役)。最初のフレームは None。

        Returns:
            GTRSeg.forward の返り値。
        """
        queries = None
        if prev is not None:  # 全部のクエリを渡す: 内容は query_updater(前の出力) + q、参照の枠は前の予測の枠(勾配を止める)
            boxes = prev[1].detach().float().clamp(1e-4, 1 - 1e-4)
            queries = (self.query_updater(prev[0]) + q0, inverse_sigmoid(boxes))
        return self.gtr(image, dn_targets, None, feats, queries)

    def train_loss(self, images: torch.Tensor, targets: list[dict], rng=None) -> tuple[torch.Tensor, dict]:
        """学習の区間の損失(ファイルの先頭の手順)。

        Args:
            images: (B, T, 3, S, S)。
            targets: 区間ごとの {"labels": (N,), "masks": (N, T, S, S) の 0/1}。
            rng: 使わない(学習の呼び方をほかの方式とそろえるため)。

        Returns:
            (区間の平均の損失(0 次元), 最後のフレームの最後の層の内訳の float)。
        """
        total, parts = 0.0, {}
        size = tuple(images.shape[-2:])
        dev = images.device
        n_clip, t_all = len(targets), images.shape[1]
        labels = [tg["labels"] for tg in targets]  # 区間ごと (N_b,)
        masks = [tg["masks"].bool() for tg in targets]  # 区間ごと (N_b, T, S, S)
        # (N_b, T) その物体がフレームに写っているか。CPU の上で使う(GPU の処理の終わりを待つのは区間ごとにここの 1 回だけ)
        present_all = [m.flatten(2).any(-1).cpu() for m in masks]
        # backbone と encoder は追跡のクエリに関係しないので、区間の T フレームをまとめて通す。区間ごとに通すのは、encoder の
        # BatchNorm の学習中の統計を、区間 1 本ずつのときと同じにするため
        feats_clip = [self._encode_clip(images[b], b) for b in range(n_clip)]  # 区間ごとに 3 × (T, C, h_l, w_l)
        n_out = len(self.gtr.layers) + 1  # 割り当てを決める出力の数(decoder の層ごと + 最初の層の前の枠)
        # 区間ごと・出力ごとの、決まった組 {クエリの番号(G·Q の中): 物体(区間の中の番号)}(物体が最初に写ったフレームで決める)
        pairs = [[{} for _ in range(n_out)] for _ in range(n_clip)]
        seen = [set() for _ in range(n_clip)]  # 区間ごとの、もう写った物体(区間の中の番号)
        clip_loss, clip_count = [0.0] * n_clip, [0] * n_clip
        prev, q0 = None, None
        contrib_sum = {}  # 損失の内訳 {出力の種類_項: 区間ごとの重みを掛けた和}
        for t in range(t_all):
            # micro-batch の区間を、このフレームでそろえて decoder に 1 回で通す(GPU の処理の起動の回数を区間の数で割る)
            frame_tg = []
            for b in range(n_clip):
                # このフレームの正解: 写っている物体 + もう写った(隠れている)物体。区間の中の番号の順
                ids = sorted(set(present_all[b][:, t].nonzero()[:, 0].tolist()) | seen[b])
                idx = torch.tensor(ids, dtype=torch.long, device=dev)
                pres_cpu = present_all[b][ids, t]  # (n,) このフレームの正解の中で、写っている物体
                pidx = pres_cpu.nonzero()[:, 0].to(dev)  # 写っている物体の、正解の中の番号
                frame_tg.append((ids, pres_cpu.to(dev), pidx, self._frame_targets(labels[b][idx], masks[b][idx, t], pidx, size)))
            # denoising の正解(写っている物体)。画像の GTR 系の batch と同じ形で、区間ごとの数は counts
            dn = {"labels": torch.cat([f[3]["labels"][f[2]] for f in frame_tg]),
                  "boxes": torch.cat([f[3]["boxes"][f[2]] for f in frame_tg]),
                  "counts": [len(f[2]) for f in frame_tg]}
            feats_t = [torch.cat([feats_clip[b][lv][t: t + 1] for b in range(n_clip)]) for lv in range(len(feats_clip[0]))]
            out = self._frame(images[:, t], dn, feats_t, prev, q0)
            if q0 is None:
                q0 = out["content"]  # 最初のフレームで選んだクエリの内容 (B, G·Q, C)(VidEoMT の q の役)
            last = out["layers"][-1]
            prev = (last["hidden"], last["boxes"])  # 次のフレームへ全部のクエリを渡す
            locals_ = [{o: j for j, o in enumerate(f[0])} for f in frame_tg]
            fixed = [[(torch.tensor(list(pairs[b][oi]), dtype=torch.long),
                       torch.tensor([locals_[b][o] for o in pairs[b][oi].values()], dtype=torch.long))
                      for b in range(n_clip)] for oi in range(n_out)]
            seen_local = [[locals_[b][o] for o in seen[b]] for b in range(n_clip)]
            losses, parts, own = self.batch_criterion.frame_loss_batch(
                out, [f[3] for f in frame_tg], [f[1] for f in frame_tg], seen_local, [0] * n_clip, 0, self.num_det,
                fixed_pairs=fixed)
            contrib = parts.pop("contrib")  # 損失の内訳 {出力の種類_項: 区間ごとの値 (B,)}(学習のログ用)
            contrib = dict(zip(contrib, torch.stack(list(contrib.values())).float().cpu()))  # CPU へは 1 回で移す
            for b in range(n_clip):
                ids = frame_tg[b][0]
                clip_loss[b] = clip_loss[b] + losses[b] * len(ids)
                clip_count[b] += len(ids)
                for name, v in contrib.items():  # 損失と同じ重みで足す(全部を足すと区間の損失になる)
                    contrib_sum.setdefault(name, [0.0] * n_clip)[b] += float(v[b]) * len(ids)
                for oi in range(n_out):  # このフレームの組(決まった組 + 新しい物体の組)を、次のフレームの決まった組にする
                    qq, tt = own[oi][b]
                    pairs[b][oi] = {int(q_): ids[int(t_)] for q_, t_ in zip(qq.tolist(), tt.tolist())}
                seen[b] |= set(present_all[b][:, t].nonzero()[:, 0].tolist())
        for b in range(n_clip):
            if clip_count[b]:
                total = total + clip_loss[b] / clip_count[b]
        if not torch.is_tensor(total):  # どの区間にも物体が写っていない: 勾配が 0 の損失(重みにつなげて backward できる形にする)
            total = self.query_updater.weight.sum() * 0
        # 損失の内訳を、返す損失と同じ割り方にする(loss_dec_mal などを全部足すと、返す損失になる)
        for name, vs in sorted(contrib_sum.items()):
            parts[f"loss_{name}"] = sum(v / clip_count[b] for b, v in enumerate(vs) if clip_count[b]) / max(n_clip, 1)
        return total / max(n_clip, 1), parts

    @staticmethod
    def _frame_targets(labels: torch.Tensor, masks: torch.Tensor, pidx: torch.Tensor, size: tuple) -> dict:
        """1 フレームの正解を、GTR 系の損失の形にする(隠れている物体は空のマスクと 0 の枠)。

        Args:
            labels: (N,)。
            masks: (N, S, S) の bool。
            pidx: 写っている物体の番号 (n,)。
            size: 入力の (縦, 横)。

        Returns:
            {"labels", "masks", "boxes", "owner", "counts"}(targets_from_masks と同じ形)。
        """
        boxes = torch.zeros(len(labels), 4, device=labels.device)
        if len(pidx):
            boxes[pidx] = targets_from_masks([{"masks": masks[pidx], "labels": labels[pidx]}], size)["boxes"]
        return {"labels": labels, "masks": masks, "boxes": boxes,
                "owner": torch.zeros(len(labels), dtype=torch.long, device=labels.device), "counts": [len(labels)]}

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """1 本の動画を推論する(VidEoMT と同じ: 規則はなく、クエリの番号が動画を通した ID)。

        Args:
            frames: (T, 3, S, S)。

        Returns:
            (各フレームのクラスの値 (T, Q, Kc)(sigmoid の前), 各フレームのマスクの値 (Q, T, h, w)(sigmoid の前))。
            video_infer.video_predictions がクエリごとにフレームで平均したクラスの値から、動画ごとに上位 100 個を出す。
        """
        logits, masks, prev, q0 = [], [], None, None
        chunk = 8  # backbone と encoder をまとめて通すフレームの数
        for t in range(len(frames)):
            if t % chunk == 0:
                feats_chunk = self.gtr.encode(frames[t: t + chunk])  # 3 × (chunk, C, h_l, w_l)
            out = self._frame(frames[t: t + 1], None, [f[t % chunk: t % chunk + 1] for f in feats_chunk], prev, q0)
            if q0 is None:
                q0 = out["content"]  # (1, Q, C)
            last = out["layers"][-1]
            prev = (last["hidden"], last["boxes"])
            logits.append(last["logits"][0])  # (Q, Kc)
            masks.append(masks_from_parts(last["mask_parts"])[0])  # (Q, h, w)
        return torch.stack(logits), torch.stack(masks, dim=1)  # (T, Q, Kc), (Q, T, h, w)
