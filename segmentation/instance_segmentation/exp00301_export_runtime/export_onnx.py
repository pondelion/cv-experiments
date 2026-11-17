"""exp00301: 学習したモデルを ONNX に書き出し、ONNX Runtime の出力が PyTorch と一致するかを確かめる。

学習の記録(`{exp}/runs/{名前}/config.json` と、評価に使った重み `best.pt`)から学習のときと同じモデルを作って重みを読み、
推論の出力をテンソルだけで返す形に包んで書き出す(`torch.onnx.export` の dynamo=True、入力は batch 1・512 × 512 の固定)。

書き出す出力(方式ごと):
    gtr_seg:  logits (1, Q, K)(sigmoid の前)、boxes (1, Q, 4)(0〜1 の cx, cy, w, h)、masks (1, Q, S/4, S/4)(sigmoid の前)。
              最後の decoder の層・1 組目のクエリ(学習のときの評価と同じ。後処理は lib_common/postprocess.py の
              rfdetr_to_predictions)
    yolo_seg: logits (1, A, K)、boxes (1, A, 4)(入力の画素の x0, y0, x1, y1)、coef (1, A, M)、protos (1, M, S/4, S/4)。
              1 対 1 の head(NMS なし。後処理は yolo_to_predictions)

確認: 評価のセット(既定は coco_val_test_cls8)の先頭の画像を評価の変換(transforms.eval_transforms)にかけて入れ、PyTorch fp32 と
ONNX Runtime(CPU・CUDA)の出力の差の最大を出す。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00301_export_runtime/export_onnx.py \
        --run segmentation/instance_segmentation/exp00103_latest_methods/runs/gtr_hybrid_r50_cls8_2k_ep100
"""
import argparse
import json
from pathlib import Path

import common

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from lib_common.datasets import load_set, set_path  # noqa: E402
from segmentation.instance_segmentation.lib_common.data import InstanceDataset  # noqa: E402
from segmentation.instance_segmentation.lib_common.models.gtr_seg import GTRSeg  # noqa: E402
from segmentation.instance_segmentation.lib_common.models.rfdetr_seg import masks_from_parts  # noqa: E402
from segmentation.instance_segmentation.lib_common.models.yolo_seg import YOLOSeg, decode_boxes  # noqa: E402
from segmentation.instance_segmentation.lib_common.transforms import eval_transforms  # noqa: E402


class GTRExport(nn.Module):
    """GTRSeg の推論の出力を、テンソル 3 つで返す形に包む。

    Args:
        model: 評価のときの GTRSeg(eval にしたもの)。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """推論する。

        Args:
            images: 正規化した画像 (1, 3, S, S)。

        Returns:
            (クラスの値 (1, Q, K), 枠 (1, Q, 4), マスクの値 (1, Q, S/4, S/4))。
        """
        last = self.model(images)["layers"][-1]
        return last["logits"], last["boxes"], masks_from_parts(last["mask_parts"])


class YOLOExport(nn.Module):
    """YOLOSeg の推論の出力(1 対 1 の head、NMS なし)を、テンソル 4 つで返す形に包む。

    Args:
        model: 評価のときの YOLOSeg(eval にしたもの)。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """推論する。

        Args:
            images: 正規化した画像 (1, 3, S, S)。

        Returns:
            (クラスの値 (1, A, K)(A = 3 段の全部の位置), 枠 (1, A, 4)(入力の画素の x0, y0, x1, y1),
             マスクの係数 (1, A, M), 原型のマスク (1, M, S/4, S/4))。
        """
        out = self.model(images)
        o = out["o2o"]
        return o["logits"], decode_boxes(o["ltrb"], out["anchors"], out["strides"]), o["coef"], out["protos"]


def psa_forward_2d(self, x: torch.Tensor) -> torch.Tensor:
    """YOLOSeg の PSABlock の forward を、注意の頭ごとの 2 次元の行列の掛け算に分けて計算する版(書き出しのときだけ差し替える)。

    元の forward(scaled_dot_product_attention。ONNX では 4 次元の行列の掛け算 (1, 頭の数, hw, d) になる)と計算は同じ。
    ブラウザの ONNX Runtime Web の WebGPU は、Windows の Chrome の既定のシェーダーの変換(DXC)で、3 次元以上の行列の掛け算の
    シェーダーの準備に極端に時間がかかる(3 次元で約 24 秒、4 次元では終わらない。2 次元はすぐ終わる。exp00304 の README)。
    batch 1 を前提に、頭ごとに (hw, d) × (d, hw) の 2 次元にする。

    Args:
        x: (1, C, h, w)。

    Returns:
        (1, C, h, w)。
    """
    from einops import rearrange
    h, w = x.shape[-2:]
    q, k, v = rearrange(self.qkv(x), "b (three n d) h w -> three (b n) (h w) d", three=3, n=self.heads)  # 3 × (頭, hw, d)
    scale = q.shape[-1] ** -0.5
    heads = [torch.softmax(q[i] @ k[i].T * scale, dim=-1) @ v[i] for i in range(self.heads)]  # 頭ごとに (hw, d)
    a = rearrange(torch.stack(heads), "n (h w) d -> 1 (n d) h w", h=h, w=w)  # -> (1, C, h, w)
    x = x + self.proj(a + self.pe(rearrange(v, "n (h w) d -> 1 (n d) h w", h=h, w=w)))
    return x + self.ffn(x)


def mha_forward_2d(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args, **kwargs):
    """nn.MultiheadAttention(batch_first)の forward を、頭ごとの 2 次元の行列の掛け算で計算する版(書き出しのときだけ差し替える)。

    推論のとき(マスクなし・batch 1)と計算は同じ。元の forward は ONNX で 3〜4 次元の行列の掛け算になり、WebGPU のシェーダーの
    準備が極端に遅い(psa_forward_2d の説明)。GTR の encoder の AIFI と decoder の自己注意で使う。

    Args:
        query: (1, M, C)。
        key: (1, N, C)。
        value: (1, N, C)。

    Returns:
        (出力 (1, M, C), None)(元の forward の need_weights=False のときと同じ形)。
    """
    assert self.batch_first and query.shape[0] == 1 and kwargs.get("attn_mask") is None
    wq, wk, wv = self.in_proj_weight.chunk(3)  # 各 (C, C)
    bq, bk, bv = self.in_proj_bias.chunk(3)
    q, k, v = query[0] @ wq.T + bq, key[0] @ wk.T + bk, value[0] @ wv.T + bv  # (M, C), (N, C), (N, C)
    d = q.shape[-1] // self.num_heads
    heads = [torch.softmax(q[:, i * d:(i + 1) * d] @ k[:, i * d:(i + 1) * d].T * d ** -0.5, dim=-1)
             @ v[:, i * d:(i + 1) * d] for i in range(self.num_heads)]  # 頭ごとに (M, d)
    return self.out_proj(torch.cat(heads, dim=-1))[None], None  # (M, C) -> (1, M, C)


def deform_forward_low_rank(self, query: torch.Tensor, ref: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
    """GTR の MSDeformableAttention の forward で、点の位置を 3 次元のテンソルで計算する版(書き出しのときだけ差し替える)。

    元の forward と計算は同じ。元は位置を (B, M, 頭, P, 2) の 5 次元で、いくつもの軸を広げて(broadcast)掛け・足すが、
    WebGPU ではこの形の演算のシェーダーの準備が終わらない(exp00304 の README)。(B·M, 頭·P, 2) の 3 次元で計算してから並べ替える。

    Args:
        query: (B, M, C)。
        ref: 参照の枠 (B, M, 4) cx, cy, w, h(0〜1)。
        values: 段ごとの特徴 [(B·heads, C/heads, h_l, w_l)]。

    Returns:
        (B, M, C)。
    """
    import torch.nn.functional as F
    from einops import rearrange
    b, m, _ = query.shape
    p = sum(self.points)
    off = rearrange(self.sampling_offsets(query), "b m (hp two) -> (b m) hp two", two=2)  # (B·M, 頭·P, 2)
    attn = rearrange(self.attention_weights(query), "b m (h p) -> b m h p", h=self.heads).softmax(-1)
    r = rearrange(ref, "b m four -> (b m) 1 four")  # (B·M, 1, 4)
    scale = self.point_scale.repeat(self.heads)[:, None] * self.offset_scale  # (頭·P, 1)
    loc = r[..., :2] + off * scale * r[..., 2:]  # (B·M, 頭·P, 2)
    grid = rearrange(2 * loc - 1, "(b m) (h p) two -> (b h) m p two", b=b, h=self.heads)  # grid_sample の座標(−1〜1)
    sampled = [F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
               for v, g in zip(values, grid.split(self.points, dim=2))]  # 段ごとに (B·h, c, M, P_l)
    sampled = torch.cat(sampled, dim=-1)  # (B·h, c, M, P)
    w = rearrange(attn, "b m h p -> (b h) 1 m p")
    out = (sampled * w).sum(-1)  # (B·h, c, M)
    return rearrange(out, "(b h) c m -> b m (h c)", b=b)


def build(run_dir: Path) -> tuple[nn.Module, dict, list[str]]:
    """学習の記録からモデルを作って重みを読み、書き出し用に包む。

    Args:
        run_dir: 学習の出力のフォルダ(config.json と best.pt がある)。

    Returns:
        (包んだモデル(eval・CPU), 学習の設定, 出力の名前の並び)。
    """
    cfg = json.loads((run_dir / "config.json").read_text())
    num_classes = len(load_set(Path(cfg["train_json"]).stem)["categories"])  # train_json の名前 = セット名
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.get("model", state).items()}
    if cfg["arch"] == "gtr_seg":  # 学習の train.build_model と同じ引数(gradient checkpointing は推論では使わない)
        model = GTRSeg(num_classes, cfg["backbone"], encoder=cfg.get("gtr_encoder", "c2f"), pretrained=False)
        model.load_state_dict(state, strict=True)
        return GTRExport(model.eval()).eval(), cfg, ["logits", "boxes", "masks"]
    if cfg["arch"] == "yolo_seg":
        model = YOLOSeg(num_classes, cfg["backbone"], pretrained=False)
        model.load_state_dict(state, strict=True)
        return YOLOExport(model.eval()).eval(), cfg, ["logits", "boxes", "coef", "protos"]
    raise ValueError(f"書き出しに対応していない方式: {cfg['arch']}")


def main() -> None:
    """書き出しと、出力の一致の確認。"""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="学習の出力のフォルダ(runs/{名前})")
    p.add_argument("--out", default=None, help="書き出す ONNX のパス(省略すると exp00301 の runs/{名前}/model.onnx)")
    p.add_argument("--check-set", default="coco_val_test_cls8")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--webgpu", action="store_true",
                   help="ブラウザの WebGPU で準備が終わらない形の演算を、同じ計算の別の形に差し替えて書き出す"
                        "(注意を頭ごとの 2 次元の行列の掛け算に、GTR の点の位置の計算を 3 次元に)")
    a = p.parse_args()
    run_dir = Path(a.run)
    model, cfg, names = build(run_dir)
    if a.webgpu:
        from segmentation.instance_segmentation.lib_common.models.gtr_seg import MSDeformableAttention
        from segmentation.instance_segmentation.lib_common.models.yolo_seg import PSABlock
        PSABlock.forward = psa_forward_2d
        nn.MultiheadAttention.forward = mha_forward_2d
        MSDeformableAttention.forward = deform_forward_low_rank
    out = Path(a.out) if a.out else common.RUNS_DIR / run_dir.name / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = InstanceDataset(str(set_path(a.check_set)), eval_transforms(cfg["size"]))
    x = ds[0]["image"][None].float()  # (1, 3, S, S)
    with torch.no_grad():
        ref = [t.numpy() for t in model(x)]
    torch.onnx.export(model, (x,), str(out), dynamo=True, opset_version=a.opset, input_names=["images"],
                      output_names=names)
    print(f"書き出した: {out}({out.stat().st_size / 1e6:.1f}MB。重みが別のファイルなら {out}.data も)", flush=True)

    import onnxruntime as ort
    try:
        ort.preload_dlls()  # torch の CUDA・cuDNN を ONNX Runtime の CUDA で使う
    except AttributeError:
        pass
    def by_score(outs: list[np.ndarray], k: int = 100) -> list[np.ndarray]:
        """クエリ(位置)をクラスの値の最大の大きい順に並べ、上位 k 個を取り出す(protos のように並びのない出力はそのまま)。

        GTR 系は上位の候補を選ぶ処理(TopK)でクエリを選ぶので、計算の細かい差で同点に近い候補の順が入れ替わると、出力の
        クエリの順も入れ替わる(decoder はクエリの順によらないので、同じ予測が別の順に並ぶ)。そのため順によらずに比べる。

        Args:
            outs: 出力の並び。最初が logits (1, Q, K)。2 番目の次元が Q の出力を並べ替える(YOLO の protos (1, M, h, w) は
                並べ替えない)。
            k: 比べる上位の数(評価の後処理の上位の数と同じ 100)。

        Returns:
            並べ替えた出力の並び(並べ替えたものは (k, ...)、並べ替えないものは (...))。
        """
        q = outs[0].shape[1]
        order = np.argsort(-outs[0][0].max(-1))[:k]
        return [o[0][order] if o.shape[1] == q else o[0] for o in outs]

    ref_sorted = by_score(ref)
    for provider, opts in [("CPUExecutionProvider", {}), ("CUDAExecutionProvider", {"use_tf32": "0"}),
                           ("CUDAExecutionProvider", {"use_tf32": "1"})]:
        sess = ort.InferenceSession(str(out), providers=[(provider, opts)])
        got = sess.run(None, {"images": x.numpy()})
        diffs = {n: float(np.abs(g - r).max()) for n, g, r in zip(names, by_score(got), ref_sorted)}
        print(f"{provider} {opts}(実際に使った: {sess.get_providers()[0]}): クラスの値の上位 100 個のクエリで比べた"
              f"出力の差の最大 {diffs}(値の大きさの最大 {[float(np.abs(r).max()) for r in ref_sorted]})", flush=True)


if __name__ == "__main__":
    main()
