"""Qwen3-VL の画像の部分(vision encoder)を、ViTAdapter に入れられる timm の ViT と同じ使い方で包む。

transformers の `Qwen3VLVisionModel` をそのまま使い(重みは `Qwen/Qwen3-VL-2B-Instruct` の `model.visual.*`)、最後のパッチの
合成(merger・DeepStack の merger)は使わない。公式の前処理(Qwen3-VL の image processor)と同じく:
    - 画像は時間の方向に 2 枚重ねる(パッチの埋め込みが 2 フレーム × 16 × 16 の 3 次元の畳み込みのため)
    - パッチは、2 × 2 のパッチのまとまりの順に並べる(位置の埋め込みと RoPE の番号がこの順で作られる)
    - 正規化は平均 0.5・標準偏差 0.5

形の記号: B = 画像の数、h・w = 縦・横のパッチの数(H/16・W/16)、D = 幅(1024)、m = まとめる大きさ(2)、t = 時間の 2 枚。
"""
import json
from pathlib import Path

import torch
from einops import rearrange
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

REPO = "Qwen/Qwen3-VL-2B-Instruct"


class Qwen3VLVisionViT(nn.Module):
    """Qwen3-VL の vision encoder を、(B, 3, H, W) の画像を受けて途中の層の特徴を返す形にする。

    Args:
        pretrained: Qwen3-VL-2B の重みを読むか。
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        path = Path(snapshot_download(REPO, allow_patterns=["config.json", "*.safetensors*"], local_files_only=True))  # 手元の cache の重み
        vcfg = Qwen3VLVisionConfig(**json.loads((path / "config.json").read_text())["vision_config"])
        vcfg._attn_implementation = "sdpa"
        self.model = Qwen3VLVisionModel(vcfg)
        self.model.merger = nn.Identity()  # 言語モデルの幅にまとめる部分は使わない
        self.model.deepstack_merger_list = nn.ModuleList()
        self.model.deepstack_visual_indexes = []
        if pretrained:
            sd = {}
            for f in sorted(path.glob("*.safetensors")):
                with safe_open(str(f), "pt") as g:
                    for k in g.keys():
                        if k.startswith("model.visual.") and ".merger" not in k and "deepstack_merger" not in k:
                            sd[k[len("model.visual."):]] = g.get_tensor(k)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            assert not missing and not unexpected, (missing, unexpected)
        self.model.float()
        self.embed_dim = vcfg.hidden_size
        self.patch, self.m, self.t = vcfg.patch_size, vcfg.spatial_merge_size, vcfg.temporal_patch_size
        self.pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}  # 公式の前処理の正規化

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """vision encoder の層で gradient checkpointing を使う。"""
        if enable:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.model.gradient_checkpointing_disable()

    def forward_intermediates(self, x: torch.Tensor, indices: list[int], norm: bool = False, output_fmt: str = "NCHW",
                              intermediates_only: bool = True) -> list[torch.Tensor]:
        """Args:
            x: (B, 3, H, W)(H・W は 32 の倍数)。
            indices: 取り出す層の番号(0 始まり)。
            norm: 使わない(Qwen3-VL の vision encoder は最後の LayerNorm を持たない。ViTAdapter には norm=False で渡す)。

        Returns:
            各層の (B, D, h, w)。
        """
        assert output_fmt == "NCHW" and intermediates_only and not norm
        b, _, hh, ww = x.shape
        h, w = hh // self.patch, ww // self.patch
        # (B, 3, H, W) -> 時間に 2 枚重ね、2×2 のまとまりの順のパッチの並び (B·h·w, 3·t·p·p)(公式の前処理と同じ並び)
        patches = rearrange(x.unsqueeze(2).expand(-1, -1, self.t, -1, -1),
                            "b c t (hb mh p) (wb mw q) -> (b hb wb mh mw) (c t p q)", mh=self.m, mw=self.m, p=self.patch,
                            q=self.patch)
        grid = torch.tensor([[1, h, w]] * b, device=x.device)
        mdl, keep, outs = self.model, set(indices), {}
        hooks = [mdl.blocks[i].register_forward_hook(lambda _m, _i, o, i=i: outs.__setitem__(i, o[0] if isinstance(o, tuple) else o))
                 for i in keep]
        try:
            last = max(indices)
            saved = mdl.blocks
            mdl.blocks = nn.ModuleList(list(saved)[: last + 1])  # 取り出す最後の層より後ろは計算しない
            mdl(patches, grid_thw=grid)
        finally:
            mdl.blocks = saved
            for hk in hooks:
                hk.remove()
        # (B·h·w, D)(2×2 のまとまりの順)-> (B, D, h, w)
        return [rearrange(outs[i], "(b hb wb mh mw) d -> b d (hb mh) (wb mw)", b=b, hb=h // self.m, wb=w // self.m, mh=self.m,
                          mw=self.m) for i in indices]
