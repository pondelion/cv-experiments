"""自己教師ありの事前学習を 1 つ行う(画像: DINOv3、動画: DoRA・V-JEPA 2.1)。

使い方(例):
    python pretrain.py --method dinov3 --name dinov3_synth_scratch --backbone vit_small_patch16_224
    python pretrain.py --method dinov3 --name dinov3_synth_from_dinov3 --backbone vit_small_patch16_dinov3.lvd1689m --pretrained
    python pretrain.py --method dora --name dora_k400 --backbone deit3_small_patch16_224.fb_in1k --pretrained
    python pretrain.py --method vjepa --name vjepa_k400 --backbone deit3_small_patch16_224.fb_in1k --pretrained

`--set key=value` で各手法の設定(DINOv3Config・DoRAConfig・VJEPAConfig)の項目を上書きする(値は JSON として読む)。
重みは runs/{name}/ に保存する(後の学習の `--init-weights` に渡す)。
"""
import argparse
import json
from pathlib import Path

import common
from lib_common.ssl import dora, vjepa
from lib_common.ssl.dinov3 import train as dinov3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["dinov3", "dora", "vjepa"])
    p.add_argument("--name", required=True)
    p.add_argument("--backbone", default="vit_small_patch16_224")
    p.add_argument("--pretrained", action="store_true", help="timm の学習済みの重みから始める(追加の事前学習)")
    p.add_argument("--max-files", type=int, default=None, help="使う画像・動画の数(確かめるとき)")
    p.add_argument("--max-steps", type=int, default=None, help="途中で止める step(確かめるとき)")
    p.add_argument("--set", nargs="*", default=[], help="設定の上書き key=value")
    a = p.parse_args()
    over = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
    base = dict(name=a.name, out_dir=str(common.RUNS_DIR), backbone=a.backbone, pretrained=a.pretrained)
    if a.method == "dinov3":
        files = sorted(str(f) for d in common.SYNTH_IMAGES for f in Path(d).glob("*.jpg"))[: a.max_files]
        cfg = dinov3.DINOv3Config(**base, max_steps=a.max_steps, **over)
        dinov3.train_dinov3(cfg, dinov3.ImageListDataset(files, cfg.crops))
        return
    files = common.kinetics_files()[: a.max_files]
    print(f"動画 {len(files)} 本", flush=True)
    if a.method == "dora":
        cfg = dora.DoRAConfig(**base, max_steps=a.max_steps, **over)
        dora.train_dora(cfg, dora.VideoClipDataset(files, cfg))
    else:
        cfg = vjepa.VJEPAConfig(**base, **over)
        vjepa.train_vjepa(cfg, vjepa.VideoClips(files, cfg), vjepa.VideoClips(files, cfg, as_image=True), max_steps=a.max_steps)


if __name__ == "__main__":
    main()
