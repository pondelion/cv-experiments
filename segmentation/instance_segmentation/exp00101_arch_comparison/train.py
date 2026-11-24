"""exp00101: 自作のモデルの学習(COCO train の実データ。既定は 8 クラス・2,000 枚)。

方式(--arch):
    mask2former: 自作の Mask2Former(segmentation/instance_segmentation/lib_common/models/mask2former.py)。
        学習の設定は公式の Mask2Former のもの(lib_common/train.py の TrainConfig の既定の値)
    rfdetr_seg: 自作の RF-DETR-Seg(segmentation/instance_segmentation/lib_common/models/rfdetr_seg.py)。
        学習の設定は公式の RF-DETR-Seg のもの(lib_common/train.py の RFDETR_RECIPE)
    yolo_seg: 自作の YOLO26-seg 系(segmentation/instance_segmentation/lib_common/models/yolo_seg.py)。
        学習の設定は YOLO 系の公開の既定の値(lib_common/train.py の YOLO_RECIPE)
    gtr_seg: 自作の GTR-Seg 系(segmentation/instance_segmentation/lib_common/models/gtr_seg.py)。
        学習の設定は公式の GTR-Seg のもの、データ拡張は基本の lsj(lib_common/train.py の GTR_RECIPE)
    maskrcnn: Mask R-CNN(torchvision の実装、backbone は timm。segmentation/instance_segmentation/lib_common/models/maskrcnn.py)。
        学習の設定は detectron2 の new baselines のもの(lib_common/train.py の MASKRCNN_RECIPE)。exp00103 の参考の値
学習・評価の流れは segmentation/instance_segmentation/lib_common/train.py。

使い方:
    .venv/bin/python -u segmentation/instance_segmentation/exp00101_arch_comparison/train.py --name m2f_r50_cls8_2k
    .venv/bin/python -u segmentation/instance_segmentation/exp00101_arch_comparison/train.py --name rfdetr_self_r50_cls8_2k --arch rfdetr_seg
    (動作確認: --max-steps 30 --eval-n 20)
"""
import argparse
import json

import common

from lib_common.datasets import set_path
from segmentation.instance_segmentation.lib_common.train import (EOMT_RECIPE, GTR_RECIPE, MASKRCNN_RECIPE, PMT_RECIPE,
                                                                 RFDETR_RECIPE, YOLO_RECIPE,
                                                                 TrainConfig,
                                                                 evaluate_saved, train)


def main() -> None:
    """引数で TrainConfig の項目を上書きして学習する。"""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--arch", default="mask2former", choices=["mask2former", "rfdetr_seg", "yolo_seg", "gtr_seg", "eomt", "pmt",
                                                                "maskrcnn"])
    p.add_argument("--train-set", default="coco_train_cls8_2k")  # 最初の 2 本は coco_train_10pct(20 クラス)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--epoch-size", type=int, default=None,
                   help="1 エポックの画像の枚数(省略すると学習のデータを 1 周。指定すると毎エポック無作為に選ぶ)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--backbone", default=None, help="timm のモデル名(省略すると方式の設定。なければ resnet50.a1_in1k)")
    p.add_argument("--max-steps", type=int, default=None)
    # 最初のセット(dev)で途中の評価・最良の重み・PQ の閾値を選び、精度は test・voc で見る(最初の 2 本は 20 クラスの同じ 3 つ)
    p.add_argument("--eval-sets", nargs="+", default=["coco_val_dev_cls8", "coco_val_test_cls8", "voc_val_cls8"])
    p.add_argument("--test-sets", nargs="+", default=None,
                   help="学習の途中に全部の画像で評価して記録する test の役のセット(最良の重みの選択には使わない)")
    p.add_argument("--test-every", type=int, default=10, help="--test-sets の評価の間隔(エポック。--eval-every の倍数)")
    p.add_argument("--video-dev-set", default=None,
                   help="MinVIS の照合をかけて動画の AP を測る dev の動画のセット(あれば最良の重みは dev の動画の AP で選ぶ)")
    p.add_argument("--video-test-set", default=None, help="同じく test の役の動画のセット(--test-every ごと。記録だけ)")
    p.add_argument("--schedule", default=None, help="学習率の変化(train.build_scheduler の名前。constant なら warmup の後は一定)")
    p.add_argument("--eval-only", action="store_true", help="学習せず、runs/{名前}/ の重み(best.pt、なければ last.pt)で評価し直す")
    p.add_argument("--augmentation", default=None,
                   help="学習のデータ拡張(transforms.build_train_transforms の名前。省略すると方式の公式の設定)")
    p.add_argument("--no-multi-scale", action="store_true", help="rfdetr_seg: まとめごとの一辺の切り替えをしない(過学習の確認用)")
    p.add_argument("--model-kwargs", default=None,
                   help='モデルを作るときに足す引数の JSON(exp00104 の軽い構成。例: \'{"dim": 128, "mask_stride": 8}\')')
    p.add_argument("--eval-resize", choices=["pad", "square"], default=None,
                   help="評価の変換(省略すると方式の設定。比較の基本の条件は pad)")
    p.add_argument("--no-compile", action="store_true", help="torch.compile を使わない")
    p.add_argument("--compile-backbone", action="store_true",
                   help="モデル全体を compile しない方式(gtr_seg など)でも、形が変わらない backbone だけを torch.compile する")
    p.add_argument("--ifan", action="store_true", help="eomt: iFAN の学習の仕組みと推論の点数を足す")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation steps(micro-batch size = batch-size / grad-accum。batch size は --batch-size のまま)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="gtr_seg: backbone とマスクの head で gradient checkpointing を使う(結果は同じで GPU のメモリが減る)")
    p.add_argument("--lr", type=float, default=None, help="学習率(backbone 以外。省略すると方式の設定)")
    p.add_argument("--clip", type=float, default=None, help="勾配の大きさの上限(省略すると方式の設定)")
    p.add_argument("--weight-decay", type=float, default=None, help="weight decay(省略すると方式の設定)")
    p.add_argument("--backbone-lr-mult", type=float, default=None, help="backbone の学習率の倍率(省略すると方式の設定)")
    p.add_argument("--pixel-decoder", choices=["fpn", "fpn_m2f", "msdeform", "unet", "unet_gn", "unet_gn_wide"], default=None,
                   help="mask2former の pixel decoder(省略すると fpn)")
    p.add_argument("--mask-pos", choices=["none", "sine"], default=None,
                   help="mask2former のマスクを作る画素の特徴に位置の情報を足すか(省略すると none)")
    p.add_argument("--gtr-encoder", choices=["c2f", "hybrid"], default=None,
                   help="gtr_seg の encoder(省略すると c2f。hybrid は RT-DETR の hybrid encoder)")
    p.add_argument("--gtr-mask-loss", choices=["points", "box_crop"], default=None,
                   help="gtr_seg のマスクの損失(省略すると points。box_crop は D-FINE-seg の掛け方)")
    p.add_argument("--optimizer", choices=["adamw", "sgd", "musgd"], default=None, help="最適化の手法(省略すると方式の設定)")
    p.add_argument("--momentum", type=float, default=None,
                   help="sgd の momentum / yolo_seg の adamw の 1 次の moment の係数(省略すると方式の設定)")
    p.add_argument("--max-vram-gb", type=float, default=12.0)
    p.add_argument("--warmup", type=int, default=None, help="学習率を上げていく回数(省略すると方式の公式の設定)")
    a = p.parse_args()
    recipe = dict({"rfdetr_seg": RFDETR_RECIPE, "yolo_seg": YOLO_RECIPE, "gtr_seg": GTR_RECIPE,
                   "eomt": EOMT_RECIPE, "pmt": PMT_RECIPE, "maskrcnn": MASKRCNN_RECIPE}.get(a.arch, {}))
    recipe["backbone"] = a.backbone or recipe.get("backbone", "resnet50.a1_in1k")
    if a.augmentation is not None:
        recipe["augmentation"] = a.augmentation
    if a.warmup is not None:
        recipe["warmup"] = a.warmup
    if a.schedule is not None:
        recipe["schedule"] = a.schedule
    if a.test_sets:
        recipe["test_sets"], recipe["test_every"] = a.test_sets, a.test_every
    if a.video_dev_set:
        recipe["video_dev_set"], recipe["video_test_set"] = a.video_dev_set, a.video_test_set
    if a.no_multi_scale:
        recipe["multi_scale"] = False
    if a.no_compile:
        recipe["compile"] = False
    if a.grad_checkpoint:
        recipe["grad_checkpoint"] = True
    if a.grad_accum > 1:
        recipe["grad_accum"] = a.grad_accum
    if a.ifan:
        recipe["ifan"] = True
    if a.eval_resize is not None:
        recipe["eval_resize"] = a.eval_resize
    if a.model_kwargs:
        recipe["model_kwargs"] = json.loads(a.model_kwargs)
    for k in ("lr", "clip", "weight_decay", "backbone_lr_mult", "pixel_decoder", "mask_pos", "optimizer", "momentum", "gtr_encoder", "gtr_mask_loss"):
        if getattr(a, k) is not None:
            recipe[k] = getattr(a, k)
    cfg = TrainConfig(name=a.name, train_json=str(set_path(a.train_set)), out_dir=str(common.RUNS_DIR),
                      eval_sets=a.eval_sets,
                      results_dir=str(common.RESULTS_DIR), epochs=a.epochs, epoch_size=a.epoch_size, eval_every=a.eval_every, eval_n=a.eval_n,
                      batch_size=a.batch_size, size=a.size, max_steps=a.max_steps,
                      max_vram_gb=a.max_vram_gb, compile_backbone=a.compile_backbone, **recipe)
    evaluate_saved(cfg) if a.eval_only else train(cfg)


if __name__ == "__main__":
    main()
