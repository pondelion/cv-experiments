"""インスタンスセグメンテーションの画像の変換(データ拡張を含む)。albumentations で作り、InstanceDataset に外から渡す。

どれも画像と、物体ごとのマスク(masks)に同じ幾何の変換をかける(マスクは最近傍で拡大縮小する)。
正規化は ImageNet の平均・標準偏差(timm の backbone の事前学習と同じ)。

- train_transforms: Mask2Former の COCO の学習と同じ考え方の LSJ(Large Scale Jitter)と左右の反転
    長い辺を S × 0.1〜2.0 の大きさにランダムに拡大縮小 → S × S に足りない所を左上に寄せて 0 で埋める → S × S をランダムに切り抜く
- eval_transforms: 長い辺を S にそろえ、S × S になるよう右と下を 0 で埋める
    (予測を元の大きさに戻すときは、倍率 S / 元の長い辺 と、左上に寄せたことを使う。postprocess.to_predictions)
- rfdetr_train_transforms / square_eval_transforms: RF-DETR の公式の学習・評価の変換(縦横の比を保たない正方形)

形の記号: S = 入力の一辺の画素数(size)
"""
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _pad(size: int) -> A.PadIfNeeded:
    """S × S に足りない所を、左上に寄せて 0 で埋める変換を作る(画像もマスクも)。

    Args:
        size: 入力の一辺の画素数 S。

    Returns:
        A.PadIfNeeded。
    """
    return A.PadIfNeeded(min_height=size, min_width=size, position="top_left", border_mode=cv2.BORDER_CONSTANT,
                         fill=0, fill_mask=0)


def train_transforms(size: int = 512, scale_range: tuple[float, float] = (0.1, 2.0)) -> A.Compose:
    """学習の変換(LSJ + 左右の反転)。

    Args:
        size: 入力の一辺の画素数 S。
        scale_range: 長い辺を S の何倍にするかの範囲(Mask2Former の LSJ は 0.1〜2.0)。

    Returns:
        A.Compose。呼ぶと {"image": (3, S, S) の float32 の tensor, "masks": (物体の数, S, S) の tensor} を返す。
    """
    # 長い辺の候補を 1 画素刻みで並べ、その中からランダムに選ばせる(拡大縮小を 1 回で済ませる)
    sizes = list(range(max(1, round(size * scale_range[0])), round(size * scale_range[1]) + 1))
    return A.Compose([
        A.LongestMaxSize(max_size=sizes, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        _pad(size),
        A.RandomCrop(height=size, width=size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def batch_scales(size: int = 512, step: int = 16, offsets: range = range(-5, 6)) -> list[int]:
    """RF-DETR の、まとめ(バッチ)ごとに選ぶ入力の一辺の候補(公式の compute_multi_scale_scales。expanded_scales、
    patch 16・窓 1 のとき)。

    Args:
        size: 基準の一辺 S。
        step: 候補の刻み。
        offsets: S / step からのずれの範囲。

    Returns:
        候補の一辺の並び(S = 512 なら 432, 448, ..., 592)。
    """
    return [(size // step + o) * step for o in offsets if (size // step + o) * step >= 2 * step]


def rfdetr_train_transforms(size: int = 512) -> A.Compose:
    """RF-DETR の公式の学習の変換(公式の make_coco_transforms_square_div_64、multi_scale="per-batch" のとき)。

    次のどちらかを半分ずつの確率で選び、正方形 M × M(M = batch_scales(S) の最大。S = 512 なら 592)にしてから左右の反転。
    縦横の比は保たない(公式と同じ)。まとめごとの大きさの切り替え(432〜592)は学習の繰り返しの中で GPU で行う。
        A: そのまま M × M に拡大縮小
        B: 短い辺を 400・500・600 のどれかにし、一辺 384〜(短い辺) の正方形をランダムな位置で切り抜いて M × M にする
    (公式は切り抜く一辺を [384, min(600, 縦, 横)] から一様に選ぶ。短い辺ごとに範囲を分けた 3 通りにすると同じ選び方になる)

    Args:
        size: 基準の一辺 S。

    Returns:
        A.Compose。返り値は train_transforms と同じ形(一辺は M)。
    """
    m = max(batch_scales(size))
    lin = dict(interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
    crop = A.OneOf([A.Sequential([A.SmallestMaxSize(max_size=s, **lin),
                                  A.RandomSizedCrop(min_max_height=(384, s), size=(m, m), w2h_ratio=1.0, **lin)])
                    for s in (400, 500, 600)], p=1.0)
    return A.Compose([
        A.OneOf([A.Resize(m, m, **lin), crop], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def square_eval_transforms(size: int = 512) -> A.Compose:
    """RF-DETR の公式の評価の変換(縦横の比を保たずに S × S に拡大縮小する)。

    予測を元の大きさに戻すときは、S × S をそのまま元の (縦, 横) に拡大縮小する(postprocess.rfdetr_to_predictions)。

    Args:
        size: 入力の一辺の画素数 S。

    Returns:
        A.Compose(返り値は eval_transforms と同じ形)。
    """
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


YOLO_FILL = 114  # YOLO 系の埋める色(灰色)


def yolo_train_transforms(size: int = 512, mosaic: bool = True) -> A.Compose:
    """YOLO 系の学習の変換(Ultralytics の YOLO の既定のデータ拡張の考え方。値は公開の既定の値: scale 0.5、translate 0.1、
    hsv_h 0.015・hsv_s 0.7・hsv_v 0.4、fliplr 0.5、mosaic 1.0)。

    mosaic=True: 4 枚を 2S × 2S の画面に並べ(各画像は長い辺を S)、中心の近く(画面の 0.25〜0.75)の S × S を切り出す
        → ランダムな拡大縮小(0.5〜1.5 倍)と平行移動(±10%)→ 色の揺らし → 左右の反転。data.MosaicInstanceDataset と使う
        (マスクは物体の番号の地図 mask で受け渡す)
    mosaic=False: 長い辺を S にして灰色で埋める → 同じ拡大縮小・平行移動・色・反転(学習の最後の close_mosaic エポック。
        InstanceDataset と使う)

    Args:
        size: 入力の一辺の画素数 S。
        mosaic: Mosaic を使うか。

    Returns:
        A.Compose。返り値は train_transforms と同じ形(mosaic=True のときマスクは "mask"(S, S)の物体の番号)。
    """
    lin = dict(interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
    first = [A.Mosaic(grid_yx=(2, 2), target_size=(size, size), cell_shape=(size, size), center_range=(0.25, 0.75),
                      fit_mode="contain", fill=YOLO_FILL, fill_mask=0, p=1.0, **lin)] if mosaic else [
        A.LongestMaxSize(max_size=size, **lin),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, fill=YOLO_FILL, fill_mask=0)]
    return A.Compose(first + [
        A.Affine(scale=(0.5, 1.5), translate_percent=(-0.1, 0.1), border_mode=cv2.BORDER_CONSTANT, fill=YOLO_FILL,
                 fill_mask=0, p=1.0, **lin),
        A.ColorJitter(brightness=0.4, contrast=0.0, saturation=0.7, hue=0.015, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def build_train_transforms(name: str, size: int = 512) -> A.Compose:
    """名前から学習の変換を作る(TrainConfig.augmentation。データ拡張の条件を比べるときは、ここに条件を足す)。

    Args:
        name: "lsj"(train_transforms。LSJ 0.1〜2.0 倍 + 左右の反転)/ "rfdetr"(rfdetr_train_transforms。RF-DETR の公式)/
            "none"(データ拡張なし。eval_transforms と同じ。少数の画像に過学習させて実装を確かめるときなど)/
            "none_square"(データ拡張なし。square_eval_transforms と同じ)/ "yolo_mosaic"・"yolo"
            (yolo_train_transforms の Mosaic あり・なし。"yolo_mosaic" は data.MosaicInstanceDataset と使う)。
        size: 入力の一辺の画素数 S。

    Returns:
        A.Compose。
    """
    builders = {"lsj": train_transforms, "rfdetr": rfdetr_train_transforms, "none": eval_transforms,
                "none_square": square_eval_transforms, "yolo_mosaic": yolo_train_transforms,
                "yolo": lambda s: yolo_train_transforms(s, mosaic=False)}
    if name not in builders:
        raise ValueError(f"知らないデータ拡張の名前: {name}")
    return builders[name](size)


def eval_transforms(size: int = 512) -> A.Compose:
    """評価の変換(長い辺を S にそろえて、右と下を埋める)。

    Args:
        size: 入力の一辺の画素数 S。

    Returns:
        A.Compose(返り値は train_transforms と同じ形)。
    """
    return A.Compose([
        A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        _pad(size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
