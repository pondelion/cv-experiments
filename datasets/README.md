# datasets

CV の実験で使うデータセットの置き場。データは git の対象外(`.gitignore`)で、この README だけを git で管理する。

| ディレクトリ | データセット | 中身 | 置き方 |
|---|---|---|---|
| `coco/` | COCO 2017 | `train2017/`(118,287 枚)・`val2017/`(5,000 枚)・`annotations/`(instances・person_keypoints・captions) | 公式の zip を展開 |
| `voc/VOCdevkit/VOC2012/` | Pascal VOC 2012 trainval | 画像 17,125 枚。インスタンスのマスク(`SegmentationObject`)は train 1,464 枚・val 1,449 枚 | 配布の tar を展開(ミラー pjreddie.com から取得し、MD5 が公式の値と一致することを確認) |
| `sets/` | 上のデータを、実験のクラス(VOC の 20 クラス)にそろえた COCO 形式の JSON | 評価のセット・実データの学習のセット | `lib_common/datasets.py` で作る |

## 出典・利用条件

- COCO: https://cocodataset.org/ 。ラベルは CC BY 4.0、画像は Flickr の各画像の利用条件に従う
- Pascal VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ 。画像は Flickr の各画像の利用条件に従う
