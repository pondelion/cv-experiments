# datasets

CV の実験で使うデータセットの置き場。データは git の対象外(`.gitignore`)で、この README だけを git で管理する。

| ディレクトリ | データセット | 中身 | 置き方 |
|---|---|---|---|
| `coco/` | COCO 2017 | `train2017/`(118,287 枚)・`val2017/`(5,000 枚)・`annotations/`(instances・person_keypoints・captions) | 公式の zip を展開 |
| `voc/VOCdevkit/VOC2012/` | Pascal VOC 2012 trainval | 画像 17,125 枚。インスタンスのマスク(`SegmentationObject`)は train 1,464 枚・val 1,449 枚 | 配布の tar を展開(ミラー pjreddie.com から取得し、MD5 が公式の値と一致することを確認) |
| `sets/` | 上のデータを、実験のクラス(VOC の 20 クラス)にそろえた COCO 形式の JSON | 評価のセット・実データの学習のセット | `lib_common/datasets.py` で作る |
| `vipseg/VIPSeg/` | VIPSeg(動画のパノプティックセグメンテーション) | `imgs/{動画}/`(フレーム)・`panomasks/{動画}/`(16 ビットの PNG。物体は「クラス × 100 + 物体の番号」、クラスの値は 1 始まりで 0 は VOID)・`train.txt`(2,806 本)・`val.txt`(343 本)・`test.txt`(387 本、正解なし)・`VIPSeg_720P/panoVIPSeg_categories.json`(124 クラス、id は 0 始まり) | 公式の Google Drive の `VIPSeg.tar` を展開 |
| `lvvis/` | LV-VIS(オープンボキャブラリーの動画インスタンスセグメンテーション) | `train/JPEGImages/`(3,076 本)・`val/JPEGImages/`(837 本)・`train_instances.json`・`val_instances.json`・`val_instances_oracle.json`(YouTube-VIS 2019 と同じ形式) | 公式の Google Drive の zip を展開 |

## 出典・利用条件

- COCO: https://cocodataset.org/ 。ラベルは CC BY 4.0、画像は Flickr の各画像の利用条件に従う
- Pascal VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ 。画像は Flickr の各画像の利用条件に従う
- VIPSeg: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset 。非商用の研究に限る
- LV-VIS: https://github.com/haochenheheda/LVVIS 。CC BY-NC-SA 4.0(非商用の研究に限る)
