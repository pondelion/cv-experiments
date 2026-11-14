# cv-experiments

CV の基本タスク(セグメンテーション・物体検出・キーポイント検出・追跡・OCR など)を、スクラッチ実装で段階的に検証していく実験のリポジトリ。

| ディレクトリ | 中身 |
|---|---|
| [segmentation/](segmentation/) | セグメンテーションの実験(例: [segmentation/instance_segmentation/](segmentation/instance_segmentation/)) |
| [object_detection/](object_detection/) | 物体検出の実験 |
| [keypoint_detection/](keypoint_detection/) | キーポイント検出の実験 |
| [mot/](mot/) | 複数物体の追跡の実験 |
| [ocr/](ocr/) | OCR の実験 |
| [datasets/](datasets/) | データセット(評価のデータセット・合成データ)の置き場 |
| [vendor/](vendor/) | 手で取ってくる他の人のコード・モデルのファイルの置き場 |

実験の方針はタスクごとの README に書く(例: [segmentation/instance_segmentation/README.md](segmentation/instance_segmentation/README.md#方針))。
