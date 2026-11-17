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
| [site/](site/) | デモのページ(GitHub Pages)の中身 |

## デモのページ(GitHub Pages)

リポジトリのトップの `index.html` がデモのページ。学習したモデル(ONNX)をブラウザの中だけで動かし(ONNX Runtime Web。WebGPU、
使えなければ WASM)、動画に結果を重ねて表示する。動画は次から選べる。

- YouTube: URL を入れると公式の埋め込みで再生し、このタブの画面を埋め込みの部分だけ取り込んで(`getDisplayMedia` + Region Capture)
  推論する。動画のファイルは取ってこない。PC 版の Chrome・Edge(104 以降)だけで動く。結果は埋め込みに重ねず横に並べる
  (重ねると、描いた結果も取り込まれるため)
- 手元の動画・画像のファイル(画像は 1 回だけ推論し、スコアの閾値を変えると描き直す)、カメラ

中身:

| ファイル | 中身 |
|---|---|
| `site/registry.json` | タスクとモデルの一覧(表示名、ONNX の場所、入力の大きさ・正規化、クラス名、出力の形 `format` など) |
| `site/app.js` | 画面と、推論の繰り返し(タスクによらない) |
| `site/lib/` | タスクによらない部品: `runtime.js` と `ort_worker.js`(ONNX Runtime Web を Web Worker の中で動かす。準備や推論が遅くてもページは固まらない)、`sources.js`(YouTube・動画・カメラ)、`image.js`(前処理など) |
| `site/tasks/{タスク}.js` | タスクごとの前処理・後処理・描画(`createTask(model)` が `{preprocess, postprocess, draw}` を返す) |
ONNX のファイルは GitHub ではなく Hugging Face のモデルのリポジトリ
[tkyk1234/cv-experiments-models](https://huggingface.co/tkyk1234/cv-experiments-models) に置き、ページはそこから直接読む
(実験が増えても GitHub の容量の上限に当たらないように。Hugging Face のファイルはブラウザから読める)。フォルダの階層はこのリポジトリの
タスクの階層に合わせる(例: `segmentation/instance_segmentation/`)。

- **モデルを足すとき**: ONNX を Hugging Face のリポジトリの同じタスクのフォルダに上げ、`site/registry.json` のそのタスクの `models` に
  1 項目足す(`onnx` は `https://huggingface.co/tkyk1234/cv-experiments-models/resolve/main/{タスクの階層}/{ファイル名}`)。出力の形が
  新しいときは、`site/tasks/{タスク}.js` の後処理に `format` の分岐を足す
- **タスクを足すとき**: `site/tasks/{タスク}.js` を作り、`registry.json` の `tasks` に 1 項目足す(`module` にそのファイルを書く)
- 公開: GitHub の Settings → Pages で「Deploy from a branch」・`main` ブランチの `/ (root)` を選ぶ
- 手元で見るとき: リポジトリのトップで `python -m http.server` を動かし、`http://localhost:8000/` を開く(ファイルを直接開くと、
  ブラウザの制限で一覧の JSON とモデルを読めない。画面の取り込みも `localhost` か https でしか使えない)

実験の方針はタスクごとの README に書く(例: [segmentation/instance_segmentation/README.md](segmentation/instance_segmentation/README.md#方針))。
