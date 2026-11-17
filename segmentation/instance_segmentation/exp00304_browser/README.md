# exp00304: ブラウザで動かす(ONNX Runtime Web の WebGPU・WASM)

## 目的

学習したモデルを、ブラウザの中だけで動かす(サーバーなし)。GitHub Pages のデモのページ(リポジトリ直下の `index.html` と `site/`)に
載せる。動画の入力は YouTube の URL(埋め込み)・手元の動画と画像のファイル・カメラ。

## 使ったもの

| 部分 | 中身 |
|---|---|
| 実行 | onnxruntime-web 1.31.1(jsDelivr から読む)。WebGPU を先に試し、使えなければ WASM(CPU) |
| モデル | exp00301 の `export_onnx.py --webgpu` で書き出した fp32 の ONNX を、exp00302 の `quantize_weights.py` で重みだけ INT8 にしたもの。Hugging Face の [tkyk1234/cv-experiments-models](https://huggingface.co/tkyk1234/cv-experiments-models) の `segmentation/instance_segmentation/` に置き、ページはそこから直接読む |
| ページのコード | `site/app.js`(画面)、`site/lib/runtime.js`・`site/lib/ort_worker.js`(ONNX Runtime Web を Web Worker で動かす)、`site/lib/sources.js`(動画の入手)、`site/tasks/instance_segmentation.js`(前後の処理と描画)、`coi-sw.js`(下の問題 4) |
| 確かめ方 | Windows の Chrome を headless で動かし、ページの自己テスト(URL に `?selftest=画像のパス&backend=webgpu&model=...&onnx=...`)の結果の表示を、Chrome のデバッグの接続(DevTools Protocol)で読む。1 回ごとに制限時間(30〜120 秒)をつけ、止まっても待ち続けないようにした |

## 起きた問題と直し方

| # | 問題 | 原因 | 直し方 |
|---|---|---|---|
| 1 | GitHub のファイルの上限(1 ファイル 100MB)を超える。実験が増えるとリポジトリの容量も足りなくなる | fp32 の ONNX が GTR 系 146MB・YOLO26-seg 系 180MB | 重みを INT8 にして 42MB・46MB(問題 8)。置き場所は Hugging Face のモデルのリポジトリにした(公開のファイルはブラウザから直接読める。Google Drive の公開のリンクはブラウザから読めないという報告が多いので使わない) |
| 2 | モデルの準備や推論の間、ページ全体が固まる | ONNX Runtime Web をページの処理と同じ流れ(メインスレッド)で動かしていた | Web Worker の中で動かす。準備に 120 秒の制限時間をつけ、超えたら Worker を止めて次の実行の場所を試す |
| 3 | YouTube の動画を推論に入れられない | 別のサイトの動画の画素は読めない(ブラウザの決まり)。動画のファイルを取ってくるのは規約上も避ける | 公式の埋め込みで再生し、このタブの画面の取り込み(`getDisplayMedia`)を、埋め込みの場所だけに切り抜く(Region Capture: `CropTarget.fromElement` → `track.cropTo`)。結果は埋め込みに重ねると取り込まれてしまうので、横に並べて描く。PC 版の Chrome・Edge だけで動く |
| 4 | WASM が遅い(1 コマ 2.7 秒、0.4 コマ毎秒) | WASM が 1 つのスレッドで動いていた。複数のスレッドには、ページが cross-origin isolated(COOP・COEP のヘッダー)である必要があるが、GitHub Pages ではヘッダーを足せない | service worker(`coi-sw.js`)で応答にヘッダーを足す(初めて開いたときに 1 回だけ読み込み直す)。COEP は `credentialless` にし、YouTube の iframe も `credentialless` にする。スレッドの数は CPU のコアの数 − 1(最大 8)。8 スレッドで 1 コマ 1.18 秒(YOLO26-seg 系、問題 8 の静的な INT8 のとき) |
| 5 | 再生を止めた・共有を切った後に止まる、最初のコマのまま変わらない、推論のエラーが表示されない | コマの待ち方と、エラーの扱いが足りなかった | `requestVideoFrameCallback` で次のコマを待つ(1 秒で打ち切る)。取り込みの終わり(track の `ended`)で止める。ループの中のエラーを捕まえて状態の欄に出す |
| 6 | WebGPU でモデルの準備が終わらない | 下の「WebGPU で準備が終わらない原因」 | exp00301 の `export_onnx.py --webgpu` で、同じ計算を次元の少ない形に差し替えて書き出す |
| 7 | ONNX を読めない(`optimizer_execution_frame.cc` のエラー) | 書き出した ONNX の重みが別のファイル(`model.onnx.data`)に分かれていた。ブラウザにはグラフのファイルしか渡していない | 重みを含めた 1 つのファイルにする(量子化した ONNX は 1 つのファイルで保存される) |
| 8 | 静的な INT8 が WebGPU で遅い(1 回の推論 GTR 系 4.2 秒・YOLO26-seg 系 2.4 秒) | 静的な量子化は計算も 8 ビットの整数で行うが、その演算(`QuantizeLinear`・`QLinearConv` など)に WebGPU が対応しておらず、その部分が CPU で動く | 重みだけの INT8 にする(下の「モデルの形式」) |
| 9 | GTR 系の予測が正しくない(スコアがほぼ 0) | ブラウザではなく静的な量子化で壊れていた(Python の ONNX Runtime でも同じ)。Gather・GatherElements の量子化が原因(exp00302 の README) | 問題 8 で重みだけの INT8 にしたので、サイトのモデルには当てはまらない。静的な量子化をするときは、この 2 つを fp32 のまま残す |

途中で間違えたこと: 問題 6 の原因が分かる前に、モデルの形式(静的な INT8 → 重みだけの INT8)を変えて直そうとした。fp32 でも同じように
止まったので、形式は原因ではなかった。原因の演算を 1 つずつ切り分けるまで、直し方を決めないこと。

## WebGPU で準備が終わらない原因

演算 1 つだけの小さい ONNX を作り、1 つずつ WebGPU で試した(Windows の Chrome、GPU のシェーダーの変換は Chrome の既定の DXC と、
古い FXC(`--disable-dawn-features=use_dxc`)の両方)。

| 演算(試した形) | DXC(既定) | FXC |
|---|---|---|
| 行列の掛け算 2 次元 | すぐ終わる | すぐ終わる |
| 行列の掛け算 3 次元(batch のある掛け算) | 1 回目の推論に約 24 秒 | すぐ終わる |
| 行列の掛け算 4 次元(注意の (1, 頭, n, d)) | 終わらない | — |
| 要素ごとの掛け算、5 次元どうしで形が同じ | すぐ終わる | すぐ終わる |
| 要素ごとの掛け算、5 次元で別の形を広げる(broadcast。例 (1, 300, 8, 12, 2) × (12, 1)) | 終わらない | 終わらない |
| 同、4 次元で broadcast | 1 回目の推論に 2.5〜3.7 秒 | 同じくらい |
| 同、3 次元で broadcast、スカラー・1 次元のベクトルとの broadcast | すぐ終わる | すぐ終わる |
| 畳み込み、Linear(3 次元の入力 × 2 次元の重み)、GridSample、並べ替え | すぐ終わる | すぐ終わる |

ONNX Runtime Web の WebGPU は、演算の形(次元の数・広げ方)ごとに GPU のシェーダーを作る。次元の多い行列の掛け算と broadcast は、
そのシェーダーの準備が極端に遅いか、終わらない。モデルの中でこれにあたる場所:

- YOLO26-seg 系: PSA の注意(4 次元の行列の掛け算)
- GTR 系: encoder の AIFI と decoder の自己注意(`nn.MultiheadAttention`。3〜4 次元の行列の掛け算)と、deformable な注意で点の位置を
  計算する部分(5 次元の broadcast)

切り分けの順: モデル全体 → 部品ごと(backbone・encoder・クエリの選び出し・decoder の自己注意・deformable な注意)に切った ONNX →
止まった部品の中の 1 つずつの計算。「GridSample が怪しい」などの見込みは、切り分けると外れていた(GridSample だけの ONNX は
すぐ終わり、止まっていたのはその前の点の位置の計算だった)。

`export_onnx.py --webgpu` は、書き出すときだけ次の forward を差し替える(学習のコードは変えない)。

| 差し替え | 中身 |
|---|---|
| `psa_forward_2d`(YOLO26-seg 系の PSA) | 注意を、頭ごとの 2 次元の行列の掛け算 (hw, d) × (d, hw) に分ける |
| `mha_forward_2d`(`nn.MultiheadAttention`) | 同じく頭ごとの 2 次元に分ける(batch 1・マスクなしの推論のとき) |
| `deform_forward_low_rank`(GTR 系の deformable な注意) | 点の位置を (B·M, 頭·点, 2) の 3 次元で計算してから、grid_sample の形に並べ替える |

| 候補 | 差し替えの前と後の出力の差の最大(PyTorch fp32) | WebGPU(DXC)でのモデルの準備 | 1 回の推論(2 回目) |
|---|---|---|---|
| 自作の GTR 系 + hybrid encoder | クラスの値 4e-6・枠 4e-7・マスク 4e-5(上位 100 個の並びも同じ) | 1.7 秒 | 0.58 秒 |
| 自作の YOLO26-seg 系 | 6e-5 | 2.0 秒 | 0.57 秒 |

(fp32 の 1 つのファイルで試した値。時間は 1 回ずつ測っただけの目安)

## モデルの形式

| 作り方(exp00302) | 8 ビットにするもの | 計算 | WebGPU | WASM(8 スレッド) |
|---|---|---|---|---|
| 静的な量子化(`quantize.py`) | 重みと、計算の途中の値 | 8 ビットの整数で計算する | 整数の計算に WebGPU が対応せず、その部分が CPU で動く(GTR 系 4.2 秒・YOLO26-seg 系 2.4 秒) | 約 1.0 秒 |
| 重みだけ(`quantize_weights.py`) | ファイルの中の重みだけ | 読み込むときに fp32 に戻して計算する | GPU だけで動く(GTR 系 0.54 秒・YOLO26-seg 系 0.57 秒) | 0.96〜1.35 秒 |

(1 回の推論の時間。同じ画像で 1 回ずつ測っただけの目安。ファイルの大きさはどちらも 40MB 前後)

重みだけの INT8 は WebGPU でも WASM でも動き、調整の画像も要らないので、サイトはこれ 1 つのファイルにした
(GTR 系 42MB・YOLO26-seg 系 46MB)。ブラウザの出力は、同じ画像の Python の ONNX Runtime の出力とほぼ同じ
(GTR 系の上位の検出: ブラウザ chair 0.50・bottle 0.38)。精度(exp00302。ONNX Runtime の CUDA で測った): test の mask AP は GTR 系 31.91・YOLO26-seg 系 30.73 で、fp32
(31.83・30.71)と変わらない。

## 残っていること

- 速さの計測(遅延の中央値など)はまだしていない。上の時間は動作の確かめのときの 1 回ずつの値
- CPU(WASM)では R50 の backbone が重い(1 コマ約 1 秒)。軽い backbone は exp00104 で「軽量・高速と精度のバランス重視」のカテゴリとして比べる
