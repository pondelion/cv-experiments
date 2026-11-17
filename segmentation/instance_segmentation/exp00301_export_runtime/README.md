# exp00301: 書き出しと実行の環境

## 目的

学習したモデルを、PyTorch のまま・ONNX に書き出して ONNX Runtime・TensorRT で動かしたときの、速さと精度を比べる。
後の exp00302(量子化。INT8 まで小さくする)と exp00304(ブラウザの ONNX Runtime Web で動かす。GitHub Pages のデモ)の土台にする。
ここでは、候補のモデルが **ONNX に書き出せて、PyTorch と同じ出力になり、ONNX Runtime・TensorRT で動くか** を確かめる。

## 候補のモデル

精度が高く、ファイルが小さく、ブラウザで動かしやすいもの(学習はどれも coco_train_cls8_2k・R50・入力 512・100 エポック)。

| 候補 | 学習した exp | test の mask AP | パラメータの数 | fp32 の大きさの目安 | ブラウザで動かすときの注意 |
|---|---|---|---|---|---|
| 自作の GTR 系 + hybrid encoder | exp00103(A) | 31.8 | 3,600 万 | 144MB | deformable な注意で双線形のサンプリング(ONNX の `GridSample`)を使う。onnxruntime-web の WebGPU は `GridSample`(opset 16〜19)に対応 |
| 自作の YOLO26-seg 系 | exp00101 | 30.7 | 4,690 万 | 188MB | 畳み込みと普通の注意だけ。NMS なし |

大きさの目安はパラメータの数 × 4 バイト。fp16 で半分、INT8 で 1/4。GitHub の 1 ファイルの上限は 100MB なので、INT8 まで小さくする
(exp00302)。

## 比べる実行の形式

| 形式 | 中身 |
|---|---|
| PyTorch fp32 | 学習のコードのモデルをそのまま |
| PyTorch bf16 | 自動の型変換(autocast)で bf16 |
| PyTorch bf16 + torch.compile | 上を torch.compile |
| ONNX Runtime GPU fp32 | ONNX に書き出し、onnxruntime-gpu(CUDA)で動かす |
| ONNX Runtime CPU fp32 | 同じ ONNX を CPU で動かす |
| TensorRT fp16 | 同じ ONNX から TensorRT の fp16 のエンジンを作って動かす |

## 測るもの

- 書き出しの確認: ONNX に書き出せるか、ONNX Runtime の出力が PyTorch fp32 と一致するか(出力の差の最大)
- 遅延(インスタンスセグメンテーションの README の「速度の指標」): 前後の処理込みと、モデルだけ。batch 1・入力 512、warmup を捨てて
  複数回測った中央値と p90。GPU は処理の終わりを待ってから時刻を取る
- 精度: 実際に動かす形式で test(`coco_val_test_cls8`・`voc_val_cls8`)の mask AP を測る(速さと精度で形式を混ぜない)
- GPU のメモリ(推論中の最大)、パラメータの数・FLOPs

## 進め方

1. 実行の環境を用意する(onnx・onnxscript・onnxruntime-gpu。TensorRT は後回し)
2. 候補を ONNX に書き出し、ONNX Runtime の出力を PyTorch と比べる(書き出せない部品があれば、計算が同じ形に書き換える)
3. 形式ごとに遅延と精度を測る

## 結果

### 1. 実行の環境

| パッケージ | バージョン | 備考 |
|---|---|---|
| torch | 2.11.0+cu128 | もとから(変えていない) |
| onnx | 1.23.2 | 足した |
| onnxscript | 0.7.2 | 足した(torch.onnx.export の dynamo=True の書き出しで使う) |
| onnxruntime-gpu | 1.32.0 | 足した。使える実行の場所: TensorRT・CUDA・CPU |
| TensorRT | 入れていない | TensorRT fp16 の形式は後回し(ONNX Runtime の TensorRT の実行の場所も、TensorRT の本体がないので使えない) |

確認: 小さいモデル(畳み込み + ReLU)を `torch.onnx.export(dynamo=True, opset_version=18)` で書き出し、ONNX Runtime の CUDA と CPU で
動かして、PyTorch と出力の差の最大が 4.8e-7 になることを確かめた。GPU は RTX 5090。

### 2. ONNX への書き出しと、出力の一致

`export_onnx.py` で、学習の記録(config.json・best.pt)からモデルを作って書き出した(opset 18、batch 1・512 × 512 の固定)。
coco_val_test_cls8 の先頭の画像を入れ、PyTorch fp32 と ONNX Runtime の出力を比べた。どちらの候補も、書き換えなしで書き出せた。

| 候補 | ONNX の大きさ(グラフ + 重み) | ONNX Runtime CPU | ONNX Runtime CUDA(TF32 なし) | ONNX Runtime CUDA(TF32 あり) |
|---|---|---|---|---|
| 自作の GTR 系 + hybrid encoder | 2.2MB + 143MB | 一致(クラスの値 2e-5・枠 3e-6・マスク 1e-3) | 一致(8e-4・3e-5・1e-2) | ずれる(3.5・0.83・82) |
| 自作の YOLO26-seg 系 | 1.1MB + 179MB | 一致(クラスの値 4e-5・枠 1e-3 画素・係数 4e-6・原型 3e-5) | 一致(1e-3・2e-2 画素・1e-4・4e-3) | ずれる(10・275 画素・1.2・2e-2) |

数は出力の差の最大(値の大きさの最大は、GTR 系のクラスの値 7.3・マスク 104、YOLO26-seg 系のクラスの値 17・枠 510 画素)。

- 比べ方: クラスの値の最大の大きい順に並べた上位 100 個(評価の後処理と同じ数)で比べた。GTR 系は上位の候補を選ぶ処理(TopK)で
  クエリを選ぶので、計算の細かい差で同点に近い候補の順が入れ替わると、出力のクエリの順も入れ替わる(番号どうしで比べると、
  CUDA で大きな差に見えた)
- TF32(行列の掛け算を少し粗い精度で行う CUDA の設定。ONNX Runtime の CUDA では既定で使う)ありでは、上位 100 個の中身がずれる。
  mask AP はほぼ変わらない(下の「4. 精度」)ので、境目の候補が入れ替わっただけで、精度は落ちていない
- ONNX Runtime の CUDA で、cuDNN の一部の計算の手順が見つからないというエラーの表示が出るが、別の手順で計算を続け、結果は上のとおり

### 3. ブラウザ用の書き出し(`--webgpu`)

`export_onnx.py --webgpu` は、ブラウザの WebGPU で準備が終わらない形の演算を、計算が同じで次元の少ない形に差し替えて書き出す。
原因・差し替えの中身・確かめた結果は exp00304 の README。

### 4. 精度

`eval_onnx.py` で、学習のときの最後の評価と同じ手順(dev で PQ の閾値を選び、test・voc で測る)で測った。値は %。

| 候補 | 形式 | test mask AP | voc mask AP | test box AP | voc box AP | test PQ | voc PQ |
|---|---|---|---|---|---|---|---|
| 自作の GTR 系 + hybrid encoder | PyTorch fp32(学習のときの評価) | 31.85 | 38.41 | 34.42 | 48.71 | 41.17 | 48.37 |
| | ONNX Runtime CUDA(TF32 なし) | 31.83 | 38.35 | 34.42 | 48.82 | 41.45 | 48.05 |
| | ONNX Runtime CUDA(TF32 あり) | 31.81 | 38.35 | 34.47 | 48.86 | 41.37 | 48.13 |
| 自作の YOLO26-seg 系 | PyTorch fp32(学習のときの評価) | 30.72 | 35.75 | 33.63 | 45.13 | 42.05 | 47.54 |
| | ONNX Runtime CUDA(TF32 なし) | 30.71 | 35.77 | 33.73 | 45.24 | 42.59 | 47.66 |
| | ONNX Runtime CUDA(TF32 あり) | 30.73 | 35.76 | 33.74 | 45.24 | 42.63 | 47.73 |

- ONNX にしても、mask AP・box AP は PyTorch と 0.1 ポイント以内で同じ。TF32 のあり・なしでも変わらない
- PQ は最大 0.6 ポイントほど違う。PQ は dev で選ぶ閾値で決まる値なので、その選ばれた閾値が違ったのかもしれないが、確かめていない
- GTR 系の TF32 なしの結果の JSON は、保存のフォルダを作っていなかった不具合で残っていない(値はログ `runs/gtr_hybrid_ort_cuda_fp32_tf32_0.log` から)
- 速さ(遅延)はまだ測っていない
