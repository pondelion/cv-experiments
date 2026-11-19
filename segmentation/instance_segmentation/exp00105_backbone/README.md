# exp00105: backbone の比較

## 目的

backbone だけを差し替え、事前学習の重みの効果を比べる。次の 2 つのカテゴリに分ける。

- **精度重視**: 大きめ(Base 程度)の backbone。exp00103 で最良だった方式(自作の GTR 系 + hybrid encoder。R50 で test 31.8)の上で、
  強い事前学習(DINOv3 など)でどこまで上がるか
- **速度バランス型**: CPU(エッジ機器・ブラウザ)でも動かせる軽い backbone。**exp00104 で決める軽い構成の上で**比べる。今の GTR 系の構成は
  backbone より後ろだけで約 19 GMACs あり、軽い backbone にしても全体は R50 のときの半分程度にしかならない(下の「候補」)ので、
  先に exp00104(backbone は R50 に固定)で、方式ごとに backbone より後ろを軽くした構成を比べて決める。速度バランス型の候補の表は、
  exp00104 の結果を見て直す

## 共通の条件

exp00101 の統一の条件で学習する(train `coco_train_cls8_2k`、val `coco_val_dev_cls8` の先頭 500 枚で最良の重みを選ぶ、test
`coco_val_test_cls8`・`voc_val_cls8`、データ拡張 `lsj`、100 エポック、batch 16、入力 512)。

- 方式と学習の設定は、exp00103 の A(`--arch gtr_seg --gtr-encoder hybrid`、`GTR_RECIPE`)のまま。**変えるのは backbone だけ**
  (backbone の学習率は、ほかの部分の 0.03 倍のまま。ViT 系で合わないと分かったら、層ごとの学習率の減衰を足す版を 1 本だけ別に足す)
- backbone は全部の層を学習する(backbone を固定する・LoRA は、下の「後の段階」)
- 基準: R50(`resnet50.a1_in1k`)の exp00103 の A(test 31.8・voc 38.4)。同じ条件なので学習し直さない
- 古典の方式との比べ: 精度重視の最良の backbone(と、時間があれば 2〜3 番目)で、Mask R-CNN(exp00103 の G。R50 で test 29.6)も
  学習する。R50 では最新の方式との差が 2 前後と小さかったので、強い backbone で差が開くかを見る

## 候補

パラメータの数と計算量(GMACs: 掛け算と足し算の組の回数 × 10⁹)は、入力 512 × 512 で backbone だけを測った値(timm で重みを読まずに作り、
`torch.utils.flop_counter` で数えた)。stride は backbone が出す特徴の、入力に対する縮小の倍率。

GTR 系 + hybrid encoder の backbone より後の部分(encoder・decoder・マスクの head)は、backbone によらずパラメータ約 1,200 万・
約 19 GMACs ある(R50 のとき全体 3,600 万・40.2 GMACs、MobileNetV4-M のとき 1,910 万・22.4 GMACs)。

### 精度重視

| 候補 | timm の名前(重み) | 事前学習 | パラメータ | GMACs | stride | 注意 |
|---|---|---|---|---|---|---|
| (基準)ResNet-50 | `resnet50.a1_in1k` | ImageNet-1K | 2,350 万 | 21.4 | 2〜32 | 比べる基準。結果は exp00103 の A(GTR 系 + hybrid encoder、test 31.8)を使い、学習し直さない |
| ConvNeXt-B(DINOv3) | `convnext_base.dinov3_lvd1689m` | DINOv3(自己教師あり、LVD-1689M)から蒸留 | 8,760 万 | 80.2 | 4〜32 | DINOv3 の ConvNeXt は V2 ではなく元の ConvNeXt の形(DINOv3 の公開の重みに V2 はない) |
| ConvNeXt V2-B | `convnextv2_base.fcmae_ft_in22k_in1k` | FCMAE(自己教師あり)→ ImageNet-22K → 1K | 8,770 万 | 80.2 | 4〜32 | ConvNeXt-B(DINOv3)と同じ大きさで、事前学習だけが違う |
| Swin-B | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | ImageNet-22K → 1K | 8,670 万 | 86.8 | 4〜32 | 特徴が (B, h, w, C) の並びで出るので並べ替える |
| ViT-B/16(DINOv3) | `vit_base_patch16_dinov3.lvd1689m` | DINOv3 | 8,560 万 | 88.0 | 16 だけ | 1 段の特徴しかない(下の「実装の注意」) |
| MaxViT-S | `maxvit_small_tf_512.in1k` | ImageNet-1K(入力 512 で調整済み) | 6,780 万 | 66.9 | 2〜32 | Base(1 億 1,850 万・137.4)は重く学習に約 10 時間かかるので、Small にする |
| EVA-02-B/16 | `eva02_base_patch16_clip_224.merged2b` | EVA-CLIP(画像と文の対応、Merged-2B) | 8,650 万 | 87.7 | 16 だけ | EVA-02 の MIM の版はパッチ 14 で、512 が割り切れない(36.6 個)ので、パッチ 16 の CLIP の版にする |
| Qwen3-VL の画像の部分 | (timm ではない)`Qwen/Qwen3-VL-2B-Instruct` の vision encoder | Qwen3-VL の学習(画像と文) | 約 3 億 | 約 360(層の幅と数からの見積もり) | 16 だけ | 24 層・幅 1024・パッチ 16(2B と 4B で同じ)。5・11・17 層目の特徴を言語モデルに渡す作り(DeepStack)なので、その層の特徴を使える。ViT-L に近い大きさで、学習が最も遅い(1 本 約 20 時間以上)。最後に回し、学習するかはそれまでの結果を見て決める |
| (足す)SigLIP2-B/16 | `vit_base_patch16_siglip_512.v2_webli` | SigLIP2(画像と文の対応、WebLI。入力 512) | 8,640 万 | 87.6 | 16 だけ | 画像と文の対応で学習した ViT の Base の版。Qwen3-VL の画像の部分(幅 1024・24 層は SigLIP2-L と同じ大きさ。Qwen3-VL が SigLIP2 から作ったかは確かめていない)と比べて、VLM として学習した効果の目安にする |
| (足す)ViT-S/16(DINOv3) | `vit_small_patch16_dinov3.lvd1689m` | DINOv3 | 2,160 万 | 22.2 | 16 だけ | R50 と同じ大きさ。exp00103 の ViT-S(ImageNet-1K の DeiT3)と比べると、同じ大きさの ViT で事前学習だけの差が見られる |
| (足す・余裕があれば)C-RADIOv3-B | `vit_base_patch16_radio_v3.c_radio` | 複数の先生(DINOv2・SigLIP・SAM など)からの蒸留 | 8,650 万 | 88.3 | 16 だけ | 密な予測(セグメンテーションなど)向けに作られた backbone。ライセンスは NVIDIA の独自のもの |

### 速度バランス型

| 候補 | 重み(timm の名前、または出どころ) | 事前学習 | パラメータ | GMACs | stride | 注意 |
|---|---|---|---|---|---|---|
| (基準)ResNet-50 | `resnet50.a1_in1k` | ImageNet-1K | 2,350 万 | 21.4 | 2〜32 | 比べる基準。結果は exp00104 の G1 + E2(64) + E3(同じ方式、test 31.4)を使い、学習し直さない |
| MobileNetV4-Conv-M | `mobilenetv4_conv_medium.e500_r256_in1k` | ImageNet-1K | 720 万 | 4.3 | 2〜32 | 畳み込みだけの最新の MobileNet(V5 は下) |
| MobileNetV4-Hybrid-M | `mobilenetv4_hybrid_medium.e500_r224_in1k` | ImageNet-1K | 860 万 | 5.6 | 2〜32 | 注意の層(Mobile MQA)を含む版 |
| MobileNetV3-L | `mobilenetv3_large_100.ra4_e3600_r224_in1k` | ImageNet-1K | 300 万 | 1.1 | 2〜32 | 古典の軽い backbone の参考。正規化は平均・標準偏差 0.5 |
| EfficientNet-B2 | `efficientnet_b2.ra_in1k` | ImageNet-1K | 720 万 | 3.3 | 2〜32 | |
| EfficientNetV2-S | `tf_efficientnetv2_s.in21k_ft_in1k` | ImageNet-21K → 1K | 1,980 万 | 14.8 | 2〜32 | 正規化は平均・標準偏差 0.5 |
| RegNetY-1.6GF | `regnety_016.tv2_in1k` | ImageNet-1K | 1,030 万 | 8.4 | 2〜32 | |
| RegNetY-3.2GF | `regnety_032.tv2_in1k` | ImageNet-1K | 1,790 万 | 16.6 | 2〜32 | R50 に近い計算量で、どちらが良いか |
| FastViT-SA12 | `fastvit_sa12.apple_dist_in1k` | ImageNet-1K(蒸留) | 1,040 万 | 7.7 | 4〜32 | 推論のときに畳み込みをまとめられる作り(re-parameterize) |
| HGNetV2-B2 / B4 | `hgnetv2_b2.ssld_stage2_ft_in1k` / `hgnetv2_b4.ssld_stage2_ft_in1k` | ImageNet-22K → 1K(蒸留) | 600 万 / 1,360 万 | 5.9 / 14.3 | 4〜32 | D-FINE・DEIM など最近の速い DETR 系の検出器の標準の backbone |
| ConvNeXt-T(DINOv3) | `convnext_tiny.dinov3_lvd1689m` | DINOv3 から蒸留 | 2,780 万 | 23.3 | 4〜32 | R50 と同じ計算量で、強い事前学習の効果を見る |
| ResNet-18 | `resnet18.a1_in1k` | ImageNet-1K | 1,120 万 | 9.5 | 2〜32 | 古典の軽い backbone の参考(R50 と同じ系統) |
| TinyViT-5M / 11M | `tiny_vit_5m_224.dist_in22k_ft_in1k` / `tiny_vit_11m_224.dist_in22k_ft_in1k` | ImageNet-22K から蒸留 → 1K | 510 万 / 1,050 万 | 7.6 / 12.6 | 4〜32 | 端末向けの SAM(MobileSAM)の画像の部分 |
| RepViT-M1.5 | `repvit_m1_5.dist_450e_in1k` | ImageNet-1K(蒸留) | 1,360 万 | 11.9 | 4〜32 | 端末向けの SAM(RepViT-SAM)の画像の部分。推論のときに枝をまとめる作り |
| EdgeNeXt-S | `edgenext_small.usi_in1k` | ImageNet-1K | 530 万 | 5.0 | 4〜32 | 畳み込みと注意を混ぜた端末向けの backbone |
| StarNet-S3 | `starnet_s3.in1k` | ImageNet-1K | 550 万 | 3.9 | 4〜32 | CVPR 2024 |
| EfficientFormerV2-S2 | `efficientformerv2_s2.snap_dist_in1k` | ImageNet-1K(蒸留) | 1,220 万 | 7.2 | 4〜32 | 端末向けの畳み込みと注意の混ぜ合わせ |
| MobileOne-S2 | `mobileone_s2.apple_in1k` | ImageNet-1K | 580 万 | 6.8(学習の形) | 2〜32 | 推論のときに枝をまとめる作り(Apple) |
| DINOv3 ViT-Tiny | `vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m` | DINOv3 の系統(LVD-1689M) | 550 万 | 5.6 | 16 だけ | DEIMv2-S の backbone と同じ大きさ。ViT 用のアダプタが要る(下の「実装の注意」) |
| DEIMv2 の ViT-Tiny | timm ではない: DEIMv2 の公式の重み `vitt_distill.pt`(Google Drive で公開) | DINOv3-S からの蒸留(検出では学習していない) | 約 550 万 | 約 5.6 | 16 だけ | DEIMv2-S の backbone そのもの。ViT 用のアダプタが要る |

候補は、指定のもの(MobileNet・EfficientNet・RegNetY・FastViT)に、下の「論文の軽量モデルが使う backbone(下調べ)」と timm の下調べから
足したもの。ViT 用のアダプタが要る 2 つは、アダプタを実装してから学習する。

### 論文の軽量モデルが使う backbone(下調べ。2026-11)

| 論文のモデル | backbone | 重み | 扱い |
|---|---|---|---|
| D-FINE・DEIM の N・S | HGNetV2-B0〜B2 | timm にある | HGNetV2-B2・B4 を候補に入れた |
| DEIMv2 の Atto〜N | HGNetV2-B0 の段を削ったもの | 公式のリポジトリ | 削っただけで HGNetV2 と同じ系統なので、候補の HGNetV2 で代える |
| DEIMv2 の S・M | DINOv3-S から蒸留した ViT-Tiny(幅 192・12 層)に、細かい解像度の情報を足す小さい CNN(STA)を付け、3・7・11 層目の特徴から stride 8・16・32 を作る | ViT-Tiny の重みは Google Drive で公開 | 候補に入れた(DEIMv2 の ViT-Tiny) |
| GTR の S | ゲート付きの線形注意の ViT(蒸留) | backbone だけの事前学習の重みは公開されていない(COCO・Objects365 などで学習した検出器の重みだけ) | 使わない(初期値に検出・セグメンテーションで学習した重みは使わない決まり) |
| ECInsSeg(EdgeCrafter) | 蒸留した小さい ViT(ECViT) | 同じく検出器の重みだけ | 使わない(同上) |
| RF-DETR-Seg の Nano | DINOv2 の ViT-S(窓に分けた注意) | timm に DINOv2 の ViT-S | ViT-S(2,160 万)は軽い候補としては重いので、精度重視の DINOv3 ViT-S で代える |
| YOLO26 の n・s | 自前の CSP の backbone | ultralytics の ImageNet の分類の重み(AGPL) | 入れない(ライセンスと、vendor のコードを使う必要があるため) |
| 端末向けの SAM(MobileSAM・EfficientViT-SAM・RepViT-SAM) | TinyViT・EfficientViT・RepViT | timm にある | TinyViT・RepViT を候補に入れた。EfficientViT-L1 は 27.0 GMACs で軽くないので入れない |

ほかに timm で確かめて入れなかったもの: SHViT(stride 8 の段がない)、ConvNeXt V2-Nano(12.8 GMACs。軽い候補は ConvNeXt-T の DINOv3 の版で見る)、
GhostNetV3(1.5 GMACs。MobileNetV3-L と同じ大きさの古典の参考は 1 つで足りる)。

扱わないもの:

- MobileNetV5(`mobilenetv5_300m.gemma3n`、Gemma 3n の画像の部分): 最新の MobileNet だが、2 億 7,890 万パラメータ・166.2 GMACs で
  速度バランス型にならない。精度重視にも入れない(精度重視は Base 程度の大きさの backbone で比べる)
- 1 億を大きく超える backbone(ViT-L 以上の DINOv3 など): Qwen3-VL の画像の部分(約 3 億)だけを大きいものの代表にする

## 実装の注意

- **使う特徴の段**: 今の GTR 系は timm の `out_indices=(2, 3, 4)`(R50 で stride 8・16・32)で選んでいるが、backbone によって段の
  番号が違う(ConvNeXt・FastViT などは stride 4 から始まるので 1・2・3)。`feature_info.reduction()` で stride 8・16・32 の段を
  選ぶように直す(R50 では同じ段になるので、exp00103 の結果は変わらない)
- **1 段の特徴しかない ViT 系**(DINOv3 ViT・EVA-02・SigLIP2・Qwen3-VL・C-RADIO と、速度バランス型の ViT-Tiny 2 つ): ViT の特徴(stride 16)から
  stride 8・16・32 の 3 段を作るアダプタを `lib_common/backbones/` に実装する(timm の backbone と同じく、3 段の特徴の並びを返す形にして、
  GTR 系からはほかの backbone と同じに使えるようにする)。作り方は DEIMv2 の S・M と同じにする: ViT の途中の層(12 層なら 3・7・11 層目)の
  特徴をそれぞれ 2 倍・1 倍・0.5 倍に拡大縮小し(GTR の公式の GTREncoder の scale_factors と同じ考え方)、細かい解像度の情報を、画像から
  作る小さい CNN(DEIMv2 の STA)の特徴で足す。比べるために、ViTDet の simple feature pyramid(最後の層だけから、逆畳み込みで stride 8、
  そのままで stride 16、max pooling で stride 32)も選べるようにする
- **DEIMv2 の ViT-Tiny の重み**: 公式の重みのファイル(`vitt_distill.pt`)を取ってきて、自作の ViT-Tiny(timm の ViT の作りと同じなら timm に
  読み込む)に読み込む。読み込めた重みの名前と形を確かめ、読めない重みがあれば実装の違いとして README に書く
- **入力の正規化**: データの変換は ImageNet の平均・標準偏差で正規化している。事前学習の正規化が違う backbone(SigLIP2・Qwen3-VL は
  平均 0.5・標準偏差 0.5、EVA-02 CLIP は CLIP の値)は、backbone の入口で正規化をし直す(timm の `pretrained_cfg` の値)
- **Qwen3-VL の画像の部分**: transformers の Qwen3-VL の vision encoder を取り出し、画像を (B, 3, 512, 512) で受けて
  パッチの並びに直して入れる。最後の合成(2 × 2 のパッチをまとめて言語モデルの幅にする merger)は使わず、その前の stride 16 の特徴を使う。
  重みは 2B の版(4B と画像の部分の作りは同じ)
- **GPU のメモリ**(上限 12GB): Base 以上は gradient checkpointing(timm の `set_grad_checkpointing`)と勾配の累積で収める
- **ライセンス**: DINOv3 の重みは独自のライセンス(DINOv3 License)、C-RADIO は NVIDIA の独自のライセンス。学習したモデルをデモ
  (Hugging Face)に上げるときは、そのライセンスに従う

## 測るもの

- 精度: test(`coco_val_test_cls8`・`voc_val_cls8`)の mask AP・box AP・PQ、クラスごとの mask AP、val の推移
- 大きさ: パラメータの数、GMACs(backbone だけと、モデル全体)
- 学習の速さ(枚/秒。GPU をほかの学習と分け合った時間を含む)
- 推論の速さ(速度バランス型は特に): GPU(PyTorch)、CPU(ONNX Runtime)、ブラウザ(exp00304 のデモ。WASM・WebGPU)。ほかの学習で GPU・CPU が
  混んでいる間は測らず、まとめて後で測る

## 進め方

1 本ずつ順に学習する(GPU のメモリの上限 12GB)。学習の時間の目安は、R50 の 2.6 時間(21.3 枚/秒)をモデル全体の計算量の比で
割り増した見積もり(まだ測っていない。各 backbone の動作確認のときに速さを測って直す)。

| 段 | 中身 | 本数 | 目安の時間 |
|---|---|---|---|
| 1 | 実装(特徴の段の選び方、ViT 系の simple feature pyramid、正規化のし直し、Qwen3-VL の取り出し)と、全部の候補の動作確認(数十回の更新) | — | — |
| 2 | 速度バランス型(指定の 8 本 + 足した 4 本)。方式は exp00104 の G1 + E2(64) + E3(`{"dim": 192, "ffn": 512, "mask_dim": 64, "enc_block": "dw"}`)。比べる相手は同じ方式の R50(exp00104 の `gtr_g1_e2e3`、test 31.4)。batch 16、1 本あたりの GPU のメモリの上限 15GB で 2 本ずつ同時に流す(`run_light.sh`。メモリが足りないときだけ 8 枚 × 2 回の勾配の累積) | 12 | 軽い構成の速さで見積もり直す(今の GTR 系の構成なら 1 本 1.5〜2.5 時間) |
| 2 の続き | 速度バランス型の残り(TinyViT-5M・11M、RepViT-M1.5、EdgeNeXt-S、StarNet-S3、EfficientFormerV2-S2、MobileOne-S2)。条件は上と同じ | 7 | 2 本ずつで計 約 7 時間 |
| 2 の続き | ViT 用のアダプタを実装し、DINOv3 ViT-Tiny・DEIMv2 の ViT-Tiny | 2 | 計 約 2 時間 |
| 3 | 精度重視(指定の 6 本 + 足した 3 本。Qwen3-VL は最後に回し、学習するかはそれまでの結果を見て決める) | 9 | Base 1 本 6〜7 時間、MaxViT-S 約 5 時間、Qwen3-VL 約 20 時間以上、計 約 75 時間(Qwen3-VL を除くと約 55 時間) |
| 4 | 精度重視の最良の backbone で Mask R-CNN | 1〜3 | 1 本あたり、同じ backbone の GTR 系の学習と同じ程度の見込み(R50 では GTR 系より速かった) |

## 後の段階(結果を見て決める)

- **backbone を固定する / LoRA**: 精度重視の最良の backbone で、全部を学習する場合と比べる(合成データでは、固定した方がドメインの差に
  強い可能性がある。インスタンスセグメンテーションの README の計画)

## 結果

(まだ)
