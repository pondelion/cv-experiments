"""動画インスタンスセグメンテーションのモデル(exp005xx)。画像のモデルは 1 つ上の models/ に置く。

どのモデルも、ファイルの先頭に「MOT(動画全体での物体の ID の管理・識別)の仕組みを持つか、持つならどの形か」を書く。
    - videomt.py: VidEoMT。学習して身に付ける形(クエリを次のフレームに渡す。クエリの番号 = ID)
    - minvis.py: MinVIS。推論の照合の形(画像の Mask2Former + 隣のフレームのクエリの埋め込みの照合)
    - dvis.py: DVIS(online)。推論の照合 + 学習する追跡の部(referring tracker)
    - ctvis.py: CTVIS。推論の照合(対照学習で学んだ reid の埋め込み)
    - dvis_plus.py: DVIS++(online)。DVIS と同じ形(segmenter = CTVIS、学習中に照合をわざと乱す)
    - dvis_daq.py: DVIS-DAQ(online)。追跡のクエリ + 新しい物体のクエリ(点数で ID を足し、続けて低い ID を終える。上限なし)
    - cavis.py: CAVIS(online)。DVIS と同じ形(文脈の特徴を足した照合と追跡の部)
    - sam2/: SAM 2(マスクを与えて物体ごとの記憶で伝搬。物体ごとに ID が固定)と、SAM 3 型の組み立て(tracker.py。検出との照合で
      ID を足し・除く)
    - deva/: DEVA(全部の物体を 1 つの記憶で伝搬 + 数フレームごとに画像のモデルの結果と投票・照合で ID を足し・除く)
    - image_adapter.py: 画像のモデル(Mask2Former・GTR 系)を同じ形の出力にそろえる包み(MOT の仕組みはない)

推論の出力は predict_video(frames (T, 3, S, S)) → (クラスの値 (T, Q, K+1)(DVIS-DAQ は動画ごとの (R, K+1)), マスクの値 (Q, T, h, w))
(クエリの番号 = ID)で、lib_common/video_infer.py の video_predictions で評価の予測の形にする。SAM 3 型・DEVA は track_video で
ID ごとの予測を直接作る。
"""
