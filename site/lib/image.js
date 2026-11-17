// 画像の前処理の共通の部品(タスクによらない)。

/**
 * 1 コマを、長い辺を size にそろえて左上に寄せ、残りを 0(黒)で埋めた size × size にし、正規化した CHW の float32 にする。
 * 学習の評価の変換(segmentation/instance_segmentation/lib_common/transforms.py の eval_transforms)と同じ:
 * LongestMaxSize → 左上に寄せて 0 で埋める → (値 / 255 − 平均) / 標準偏差。
 * @param {CanvasImageSource} frame 元の 1 コマ(video・canvas など)。
 * @param {number} width 元の幅。
 * @param {number} height 元の高さ。
 * @param {number} size モデルの入力の一辺。
 * @param {number[]} mean RGB ごとの平均。
 * @param {number[]} std RGB ごとの標準偏差。
 * @param {OffscreenCanvas} work 作業用のキャンバス(size × size)。
 * @returns {{data: Float32Array, scale: number, rw: number, rh: number}} 入力 (3·size·size)、縮めた倍率、画像の部分の幅・高さ。
 */
export function letterbox(frame, width, height, size, mean, std, work) {
  const scale = size / Math.max(width, height);
  const rw = Math.round(width * scale);
  const rh = Math.round(height * scale);
  const ctx = work.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(frame, 0, 0, rw, rh);
  const px = ctx.getImageData(0, 0, size, size).data; // (size·size·4) の RGBA
  const n = size * size;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    // 埋めた部分も (0 − 平均) / 標準偏差 になる(学習のときと同じく、正規化の前に 0 で埋める)
    data[i] = (px[4 * i] / 255 - mean[0]) / std[0];
    data[n + i] = (px[4 * i + 1] / 255 - mean[1]) / std[1];
    data[2 * n + i] = (px[4 * i + 2] / 255 - mean[2]) / std[2];
  }
  return { data, scale, rw, rh };
}

/** @param {number} x @returns {number} sigmoid。 */
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/**
 * 値の並びの上位 k 個の番号を、大きい順に返す。
 * @param {Float32Array|number[]} values 値。
 * @param {number} k 個数。
 * @returns {number[]} 番号。
 */
export function topK(values, k) {
  const idx = Array.from(values.keys());
  idx.sort((a, b) => values[b] - values[a]);
  return idx.slice(0, k);
}

/**
 * 番号から、見分けやすい色を返す(色相を黄金比で回す)。
 * @param {number} i 番号。
 * @returns {[number, number, number]} RGB(0〜255)。
 */
export function color(i) {
  const h = (i * 0.618033988749895) % 1;
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return Math.round(255 * (0.55 - 0.45 * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}
