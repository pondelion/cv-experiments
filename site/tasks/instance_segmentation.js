// インスタンスセグメンテーションのタスク: 前処理・後処理・描画。
//
// モデルの出力の形(registry.json の format)ごとの後処理は、学習の評価の後処理
// (segmentation/instance_segmentation/lib_common/postprocess.py)と同じ手順にする。
//   gtr_seg:  logits (1, Q, K)・boxes (1, Q, 4)(入力の中の 0〜1 の cx, cy, w, h)・masks (1, Q, S/4, S/4)
//             → rfdetr_to_predictions と同じ(クラスの確率 = sigmoid、Q × K から上位を選ぶ、マスクの値 > 0 が物体)
//   yolo_seg: logits (1, A, K)・boxes (1, A, 4)(入力の画素の x0, y0, x1, y1)・coef (1, A, M)・protos (1, M, S/4, S/4)
//             → yolo_to_predictions と同じ(係数 × 原型のマスク、枠の外を消す、値 > 0 が物体)
// どの形も、出力は「検出 = {label, score, box(元の画像の画素の x0, y0, x1, y1), mask(S/4 × S/4 の 0/1)}」にそろえる。

import { color, letterbox, sigmoid, topK } from "../lib/image.js";

const MAX_CANDIDATES = 100; // 学習の評価の後処理と同じく、上位 100 個の (クエリ, クラス) から選ぶ

/**
 * 1 つのモデルの前処理・後処理・描画を作る。
 * @param {object} model registry.json のモデルの項目。
 * @returns {{preprocess: Function, postprocess: Function, draw: Function}}
 */
export function createTask(model) {
  const size = model.input_size;
  const ms = size / 4; // マスクの一辺(入力の 1/4)
  const work = new OffscreenCanvas(size, size);
  const maskCanvas = new OffscreenCanvas(ms, ms);

  /**
   * 1 コマをモデルの入力にする。
   * @param {CanvasImageSource} frame 1 コマ。
   * @param {number} width 幅。
   * @param {number} height 高さ。
   * @returns {{feeds: object, meta: object}} 入力と、後処理で使う情報。
   */
  function preprocess(frame, width, height) {
    const { data, scale, rw, rh } = letterbox(frame, width, height, size, model.mean, model.std, work);
    // 入力は {型, 値, 形}(推論は Web Worker の中で行うので、ここでは ONNX Runtime の Tensor にしない)
    return { feeds: { images: { type: "float32", data, dims: [1, 3, size, size] } }, meta: { scale, rw, rh, width, height } };
  }

  /**
   * モデルの出力を検出の並びにする。
   * @param {object} out セッションの出力(名前 → Tensor)。
   * @param {object} meta preprocess の meta。
   * @param {{threshold: number}} opts スコアの閾値。
   * @returns {object[]} 検出の並び(スコアの大きい順)。
   */
  function postprocess(out, meta, opts) {
    const logits = out.logits.data;
    const [, n, k] = out.logits.dims; // n = クエリ(位置)の数、k = クラスの数
    const probs = logits.map(sigmoid); // (n·k)
    const dets = [];
    for (const idx of topK(probs, MAX_CANDIDATES)) {
      const score = probs[idx];
      if (score < opts.threshold) break;
      const q = Math.floor(idx / k);
      const label = idx % k;
      let box; // 入力(size × size)の画素の x0, y0, x1, y1
      let maskLogit; // (ms·ms) のマスクの値(sigmoid の前)
      if (model.format === "gtr_seg") {
        const b = out.boxes.data.subarray(4 * q, 4 * q + 4);
        const w = Math.max(b[2], 0) * size, h = Math.max(b[3], 0) * size;
        box = [b[0] * size - w / 2, b[1] * size - h / 2, b[0] * size + w / 2, b[1] * size + h / 2];
        maskLogit = out.masks.data.subarray(q * ms * ms, (q + 1) * ms * ms);
      } else if (model.format === "yolo_seg") {
        box = Array.from(out.boxes.data.subarray(4 * q, 4 * q + 4));
        const m = out.protos.dims[1];
        const coef = out.coef.data.subarray(m * q, m * (q + 1));
        const protos = out.protos.data;
        maskLogit = new Float32Array(ms * ms);
        for (let j = 0; j < m; j++) {
          const c = coef[j];
          const base = j * ms * ms;
          for (let p = 0; p < ms * ms; p++) maskLogit[p] += c * protos[base + p];
        }
      } else {
        throw new Error(`知らない出力の形: ${model.format}`);
      }
      const mask = new Uint8Array(ms * ms);
      const [bx0, by0, bx1, by1] = box.map((v) => v / 4); // マスクの座標
      for (let y = 0; y < ms; y++) {
        for (let x = 0; x < ms; x++) {
          const p = y * ms + x;
          const inBox = !model.crop_to_box || (x >= bx0 && x < bx1 && y >= by0 && y < by1);
          mask[p] = inBox && maskLogit[p] > 0 ? 1 : 0;
        }
      }
      dets.push({ label, score, mask, box: box.map((v) => v / meta.scale) });
    }
    return dets;
  }

  /**
   * 検出を、元の 1 コマを描いたキャンバスの上に重ねて描く。
   * @param {CanvasRenderingContext2D} ctx 出力のキャンバス(元の 1 コマの大きさ)。
   * @param {object[]} dets postprocess の検出。
   * @param {object} meta preprocess の meta。
   */
  function draw(ctx, dets, meta) {
    const mctx = maskCanvas.getContext("2d");
    const img = mctx.createImageData(ms, ms);
    dets.forEach((d, i) => { // 後の(スコアの低い)検出は上書きしない: スコアの高いものを手前に
      const [r, g, b] = color(d.label * 7 + i);
      for (let p = 0; p < ms * ms; p++) {
        if (d.mask[p] && img.data[4 * p + 3] === 0) {
          img.data.set([r, g, b, 120], 4 * p);
        }
      }
    });
    mctx.putImageData(img, 0, 0);
    // マスクは入力(size × size)の 1/4。画像の部分(左上の rw × rh)だけを元の大きさに広げて重ねる
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(maskCanvas, 0, 0, meta.rw / 4, meta.rh / 4, 0, 0, meta.width, meta.height);
    ctx.font = `${Math.max(12, Math.round(meta.width / 60))}px sans-serif`;
    ctx.lineWidth = Math.max(2, meta.width / 400);
    dets.forEach((d, i) => {
      const [r, g, b] = color(d.label * 7 + i);
      const [x0, y0, x1, y1] = d.box;
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      const text = `${model.classes[d.label] ?? d.label} ${d.score.toFixed(2)}`;
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x0, Math.max(0, y0 - 18), tw + 8, 18);
      ctx.fillStyle = "#000";
      ctx.fillText(text, x0 + 4, Math.max(13, y0 - 5));
    });
  }

  return { preprocess, postprocess, draw };
}
