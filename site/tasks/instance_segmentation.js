// インスタンスセグメンテーションのタスク: 前処理・後処理・描画。
//
// モデルの出力の形(registry.json の format)ごとの後処理は、学習の評価の後処理
// (segmentation/instance_segmentation/lib_common/postprocess.py)と同じ手順にする。
//   gtr_seg:  logits (1, Q, K)・boxes (1, Q, 4)(入力の中の 0〜1 の cx, cy, w, h)・masks (1, Q, S/4, S/4)
//             → rfdetr_to_predictions と同じ(クラスの確率 = sigmoid、Q × K から上位を選ぶ、マスクの値 > 0 が物体)
//   yolo_seg: logits (1, A, K)・boxes (1, A, 4)(入力の画素の x0, y0, x1, y1)・coef (1, A, M)・protos (1, M, S/4, S/4)
//             → yolo_to_predictions と同じ(係数 × 原型のマスク、枠の外を消す、値 > 0 が物体)
// どの形も、出力は「検出 = {label, score, box(元の画像の画素の x0, y0, x1, y1), logit(S/4 × S/4 のマスクの値。sigmoid の前)}」に
// そろえる。マスクは、学習の評価(postprocess.py)と同じく、1/4 の値を元の 1 コマの大きさに bilinear で拡大してから、値 > 0 を物体にする
// (先に 1/4 のまま 0/1 にして引き伸ばすと、ふちが 1/4 の画素の大きさで角ばる)。

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
  let overlay = null; // 元の 1 コマの大きさのマスクの重ね絵(大きさが変わったら作り直す)
  let overlayImg = null;
  // 色を前のコマから引き継ぐための状態(追跡はしない。前のコマの同じクラスで枠の重なりが最も大きい検出の色を使う)
  let prevDets = []; // [{label, box, colorId}]
  let nextColorId = 0;

  /**
   * 前のコマの検出と枠の重なり(IoU)で対応付けて、検出ごとの色の番号を決める(同じ物体の色がコマごとに変わってちらつかないように)。
   * 同じクラスで IoU が 0.3 以上の組を、IoU の大きい順に 1 対 1 で選ぶ。対応のない検出には新しい番号を付ける。
   * @param {object[]} dets 今のコマの検出(box は元の 1 コマの画素の x0, y0, x1, y1)。colorId を書き込む。
   */
  function assignColors(dets) {
    const iou = (a, b) => {
      const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
      if (w <= 0 || h <= 0) return 0;
      const inter = w * h;
      return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
    };
    const pairs = [];
    dets.forEach((d, i) => prevDets.forEach((p, j) => {
      if (p.label === d.label) {
        const v = iou(d.box, p.box);
        if (v >= 0.3) pairs.push([v, i, j]);
      }
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    const usedPrev = new Set();
    for (const [, i, j] of pairs) {
      if (dets[i].colorId === undefined && !usedPrev.has(j)) {
        dets[i].colorId = prevDets[j].colorId;
        usedPrev.add(j);
      }
    }
    for (const d of dets) if (d.colorId === undefined) d.colorId = nextColorId++;
    prevDets = dets.map((d) => ({ label: d.label, box: d.box, colorId: d.colorId }));
  }

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
      // 出力の配列は次の推論で使い回されることがあるので、マスクの値は写しておく
      dets.push({ label, score, logit: Float32Array.from(maskLogit), box: box.map((v) => v / meta.scale) });
    }
    assignColors(dets);
    return dets;
  }

  /**
   * 検出を、元の 1 コマを描いたキャンバスの上に重ねて描く。
   * @param {CanvasRenderingContext2D} ctx 出力のキャンバス(元の 1 コマの大きさ)。
   * @param {object[]} dets postprocess の検出。
   * @param {object} meta preprocess の meta。
   */
  function draw(ctx, dets, meta) {
    const { width, height } = meta;
    if (!overlay || overlay.width !== width || overlay.height !== height) {
      overlay = new OffscreenCanvas(width, height);
      overlayImg = overlay.getContext("2d").createImageData(width, height);
    }
    const px = overlayImg.data;
    px.fill(0);
    // 元の 1 コマの画素 (X, Y) の中心 → 入力の画像の部分(左上の rw × rh)の座標 → 1/4 のマスクの座標(画素の中心が整数)
    const sx = meta.rw / width / 4, sy = meta.rh / height / 4;
    dets.forEach((d) => { // 後の(スコアの低い)検出は上書きしない: スコアの高いものを手前に
      const [r, g, b] = color(d.colorId);
      const lg = d.logit;
      // 描く範囲: 1/4 のマスクで値が 0 に近い(> -4)画素を含む矩形を、元の 1 コマの座標にしたもの(全体を回さずに済ませる)
      let mx0 = ms, my0 = ms, mx1 = -1, my1 = -1;
      for (let y = 0; y < ms; y++) {
        for (let x = 0; x < ms; x++) {
          if (lg[y * ms + x] > -4) {
            if (x < mx0) mx0 = x;
            if (x > mx1) mx1 = x;
            if (y < my0) my0 = y;
            if (y > my1) my1 = y;
          }
        }
      }
      if (mx1 < 0) return;
      let X0 = Math.max(0, Math.floor((mx0 - 0.5) / sx)), X1 = Math.min(width, Math.ceil((mx1 + 1.5) / sx));
      let Y0 = Math.max(0, Math.floor((my0 - 0.5) / sy)), Y1 = Math.min(height, Math.ceil((my1 + 1.5) / sy));
      if (model.crop_to_box) { // 枠の外を消す(yolo_to_predictions と同じ)
        X0 = Math.max(X0, Math.floor(d.box[0])); X1 = Math.min(X1, Math.ceil(d.box[2]));
        Y0 = Math.max(Y0, Math.floor(d.box[1])); Y1 = Math.min(Y1, Math.ceil(d.box[3]));
      }
      for (let Y = Y0; Y < Y1; Y++) {
        const v = Math.min(Math.max((Y + 0.5) * sy - 0.5, 0), ms - 1); // bilinear の縦の位置(端は端の値を使う)
        const v0 = Math.floor(v), v1 = Math.min(v0 + 1, ms - 1), fv = v - v0;
        for (let X = X0; X < X1; X++) {
          const o = 4 * (Y * width + X);
          if (px[o + 3] !== 0) continue;
          const u = Math.min(Math.max((X + 0.5) * sx - 0.5, 0), ms - 1);
          const u0 = Math.floor(u), u1 = Math.min(u0 + 1, ms - 1), fu = u - u0;
          const top = lg[v0 * ms + u0] * (1 - fu) + lg[v0 * ms + u1] * fu;
          const bot = lg[v1 * ms + u0] * (1 - fu) + lg[v1 * ms + u1] * fu;
          if (top * (1 - fv) + bot * fv > 0) {
            px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 120;
          }
        }
      }
    });
    overlay.getContext("2d").putImageData(overlayImg, 0, 0);
    ctx.drawImage(overlay, 0, 0);
    ctx.font = `${Math.max(12, Math.round(meta.width / 60))}px sans-serif`;
    ctx.lineWidth = Math.max(2, meta.width / 400);
    dets.forEach((d) => {
      const [r, g, b] = color(d.colorId);
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
