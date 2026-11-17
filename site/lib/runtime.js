// ONNX Runtime Web でモデルを読み込んで動かす部分(タスクによらない)。
//
// - ONNX Runtime Web は Web Worker(ort_worker.js)の中で動かす。モデルの準備や推論が遅くても止まっても、ページは固まらない
// - 実行の場所は WebGPU を先に試し、使えなければ WASM(CPU)にする。モデルの準備が LOAD_TIMEOUT_MS を超えたら、その Worker を
//   止めて次の実行の場所を試す
// - モデルのファイルは読み込みの進み方を表示しながら取ってくる(大きいファイルでも止まって見えないように)

const LOAD_TIMEOUT_MS = 120000;

/**
 * ファイルを取ってくる(読み込んだ割合を onProgress に渡す)。
 * @param {string} url ファイルの場所(リポジトリのトップからの相対パス)。
 * @param {(ratio: number|null, loadedBytes: number) => void} onProgress 進み方(全体の大きさが分からないときは ratio が null)。
 * @returns {Promise<ArrayBuffer>} ファイルの中身。
 */
export async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`モデルのファイルを読めない: ${url}(HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(total ? loaded / total : null, loaded);
  }
  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) { buf.set(c, offset); offset += c.length; }
  return buf.buffer;
}

/** Web Worker の中の推論のセッションを、ページから使う形(run(feeds) → 出力)にしたもの。 */
class WorkerSession {
  /** @param {Worker} worker 準備の済んだ Worker。 */
  constructor(worker) {
    this.worker = worker;
    this.nextId = 0;
    this.pending = new Map(); // 推論の番号 → {resolve, reject}
    worker.onmessage = ({ data: msg }) => {
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.type === "result") p.resolve(msg.outputs);
      else p.reject(new Error(msg.message));
    };
  }

  /**
   * 推論する。
   * @param {Object<string, {type: string, data: Float32Array, dims: number[]}>} feeds 入力(名前 → 値)。
   * @returns {Promise<Object<string, {data: Float32Array, dims: number[]}>>} 出力(名前 → 値)。
   */
  run(feeds) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, feeds }, Object.values(feeds).map((t) => t.data.buffer));
    });
  }
}

/**
 * 1 つの実行の場所で、Worker の中にモデルを準備する(制限時間つき)。
 * @param {ArrayBuffer} model ONNX のファイルの中身(Worker に写して渡す)。
 * @param {"webgpu"|"wasm"} ep 実行の場所。
 * @param {string} optLevel グラフの最適化の段階。
 * @returns {Promise<{worker: Worker, threads: number|null}>} 準備の済んだ Worker と、WASM のスレッドの数。
 */
function loadInWorker(model, ep, optLevel) {
  const url = new URL("./ort_worker.js", import.meta.url);
  const ortVersion = new URLSearchParams(location.search).get("ort"); // 動作の確かめ用: ページの ?ort= を Worker に渡す
  if (ortVersion) url.searchParams.set("ort", ortVersion);
  const log = new URLSearchParams(location.search).get("log");
  if (log) url.searchParams.set("log", log);
  if (new URLSearchParams(location.search).has("nosubgroups")) url.searchParams.set("nosubgroups", "1");
  const worker = new Worker(url, { type: "module" });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      worker.terminate();
      reject(new Error(`${ep} でモデルの準備が ${LOAD_TIMEOUT_MS / 1000} 秒で終わらない`));
    }, LOAD_TIMEOUT_MS);
    worker.onmessage = ({ data: msg }) => {
      clearTimeout(timer);
      if (msg.type === "loaded") resolve({ worker, threads: msg.threads });
      else { worker.terminate(); reject(new Error(`${ep} でモデルを準備できない: ${msg.message}`)); }
    };
    worker.onerror = (e) => { clearTimeout(timer); worker.terminate(); reject(new Error(`${ep} の Worker のエラー: ${e.message}`)); };
    worker.postMessage({ type: "load", model: model.slice(0), ep, optLevel });
  });
}

/**
 * モデルを読み込んで、推論のセッションを作る。
 * @param {ArrayBuffer} model ONNX のファイルの中身。
 * @param {"auto"|"webgpu"|"wasm"} backend 実行の場所(auto は WebGPU → WASM の順に試す)。
 * @param {"disabled"|"basic"|"extended"|"all"} optLevel グラフの最適化の段階。
 * @param {(text: string) => void} onStatus 状態の表示(どの実行の場所を試しているか)。
 * @returns {Promise<{session: WorkerSession, backend: string}>} セッションと、実際に使った実行の場所。
 */
export async function createSession(model, backend = "auto", optLevel = "all", onStatus = () => {}) {
  const order = backend === "auto" ? (navigator.gpu ? ["webgpu", "wasm"] : ["wasm"]) : [backend];
  const errors = [];
  for (const ep of order) {
    try {
      onStatus(`${ep} でモデルを準備中…(初回は時間がかかります。最大 ${LOAD_TIMEOUT_MS / 1000} 秒)`);
      const { worker, threads } = await loadInWorker(model, ep, optLevel);
      return { session: new WorkerSession(worker), backend: threads ? `${ep}・${threads} スレッド` : ep };
    } catch (e) {
      console.warn(e);
      errors.push(e.message);
    }
  }
  throw new Error(errors.join(" / "));
}
