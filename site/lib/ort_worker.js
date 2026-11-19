// ONNX Runtime Web を動かす Web Worker(ページの画面とは別の処理の流れ)。
// モデルの準備や推論に時間がかかったり止まったりしても、ページの操作は止まらない。
//
// やり取り(postMessage):
//   ページ → Worker: {type: "load", model: ArrayBuffer, ep: "webgpu"|"wasm", optLevel}
//                    {type: "run", id, feeds: {名前: {type, data, dims}}}
//   Worker → ページ: {type: "loaded", ep} / {type: "error", message, id?}
//                    {type: "result", id, outputs: {名前: {data, dims}}}

// onnxruntime-web の版(Worker の URL の ?ort= で変えられる。動作の確かめ用)
const ORT_VERSION = new URLSearchParams(self.location.search).get("ort") ?? "1.31.1";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
let ort = null;
let session = null;

self.onmessage = async ({ data: msg }) => {
  try {
    if (!ort) {
      ort = await import(`${ORT_DIST}ort.webgpu.min.mjs`);
      ort.env.wasm.wasmPaths = ORT_DIST;
      // WASM のスレッドの数: cross-origin isolated のときだけ複数にできる。画面の処理の分を 1 つ残し、多すぎても速くならないので 8 まで
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(8, navigator.hardwareConcurrency - 1)) : 1;
      const threads = new URLSearchParams(self.location.search).get("threads"); // 動作の確かめ用: ?threads=N でスレッドの数を決める
      if (threads && self.crossOriginIsolated) ort.env.wasm.numThreads = Number(threads);
      const log = new URLSearchParams(self.location.search).get("log"); // 動作の確かめ用: ?log=verbose で詳しい記録
      if (log) { ort.env.logLevel = log; ort.env.debug = log === "verbose"; }
    }
    if (msg.type === "load") {
      const gpu = new URLSearchParams(self.location.search).get("gpu"); // 動作の確かめ用: ?gpu=high で性能の高い GPU を頼む
      if (msg.ep === "webgpu" && gpu) ort.env.webgpu.powerPreference = gpu === "high" ? "high-performance" : "low-power";
      let adapter = null; // 使う GPU の名前(ページの表示に出し、どの GPU で動いているかを分かるようにする)
      if (msg.ep === "webgpu") {
        const a = await navigator.gpu.requestAdapter(gpu ? { powerPreference: ort.env.webgpu.powerPreference } : undefined);
        adapter = [a?.info?.vendor, a?.info?.architecture, a?.info?.description].filter(Boolean).join(" ") || null;
      }
      if (msg.ep === "webgpu" && new URLSearchParams(self.location.search).has("nosubgroups")) {
        // 動作の確かめ用: subgroups(GPU の同じ組のスレッドどうしで値を受け渡す機能)を持たない GPU の装置を作って渡す。
        // onnxruntime-web は装置にある機能を見て計算の方法を選ぶので、subgroups を使う計算(MatMulSubgroup など)を避けられる
        const adapter = await navigator.gpu.requestAdapter();
        const requiredLimits = {};
        for (const k in adapter.limits) requiredLimits[k] = adapter.limits[k];
        ort.env.webgpu.device = await adapter.requestDevice({
          requiredFeatures: [...adapter.features].filter((f) => !f.includes("subgroup")), requiredLimits,
        });
      }
      session = await ort.InferenceSession.create(msg.model, {
        executionProviders: [msg.ep], graphOptimizationLevel: msg.optLevel,
        // 重みだけ INT8 のモデル(exp00302 の quantize_weights.py)の、INT8 の重みを fp32 に戻す処理(DequantizeLinear)を、
        // 読み込むときに 1 回だけ計算しておく(定数にまとめる)。既定では、量子化したモデルの最適化のためにこの処理を残すので、
        // 推論のたびに重みを戻す計算が走る(exp00304 の README の問題 10)
        // 動作の確かめ用: ページの URL に &keepqdq を付けると、この設定を入れない(修正の前と比べるため)
        extra: new URLSearchParams(self.location.search).has("keepqdq") ? {} : { session: { disable_quant_qdq: "1" } },
      });
      self.postMessage({ type: "loaded", ep: msg.ep, threads: msg.ep === "wasm" ? ort.env.wasm.numThreads : null, adapter });
    } else if (msg.type === "run") {
      const feeds = {};
      for (const [name, t] of Object.entries(msg.feeds)) feeds[name] = new ort.Tensor(t.type, t.data, t.dims);
      const t0 = performance.now();
      const result = await session.run(feeds);
      const t1 = performance.now();
      const outputs = {};
      const transfer = [];
      for (const [name, t] of Object.entries(result)) {
        const data = await t.getData(); // WebGPU の出力も、ここで CPU の配列にする
        outputs[name] = { data, dims: t.dims };
        transfer.push(data.buffer);
      }
      // 時間の内訳(Worker の中で測った、推論そのものと、出力を JS の配列に写す時間)。ページの自己テストで使う
      self.postMessage({ type: "result", id: msg.id, outputs, timing: { run: t1 - t0, copy: performance.now() - t1 } }, transfer);
    }
  } catch (e) {
    self.postMessage({ type: "error", id: msg.id, message: String(e?.message ?? e) });
  }
};
