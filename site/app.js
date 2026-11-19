// デモのページの画面と推論の繰り返し(タスクによらない)。
//
// 流れ: registry.json を読む → タスク・モデルを選ぶ → 入手先(YouTube・手元の動画・カメラ)から <video> を得る →
// 1 コマずつ、タスクの前処理 → ONNX Runtime Web で推論 → タスクの後処理 → 元の 1 コマに結果を重ねて描く、を繰り返す。
// タスクごとの処理は registry.json の module(site/tasks/*.js)に分けてあり、createTask(model) が
// {preprocess, postprocess, draw} を返す決まり。タスク・モデルを足すときは registry.json と site/tasks/ だけを変える。

import { createSession, fetchWithProgress } from "./lib/runtime.js";
import { cameraVideo, captureElement, embedYouTube, fileVideo, imageFile, parseYouTubeId } from "./lib/sources.js";

const $ = (id) => document.getElementById(id);
const state = { registry: null, task: null, model: null, source: "youtube", running: false, stopSource: null };
const sessions = new Map(); // モデルの id → {session, backend, task}(読み込みは 1 回だけ)

/** @param {string} text 状態の表示。 */
const status = (text) => { $("status").textContent = text; };

/** registry.json を読み、選択肢を作る。 */
async function init() {
  state.registry = await (await fetch("site/registry.json")).json();
  $("description").textContent = state.registry.description;
  $("repo").href = state.registry.repo_url;
  const taskSel = $("task");
  for (const t of state.registry.tasks) taskSel.add(new Option(t.title, t.id));
  taskSel.onchange = () => selectTask(taskSel.value);
  $("model").onchange = () => selectModel($("model").value);
  $("threshold").oninput = () => { $("threshold-value").textContent = Number($("threshold").value).toFixed(2); };
  document.querySelectorAll(".tab").forEach((b) => { b.onclick = () => selectSource(b.dataset.source); });
  $("yt-load").onclick = loadYouTube;
  $("start").onclick = start;
  $("stop").onclick = stop;
  selectTask(state.registry.tasks[0].id);
  selectSource("youtube");
}

/** @param {string} id タスクの id。 */
function selectTask(id) {
  state.task = state.registry.tasks.find((t) => t.id === id);
  const sel = $("model");
  sel.innerHTML = "";
  for (const m of state.task.models) sel.add(new Option(m.title, m.id));
  selectModel(state.task.models[0].id);
}

/** @param {string} id モデルの id。 */
function selectModel(id) {
  cleanupIdle(); // 前のモデルの静止画の描き直しを止める
  state.model = state.task.models.find((m) => m.id === id);
  $("model-info").textContent = `${state.model.info}(クラス: ${state.model.classes.join(", ")})`;
  $("threshold").value = state.model.score_threshold;
  $("threshold").oninput();
}

/** @param {"youtube"|"file"|"camera"} name 入手先。 */
function selectSource(name) {
  if (state.running) stop();
  cleanupIdle();
  state.source = name;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.source === name));
  document.querySelectorAll(".source").forEach((s) => { s.hidden = s.dataset.source !== name; });
  $("input-view").innerHTML = "";
}

/** YouTube の URL を読み、埋め込みを置く。 */
function loadYouTube() {
  const id = parseYouTubeId($("yt-url").value);
  if (!id) { status("YouTube の URL を読めない"); return; }
  embedYouTube($("input-view"), id);
  status("埋め込みました。再生してから「推論を始める」を押してください");
}

/**
 * 選んでいるモデルのセッションを用意する(読み込み済みなら使い回す)。
 * @returns {Promise<{session: object, backend: string, task: object}>}
 */
async function prepareModel() {
  const q = new URLSearchParams(location.search); // 動作の確かめ用: &onnx={ファイル}・&opt={最適化の段階} で差し替えられる
  const onnx = q.get("onnx") ?? state.model.onnx;
  const key = `${state.model.id}:${$("backend").value}:${onnx}`;
  if (sessions.has(key)) return sessions.get(key);
  const buf = await fetchWithProgress(onnx, (ratio, bytes) => {
    status(`モデルを読み込み中… ${ratio === null ? `${(bytes / 1e6).toFixed(1)}MB` : `${Math.round(ratio * 100)}%`}`);
  });
  const { session, backend } = await createSession(buf, $("backend").value, q.get("opt") ?? "all", status);
  const { createTask } = await import(`./${state.task.module}`);
  const entry = { session, backend, task: createTask(state.model) };
  sessions.set(key, entry);
  return entry;
}

/**
 * 入手先から <video>(静止画のファイルなら <img>)を得る。
 * @returns {Promise<{video?: HTMLVideoElement, image?: HTMLImageElement, stop: () => void}>}
 */
async function openSource() {
  const view = $("input-view");
  if (state.source === "youtube") {
    if (!view.querySelector("iframe")) throw new Error("先に YouTube の URL を入れて「埋め込む」を押してください");
    return captureElement(view);
  }
  let src;
  if (state.source === "file") {
    const file = $("file").files[0];
    if (!file) throw new Error("動画か画像のファイルを選んでください");
    src = file.type.startsWith("image/") ? await imageFile(file) : await fileVideo(file);
  } else {
    src = await cameraVideo();
  }
  view.innerHTML = "";
  const el = src.video ?? src.image;
  el.className = "fill";
  view.appendChild(el); // 手元の動画・画像・カメラは入力の欄にそのまま表示する
  return src;
}

/** 推論を始める。 */
async function start() {
  if (state.running) return;
  cleanupIdle();
  $("start").disabled = true;
  try {
    const { session, backend, task } = await prepareModel();
    const src = await openSource();
    state.stopSource = src.stop;
    if (src.image) { // 静止画: 1 回だけ推論する
      await runImage(src.image, session, task);
      status(`推論しました(${backend})。スコアの閾値を変えると描き直します`);
      $("start").disabled = false;
      return;
    }
    state.running = true;
    $("stop").disabled = false;
    status(`推論中(${backend})`);
    // 共有の停止(ブラウザの「共有を停止」)やカメラの切断で映像が終わったら、推論も止める
    src.video.srcObject?.getVideoTracks().forEach((t) => t.addEventListener("ended", () => { if (state.running) stop(); }));
    loop(src.video, session, task);
  } catch (e) {
    console.error(e);
    status(`止まりました: ${e.message ?? e}`);
    $("start").disabled = false;
  }
}

/** 推論を止める。 */
function stop() {
  state.running = false;
  state.stopSource?.();
  state.stopSource = null;
  $("start").disabled = false;
  $("stop").disabled = true;
  $("fps").textContent = "";
  status("止めました");
}

/**
 * 動画に新しいコマが来るまで待つ(同じコマを何度も推論しないため)。止めたときや映像が終わったときは、1 秒で待つのをやめる。
 * requestVideoFrameCallback がないブラウザでは、画面の描き直し 1 回分だけ待つ。
 * @param {HTMLVideoElement} video 動画。
 * @returns {Promise<void>}
 */
function nextFrame(video) {
  return new Promise((resolve) => {
    if (!video.requestVideoFrameCallback) { requestAnimationFrame(() => resolve()); return; }
    const timer = setTimeout(resolve, 1000);
    video.requestVideoFrameCallback(() => { clearTimeout(timer); resolve(); });
  });
}

/**
 * 1 枚(1 コマ)の推論の出力から、検出を作って出力のキャンバスに描く。
 * @param {HTMLCanvasElement} frame 推論に使った 1 コマ。
 * @param {object} result セッションの出力。
 * @param {object} meta 前処理の meta。
 * @param {object} task createTask の返り値。
 * @returns {object[]} 検出の並び。
 */
function render(frame, result, meta, task) {
  const t0 = performance.now();
  const dets = task.postprocess(result, meta, { threshold: Number($("threshold").value) });
  const t1 = performance.now();
  const out = $("output");
  out.width = frame.width; out.height = frame.height;
  const ctx = out.getContext("2d");
  ctx.drawImage(frame, 0, 0);
  task.draw(ctx, dets, meta);
  render.lastTiming = { post: t1 - t0, draw: performance.now() - t1 }; // 後処理と描画の時間(動画のループで段ごとの時間を出すため)
  return dets;
}

/**
 * 静止画を 1 回だけ推論して描く。スコアの閾値を変えたときは、推論をやり直さずに描き直す。
 * @param {HTMLImageElement} image 画像。
 * @param {object} session ONNX Runtime Web のセッション。
 * @param {object} task createTask の返り値。
 */
async function runImage(image, session, task) {
  const frame = document.createElement("canvas");
  frame.width = image.naturalWidth; frame.height = image.naturalHeight;
  frame.getContext("2d").drawImage(image, 0, 0);
  const t0 = performance.now();
  const { feeds, meta } = task.preprocess(frame, frame.width, frame.height);
  const result = await session.run(feeds);
  const dets = render(frame, result, meta, task);
  $("fps").textContent = `(${(performance.now() - t0).toFixed(0)} ms、検出 ${dets.length} 個)`;
  const slider = $("threshold");
  const base = slider.oninput;
  slider.oninput = () => { // 閾値の表示を更新してから、同じ出力で描き直す
    base();
    const n = render(frame, result, meta, task).length;
    $("fps").textContent = `(検出 ${n} 個)`;
  };
  const prev = state.stopSource; // 画像のファイルの後片付け(imageFile の stop)も続けて行う
  state.stopSource = () => { slider.oninput = base; prev?.(); };
}

/** 動いていない入手先(静止画の描き直しなど)の後片付けをする。 */
function cleanupIdle() {
  if (state.running) return;
  state.stopSource?.();
  state.stopSource = null;
}

/**
 * 1 コマずつ推論して描く繰り返し。推論が終わるたびに、その時点の最新の 1 コマを使う(推論より速いコマは飛ばす)。
 * 1 コマの時間を段ごと(取り込み・前処理・推論・後処理・描画・次のコマを待つ時間)に分けて、直近 20 コマの平均を表示する。
 * @param {HTMLVideoElement} video 入力の動画。
 * @param {object} session ONNX Runtime Web のセッション。
 * @param {object} task createTask の返り値。
 * @param {{maxFrames?: number, onDone?: (avg: object) => void}} opts 動作の確かめ用: maxFrames コマで止め、段ごとの平均を onDone に渡す。
 */
async function loop(video, session, task, opts = {}) {
  const frame = document.createElement("canvas"); // 推論に使った 1 コマ(結果はこの 1 コマに重ねる)
  const out = $("output");
  const hist = []; // 直近 20 コマの段ごとの時間(ms)
  let count = 0;
  let waitMs = 0; // 前のコマの後に、次のコマを待った時間
  // 動画のコマが新しく出るたびに、その番号を記録する。推論の間に新しいコマが来ていれば、待たずにすぐ次の推論を始める
  // (推論がコマの間隔より遅いとき)。来ていなければ次のコマを待つ(推論が速いときに、同じコマを何度も推論しないため)
  let latest = 0;
  const watch = () => video.requestVideoFrameCallback((now, m) => { latest = m.presentedFrames; if (state.running) watch(); });
  if (video.requestVideoFrameCallback) watch();
  const avg = () => {
    const a = {};
    for (const k of Object.keys(hist[0] ?? {})) a[k] = hist.reduce((s, x) => s + x[k], 0) / hist.length;
    return a;
  };
  try {
    while (state.running) {
      const w = video.videoWidth, h = video.videoHeight;
      let used = -1; // このコマの番号(推論の後に、新しいコマが来たかを比べる)
      if (video.readyState >= 2 && w && h) {
        used = latest;
        const t0 = performance.now();
        frame.width = w; frame.height = h;
        frame.getContext("2d").drawImage(video, 0, 0, w, h);
        if (count === 0) { // 1 回目: 取り込めているかが分かるよう、推論の前に取り込んだコマをそのまま描く
          out.width = w; out.height = h;
          out.getContext("2d").drawImage(frame, 0, 0);
          status(`取り込み ${w} × ${h}。1 回目の推論中…(WebGPU は初回の準備に時間がかかることがあります)`);
        }
        const tCap = performance.now();
        const { feeds, meta } = task.preprocess(frame, w, h);
        const tPre = performance.now();
        const result = await session.run(feeds);
        const tInf = performance.now();
        const dets = render(frame, result, meta, task);
        const tEnd = performance.now();
        count += 1;
        if (count === 1) status(`推論中(1 回目の推論 ${(tInf - tPre).toFixed(0)} ms)`);
        else { // 1 回目は準備を含むので平均に入れない
          hist.push({ total: tEnd - t0 + waitMs, capture: tCap - t0, pre: tPre - tCap, infer: tInf - tPre,
                      inferWorker: session.lastTiming?.run ?? 0, post: render.lastTiming.post, draw: render.lastTiming.draw,
                      wait: waitMs });
          if (hist.length > 20) hist.shift();
          const a = avg();
          $("fps").textContent = `${(1000 / a.total).toFixed(1)} fps(1 コマ ${a.total.toFixed(0)} ms: 取り込み ${a.capture.toFixed(0)}・`
            + `前処理 ${a.pre.toFixed(0)}・推論 ${a.infer.toFixed(0)}(Worker の中 ${a.inferWorker.toFixed(0)})・後処理 ${a.post.toFixed(0)}・`
            + `描画 ${a.draw.toFixed(0)}・次のコマを待つ ${a.wait.toFixed(0)} ms、検出 ${dets.length} 個)`;
        }
        if (opts.maxFrames && count >= opts.maxFrames) { opts.onDone?.(avg()); break; }
      } else if (count === 0) {
        status(`取り込みの最初のコマを待っています(readyState ${video.readyState}、${w} × ${h})`);
      }
      const tw = performance.now();
      if (!(video.requestVideoFrameCallback && latest > used && used >= 0)) await nextFrame(video); // 新しいコマがなければ待つ
      waitMs = performance.now() - tw;
    }
  } catch (e) { // 繰り返しの中のエラーは、画面とコンソールに出して止める(出さないと「推論中」のまま止まって見える)
    console.error(e);
    stop();
    status(`推論の途中で止まりました: ${e.message ?? e}(詳しくは開発者ツールのコンソール)`);
  }
}

/**
 * 動作の確かめ: URL に ?selftest={画像の場所}(と &model={id}・&backend={auto|webgpu|wasm}・&repeat={回数})があれば、その画像で推論し、
 * かかった時間・検出の数・エラーを #status に書く(ブラウザを外から動かして結果を読むため)。
 */
async function selfTest() {
  const q = new URLSearchParams(location.search);
  if (q.has("selftestvideo")) return selfTestVideo(q);
  if (!q.has("selftest")) return;
  try {
    if (q.get("model")) { $("model").value = q.get("model"); selectModel(q.get("model")); }
    if (q.get("backend")) $("backend").value = q.get("backend");
    const image = new Image();
    image.src = q.get("selftest");
    await image.decode();
    const t0 = performance.now();
    const { session, backend, task } = await prepareModel();
    const t1 = performance.now();
    const frame = document.createElement("canvas");
    frame.width = image.naturalWidth; frame.height = image.naturalHeight;
    frame.getContext("2d").drawImage(image, 0, 0);
    const { feeds, meta } = task.preprocess(frame, frame.width, frame.height);
    const result = await session.run(feeds);
    const t2 = performance.now();
    const result2 = await session.run(task.preprocess(frame, frame.width, frame.height).feeds);
    const t3 = performance.now();
    // &repeat=N: さらに N 回推論して、推論の時間(前処理を含まない)の中央値を出す(速さを比べるため)
    const times = [], detail = [];
    for (let i = 0; i < Number(q.get("repeat") ?? 0); i++) {
      const f = task.preprocess(frame, frame.width, frame.height).feeds;
      const s = performance.now();
      await session.run(f);
      const total = performance.now() - s;
      times.push(total);
      const w = session.lastTiming ?? {};  // Worker の中の推論そのもの(run)と、出力を写す時間(copy)
      detail.push(`${total.toFixed(0)}(${(w.run ?? 0).toFixed(0)}+${(w.copy ?? 0).toFixed(0)})`);
    }
    if (detail.length) console.log(`SELFTEST TIMES ${detail.join(" ")}`);  // 回ごとの 全体(推論 + 写す) ms
    times.sort((a, b) => a - b);
    const median = times.length ? ` median=${times[Math.floor(times.length / 2)].toFixed(0)}ms(n=${times.length})` : "";
    if (q.has("raw")) { // 出力の形が違うモデル(確かめ用の小さいモデルなど)は、時間と出力の名前だけを書く
      status(`SELFTEST OK backend=${backend} load=${(t1 - t0).toFixed(0)}ms run1=${(t2 - t1).toFixed(0)}ms `
        + `run2=${(t3 - t2).toFixed(0)}ms${median} outputs=${Object.keys(result2).join(",")}`);
      return;
    }
    const dets = render(frame, result2, meta, task);
    status(`SELFTEST OK backend=${backend} load=${(t1 - t0).toFixed(0)}ms run1=${(t2 - t1).toFixed(0)}ms `
      + `run2=${(t3 - t2).toFixed(0)}ms${median} dets=${dets.length} `
      + dets.slice(0, 5).map((d) => `${state.model.classes[d.label]}:${d.score.toFixed(2)}`).join(","));
    void result;
  } catch (e) {
    console.error(e);
    status(`SELFTEST ERROR ${e.message ?? e}`);
  }
}

/**
 * 動作の確かめ(動画): URL に ?selftestvideo={動画の場所}(と &frames={コマ数}・&model・&backend)があれば、その動画を繰り返し再生しながら
 * 実際の推論の繰り返し(loop)を frames コマだけ動かし、段ごとの時間の平均を #status に書く。
 * @param {URLSearchParams} q ページの URL の引数。
 */
async function selfTestVideo(q) {
  try {
    if (q.get("model")) { $("model").value = q.get("model"); selectModel(q.get("model")); }
    if (q.get("backend")) $("backend").value = q.get("backend");
    const video = document.createElement("video");
    video.muted = true; video.loop = true; video.playsInline = true;
    video.src = q.get("selftestvideo");
    $("input-view").replaceChildren(video); // 実際の使い方と同じく、入力の動画を画面に出しておく
    await video.play();
    const { session, backend, task } = await prepareModel();
    state.running = true;
    await loop(video, session, task, {
      maxFrames: Number(q.get("frames") ?? 30),
      onDone: (a) => status(`SELFTEST VIDEO OK backend=${backend} `
        + Object.entries(a).map(([k, v]) => `${k}=${v.toFixed(0)}`).join(" ")),
    });
    state.running = false;
  } catch (e) {
    console.error(e);
    status(`SELFTEST ERROR ${e.message ?? e}`);
  }
}

init().then(selfTest).catch((e) => { console.error(e); status(`読み込めませんでした: ${e.message ?? e}`); });
