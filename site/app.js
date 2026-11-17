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
  const dets = task.postprocess(result, meta, { threshold: Number($("threshold").value) });
  const out = $("output");
  out.width = frame.width; out.height = frame.height;
  const ctx = out.getContext("2d");
  ctx.drawImage(frame, 0, 0);
  task.draw(ctx, dets, meta);
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
 * @param {HTMLVideoElement} video 入力の動画。
 * @param {object} session ONNX Runtime Web のセッション。
 * @param {object} task createTask の返り値。
 */
async function loop(video, session, task) {
  const frame = document.createElement("canvas"); // 推論に使った 1 コマ(結果はこの 1 コマに重ねる)
  const out = $("output");
  const times = [];
  let count = 0;
  try {
    while (state.running) {
      const w = video.videoWidth, h = video.videoHeight;
      if (video.readyState >= 2 && w && h) {
        const t0 = performance.now();
        frame.width = w; frame.height = h;
        frame.getContext("2d").drawImage(video, 0, 0, w, h);
        if (count === 0) { // 1 回目: 取り込めているかが分かるよう、推論の前に取り込んだコマをそのまま描く
          out.width = w; out.height = h;
          out.getContext("2d").drawImage(frame, 0, 0);
          status(`取り込み ${w} × ${h}。1 回目の推論中…(WebGPU は初回の準備に時間がかかることがあります)`);
        }
        const { feeds, meta } = task.preprocess(frame, w, h);
        const result = await session.run(feeds);
        const t1 = performance.now();
        const dets = render(frame, result, meta, task);
        count += 1;
        if (count === 1) status(`推論中(1 回目の推論 ${(t1 - t0).toFixed(0)} ms)`);
        times.push(performance.now() - t0);
        if (times.length > 20) times.shift();
        const ms = times.reduce((a, b) => a + b, 0) / times.length;
        $("fps").textContent = `${(1000 / ms).toFixed(1)} fps(1 コマ ${ms.toFixed(0)} ms、検出 ${dets.length} 個)`;
      } else if (count === 0) {
        status(`取り込みの最初のコマを待っています(readyState ${video.readyState}、${w} × ${h})`);
      }
      await nextFrame(video);
    }
  } catch (e) { // 繰り返しの中のエラーは、画面とコンソールに出して止める(出さないと「推論中」のまま止まって見える)
    console.error(e);
    stop();
    status(`推論の途中で止まりました: ${e.message ?? e}(詳しくは開発者ツールのコンソール)`);
  }
}

/**
 * 動作の確かめ: URL に ?selftest={画像の場所}(と &model={id}・&backend={auto|webgpu|wasm})があれば、その画像で 1 回だけ推論し、
 * かかった時間・検出の数・エラーを #status に書く(ブラウザを外から動かして結果を読むため)。
 */
async function selfTest() {
  const q = new URLSearchParams(location.search);
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
    if (q.has("raw")) { // 出力の形が違うモデル(確かめ用の小さいモデルなど)は、時間と出力の名前だけを書く
      status(`SELFTEST OK backend=${backend} load=${(t1 - t0).toFixed(0)}ms run1=${(t2 - t1).toFixed(0)}ms `
        + `run2=${(t3 - t2).toFixed(0)}ms outputs=${Object.keys(result2).join(",")}`);
      return;
    }
    const dets = render(frame, result2, meta, task);
    status(`SELFTEST OK backend=${backend} load=${(t1 - t0).toFixed(0)}ms run1=${(t2 - t1).toFixed(0)}ms `
      + `run2=${(t3 - t2).toFixed(0)}ms dets=${dets.length} `
      + dets.slice(0, 5).map((d) => `${state.model.classes[d.label]}:${d.score.toFixed(2)}`).join(","));
    void result;
  } catch (e) {
    console.error(e);
    status(`SELFTEST ERROR ${e.message ?? e}`);
  }
}

init().then(selfTest).catch((e) => { console.error(e); status(`読み込めませんでした: ${e.message ?? e}`); });
