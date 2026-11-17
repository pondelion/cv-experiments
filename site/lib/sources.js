// 推論に入れる動画の入手(タスクによらない)。どの入手先も、1 コマずつ読める <video> を返す形にそろえる。
//
// - youtube: YouTube の公式の埋め込みで再生し、このタブの画面の取り込み(getDisplayMedia)を、埋め込みの部分だけに
//   切り抜いて(Region Capture: CropTarget.fromElement → track.cropTo)読む。動画のファイルは取ってこない。
//   PC 版の Chrome・Edge(104 以降)だけで動く。切り抜く場所の上に重なった画素も取り込まれるので、結果は埋め込みに
//   重ねず、横に並べて表示する
// - file: 手元の動画・画像のファイル(画像は <video> ではなく <img> を返し、推論は 1 回だけ)
// - camera: カメラ

/**
 * YouTube の URL から動画の ID を取り出す。
 * @param {string} url youtube.com/watch?v=・youtu.be/・/shorts/・/embed/ の URL か、ID そのもの。
 * @returns {string|null} 動画の ID(11 文字)。読めなければ null。
 */
export function parseYouTubeId(url) {
  const s = url.trim();
  if (/^[\w-]{11}$/.test(s)) return s;
  try {
    const u = new URL(s);
    if (u.hostname.endsWith("youtu.be")) return u.pathname.slice(1, 12) || null;
    if (u.searchParams.get("v")) return u.searchParams.get("v").slice(0, 11);
    const m = u.pathname.match(/\/(shorts|embed|live)\/([\w-]{11})/);
    return m ? m[2] : null;
  } catch {
    return null;
  }
}

/**
 * Region Capture(タブの取り込みを、ページの一部の要素の場所に切り抜く)が使えるか。
 * @returns {boolean}
 */
export function regionCaptureSupported() {
  return "CropTarget" in self && "fromElement" in CropTarget && !!navigator.mediaDevices?.getDisplayMedia;
}

/**
 * MediaStream を流す <video> を作る(画面には出さない)。
 * @param {MediaStream} stream 動画の流れ。
 * @returns {Promise<HTMLVideoElement>} 再生を始めた <video>。
 */
async function videoFromStream(stream) {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  return video;
}

/**
 * YouTube の埋め込みを置く。
 * @param {HTMLElement} container 埋め込みを入れる要素(この要素の場所が取り込みの切り抜きになる)。
 * @param {string} id 動画の ID。
 */
export function embedYouTube(container, id) {
  container.innerHTML = "";
  const iframe = document.createElement("iframe");
  // ページは COEP の credentialless(coi-sw.js)なので、別のサイトの iframe は credentialless にしないと読み込めない
  iframe.credentialless = true;
  iframe.src =`https://www.youtube.com/embed/${id}?playsinline=1&rel=0`;
  iframe.title = "YouTube の動画";
  iframe.allow = "autoplay; encrypted-media; picture-in-picture";
  iframe.allowFullscreen = false;
  container.appendChild(iframe);
}

/**
 * このタブの取り込みを始め、container の場所だけに切り抜いた <video> を返す。
 * @param {HTMLElement} container 切り抜く場所の要素(YouTube の埋め込みを入れた要素)。
 * @returns {Promise<{video: HTMLVideoElement, stop: () => void}>}
 */
export async function captureElement(container) {
  if (!regionCaptureSupported()) {
    throw new Error("このブラウザは Region Capture に対応していない(PC 版の Chrome・Edge 104 以降で開いてください)");
  }
  const cropTarget = await CropTarget.fromElement(container);
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: 30 }, audio: false, preferCurrentTab: true, selfBrowserSurface: "include",
  });
  const [track] = stream.getVideoTracks();
  if (track.getSettings().displaySurface && track.getSettings().displaySurface !== "browser") {
    track.stop();
    throw new Error("「このタブ」を選んでください(ほかの画面・ウィンドウでは切り抜けない)");
  }
  await track.cropTo(cropTarget);
  const video = await videoFromStream(stream);
  return { video, stop: () => stream.getTracks().forEach((t) => t.stop()) };
}

/**
 * 手元の動画のファイルを再生する <video> を返す(繰り返し再生)。
 * @param {File} file 動画のファイル。
 * @returns {Promise<{video: HTMLVideoElement, stop: () => void}>}
 */
export async function fileVideo(file) {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.loop = true;
  const url = URL.createObjectURL(file);
  video.src = url;
  await video.play();
  return { video, stop: () => { video.pause(); URL.revokeObjectURL(url); } };
}

/**
 * 手元の画像のファイルを読む(静止画。推論は 1 回だけ)。
 * @param {File} file 画像のファイル。
 * @returns {Promise<{image: HTMLImageElement, stop: () => void}>}
 */
export async function imageFile(file) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.src = url;
  await image.decode();
  return { image, stop: () => URL.revokeObjectURL(url) };
}

/**
 * カメラの映像の <video> を返す。
 * @returns {Promise<{video: HTMLVideoElement, stop: () => void}>}
 */
export async function cameraVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  const video = await videoFromStream(stream);
  return { video, stop: () => stream.getTracks().forEach((t) => t.stop()) };
}
