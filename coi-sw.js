// デモのページを「cross-origin isolated」にする service worker。
//
// ONNX Runtime Web の WASM(CPU)は、ページが cross-origin isolated(COOP・COEP のヘッダーが付いた状態)のときだけ、
// 複数のスレッドで計算できる(SharedArrayBuffer を使うため)。GitHub Pages はヘッダーを足せないので、この service worker が
// ページの読み込みの応答に COOP・COEP のヘッダーを足す。COEP は credentialless(別のサイトの読み込みは、認証の情報なしで
// 行えばよい形)にして、CDN(jsdelivr)のスクリプトや YouTube の埋め込み(iframe に credentialless を付ける)も使えるようにする。
// リポジトリのトップに置く(service worker が受け持てるのは、自分の場所より下のページだけのため)。

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith((async () => {
    const res = await fetch(req);
    if (res.status === 0) return res; // 中身の見えない応答(別のサイトの no-cors)はそのまま
    const headers = new Headers(res.headers);
    headers.set("Cross-Origin-Embedder-Policy", "credentialless");
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
  })());
});
