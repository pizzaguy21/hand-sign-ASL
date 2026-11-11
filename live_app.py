"""Mode video huruf ASL: kamera HP -> WebSocket -> ResNet50 di GPU laptop.

Jalankan dari folder proyek:
    python -u live_app.py
    tools\\cloudflared\\cloudflared.exe tunnel --url http://127.0.0.1:8502 --no-autoupdate
Buka link trycloudflare.com di HP, lalu masukkan kode akses yang tercetak saat live_app.py mulai.
Server hanya mendengarkan di 127.0.0.1. Frame diproses di memori dan tidak disimpan.
"""
import asyncio
import base64
import collections
import json
import secrets
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect

import demo_webcam as demo

ROOT = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 8502
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_SIDE = 1280
MAX_CLIENTS = 2
SMOOTH_FRAMES = 5
TOP_K = 3
CHOICES = ["utama"] + demo.PEOPLE
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
ACCESS_CODE = "".join(secrets.choice(ALPHABET) for _ in range(8))


class Engine:
    """Satu model per bobot dan satu detektor; dikunci karena dipakai dari beberapa thread."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = demo.HandDetector(str(ROOT / "hand_landmarker.task"))
        self.models = {}
        self.lock = threading.Lock()

    def model(self, choice):
        if choice not in self.models:
            self.models[choice] = demo.load_model(demo.weights_path(choice), self.device)
        return self.models[choice]

    def run(self, jpeg, choice):
        """Kembalikan dict berisi 'problem' (teks) atau 'probs' dan 'view' (JPEG base64)."""
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"problem": "Frame kamera tidak terbaca."}
        if max(bgr.shape[:2]) > MAX_SIDE:
            scale = MAX_SIDE / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        with self.lock:
            landmarks = self.detector.detect(bgr)
            if landmarks is None:
                return {"problem": "Tangan tidak terdeteksi. Coba tambah cahaya dan pakai latar polos."}
            crop_landmarks = demo.to_crop_coords(landmarks, bgr.shape)
            if not demo.hand_inside(crop_landmarks):
                return {"problem": "Tangan terpotong. Masukkan seluruh tangan ke kotak hijau."}
            image = demo.frame_to_model_image(bgr, crop_landmarks)
            probs = demo.predict_probs(self.model(choice), image, self.device)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return {"probs": probs, "view": base64.b64encode(encoded.tobytes()).decode() if ok else None}


engine = None
active_clients = 0


def valid_choice(value, fallback="utama"):
    return value if value in CHOICES else fallback


def code_matches(value):
    return secrets.compare_digest(str(value).strip().upper().encode(), ACCESS_CODE.encode())


async def homepage(request):
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store"})


async def read_hello(ws):
    try:
        hello = json.loads(await asyncio.wait_for(ws.receive_text(), 15))
    except (asyncio.TimeoutError, json.JSONDecodeError, KeyError, WebSocketDisconnect):
        return None
    return hello if isinstance(hello, dict) else None


async def stream(ws):
    global active_clients
    await ws.accept()
    hello = await read_hello(ws)
    if hello is None or not code_matches(hello.get("code", "")):
        await asyncio.sleep(1.5)                      # memperlambat tebakan kode
        await ws.send_json({"ok": False, "error": "Kode akses salah."})
        await ws.close()
        return
    if active_clients >= MAX_CLIENTS:
        await ws.send_json({"ok": False, "error": "Server sedang dipakai perangkat lain."})
        await ws.close()
        return

    active_clients += 1
    choice = valid_choice(hello.get("weights"))
    recent = collections.deque(maxlen=SMOOTH_FRAMES)
    try:
        await ws.send_json({"ok": True})
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text"):
                update = json.loads(message["text"])
                choice = valid_choice(update.get("weights") if isinstance(update, dict) else None, choice)
                recent.clear()
                continue
            data = message.get("bytes")
            if not data or len(data) > MAX_FRAME_BYTES:
                await ws.send_json({"problem": "Frame kamera tidak terbaca."})
                continue
            result = await asyncio.to_thread(engine.run, data, choice)
            if "problem" in result:
                recent.clear()
                await ws.send_json({"problem": result["problem"]})
                continue
            recent.append(result["probs"])
            probs = np.mean(recent, axis=0)
            top = [[demo.LETTERS[k], round(100 * float(probs[k]), 1)] for k in np.argsort(-probs)[:TOP_K]]
            await ws.send_json({"top": top, "view": result["view"]})
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        active_clients -= 1


PAGE = """<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Demo Huruf ASL</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#101317;color:#f2f4f7;font:16px/1.45 system-ui,sans-serif}
main{max-width:520px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:0 0 4px}
.note{color:#aab2bd;font-size:13px;margin:4px 0 12px}
label{display:block;font-size:14px;margin:12px 0 4px}
input,select,button{width:100%;font:inherit;padding:12px;border-radius:8px;border:1px solid #39424e;background:#1b2027;color:inherit}
button{background:#2f7d4f;border-color:#2f7d4f;font-weight:600;margin-top:14px}
button.alt{background:#1b2027;border-color:#39424e;font-weight:400;margin-top:8px}
#stage{position:relative;width:100%;background:#000;border-radius:10px;overflow:hidden;aspect-ratio:3/4}
video{width:100%;height:100%;object-fit:cover;display:block}
#guide{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);aspect-ratio:1;border:3px solid #25d366;width:100%}
#letter{position:absolute;left:12px;top:4px;font-size:84px;font-weight:800;color:#ffe45c;text-shadow:0 2px 8px #000}
#inset{position:absolute;right:8px;top:8px;width:92px;height:92px;border-radius:6px;background:#fff;display:none}
#bars div.row{display:flex;align-items:center;gap:8px;margin:6px 0}
#bars b{width:20px}
#bars .track{flex:1;height:12px;background:#272e37;border-radius:6px;overflow:hidden}
#bars .fill{height:100%;background:#ffe45c}
#bars span{width:58px;text-align:right;font-variant-numeric:tabular-nums}
#message{min-height:24px;color:#ffb74d;margin:10px 0 2px}
#live,#error{display:none}
#error{color:#ff8a80;margin-top:12px}
</style></head><body><main>
<h1>Demo Huruf ASL</h1>
<div class="note">ResNet50, 26 huruf A-Z. Video dikirim ke laptop lewat Cloudflare dan tidak disimpan.</div>
<section id="setup">
<label for="code">Kode akses</label>
<input id="code" autocomplete="off" autocapitalize="characters" spellcheck="false">
<label for="weights">Bobot model</label>
<select id="weights">
<option value="utama">Bobot utama (demo)</option>
<option value="steven">Uji orang baru: Steven</option>
<option value="chela">Uji orang baru: Chela</option>
<option value="gea">Uji orang baru: Gea</option>
</select>
<button id="start">Mulai kamera</button>
<div id="error"></div>
</section>
<section id="live">
<div id="stage"><video id="video" playsinline muted autoplay></video><div id="guide"></div><div id="letter"></div><img id="inset" alt="Gambar yang dilihat model"></div>
<div id="message"></div>
<div id="bars"></div>
<div class="note">Persentase adalah keyakinan model, bukan akurasinya. J dan Z aslinya bergerak, jadi kurang dapat diandalkan.</div>
<label for="weights2">Bobot model</label>
<select id="weights2"></select>
<button class="alt" id="mirror">Cermin (untuk tangan kiri): mati</button>
<button class="alt" id="flip">Ganti kamera depan/belakang</button>
<button class="alt" id="stop">Berhenti</button>
</section>
</main>
<script>
const $ = id => document.getElementById(id);
const video = $("video"), canvas = document.createElement("canvas"), ctx = canvas.getContext("2d");
let ws = null, stream = null, facing = "user", running = false, mirror = false;
$("weights2").innerHTML = $("weights").innerHTML;

function showError(text) { $("error").textContent = text; $("error").style.display = "block"; }

async function openCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: {ideal: facing}, width: {ideal: 640}, height: {ideal: 480}}});
  video.srcObject = stream;
  video.style.transform = facing === "user" ? "scaleX(-1)" : "none";
  await video.play();
  const w = video.videoWidth || 480, h = video.videoHeight || 640, scale = Math.min(1, 640 / Math.max(w, h));
  canvas.width = Math.round(w * scale); canvas.height = Math.round(h * scale);
  $("stage").style.aspectRatio = w + "/" + h;
  $("guide").style.width = w > h ? "auto" : "100%";
  $("guide").style.height = w > h ? "100%" : "auto";
}

function sendFrame() {
  if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
  ctx.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? canvas.width : 0, 0);   // cermin hanya untuk gambar ke model
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => { if (blob && running && ws.readyState === WebSocket.OPEN) ws.send(blob); else if (running) setTimeout(sendFrame, 200); }, "image/jpeg", 0.92);
}

function render(data) {
  const bars = $("bars");
  bars.replaceChildren();
  if (data.problem) {
    $("letter").textContent = ""; $("message").textContent = data.problem; $("inset").style.display = "none";
    return;
  }
  $("message").textContent = "";
  $("letter").textContent = data.top[0][0];
  if (data.view) { $("inset").src = "data:image/jpeg;base64," + data.view; $("inset").style.display = "block"; }
  for (const [letter, pct] of data.top) {
    const row = document.createElement("div"); row.className = "row";
    const b = document.createElement("b"); b.textContent = letter;
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = pct + "%"; track.append(fill);
    const span = document.createElement("span"); span.textContent = pct.toFixed(1) + "%";
    row.append(b, track, span); bars.append(row);
  }
}

function stop(message) {
  running = false;
  if (ws) ws.close();
  ws = null;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  $("live").style.display = "none"; $("setup").style.display = "block";
  if (message) showError(message);
}

$("start").onclick = async () => {
  $("error").style.display = "none";
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { showError("Browser ini tidak mendukung kamera. Pakai Chrome atau Safari terbaru."); return; }
  try { await openCamera(); } catch (e) { showError("Kamera tidak bisa dibuka. Izinkan akses kamera lalu coba lagi."); return; }
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  let authed = false;
  ws.onopen = () => ws.send(JSON.stringify({code: $("code").value, weights: $("weights").value}));
  ws.onmessage = event => {
    const data = JSON.parse(event.data);
    if (!authed) {
      if (!data.ok) { stop(data.error || "Tidak bisa terhubung."); return; }
      authed = true; running = true;
      $("weights2").value = $("weights").value;
      $("setup").style.display = "none"; $("live").style.display = "block";
      sendFrame();
      return;
    }
    render(data);
    setTimeout(sendFrame, 60);
  };
  ws.onclose = () => { if (running) stop("Koneksi ke laptop terputus."); };
  ws.onerror = () => { if (!authed) stop("Tidak bisa terhubung ke laptop."); };
};
$("weights2").onchange = () => { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({weights: $("weights2").value})); };
$("mirror").onclick = () => {
  mirror = !mirror;
  $("mirror").textContent = "Cermin (untuk tangan kiri): " + (mirror ? "nyala" : "mati");
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({weights: $("weights2").value}));   // mengosongkan rata-rata frame
};
$("flip").onclick = async () => {
  facing = facing === "user" ? "environment" : "user";
  try { await openCamera(); } catch (e) { $("message").textContent = "Kamera lain tidak bisa dibuka."; }
};
$("stop").onclick = () => stop();
</script></body></html>"""

app = Starlette(routes=[Route("/", homepage), WebSocketRoute("/ws", stream)])


def main():
    global engine
    engine = Engine()
    engine.model("utama")
    print("KODE AKSES:", ACCESS_CODE, flush=True)
    uvicorn.run(app, host=HOST, port=PORT, ws_max_size=MAX_FRAME_BYTES + 1024, log_level="warning")


if __name__ == "__main__":
    main()
