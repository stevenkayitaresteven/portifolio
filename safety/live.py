"""Real-time webcam moderation — blur nudity/gore live as the camera runs.

Two front-ends over the same image pipeline
(:class:`~safety.pipeline.ExplicitContentFilter`, i.e. NudeNet + the wound
heuristic + the blur engine):

* **Browser** (default) — ``python -m safety live`` opens a page that grabs your
  webcam with ``getUserMedia``, streams frames to the server, and paints back
  the blurred frame with a live verdict pill. Works on any machine (and is the
  shareable/hosted surface), since the camera lives in the browser.
* **Native OpenCV window** — ``python -m safety live --opencv`` opens
  ``cv2.VideoCapture(0)`` and shows a blurred preview window. Lowest latency,
  no browser, but needs a local camera and a desktop session.

Frames are processed one-at-a-time (the client requests the next frame only
after the previous verdict returns), so throughput self-tunes to whatever the
box can do — a few FPS on CPU, more with a GPU-backed detector. Nothing is
stored: each frame is moderated in memory and dropped.
"""
from __future__ import annotations

import base64
import time

from .config import SafetyConfig
from .pipeline import ExplicitContentFilter

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
except Exception:  # pragma: no cover - FastAPI is an optional serving dep
    FastAPI = None  # type: ignore


# --- native OpenCV window ----------------------------------------------------
def run_opencv(config: SafetyConfig | None = None, camera: int = 0,
               width: int = 640) -> int:  # pragma: no cover - needs a camera
    """Open the webcam and show a live, blurred preview (press q to quit)."""
    import cv2

    filt = ExplicitContentFilter(config or SafetyConfig.from_env())
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"!! could not open camera {camera}")
        return 1
    print("  Sentinel live — press 'q' in the window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if width and frame.shape[1] > width:
                h = int(frame.shape[0] * width / frame.shape[1])
                frame = cv2.resize(frame, (width, h))
            clean, verdict = filt.scan_and_redact(frame)
            if verdict.explicit:
                label = "BLURRED: " + ", ".join(
                    sorted(c.value for c in verdict.categories()))
                cv2.putText(clean, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Sentinel live (q to quit)", clean)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


# --- browser app -------------------------------------------------------------
def create_live_app(config: SafetyConfig | None = None):
    """Build the FastAPI app for the browser webcam UI."""
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("the live UI needs FastAPI: pip install fastapi "
                           "uvicorn")

    app = FastAPI(title="Sentinel Live — real-time webcam moderation")
    filt = ExplicitContentFilter(config or SafetyConfig.from_env())

    def _decode(raw: bytes):
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="invalid frame")
        return img

    @app.get("/", response_class=HTMLResponse)
    def index():
        return LIVE_HTML

    @app.get("/live/health")
    def health():
        return {"status": "ok", "detectors": filt.active_detectors}

    @app.post("/live/frame")
    async def frame(request: Request):
        import cv2

        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="empty frame")
        t0 = time.perf_counter()
        img = _decode(raw)
        clean, verdict = filt.scan_and_redact(img)
        ok, buf = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:  # pragma: no cover
            raise HTTPException(status_code=500, detail="encode failed")
        return JSONResponse({
            "image": "data:image/jpeg;base64," +
                     base64.b64encode(buf.tobytes()).decode(),
            "explicit": verdict.explicit,
            "categories": sorted(c.value for c in verdict.categories()),
            "regions": len(verdict.regions),
            "severity": str(verdict.max_severity),
            "ms": round((time.perf_counter() - t0) * 1000),
        })

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True,
          config: SafetyConfig | None = None):
    """Launch the browser webcam UI with uvicorn (blocking)."""
    import uvicorn

    app = create_live_app(config)
    url = f"http://{host}:{port}"
    print(f"\n  Sentinel Live  ->  {url}\n  (allow camera access; Ctrl+C to stop)\n")
    if open_browser:
        try:
            import threading
            import webbrowser

            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")


# --- the page: webcam capture loop, plain HTML/JS, dark Sentinel look --------
LIVE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel Live</title>
<style>
  :root{--bg:#0b141a;--panel:#202c33;--line:#2a3942;--txt:#e9edef;
        --muted:#8696a0;--accent:#00a884;--danger:#f15c6d}
  *{box-sizing:border-box} html,body{height:100%;margin:0}
  body{background:var(--bg);color:var(--txt);
       font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       display:flex;flex-direction:column;align-items:center}
  header{width:100%;display:flex;align-items:center;gap:12px;padding:12px 18px;
         background:var(--panel);border-bottom:1px solid var(--line)}
  .avatar{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;
          justify-content:center;font-size:19px;
          background:linear-gradient(135deg,#00a884,#005c4b)}
  header h1{font-size:16px;margin:0} header .sub{font-size:12.5px;color:var(--muted)}
  main{flex:1;display:flex;flex-direction:column;align-items:center;
       justify-content:center;gap:14px;padding:18px;width:100%;max-width:760px}
  .stage{position:relative;width:100%;background:#000;border-radius:14px;
         overflow:hidden;aspect-ratio:4/3;border:1px solid var(--line)}
  #out{width:100%;height:100%;object-fit:contain;display:block}
  #placeholder{position:absolute;inset:0;display:flex;align-items:center;
        justify-content:center;color:var(--muted);text-align:center;padding:24px}
  .pill{position:absolute;top:12px;left:12px;display:none;align-items:center;
        gap:7px;font-size:13px;font-weight:600;padding:6px 12px;border-radius:999px;
        background:rgba(241,92,109,.18);color:var(--danger);
        border:1px solid rgba(241,92,109,.5)}
  .pill.safe{background:rgba(0,168,132,.16);color:var(--accent);
             border-color:rgba(0,168,132,.5)}
  .stat{position:absolute;bottom:10px;right:12px;font-size:11.5px;
        color:rgba(233,237,239,.7);background:rgba(0,0,0,.4);padding:3px 8px;
        border-radius:8px}
  .controls{display:flex;gap:10px;align-items:center}
  button{background:var(--accent);color:#06100d;border:none;border-radius:10px;
         padding:11px 20px;font:inherit;font-weight:600;cursor:pointer}
  button.stop{background:var(--danger);color:#fff} button:disabled{opacity:.5}
  .note{font-size:12.5px;color:var(--muted);text-align:center;max-width:560px}
  video,canvas{display:none}
</style>
</head>
<body>
<header>
  <div class="avatar">🛡️</div>
  <div><h1>Sentinel Live</h1>
    <div class="sub" id="sub">real-time webcam moderation · nudity &amp; gore blurred on the fly</div></div>
</header>
<main>
  <div class="stage">
    <img id="out" alt="moderated stream">
    <div id="placeholder">Click <b>&nbsp;Start camera&nbsp;</b> below.<br>
      Frames are moderated on the server and never stored.</div>
    <div class="pill" id="pill"></div>
    <div class="stat" id="stat"></div>
  </div>
  <div class="controls">
    <button id="start">Start camera</button>
    <button id="stop" class="stop" disabled>Stop</button>
  </div>
  <div class="note">Your browser captures the webcam; each frame is sent to the
    server, scanned for nudity/gore, and the <b>blurred</b> frame is sent back.
    Processing is one frame at a time, so the rate adapts to the hardware.</div>
</main>

<video id="cam" autoplay playsinline muted></video>
<canvas id="cap"></canvas>

<script>
const cam=document.getElementById('cam'), cap=document.getElementById('cap');
const out=document.getElementById('out'), pill=document.getElementById('pill');
const stat=document.getElementById('stat'), ph=document.getElementById('placeholder');
const startBtn=document.getElementById('start'), stopBtn=document.getElementById('stop');
const ctx=cap.getContext('2d');
let stream=null, running=false, SEND_W=480;

startBtn.onclick=async()=>{
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
  }catch(e){ document.getElementById('sub').textContent='camera blocked: '+e.message; return; }
  cam.srcObject=stream;
  await cam.play();
  ph.style.display='none';
  running=true; startBtn.disabled=true; stopBtn.disabled=false;
  loop();
};
stopBtn.onclick=()=>{
  running=false; startBtn.disabled=false; stopBtn.disabled=true;
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  pill.style.display='none'; stat.textContent='';
};

async function loop(){
  while(running){
    const w=SEND_W, h=Math.round(cam.videoHeight*w/cam.videoWidth)||360;
    cap.width=w; cap.height=h;
    ctx.drawImage(cam,0,0,w,h);
    const blob=await new Promise(r=>cap.toBlob(r,'image/jpeg',0.7));
    if(!blob){ await sleep(50); continue; }
    try{
      const t0=performance.now();
      const r=await fetch('/live/frame',{method:'POST',body:blob,
        headers:{'Content-Type':'application/octet-stream'}});
      const j=await r.json();
      out.src=j.image;
      const rtt=Math.round(performance.now()-t0);
      stat.textContent=(j.ms)+' ms scan · '+rtt+' ms total · ~'+
        (rtt?Math.max(1,Math.round(1000/rtt)):0)+' fps';
      if(j.explicit){
        pill.className='pill'; pill.style.display='inline-flex';
        pill.textContent='⚠ blurred · '+(j.categories.join(', ')||'sensitive');
      }else{
        pill.className='pill safe'; pill.style.display='inline-flex';
        pill.textContent='✓ clear';
      }
    }catch(e){ stat.textContent='error: '+e.message; await sleep(300); }
  }
}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
</script>
</body>
</html>
"""
