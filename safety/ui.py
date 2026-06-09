"""A tiny local web UI for the explicit-content filter.

Run ``python -m safety ui`` and a browser page opens where you can drag-and-drop
or pick an image. The server runs the same :class:`ExplicitContentFilter` used
everywhere else and returns:

* the **blurred** image when the picture is flagged (nudity / gore), or the
  **original**, untouched image when it's safe, and
* a **confidence score** in both cases — the model's confidence in its verdict.

It's a single self-contained HTML page (no build step, no JS framework, no CDN —
works fully offline) talking to one JSON endpoint, ``POST /process``.
"""
from __future__ import annotations

import base64

from .config import SafetyConfig
from .pipeline import ExplicitContentFilter
from .types import Verdict

# Imported at module level (not inside create_ui_app) so that FastAPI can resolve
# the string annotation `UploadFile` on the endpoint — with `from __future__
# import annotations`, annotations are strings looked up in the *module* globals.
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
except Exception:  # pragma: no cover - FastAPI is an optional serving dep
    FastAPI = None  # type: ignore


def decision_and_confidence(verdict: Verdict) -> tuple[str, str, float]:
    """Collapse a verdict into ``(decision, label, confidence)`` for display.

    * decision   -> ``"explicit"`` or ``"safe"``
    * label      -> the headline category (e.g. ``"nudity"``) or ``"safe"``
    * confidence -> 0..1, the model's confidence *in that decision*:
        - flagged: the highest score among the regions that triggered the blur
        - safe:    ``1 - max(any detection score)`` (high when nothing was seen)
    """
    if verdict.explicit and verdict.regions:
        top = max(verdict.regions, key=lambda r: r.score)
        return "explicit", top.category.value, float(top.score)
    # Safe: confidence = how sure we are nothing crossed the line.
    max_seen = max((r.score for r in verdict.all_regions), default=0.0)
    return "safe", "safe", float(1.0 - max_seen)


def _encode_data_url(image) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def create_ui_app(config: SafetyConfig | None = None):
    """Build a FastAPI app that serves the UI page and the ``/process`` endpoint."""
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError(
            "the UI needs FastAPI: pip install fastapi uvicorn python-multipart")

    app = FastAPI(title="Sentinel — explicit-content filter")
    filt = ExplicitContentFilter(config or SafetyConfig.from_env())

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {"status": "ok", "detectors": filt.active_detectors}

    @app.post("/process")
    async def process(file: UploadFile = File(...)):
        import cv2
        import numpy as np

        raw = await file.read()
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="not a valid image")

        clean, verdict = filt.scan_and_redact(img)
        decision, label, confidence = decision_and_confidence(verdict)
        return JSONResponse({
            "decision": decision,                 # "explicit" | "safe"
            "label": label,                       # headline category
            "confidence": round(confidence, 4),   # 0..1
            "categories": sorted(c.value for c in verdict.categories()),
            "reasons": verdict.reasons,
            "num_regions": len(verdict.regions),
            "whole_image": verdict.whole_image,
            "scores": {k: round(v, 4) for k, v in verdict.scores.items()},
            "image": _encode_data_url(clean),     # original if safe, blurred if not
        })

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True,
          config: SafetyConfig | None = None):
    """Launch the UI with uvicorn (blocking)."""
    import uvicorn

    app = create_ui_app(config)
    url = f"http://{host}:{port}"
    print(f"\n  Sentinel UI  ->  {url}\n  (Ctrl+C to stop)\n")
    if open_browser:
        try:
            import threading
            import webbrowser

            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")


# --- The page. Plain HTML/CSS/JS, no external assets so it works offline. ----
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel — explicit-content filter</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#262b36; --muted:#8b93a7;
          --safe:#2ecc71; --danger:#ff5c5c; --accent:#5b8cff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:#e8ebf2;
         font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:860px; margin:0 auto; padding:32px 20px 64px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 24px; font-size:14px; }
  .drop { border:2px dashed var(--line); border-radius:14px; padding:40px 20px;
          text-align:center; cursor:pointer; transition:.15s; background:var(--card); }
  .drop.drag { border-color:var(--accent); background:#1b2230; }
  .drop strong { color:#fff; }
  .drop input { display:none; }
  .hint { color:var(--muted); font-size:13px; margin-top:8px; }
  .result { margin-top:28px; display:none; }
  .badge { display:inline-flex; align-items:center; gap:8px; padding:8px 14px;
           border-radius:999px; font-weight:600; font-size:14px; }
  .badge.safe { background:rgba(46,204,113,.15); color:var(--safe);
                border:1px solid rgba(46,204,113,.4); }
  .badge.explicit { background:rgba(255,92,92,.15); color:var(--danger);
                    border:1px solid rgba(255,92,92,.4); }
  .meter { height:10px; background:#0c0e12; border-radius:999px; overflow:hidden;
           margin:14px 0 6px; border:1px solid var(--line); }
  .meter > div { height:100%; transition:width .4s; }
  .row { display:flex; justify-content:space-between; color:var(--muted);
         font-size:13px; }
  figure { margin:18px 0 0; }
  figure img { width:100%; border-radius:12px; border:1px solid var(--line);
               display:block; background:#000; }
  figcaption { color:var(--muted); font-size:13px; margin-top:8px; }
  .meta { margin-top:14px; font-size:13px; color:var(--muted); }
  .meta code { color:#cfd6e6; }
  .err { color:var(--danger); margin-top:16px; display:none; }
  .spin { display:none; margin-top:20px; color:var(--muted); }
  footer { margin-top:40px; color:var(--muted); font-size:12px; text-align:center; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Sentinel</h1>
  <p class="sub">Upload an image — it's blurred if it's explicit (nudity / gore),
     shown as-is if it's safe. Either way you get a confidence score.</p>

  <label class="drop" id="drop">
    <input type="file" id="file" accept="image/*">
    <div>📤 <strong>Drop an image here</strong> or click to choose</div>
    <div class="hint">JPEG / PNG / WebP — processed locally, nothing leaves your machine</div>
  </label>

  <div class="spin" id="spin">Analyzing…</div>
  <div class="err" id="err"></div>

  <div class="result" id="result">
    <span class="badge" id="badge"></span>
    <div class="meter"><div id="bar"></div></div>
    <div class="row"><span id="confLabel">confidence</span><span id="confVal"></span></div>
    <figure>
      <img id="img" alt="result">
      <figcaption id="cap"></figcaption>
    </figure>
    <div class="meta" id="meta"></div>
  </div>
</div>
<footer>Sentinel · defensive content-moderation tool · runs offline on CPU</footer>

<script>
const drop=document.getElementById('drop'), file=document.getElementById('file');
const result=document.getElementById('result'), badge=document.getElementById('badge');
const bar=document.getElementById('bar'), confVal=document.getElementById('confVal');
const confLabel=document.getElementById('confLabel'), img=document.getElementById('img');
const cap=document.getElementById('cap'), meta=document.getElementById('meta');
const spin=document.getElementById('spin'), err=document.getElementById('err');

['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{
  ev.preventDefault(); drop.classList.add('drag'); }));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{
  ev.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop',ev=>{ if(ev.dataTransfer.files[0]) send(ev.dataTransfer.files[0]); });
file.addEventListener('change',()=>{ if(file.files[0]) send(file.files[0]); });

async function send(f){
  err.style.display='none'; result.style.display='none'; spin.style.display='block';
  const fd=new FormData(); fd.append('file', f);
  try{
    const r=await fetch('/process',{method:'POST',body:fd});
    if(!r.ok){ const j=await r.json().catch(()=>({detail:r.statusText}));
               throw new Error(j.detail||('HTTP '+r.status)); }
    render(await r.json());
  }catch(e){ err.textContent='⚠ '+e.message; err.style.display='block'; }
  finally{ spin.style.display='none'; }
}

function render(d){
  const explicit = d.decision==='explicit';
  const pct = Math.round(d.confidence*100);
  badge.className = 'badge '+(explicit?'explicit':'safe');
  badge.textContent = explicit
    ? ('⚠ Flagged — '+d.label.toUpperCase()+' (blurred)')
    : '✓ Safe';
  confLabel.textContent = explicit
    ? ('confidence · '+d.label) : 'safe confidence';
  confVal.textContent = pct+'%';
  bar.style.width = pct+'%';
  bar.style.background = explicit ? 'var(--danger)' : 'var(--safe)';
  img.src = d.image;
  cap.textContent = explicit
    ? (d.whole_image
        ? 'entire image blurred — '+d.num_regions+' detection(s)'
        : (d.num_regions+' region(s) redacted'))
    : 'original image (unmodified)';
  const parts=[];
  if(d.categories.length) parts.push('categories: <code>'+d.categories.join(', ')+'</code>');
  const sc=Object.entries(d.scores||{});
  if(sc.length) parts.push('scores: <code>'+sc.map(([k,v])=>k+' '+v).join(', ')+'</code>');
  meta.innerHTML = parts.join(' &nbsp;·&nbsp; ');
  result.style.display='block';
}
</script>
</body>
</html>
"""
