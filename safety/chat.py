"""A WhatsApp-style chat UI with built-in multimodal content moderation.

Run ``python -m safety chat`` and a chat page opens. You can type messages and
attach **one file at a time** — image, video, audio, or text document. Every
message passes through the :class:`~safety.multimodal.MultimodalModerator`
before it lands in the conversation:

* **text** — curse words are masked in place (``f***``); model-flagged
  hostility is delivered with a warning badge (toxic-bert).
* **image** — explicit images (nudity / gore) are delivered **blurred**, with
  the category and confidence shown on the bubble.
* **video** — explicit clips are re-encoded fully blurred; clips whose *audio*
  is explicit are delivered muted.
* **audio** — voice notes are transcribed (Whisper) and moderated; explicit
  audio is **removed** and replaced by its censored transcript.
* **documents** (.txt/.md/...) — content is moderated like text; a censored
  copy is what gets shared.

The page is one self-contained HTML file (no build step, no CDN). State is
in-memory + a temp media dir — it's a demo surface for the moderation engine,
not a messaging backend.
"""
from __future__ import annotations

import mimetypes
import os
import shutil
import tempfile
import time
import uuid

from .multimodal import MultimodalModerator, MultimodalConfig, modality_for
from .multimodal.result import ACTION_BLOCK

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
except Exception:  # pragma: no cover - FastAPI is an optional serving dep
    FastAPI = None  # type: ignore

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # one WhatsApp-ish file cap


def create_chat_app(config: MultimodalConfig | None = None,
                    moderator: MultimodalModerator | None = None):
    """Build the FastAPI chat app (moderator injectable for tests)."""
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError(
            "the chat UI needs FastAPI: pip install fastapi uvicorn python-multipart")

    from contextlib import asynccontextmanager

    media_dir = tempfile.mkdtemp(prefix="sentinel_chat_")

    @asynccontextmanager
    async def _lifespan(_app):
        yield
        shutil.rmtree(media_dir, ignore_errors=True)

    app = FastAPI(title="Sentinel Chat — moderated multimodal chat",
                  lifespan=_lifespan)
    mod = moderator or MultimodalModerator(config or MultimodalConfig.from_env())
    messages: list[dict] = []
    media_index: dict[str, tuple[str, str]] = {}  # id -> (path, mime)

    def _register_media(path: str, mime: str) -> str:
        mid = uuid.uuid4().hex
        media_index[mid] = (path, mime)
        return f"/chat/media/{mid}"

    def _result_fields(res) -> dict:
        return {
            "flagged": res.flagged,
            "action": res.action,
            "categories": sorted(res.categories),
            "confidence": round(res.confidence, 4),
            "reasons": res.reasons,
            "detectors": res.detectors,
            "scanned": res.scanned,
        }

    def _push(msg: dict) -> dict:
        msg["id"] = uuid.uuid4().hex[:12]
        msg["ts"] = time.strftime("%H:%M")
        messages.append(msg)
        return msg

    @app.get("/", response_class=HTMLResponse)
    def index():
        return CHAT_HTML

    @app.get("/chat/health")
    def health():
        return mod.health()

    @app.get("/chat/messages")
    def list_messages():
        return {"messages": messages}

    @app.get("/chat/media/{mid}")
    def get_media(mid: str):
        entry = media_index.get(mid)
        if entry is None or not os.path.exists(entry[0]):
            raise HTTPException(status_code=404, detail="no such media")
        return FileResponse(entry[0], media_type=entry[1])

    @app.post("/chat/send")
    async def send(text: str = Form(""),
                   files: list[UploadFile] = File(default=[])):
        files = [f for f in files if f and f.filename]
        if len(files) > 1:
            raise HTTPException(status_code=400,
                                detail="one file at a time, please")
        text = text.strip()
        if not text and not files:
            raise HTTPException(status_code=400, detail="empty message")

        out: list[dict] = []
        caption = None
        if text:
            res = mod.moderate_text(text)
            blocked = res.action == ACTION_BLOCK
            entry = {
                "kind": "text",
                "text": "" if blocked else (res.censored_text or text),
                **_result_fields(res),
            }
            if files:           # text typed alongside a file rides as caption
                caption = entry
            else:
                out.append(_push(entry))

        if files:
            out.append(_push(await _handle_file(files[0], caption)))
        return JSONResponse({"messages": out})

    async def _handle_file(upload: UploadFile, caption: dict | None) -> dict:
        name = os.path.basename(upload.filename or "file")
        modality = modality_for(name, upload.content_type)
        if modality is None:
            raise HTTPException(
                status_code=400,
                detail="unsupported file type — send an image, video, "
                       "audio file, or text document")

        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large (100 MB max)")
        if not data:
            raise HTTPException(status_code=400, detail="empty file")

        msg: dict = {"kind": modality, "filename": name}
        if caption:
            msg["caption"] = caption["text"]
            msg["caption_flagged"] = caption["flagged"]

        if modality == "image":
            jpeg, res = mod.image.moderate_bytes(data)
            if jpeg is None:
                raise HTTPException(status_code=400, detail="not a valid image")
            path = os.path.join(media_dir, uuid.uuid4().hex + ".jpg")
            with open(path, "wb") as fh:
                fh.write(jpeg)
            msg["media"] = _register_media(path, "image/jpeg")
            msg.update(_result_fields(res))
            return msg

        # video / audio / text documents are moderated from disk
        ext = os.path.splitext(name)[1].lower() or ".bin"
        src = os.path.join(media_dir, uuid.uuid4().hex + ext)
        with open(src, "wb") as fh:
            fh.write(data)

        out_hint = None
        if modality == "video":
            out_hint = os.path.join(media_dir, uuid.uuid4().hex + ".mp4")
        delivered, res = mod.moderate_file(src, modality=modality,
                                           out_path=out_hint)
        msg.update(_result_fields(res))
        if res.transcript:
            msg["transcript"] = res.transcript
        if modality == "text":
            preview = ""
            if delivered:
                with open(delivered, "r", encoding="utf-8",
                          errors="replace") as fh:
                    preview = fh.read(600)
            msg["text"] = preview

        if delivered is None:
            # withheld entirely (explicit audio, unreadable file, ...)
            try:
                os.unlink(src)
            except OSError:
                pass
            msg["media"] = None
            return msg

        if delivered != src:
            # a redacted copy is what ships; drop the explicit original
            try:
                os.unlink(src)
            except OSError:
                pass
        mime = ("video/mp4" if modality == "video" and delivered.endswith(".mp4")
                else mimetypes.guess_type(delivered)[0]
                or upload.content_type or "application/octet-stream")
        msg["media"] = _register_media(delivered, mime)
        return msg

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True,
          config: MultimodalConfig | None = None):
    """Launch the chat UI with uvicorn (blocking)."""
    import uvicorn

    app = create_chat_app(config)
    url = f"http://{host}:{port}"
    print(f"\n  Sentinel Chat  ->  {url}\n  (Ctrl+C to stop)\n")
    if open_browser:
        try:
            import threading
            import webbrowser

            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")


# --- The page. Plain HTML/CSS/JS, WhatsApp-dark look, no external assets. -----
CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel Chat</title>
<style>
  :root{
    --bg:#0b141a; --panel:#202c33; --panel2:#111b21; --line:#2a3942;
    --sent:#005c4b; --recv:#202c33; --txt:#e9edef; --muted:#8696a0;
    --accent:#00a884; --danger:#f15c6d; --warn:#ffb02e;
  }
  *{box-sizing:border-box} html,body{height:100%}
  body{margin:0;background:var(--panel2);color:var(--txt);
       font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       display:flex;flex-direction:column}
  /* header */
  header{display:flex;align-items:center;gap:12px;padding:10px 16px;
         background:var(--panel);border-bottom:1px solid var(--line);flex:0 0 auto}
  .avatar{width:40px;height:40px;border-radius:50%;flex:0 0 auto;
          background:linear-gradient(135deg,#00a884,#005c4b);
          display:flex;align-items:center;justify-content:center;font-size:20px}
  header h1{font-size:16px;margin:0;font-weight:600}
  header .status{font-size:12.5px;color:var(--muted)}
  /* chat surface */
  #chat{flex:1 1 auto;overflow-y:auto;padding:18px 7% 12px;
        background:
          radial-gradient(circle at 25% 15%, rgba(0,168,132,.05), transparent 40%),
          radial-gradient(circle at 80% 70%, rgba(0,168,132,.04), transparent 45%),
          var(--bg)}
  .day{align-self:center;text-align:center;margin:4px auto 14px;font-size:12px;
       color:var(--muted);background:var(--panel);padding:5px 12px;border-radius:8px;
       width:fit-content}
  .msg{display:flex;flex-direction:column;align-items:flex-end;margin:3px 0}
  .bubble{position:relative;max-width:min(480px,85%);background:var(--sent);
          border-radius:10px;border-top-right-radius:2px;padding:7px 9px 8px;
          box-shadow:0 1px 1px rgba(0,0,0,.3);overflow-wrap:anywhere}
  .bubble .body{white-space:pre-wrap}
  .meta{display:flex;gap:5px;justify-content:flex-end;align-items:center;
        font-size:11px;color:rgba(233,237,239,.6);margin-top:3px}
  .tick{color:#53bdeb;font-size:13px;line-height:1}
  /* moderation pills */
  .pill{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;
        font-weight:600;border-radius:999px;padding:3px 10px;margin-bottom:6px;
        width:fit-content}
  .pill.flag{background:rgba(241,92,109,.16);color:var(--danger);
             border:1px solid rgba(241,92,109,.45)}
  .pill.warn{background:rgba(255,176,46,.14);color:var(--warn);
             border:1px solid rgba(255,176,46,.4)}
  .reasons{font-size:11.5px;color:rgba(233,237,239,.55);margin-top:5px}
  /* media */
  .bubble img,.bubble video{display:block;max-width:100%;width:340px;
        border-radius:8px;background:#000}
  .bubble audio{width:300px;max-width:100%;display:block}
  .blocked{display:flex;gap:10px;align-items:flex-start;background:rgba(241,92,109,.1);
        border:1px solid rgba(241,92,109,.35);border-radius:8px;padding:10px 12px;
        width:320px;max-width:100%}
  .blocked .ico{font-size:20px}
  .blocked b{color:var(--danger);font-size:13.5px}
  .blocked .tr{font-size:13px;color:var(--muted);font-style:italic;margin-top:4px}
  .doc{display:flex;gap:10px;align-items:center;background:rgba(255,255,255,.06);
       border-radius:8px;padding:10px 12px;width:300px;max-width:100%}
  .doc .ico{font-size:22px}
  .doc a{color:var(--accent);text-decoration:none;font-weight:600;font-size:14px}
  .doc .sub{font-size:12px;color:var(--muted)}
  .excerpt{font-size:13px;color:rgba(233,237,239,.8);margin-top:6px;
           border-left:3px solid var(--line);padding-left:8px;white-space:pre-wrap}
  .caption{margin-top:6px}
  /* composer */
  footer{flex:0 0 auto;display:flex;align-items:flex-end;gap:8px;
         padding:8px 12px;background:var(--panel)}
  .iconbtn{background:none;border:none;color:var(--muted);font-size:22px;
           cursor:pointer;padding:9px 8px;border-radius:50%;line-height:1}
  .iconbtn:hover{color:var(--txt)}
  .inputwrap{flex:1;background:var(--recv);border-radius:10px;padding:6px 12px}
  #attach-chip{display:none;align-items:center;gap:8px;font-size:13px;
        color:var(--accent);padding:4px 0 2px}
  #attach-chip button{background:none;border:none;color:var(--danger);
        cursor:pointer;font-size:14px}
  #text{width:100%;background:none;border:none;outline:none;color:var(--txt);
        font:inherit;resize:none;max-height:120px;padding:5px 0}
  #send{background:var(--accent);color:#06100d;border:none;border-radius:50%;
        width:44px;height:44px;font-size:19px;cursor:pointer;flex:0 0 auto}
  #send:disabled{opacity:.5;cursor:default}
  /* attach menu */
  #menu{position:absolute;bottom:74px;left:14px;background:var(--panel);
        border:1px solid var(--line);border-radius:14px;padding:8px;display:none;
        box-shadow:0 8px 24px rgba(0,0,0,.5);z-index:5}
  #menu button{display:flex;gap:10px;align-items:center;width:200px;
        background:none;border:none;color:var(--txt);font:inherit;cursor:pointer;
        padding:10px 12px;border-radius:9px;text-align:left}
  #menu button:hover{background:rgba(255,255,255,.06)}
  #menu .mi{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;
        justify-content:center;font-size:17px}
  .err{position:fixed;top:70px;left:50%;transform:translateX(-50%);
       background:#3b1d22;color:var(--danger);border:1px solid rgba(241,92,109,.5);
       padding:9px 16px;border-radius:10px;font-size:13.5px;display:none;z-index:9}
  .sending{font-size:12.5px;color:var(--muted);text-align:right;margin:4px 2px}
</style>
</head>
<body>
<header>
  <div class="avatar">🛡️</div>
  <div>
    <h1>Sentinel Chat</h1>
    <div class="status" id="status">multimodal moderation · image · video · audio · text</div>
  </div>
</header>

<div id="chat">
  <div class="day">Messages are scanned before delivery — explicit images &amp; video
  are <b>blurred</b>, explicit audio is <b>removed</b>, curse words are <b>masked</b>.</div>
</div>
<div class="err" id="err"></div>

<div id="menu">
  <button data-accept="image/*">  <span class="mi" style="background:#7f66ff22;color:#a99bff">🖼️</span> Photo</button>
  <button data-accept="video/*">  <span class="mi" style="background:#f15c6d22;color:#f58ea0">🎬</span> Video</button>
  <button data-accept="audio/*">  <span class="mi" style="background:#ffb02e22;color:#ffc76b">🎙️</span> Audio</button>
  <button data-accept=".txt,.md,.csv,.log,.json,text/plain">
                                  <span class="mi" style="background:#00a88422;color:#39d3b5">📄</span> Document</button>
</div>

<footer>
  <button class="iconbtn" id="clip" title="Attach (one file at a time)">📎</button>
  <div class="inputwrap">
    <div id="attach-chip"><span id="attach-name"></span>
      <button id="attach-x" title="remove">✕</button></div>
    <textarea id="text" rows="1" placeholder="Type a message"></textarea>
  </div>
  <button id="send" title="Send">➤</button>
  <input type="file" id="file" hidden>
</footer>

<script>
const chat=document.getElementById('chat'), text=document.getElementById('text');
const send=document.getElementById('send'), clip=document.getElementById('clip');
const menu=document.getElementById('menu'), fileIn=document.getElementById('file');
const chipWrap=document.getElementById('attach-chip');
const chipName=document.getElementById('attach-name');
const err=document.getElementById('err');
let attached=null;

/* ---- attach menu: choose ONE file of a chosen type ---- */
clip.onclick=e=>{e.stopPropagation();
  menu.style.display=menu.style.display==='block'?'none':'block';};
document.body.addEventListener('click',()=>menu.style.display='none');
menu.querySelectorAll('button').forEach(b=>b.onclick=e=>{
  e.stopPropagation(); menu.style.display='none';
  fileIn.accept=b.dataset.accept; fileIn.value=''; fileIn.click();});
fileIn.onchange=()=>{
  if(!fileIn.files[0]) return;
  attached=fileIn.files[0];           // exactly one — replaces any previous pick
  chipName.textContent='📎 '+attached.name+'  ('+fmtSize(attached.size)+')';
  chipWrap.style.display='flex'; text.focus();};
document.getElementById('attach-x').onclick=()=>{attached=null;
  chipWrap.style.display='none'; fileIn.value='';};
function fmtSize(n){return n>1048576?(n/1048576).toFixed(1)+' MB'
                        :n>1024?(n/1024).toFixed(0)+' KB':n+' B';}

/* ---- sending ---- */
text.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault(); doSend();}});
send.onclick=doSend;

async function doSend(){
  const t=text.value.trim();
  if(!t&&!attached) return;
  send.disabled=true;
  const note=document.createElement('div');
  note.className='sending';
  note.textContent=attached?'scanning '+attached.name+' …':'sending…';
  chat.appendChild(note); scroll();
  const fd=new FormData();
  fd.append('text',t);
  if(attached) fd.append('files',attached,attached.name);
  try{
    const r=await fetch('/chat/send',{method:'POST',body:fd});
    if(!r.ok){const j=await r.json().catch(()=>({detail:r.statusText}));
              throw new Error(j.detail||('HTTP '+r.status));}
    const j=await r.json();
    j.messages.forEach(render);
    text.value=''; attached=null; chipWrap.style.display='none'; fileIn.value='';
  }catch(e){showErr(e.message);}
  finally{note.remove(); send.disabled=false; scroll(); text.focus();}
}
function showErr(m){err.textContent='⚠ '+m; err.style.display='block';
  clearTimeout(showErr.t); showErr.t=setTimeout(()=>err.style.display='none',5000);}
function scroll(){chat.scrollTop=chat.scrollHeight;}
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML;};

/* ---- rendering ---- */
function pill(m){
  if(!m.flagged) return '';
  const cats=(m.categories||[]).join(', ');
  const conf=m.confidence?(' · '+Math.round(m.confidence*100)+'%'):'';
  if(!m.scanned) return '<span class="pill warn">⚠ not scanned — flagged for review</span>';
  const what={blur:'blurred',mask:'filtered',block:'removed',mute:'muted',
              flag:'flagged'}[m.action]||'flagged';
  return '<span class="pill flag">⚠ '+esc(what)+(cats?' · '+esc(cats):'')+conf+'</span>';
}
function reasons(m){
  return (m.flagged&&m.reasons&&m.reasons.length)
    ? '<div class="reasons">'+esc(m.reasons.slice(0,3).join(' · '))+'</div>':'';
}
function caption(m){
  if(!('caption' in m)||m.caption===undefined||m.caption===null||m.caption==='')return '';
  return '<div class="body caption">'+esc(m.caption)
    +(m.caption_flagged?' <span class="pill flag" style="margin:0">filtered</span>':'')+'</div>';
}
function render(m){
  const div=document.createElement('div'); div.className='msg';
  let inner='';
  if(m.kind==='text'){
    inner=pill(m)+'<div class="body">'+esc(m.text)+'</div>'+reasons(m);
  }else if(m.kind==='image'){
    inner=pill(m)+'<img src="'+m.media+'" alt="image">'+caption(m)+reasons(m);
  }else if(m.kind==='video'){
    inner=pill(m)+(m.media
        ? '<video controls preload="metadata" src="'+m.media+'"></video>'
        : blockedCard('Video removed',m))+caption(m)+reasons(m);
  }else if(m.kind==='audio'){
    inner=pill(m)+(m.media
        ? '<audio controls src="'+m.media+'"></audio>'
        : blockedCard('Voice message removed — explicit language',m))
        +caption(m)+reasons(m);
  }else{ // document
    inner=pill(m)+docCard(m)+caption(m)+reasons(m);
  }
  div.innerHTML='<div class="bubble">'+inner
    +'<div class="meta">'+esc(m.ts||'')+' <span class="tick">✓✓</span></div></div>';
  chat.appendChild(div); scroll();
}
function blockedCard(title,m){
  return '<div class="blocked"><span class="ico">🚫</span><div><b>'+esc(title)+'</b>'
    +(m.transcript?'<div class="tr">“'+esc(m.transcript)+'”</div>':'')+'</div></div>';
}
function docCard(m){
  const dl=m.media?('<a href="'+m.media+'" download="'+esc(m.filename||'document.txt')
      +'">'+esc(m.filename||'document')+'</a>'):'<b>'+esc(m.filename||'document')+'</b>';
  return '<div class="doc"><span class="ico">📄</span><div>'+dl
    +'<div class="sub">'+(m.flagged?'censored copy':'document')+'</div></div></div>'
    +(m.text?'<div class="excerpt">'+esc(m.text)+'</div>':'');
}

/* ---- restore history + health ---- */
(async()=>{
  try{
    const r=await fetch('/chat/messages'); const j=await r.json();
    (j.messages||[]).forEach(render);
  }catch(_){}
  try{
    const h=await fetch('/chat/health').then(r=>r.json());
    const det=(h.image&&Array.isArray(h.image.detectors))
        ? h.image.detectors.join(', '):'';
    if(det) document.getElementById('status').textContent=
        'moderators ready · image: '+det;
  }catch(_){}
})();
</script>
</body>
</html>
"""
