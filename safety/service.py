"""FastAPI router exposing the explicit-content filter over HTTP.

Mount it on any FastAPI app::

    from fastapi import FastAPI
    from safety.service import router as safety_router
    app = FastAPI()
    app.include_router(safety_router)

Endpoints
---------
``POST /safety/scan``   multipart image -> JSON verdict (no image returned)
``POST /safety/redact`` multipart image -> the blurred image (image/jpeg)
``GET  /safety/health`` which detectors are active

The filter is built once at import and reused (the NudeNet ONNX session is
loaded lazily on first request).
"""
from __future__ import annotations

import io

from .config import SafetyConfig
from .pipeline import ExplicitContentFilter

try:
    from fastapi import APIRouter, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse, Response
except Exception:  # pragma: no cover - FastAPI is optional for this module
    APIRouter = None  # type: ignore


def build_router(config: SafetyConfig | None = None):
    if APIRouter is None:  # pragma: no cover
        raise RuntimeError("fastapi is not installed")

    router = APIRouter(prefix="/safety", tags=["safety"])
    _filter = ExplicitContentFilter(config or SafetyConfig.from_env())

    def _decode(raw: bytes):
        import cv2
        import numpy as np

        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="invalid image")
        return img

    @router.get("/health")
    def health():
        return {"status": "ok", "detectors": _filter.active_detectors}

    @router.post("/scan")
    async def scan(file: UploadFile = File(...)):
        img = _decode(await file.read())
        verdict = _filter.evaluate(img)
        return JSONResponse(verdict.to_dict())

    @router.post("/redact")
    async def redact_endpoint(file: UploadFile = File(...)):
        import cv2

        img = _decode(await file.read())
        clean, verdict = _filter.scan_and_redact(img)
        ok, buf = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        headers = {
            "X-Explicit": str(verdict.explicit).lower(),
            "X-Categories": ",".join(sorted(c.value for c in verdict.categories())),
            "X-Regions": str(len(verdict.regions)),
        }
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)

    return router


# A ready-to-mount router with env-driven config. It is ``None`` when FastAPI
# (or its multipart extra, needed for file uploads) isn't installed, so importing
# this module never fails — callers should null-check or call build_router()
# themselves. Install the serving deps with: pip install fastapi python-multipart
try:
    router = build_router() if APIRouter is not None else None
except Exception:  # pragma: no cover - missing optional serving deps
    router = None
