"""Command-line interface for the explicit-content filter.

Examples
--------
Scan a folder and print a JSON verdict per image (no files written)::

    python -m safety scan ./photos --json

Blur every flagged image in a folder into ./clean, keeping a report::

    python -m safety blur ./photos -o ./clean --report report.json

Draw detection boxes (debug, no blurring) for one image::

    python -m safety debug photo.jpg -o boxed.jpg

Run ``python -m safety --help`` for all options.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

from .config import SafetyConfig
from .types import Severity

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _iter_images(path: str):
    if os.path.isfile(path):
        yield path
        return
    for ext in _IMG_EXTS:
        yield from glob(os.path.join(path, "**", f"*{ext}"), recursive=True)
        yield from glob(os.path.join(path, "**", f"*{ext.upper()}"), recursive=True)


def _config_from_args(args) -> SafetyConfig:
    cfg = SafetyConfig.from_env()
    if args.style:
        cfg.blur_style = args.style
    if args.min_severity:
        cfg.min_blur_severity = Severity[args.min_severity.upper()]
    if args.no_nudenet:
        cfg.use_nudenet = False
    if args.no_wound:
        cfg.use_wound_heuristic = False
    if args.classifier:
        cfg.use_classifier = True
        cfg.classifier_path = args.classifier
    return cfg


def _build_filter(cfg: SafetyConfig):
    from .pipeline import ExplicitContentFilter

    return ExplicitContentFilter(cfg)


def cmd_scan(args) -> int:
    from . import load_image

    filt = _build_filter(_config_from_args(args))
    print(f"# detectors: {', '.join(filt.active_detectors) or 'none'}",
          file=sys.stderr)
    flagged = 0
    results = []
    for p in _iter_images(args.path):
        try:
            verdict = filt.evaluate(load_image(p))
        except Exception as exc:
            print(f"!! {p}: {exc}", file=sys.stderr)
            continue
        flagged += int(verdict.explicit)
        entry = {"path": p, **verdict.to_dict()}
        results.append(entry)
        if args.json:
            print(json.dumps(entry))
        else:
            tag = "EXPLICIT" if verdict.explicit else "ok"
            cats = ",".join(sorted(c.value for c in verdict.categories())) or "-"
            print(f"[{tag:8}] {cats:20} {p}")
    print(f"# {flagged}/{len(results)} flagged", file=sys.stderr)
    if args.report:
        json.dump(results, open(args.report, "w"), indent=2)
    return 0


def cmd_blur(args) -> int:
    from . import load_image, save_image

    filt = _build_filter(_config_from_args(args))
    os.makedirs(args.output, exist_ok=True)
    report = []
    n = 0
    for p in _iter_images(args.path):
        try:
            img = load_image(p)
        except Exception as exc:
            print(f"!! {p}: {exc}", file=sys.stderr)
            continue
        clean, verdict = filt.scan_and_redact(img)
        out_path = os.path.join(args.output, os.path.basename(p))
        if verdict.explicit or args.copy_clean:
            save_image(out_path, clean)
        report.append({"path": p, "output": out_path, **verdict.to_dict()})
        if verdict.explicit:
            n += 1
            print(f"[BLURRED] {p} -> {out_path} "
                  f"({len(verdict.regions)} region(s))")
    print(f"# blurred {n} image(s)", file=sys.stderr)
    if args.report:
        json.dump(report, open(args.report, "w"), indent=2)
    return 0


def cmd_ui(args) -> int:
    try:
        from .ui import serve
    except Exception as exc:  # pragma: no cover - missing serving deps
        print(f"!! the UI needs FastAPI + uvicorn: pip install fastapi "
              f"uvicorn python-multipart  ({exc})", file=sys.stderr)
        return 1
    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          config=_config_from_args(args))
    return 0


def cmd_chat(args) -> int:
    try:
        from .chat import serve
    except Exception as exc:  # pragma: no cover - missing serving deps
        print(f"!! the chat UI needs FastAPI + uvicorn: pip install fastapi "
              f"uvicorn python-multipart  ({exc})", file=sys.stderr)
        return 1
    from .multimodal import MultimodalConfig

    cfg = MultimodalConfig.from_env()
    cfg.safety = _config_from_args(args)
    if args.no_hf:
        cfg.use_hf_text = cfg.use_hf_image = cfg.use_hf_audio = False
    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          config=cfg)
    return 0


def cmd_debug(args) -> int:
    from . import load_image, save_image, draw_boxes

    filt = _build_filter(_config_from_args(args))
    img = load_image(args.path)
    regions = filt.detect(img)
    boxed = draw_boxes(img, regions)
    out = args.output or "boxed.jpg"
    save_image(out, boxed)
    print(json.dumps({"path": args.path, "detections": [r.to_dict() for r in regions]},
                     indent=2))
    print(f"# wrote {out} with {len(regions)} box(es)", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m safety",
        description="Detect and redact explicit content — images (blur), video, "
                    "audio, and text (chat UI).",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--style", choices=["gaussian", "pixelate", "box", "fill"],
                        help="redaction style (default: gaussian)")
    common.add_argument("--min-severity", choices=["low", "medium", "high"],
                        help="minimum severity that triggers a blur")
    common.add_argument("--classifier", metavar="PATH",
                        help="path to a fine-tuned .onnx/.pt classifier")
    common.add_argument("--no-nudenet", action="store_true",
                        help="disable the NudeNet detector")
    common.add_argument("--no-wound", action="store_true",
                        help="disable the wound/blood heuristic")

    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", parents=[common], help="report verdicts only")
    s.add_argument("path", help="image file or directory")
    s.add_argument("--json", action="store_true", help="one JSON object per line")
    s.add_argument("--report", metavar="FILE", help="write a JSON report")
    s.set_defaults(func=cmd_scan)

    b = sub.add_parser("blur", parents=[common], help="write redacted copies")
    b.add_argument("path", help="image file or directory")
    b.add_argument("-o", "--output", default="clean", help="output directory")
    b.add_argument("--copy-clean", action="store_true",
                   help="also copy non-flagged images to the output dir")
    b.add_argument("--report", metavar="FILE", help="write a JSON report")
    b.set_defaults(func=cmd_blur)

    d = sub.add_parser("debug", parents=[common], help="draw detection boxes")
    d.add_argument("path", help="image file")
    d.add_argument("-o", "--output", help="output image (default boxed.jpg)")
    d.set_defaults(func=cmd_debug)

    u = sub.add_parser("ui", parents=[common],
                       help="launch the browser upload UI")
    u.add_argument("--host", default="127.0.0.1", help="bind host")
    u.add_argument("--port", type=int, default=8000, help="bind port")
    u.add_argument("--no-browser", action="store_true",
                   help="don't auto-open a browser tab")
    u.set_defaults(func=cmd_ui)

    c = sub.add_parser("chat", parents=[common],
                       help="launch the WhatsApp-style moderated chat UI "
                            "(image, video, audio, text)")
    c.add_argument("--host", default="127.0.0.1", help="bind host")
    c.add_argument("--port", type=int, default=8000, help="bind port")
    c.add_argument("--no-browser", action="store_true",
                   help="don't auto-open a browser tab")
    c.add_argument("--no-hf", action="store_true",
                   help="disable the Hugging Face backends (offline fallbacks only)")
    c.set_defaults(func=cmd_chat)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
