"""Export a trained checkpoint to ONNX (or TorchScript) for inference.

ONNX lets the runtime use ``onnxruntime`` alone — no PyTorch needed in
production, keeping the serving image small. The exported model takes a
``[1,3,H,W]`` NCHW float tensor (ImageNet-normalized, see
``ClassifierDetector._preprocess``) and outputs raw logits; the detector applies
the sigmoid.

Usage::

    python -m safety.train.export runs/v1/best.pt --out runs/v1/model.onnx

A ``labels.json`` is written next to the output so ``ClassifierDetector`` can map
output indices to class names.
"""
from __future__ import annotations

import argparse
import json
import os


def export(args) -> int:
    import torch

    from .model import build_model

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "mobilen_v3_large")
    input_size = ckpt.get("input_size", 224)

    net = build_model(len(classes), arch, pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    out = args.out or os.path.splitext(args.checkpoint)[0] + ".onnx"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size)

    if out.endswith(".onnx"):
        # Force the stable TorchScript-based exporter (dynamo=False) so the
        # export doesn't require the optional `onnxscript` package that newer
        # torch versions pull in for their dynamo path.
        export_kwargs = dict(
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
        )
        try:
            torch.onnx.export(net, dummy, out, dynamo=False, **export_kwargs)
        except TypeError:
            # Older torch has no `dynamo` kwarg; its default path is the legacy one.
            torch.onnx.export(net, dummy, out, **export_kwargs)
    else:  # TorchScript
        torch.jit.save(torch.jit.trace(net, dummy), out)

    # Sidecar label map for ClassifierDetector.
    json.dump(classes, open(os.path.join(os.path.dirname(os.path.abspath(out)),
                                          "labels.json"), "w"))
    print(f"exported {arch} ({len(classes)} classes) -> {out}")
    print(f"classes: {classes}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX/TorchScript")
    p.add_argument("checkpoint", help="path to best.pt / last.pt")
    p.add_argument("--out", help="output .onnx or .pt (default: alongside ckpt)")
    p.add_argument("--opset", type=int, default=17)
    return p


def main(argv=None):
    return export(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
