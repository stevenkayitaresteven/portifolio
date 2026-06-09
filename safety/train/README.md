# Fine-tuning a custom explicit-content classifier

The runtime works out of the box with the **NudeNet** box detector (nudity) and
the **wound heuristic** (gore) — no training required. Train a classifier here
when you want:

- a **learned gore/wound** model (far better precision than the red-pixel
  heuristic), and/or
- a **whole-image NSFW** score tuned to *your* data and failure cases.

> We ship **no** explicit imagery. You provide your own labeled dataset — an
> internal moderation set, a licensed corpus, etc. Handle it lawfully and store
> it securely.

## 1. Organize the data

```
data/
  train/
    safe/    img0001.jpg ...
    nudity/  ...
    gore/    ...
  val/
    safe/ ...  nudity/ ...  gore/ ...
```

Folder names become class names (sorted). Use whatever set you need — e.g.
`safe / nudity / suggestive / gore`. `ClassifierDetector` maps common names onto
the runtime taxonomy (`safe→ignored`, `nudity/nsfw→NUDITY`, `gore/wound→GORE`,
`suggestive/sexy→SUGGESTIVE`); unknown names default to NUDITY (fail-safe).

A 70/15/15 train/val/test split is a reasonable default. Aim for a few thousand
images per class to start; the head-then-fine-tune schedule below works with
less.

## 2. Train

```bash
pip install torch torchvision pillow onnx   # training/export-only deps
python -m safety.train.train \
    --data ./data --arch mobilen_v3_large \
    --epochs 8 --freeze-epochs 2 --batch-size 32 --out runs/v1
```

- **`--freeze-epochs 2`** trains only the new head first (fast, stable on small
  data), then unfreezes the backbone at a lower LR.
- Class imbalance is handled automatically via `BCEWithLogitsLoss` positive
  weights derived from the train counts.
- Loss is multi-label (sigmoid per class), so an image can be both `nudity` and
  `gore`.
- Best checkpoint is selected by **validation macro-F1**.

Outputs in `runs/v1/`: `best.pt`, `last.pt`, `labels.json`, `metrics.json`.

CPU works (slow); a GPU is auto-detected. `mobilen_v3_large` is the default for a
small, fast CPU model; `efficientnet_b0` / `resnet18` are alternatives.

## 3. Export to ONNX

```bash
pip install onnx                  # serialization dep, export-time only
python -m safety.train.export runs/v1/best.pt --out runs/v1/model.onnx
```

ONNX means inference needs only `onnxruntime` (no PyTorch in production). A
`labels.json` is written next to the model.

## 4. Use it

```bash
# CLI
python -m safety blur ./photos --classifier runs/v1/model.onnx -o ./clean

# Python
from safety import SafetyConfig, ExplicitContentFilter
cfg = SafetyConfig(use_classifier=True, classifier_path="runs/v1/model.onnx")
filt = ExplicitContentFilter(cfg)
```

The classifier runs alongside the box detectors. When a whole-image class
(nudity/gore) scores above `whole_image_threshold` (default 0.80), the **entire
image** is blurred; box detectors still localize specific regions.

## Evaluation tips

- Track **recall on the explicit classes** as the primary metric — for a safety
  filter, a missed explicit image is worse than an over-blur.
- Inspect false negatives/positives with `python -m safety debug img.jpg` to see
  what the detectors localized.
- Calibrate thresholds per category in `SafetyConfig` rather than retraining for
  small precision/recall trade-offs.
