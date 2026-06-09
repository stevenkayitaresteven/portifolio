"""Train / fine-tune the explicit-content classifier.

Usage::

    python -m safety.train.train --data ./data --arch mobilen_v3_large \
        --epochs 8 --batch-size 32 --out runs/v1

Outputs into ``--out``:
    best.pt        best checkpoint (state_dict + meta) by val F1
    last.pt        final-epoch checkpoint
    labels.json    class order (also copied next to any exported ONNX)
    metrics.json   per-epoch train/val loss + macro-F1

The loss is ``BCEWithLogitsLoss`` (multi-label). Class imbalance is handled with
positive weights derived from the train split counts. Designed to run on CPU
(slow but functional) or GPU automatically.
"""
from __future__ import annotations

import argparse
import json
import os


def evaluate(net, loader, device, criterion, num_classes):
    """Return (mean_loss, macro_f1, per_class_f1)."""
    import torch

    net.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            for c in range(num_classes):
                tp[c] += int(((preds[:, c] == 1) & (y[:, c] == 1)).sum())
                fp[c] += int(((preds[:, c] == 1) & (y[:, c] == 0)).sum())
                fn[c] += int(((preds[:, c] == 0) & (y[:, c] == 1)).sum())
    f1s = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append((2 * tp[c] / denom) if denom else 0.0)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return (total_loss / max(n, 1)), macro, f1s


def train(args) -> int:
    import torch
    from torch import nn, optim

    from .dataset import discover_classes, make_loaders
    from .model import build_model, freeze_backbone, unfreeze

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.out, exist_ok=True)

    classes = discover_classes(args.data)
    print(f"classes: {classes}")
    train_ds, val_ds, train_loader, val_loader = make_loaders(
        args.data, classes, args.input_size, args.batch_size, args.num_workers)
    print(f"train counts: {train_ds.class_counts()}")
    if val_ds:
        print(f"val   counts: {val_ds.class_counts()}")

    # Positive weights for class imbalance: neg/pos per class on the train split.
    counts = train_ds.class_counts()
    n_total = max(len(train_ds), 1)
    pos_weight = torch.tensor(
        [max(1.0, (n_total - counts[c]) / max(counts[c], 1)) for c in classes],
        dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    net = build_model(len(classes), args.arch, pretrained=not args.no_pretrained)
    net.to(device)
    if args.freeze_epochs > 0:
        freeze_backbone(net, args.arch)

    optimizer = optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=1e-4)

    json.dump(classes, open(os.path.join(args.out, "labels.json"), "w"))
    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs and epoch == args.freeze_epochs + 1:
            print("unfreezing backbone")
            unfreeze(net)
            optimizer = optim.AdamW(net.parameters(), lr=args.lr * 0.2,
                                    weight_decay=1e-4)

        net.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / max(seen, 1)

        if val_loader:
            val_loss, val_f1, per_f1 = evaluate(
                net, val_loader, device, criterion, len(classes))
        else:
            # No val split: fall back to tracking the lowest train loss so a
            # "best.pt" is still produced.
            val_loss, val_f1, per_f1 = train_loss, -train_loss, []
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_macroF1={val_f1:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_macro_f1": val_f1,
                        "per_class_f1": dict(zip(classes, per_f1))})

        ckpt = {"state_dict": net.state_dict(), "classes": classes,
                "arch": args.arch, "input_size": args.input_size}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(ckpt, os.path.join(args.out, "best.pt"))

    json.dump(history, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"done. best val macro-F1={best_f1:.4f}  ->  {args.out}/best.pt")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Fine-tune the explicit-content classifier")
    p.add_argument("--data", required=True, help="dataset root (train/ [val/])")
    p.add_argument("--out", default="runs/v1", help="output dir for checkpoints")
    p.add_argument("--arch", default="mobilen_v3_large",
                   choices=["mobilen_v3_large", "efficientnet_b0", "resnet18"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--freeze-epochs", type=int, default=2,
                   help="train only the head for this many epochs first")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="", help="cpu | cuda | '' (auto)")
    p.add_argument("--no-pretrained", action="store_true")
    return p


def main(argv=None):
    return train(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
