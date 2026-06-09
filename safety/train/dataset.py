"""Dataset + transforms for the explicit-content classifier.

Layout expected on disk (an ``ImageFolder`` per split)::

    data/
      train/
        safe/    *.jpg
        nudity/  *.jpg
        gore/    *.jpg
      val/
        safe/ ... nudity/ ... gore/ ...

Each top-level folder name under a split is one class. We train a **multi-label**
head (sigmoid per class) so an image can be both ``nudity`` and ``gore``; a plain
single-folder-per-image dataset yields one-hot targets, which is the common case.

NOTE: we deliberately ship **no** explicit imagery. You bring your own labeled
data (e.g. from an internal moderation set or a licensed dataset). Class folders
that are empty/absent are simply skipped.
"""
from __future__ import annotations

import os
from typing import Callable

# Torch is only needed for training, imported lazily by callers.
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def discover_classes(root: str) -> list[str]:
    """Class names = sorted sub-directories of the train split."""
    train_dir = os.path.join(root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"missing train split: {train_dir}")
    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    if not classes:
        raise ValueError(f"no class sub-folders under {train_dir}")
    return classes


def build_transforms(input_size: int = 224, train: bool = True):
    """ImageNet-normalized transforms; light augmentation for the train split."""
    from torchvision import transforms

    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        norm,
    ])


class MultiLabelImageFolder:
    """A thin ``torch.utils.data.Dataset`` over class sub-folders.

    Returns ``(image_tensor, target_vector)`` where the target is a float vector
    of length ``len(classes)`` (one-hot for single-folder membership).
    """

    def __init__(self, split_dir: str, classes: list[str], transform: Callable):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for c in classes:
            d = os.path.join(split_dir, c)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith(_IMG_EXTS):
                    self.samples.append((os.path.join(d, fn), self.class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        path, idx = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.zeros(len(self.classes), dtype=torch.float32)
        y[idx] = 1.0
        return x, y

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for _, idx in self.samples:
            counts[self.classes[idx]] += 1
        return counts


def make_loaders(root: str, classes: list[str], input_size: int,
                 batch_size: int, num_workers: int = 2):
    from torch.utils.data import DataLoader

    train_ds = MultiLabelImageFolder(
        os.path.join(root, "train"), classes,
        build_transforms(input_size, train=True))
    val_dir = os.path.join(root, "val")
    val_ds = MultiLabelImageFolder(
        val_dir, classes, build_transforms(input_size, train=False)
    ) if os.path.isdir(val_dir) else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers) if val_ds else None
    return train_ds, val_ds, train_loader, val_loader
