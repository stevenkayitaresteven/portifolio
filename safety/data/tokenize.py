"""Tokenise the JSONL splits with a Hugging Face tokenizer.

Kept separate from the pipeline so you can re-tokenise the same cleaned dataset
for different base models (a RoBERTa head vs. a Llama LoRA) without re-running
the expensive clean/dedup stages.
"""
from __future__ import annotations

import json
import os

from .label import CATEGORIES


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize_split(jsonl_path: str, tokenizer, max_length: int = 256) -> dict:
    """Tokenise one split into ``{input_ids, attention_mask, labels}`` lists."""
    texts, labels = [], []
    for row in _read_jsonl(jsonl_path):
        texts.append(row["text"])
        labels.append(row.get("labels") or [0] * len(CATEGORIES))
    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    enc = tokenizer(texts, truncation=True, max_length=max_length,
                    padding="max_length")
    return {"input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels}


def tokenize_dataset(data_dir: str, tokenizer_name: str = "distilroberta-base",
                     max_length: int = 256, out_dir: str | None = None) -> dict:
    """Tokenise every split found in ``data_dir`` (train/val/test.jsonl)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("pip install transformers to tokenise") from exc
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out_dir = out_dir or os.path.join(data_dir, "tokenized")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for name in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        enc = tokenize_split(path, tok, max_length)
        with open(os.path.join(out_dir, f"{name}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(enc, fh)
        summary[name] = len(enc["input_ids"])
    return summary
