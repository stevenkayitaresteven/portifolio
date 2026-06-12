"""Fine-tune a multi-label text classifier on the 12-category safety taxonomy.

Trains any Hugging Face encoder (default: ``distilroberta-base``; pass
``KoalaAI/Text-Moderation`` to continue from the moderation checkpoint) as a
**multi-label** classifier whose labels are exactly the canonical categories in
:mod:`safety.multimodal.taxonomy`. The exported checkpoint therefore plugs
straight into the chat::

    SAFETY_MM_TEXT_MODELS=runs/teamsafety python -m safety chat

Data can come from a local CSV/JSONL or a Hub dataset. Presets ship for the
key public corpora (schemas verified):

* ``jigsaw``  — Jigsaw toxic comments (toxic/hate/threat multi-label)
* ``aegis2``  — nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (12+ violation
  categories incl. criminal planning, fraud, malware, suicide, PII)
* ``liar2``   — chengxuphd/liar2 fact-checked claims → misinformation

Examples::

    python -m safety.train.finetune_text --preset aegis2 --out runs/aegis
    python -m safety.train.finetune_text --data mod.csv --text-col text \
        --labels-col labels --out runs/custom --epochs 3
    python -m safety.train.finetune_text --preset jigsaw --data train.csv \
        --eval-only --model runs/aegis        # metrics only, no training

Needs ``pip install torch transformers datasets``(+ ``accelerate``). Metrics
(per-category precision/recall/F1, micro/macro) are implemented locally — no
sklearn dependency.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field

from ..multimodal import taxonomy as T

CATEGORIES = list(T.CATEGORIES)


# --- label mapping ------------------------------------------------------------------

def map_source_label(label: str, label_map: dict[str, str] | None = None) -> str | None:
    """Map a source-dataset label to a canonical category (None = drop)."""
    raw = label.strip().lower().replace(" ", "_")
    if label_map:
        if raw in label_map:
            return label_map[raw]
        for k, v in label_map.items():
            if k in raw:
                return v
    return T.category_for_label("", raw)


# Aegis 2.0 `violated_categories` values -> taxonomy (substring match, lowered).
AEGIS_MAP: dict[str, str] = {
    "hate": T.HATE, "identity": T.HATE,
    "sexual_(minor)": T.CHILD_SAFETY, "minor": T.CHILD_SAFETY,
    "sexual": T.SEXUAL,
    "suicide": T.SELF_HARM, "self_harm": T.SELF_HARM, "self-harm": T.SELF_HARM,
    "violence": T.VIOLENCE, "threat": T.VIOLENCE,
    "criminal": T.CRIMINAL, "fraud": T.CRIMINAL, "deception": T.CRIMINAL,
    "illegal": T.CRIMINAL, "weapons": T.CRIMINAL, "substances": T.CRIMINAL,
    "theft": T.CRIMINAL,
    "malware": T.CYBERSECURITY, "hacking": T.CYBERSECURITY,
    "profanity": T.TOXIC, "harassment": T.TOXIC, "bullying": T.TOXIC,
    "pii": T.PRIVACY, "privacy": T.PRIVACY,
    "terror": T.EXTREMISM, "extremis": T.EXTREMISM, "radical": T.EXTREMISM,
    "misinformation": T.MISINFORMATION, "disinformation": T.MISINFORMATION,
    "manipulation": T.CRIMINAL,
}


@dataclass
class Example:
    text: str
    labels: set[str] = field(default_factory=set)

    def multi_hot(self) -> list[float]:
        return [1.0 if c in self.labels else 0.0 for c in CATEGORIES]


def rows_to_examples(rows, *, text_col: str, labels_col: str | None = None,
                     label_cols: list[str] | None = None,
                     label_map: dict[str, str] | None = None,
                     row_labeler=None) -> list[Example]:
    """Convert raw rows (dicts) to taxonomy-labeled examples.

    Three labeling schemes, used in this order of precedence:
    * ``row_labeler(row) -> set[str]`` — arbitrary logic (presets use this)
    * ``labels_col`` — one column holding comma/semicolon-separated labels
    * ``label_cols`` — one 0/1 column per source label
    """
    out: list[Example] = []
    for row in rows:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        labels: set[str] = set()
        if row_labeler is not None:
            labels = {c for c in row_labeler(row) if c in CATEGORIES}
        elif labels_col is not None:
            raw = str(row.get(labels_col) or "")
            for part in raw.replace(";", ",").split(","):
                if part.strip():
                    cat = map_source_label(part, label_map)
                    if cat:
                        labels.add(cat)
        elif label_cols:
            for col in label_cols:
                try:
                    positive = float(row.get(col) or 0) >= 0.5
                except (TypeError, ValueError):
                    positive = str(row.get(col)).strip().lower() in {"true", "yes"}
                if positive:
                    cat = map_source_label(col, label_map)
                    if cat:
                        labels.add(cat)
        out.append(Example(text=text, labels=labels))
    return out


# --- presets (schemas verified on the Hub) ----------------------------------------

def _aegis_labeler(row) -> set[str]:
    if str(row.get("prompt_label", "")).strip().lower() != "unsafe":
        return set()
    labels = set()
    for part in str(row.get("violated_categories") or "").split(","):
        cat = map_source_label(part, AEGIS_MAP)
        if cat:
            labels.add(cat)
    return labels


def _liar2_labeler(row) -> set[str]:
    # LIAR2 labels: 0 pants-on-fire, 1 false, 2 barely-true, 3 half-true,
    # 4 mostly-true, 5 true. The clearly-false tiers train `misinformation`.
    try:
        return {T.MISINFORMATION} if int(row.get("label", 5)) <= 1 else set()
    except (TypeError, ValueError):
        return set()


PRESETS: dict[str, dict] = {
    "jigsaw": dict(
        hf_dataset=None,   # bring the Kaggle CSV: --data train.csv
        text_col="comment_text",
        label_cols=["toxic", "severe_toxic", "obscene", "threat",
                    "insult", "identity_hate"],
    ),
    "aegis2": dict(
        hf_dataset="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        text_col="prompt",
        row_labeler=_aegis_labeler,
    ),
    "liar2": dict(
        hf_dataset="chengxuphd/liar2",
        text_col="statement",
        row_labeler=_liar2_labeler,
    ),
}


# --- data loading -------------------------------------------------------------------

def load_local_rows(path: str) -> list[dict]:
    if path.endswith((".jsonl", ".ndjson")):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_hub_rows(dataset: str, split: str) -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset(dataset, split=split))


# --- metrics (no sklearn) ------------------------------------------------------------

def multilabel_metrics(y_true: list[list[float]], y_pred: list[list[float]]) -> dict:
    """Per-category precision/recall/F1 + micro/macro aggregates."""
    per: dict[str, dict] = {}
    tp_all = fp_all = fn_all = 0
    f1s = []
    for i, cat in enumerate(CATEGORIES):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t[i] >= 0.5 and p[i] >= 0.5)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t[i] < 0.5 and p[i] >= 0.5)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t[i] >= 0.5 and p[i] < 0.5)
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[cat] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": support,
                    "confusion": {"tp": tp, "fp": fp, "fn": fn}}
        if support:
            f1s.append(f1)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r else 0.0)
    return {"per_category": per,
            "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4),
                      "f1": round(micro_f1, 4)},
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0}


# --- training -----------------------------------------------------------------------

def train(examples: list[Example], *, base_model: str, out_dir: str,
          epochs: float, batch_size: int, lr: float, max_length: int,
          eval_fraction: float = 0.1):
    import numpy as np
    import torch  # noqa: F401  (fail fast with a clear message)
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(CATEGORIES)),
        label2id={c: i for i, c in enumerate(CATEGORIES)},
        ignore_mismatched_sizes=True,  # re-headed when continuing a checkpoint
    )

    class DS(torch.utils.data.Dataset):
        def __init__(self, ex):
            self.enc = tok([e.text for e in ex], truncation=True,
                           max_length=max_length, padding=True)
            self.labels = [e.multi_hot() for e in ex]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.float)
            return item

    n_eval = max(1, int(len(examples) * eval_fraction))
    train_ds, eval_ds = DS(examples[n_eval:]), DS(examples[:n_eval])

    def compute_metrics(pred):
        probs = 1 / (1 + np.exp(-pred.predictions))
        m = multilabel_metrics(pred.label_ids.tolist(),
                               (probs >= 0.5).astype(float).tolist())
        return {"micro_f1": m["micro"]["f1"], "macro_f1": m["macro_f1"]}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir, num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, learning_rate=lr,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="micro_f1",
            logging_steps=50, report_to=[],
        ),
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    metrics = trainer.evaluate()
    json.dump(metrics, open(os.path.join(out_dir, "eval.json"), "w"), indent=2)
    print(f"\nsaved -> {out_dir}\n  use it: SAFETY_MM_TEXT_MODELS={out_dir} "
          f"python -m safety chat")
    return metrics


def evaluate_model(examples: list[Example], model_path: str,
                   max_length: int) -> dict:
    """Score an existing checkpoint against taxonomy-labeled examples."""
    from transformers import pipeline

    pipe = pipeline("text-classification", model=model_path, top_k=None,
                    truncation=True, max_length=max_length)
    y_true, y_pred = [], []
    for ex in examples:
        y_true.append(ex.multi_hot())
        rows = pipe(ex.text[:2000])
        if rows and isinstance(rows[0], list):
            rows = rows[0]
        pred = [0.0] * len(CATEGORIES)
        for r in rows:
            cat = T.category_for_label(model_path, str(r["label"]))
            if cat in CATEGORIES and float(r["score"]) >= 0.5:
                pred[CATEGORIES.index(cat)] = 1.0
        y_pred.append(pred)
    return multilabel_metrics(y_true, y_pred)


# --- CLI ----------------------------------------------------------------------------

def build_examples(args) -> list[Example]:
    preset = PRESETS.get(args.preset or "", {})
    text_col = args.text_col or preset.get("text_col") or "text"
    labels_col = args.labels_col or preset.get("labels_col")
    label_cols = (args.label_cols.split(",") if args.label_cols
                  else preset.get("label_cols"))
    label_map = None
    if args.label_map:
        label_map = dict(p.split("=", 1) for p in args.label_map.split(",") if "=" in p)

    if args.data:
        rows = load_local_rows(args.data)
    elif preset.get("hf_dataset"):
        rows = load_hub_rows(preset["hf_dataset"], args.split)
    else:
        sys.exit("need --data FILE (or a preset with a Hub dataset)")
    return rows_to_examples(rows, text_col=text_col, labels_col=labels_col,
                            label_cols=label_cols, label_map=label_map,
                            row_labeler=preset.get("row_labeler"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m safety.train.finetune_text",
        description="Fine-tune a multi-label safety classifier on the "
                    "12-category taxonomy.")
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="dataset preset (column mapping included)")
    p.add_argument("--data", help="local CSV/JSONL file")
    p.add_argument("--split", default="train", help="Hub dataset split")
    p.add_argument("--text-col", help="text column name")
    p.add_argument("--labels-col", help="column with comma-separated labels")
    p.add_argument("--label-cols", help="comma-separated 0/1 label columns")
    p.add_argument("--label-map", help="source=category overrides, comma-separated")
    p.add_argument("--model", default="distilroberta-base",
                   help="base checkpoint (or the model to --eval-only)")
    p.add_argument("--out", default="runs/text", help="output directory")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--eval-only", action="store_true",
                   help="evaluate --model on the data; no training")
    args = p.parse_args(argv)

    examples = build_examples(args)
    n_pos = sum(1 for e in examples if e.labels)
    print(f"# {len(examples)} examples ({n_pos} with >=1 category)", file=sys.stderr)

    if args.eval_only:
        print(json.dumps(evaluate_model(examples, args.model, args.max_length),
                         indent=2))
        return 0
    train(examples, base_model=args.model, out_dir=args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, max_length=args.max_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
