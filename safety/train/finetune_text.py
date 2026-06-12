"""Fine-tune a multi-label text classifier on the 12-category safety taxonomy.

Trains any Hugging Face encoder (default: ``distilroberta-base``; pass
``KoalaAI/Text-Moderation`` to continue from the moderation checkpoint) as a
**multi-label** classifier whose labels are exactly the canonical categories in
:mod:`safety.multimodal.taxonomy`. The exported checkpoint therefore plugs
straight into the chat::

    SAFETY_MM_TEXT_MODELS=runs/teamsafety python -m safety chat

Data can come from a local CSV/TSV/JSONL or a Hub dataset, and several presets
can be **mixed into one corpus** (``--mix``). Presets ship for the key public
corpora (schemas verified on the Hub):

================= ============================================== ====================
 Preset             Source                                          Trains
================= ============================================== ====================
 ``aegis2``         nvidia/Aegis-AI-Content-Safety-Dataset-2.0     12+ categories
 ``jigsaw``         Jigsaw toxic comments (Kaggle CSV)             toxic/hate/violence
 ``jigsaw_bias``    Jigsaw unintended-bias 1.8M (Kaggle CSV)       toxic/hate/sexual
 ``civil_comments`` google/civil_comments (~2M)                    toxic/hate/violence/sexual
 ``hatexplain``     Hate-speech-CNERG/hatexplain (script dataset)  hate/toxic
 ``davidson``       tdavidson/hate_speech_offensive (~25K tweets)  hate/toxic
 ``toxigen``        toxigen/toxigen-data ``annotated`` (gated)     implicit hate
 ``textdetox``      textdetox/multilingual_toxicity_dataset        toxic, 14 languages
 ``olid``           OLID/SOLID OffensEval TSV (local file)         toxic
 ``cyberbullying``  Kaggle cyberbullying_tweets.csv (local file)   hate/toxic
 ``goemotions``     google-research-datasets/go_emotions           benign negatives
 ``liar2``          chengxuphd/liar2 fact-checked claims           misinformation
================= ============================================== ====================

``goemotions`` contributes *clean counterexamples* (hostile-emotion rows are
dropped, not labeled) to reduce false positives; ``--max-clean-ratio`` keeps
clean-heavy corpora from drowning the signal.

Examples::

    # one corpus from four Hub datasets, capped and balanced
    python -m safety.train.finetune_text \
        --mix civil_comments,davidson,toxigen,goemotions \
        --max-per-source 50000 --out runs/toxicity

    python -m safety.train.finetune_text --preset aegis2 --out runs/aegis
    python -m safety.train.finetune_text --preset textdetox --split es \
        --model microsoft/mdeberta-v3-base --out runs/es   # multilingual base
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
                     row_labeler=None, row_text=None) -> list[Example]:
    """Convert raw rows (dicts) to taxonomy-labeled examples.

    Three labeling schemes, used in this order of precedence:
    * ``row_labeler(row) -> set[str] | None`` — arbitrary logic (presets use
      this); returning ``None`` drops the row entirely
    * ``labels_col`` — one column holding comma/semicolon-separated labels
    * ``label_cols`` — one 0/1 column per source label

    ``row_text(row) -> str`` overrides ``text_col`` for datasets whose text
    needs assembly (e.g. HateXplain's token lists).
    """
    out: list[Example] = []
    for row in rows:
        text = (row_text(row) if row_text is not None
                else row.get(text_col) or "").strip()
        if not text:
            continue
        labels: set[str] = set()
        if row_labeler is not None:
            raw_labels = row_labeler(row)
            if raw_labels is None:      # labeler vetoes the row
                continue
            labels = {c for c in raw_labels if c in CATEGORIES}
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


def _civil_comments_labeler(row) -> set[str]:
    # Float annotator-fraction scores per column; >=0.5 = positive.
    def hot(col):
        try:
            return float(row.get(col) or 0) >= 0.5
        except (TypeError, ValueError):
            return False
    labels = set()
    if hot("toxicity") or hot("severe_toxicity") or hot("obscene") or hot("insult"):
        labels.add(T.TOXIC)
    if hot("identity_attack"):
        labels.add(T.HATE)
    if hot("threat"):
        labels.add(T.VIOLENCE)
    if hot("sexual_explicit"):
        labels.add(T.SEXUAL)
    return labels


def _davidson_labeler(row) -> set[str]:
    # class: 0 hate speech, 1 offensive language, 2 neither.
    try:
        cls = int(row.get("class", 2))
    except (TypeError, ValueError):
        return set()
    return {0: {T.HATE, T.TOXIC}, 1: {T.TOXIC}}.get(cls, set())


def _hatexplain_labeler(row) -> set[str]:
    # Majority vote across the three annotators: 0 hate, 1 normal, 2 offensive.
    votes = (row.get("annotators") or {}).get("label") or []
    if not votes:
        return set()
    majority = max(set(votes), key=votes.count)
    return {0: {T.HATE, T.TOXIC}, 2: {T.TOXIC}}.get(majority, set())


def _hatexplain_text(row) -> str:
    return " ".join(row.get("post_tokens") or [])


def _toxigen_labeler(row) -> set[str]:
    # `annotated` config: toxicity_human is a 1-5 mean; ToxiGen statements are
    # implicit hate aimed at the row's target_group, so toxic ones train HATE.
    try:
        score = float(row.get("toxicity_human") or 0)
    except (TypeError, ValueError):
        return set()
    return {T.HATE, T.TOXIC} if score >= 3.5 else set()


def _textdetox_labeler(row) -> set[str]:
    return {T.TOXIC} if str(row.get("toxic", "0")).strip() in {"1", "1.0", "true"} \
        else set()


def _olid_labeler(row) -> set[str]:
    # OLID/SOLID TSV, subtask A: OFF / NOT (SOLID ships an `average` score).
    sub_a = str(row.get("subtask_a") or "").strip().upper()
    if sub_a:
        return {T.TOXIC} if sub_a == "OFF" else set()
    try:
        return {T.TOXIC} if float(row.get("average") or 0) >= 0.5 else set()
    except (TypeError, ValueError):
        return set()


def _cyberbullying_labeler(row) -> set[str]:
    # Kaggle cyberbullying_tweets.csv: cyberbullying_type in {not_cyberbullying,
    # gender, religion, ethnicity, age, other_cyberbullying}.
    kind = str(row.get("cyberbullying_type") or "").strip().lower()
    if not kind or kind == "not_cyberbullying":
        return set()
    if kind in {"gender", "religion", "ethnicity"}:
        return {T.HATE, T.TOXIC}
    return {T.TOXIC}


# GoEmotions simplified label ids (0-27) that are safe to treat as benign.
# Hostile emotions (anger 2, annoyance 3, disapproval 10, disgust 11) are
# DROPPED, not labeled — emotion is not toxicity ground truth.
_GOEMOTIONS_HOSTILE = {2, 3, 10, 11}


def _goemotions_labeler(row) -> set[str] | None:
    labels = row.get("labels") or []
    if any(l in _GOEMOTIONS_HOSTILE for l in labels):
        return None        # drop: ambiguous for safety training
    return set()           # clean counterexample (reduces false positives)


PRESETS: dict[str, dict] = {
    # --- multi-category safety -------------------------------------------------
    "aegis2": dict(
        hf_dataset="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        text_col="prompt",
        row_labeler=_aegis_labeler,
    ),
    # --- toxicity / hate / offensive -------------------------------------------
    "jigsaw": dict(
        hf_dataset=None,   # bring the Kaggle CSV: --data train.csv
        text_col="comment_text",
        label_cols=["toxic", "severe_toxic", "obscene", "threat",
                    "insult", "identity_hate"],
    ),
    "jigsaw_bias": dict(
        hf_dataset=None,   # Kaggle unintended-bias CSV: --data train.csv
        text_col="comment_text",
        label_cols=["target", "severe_toxicity", "obscene", "threat",
                    "insult", "identity_attack", "sexual_explicit"],
        label_map={"target": T.TOXIC, "severe_toxicity": T.TOXIC,
                   "obscene": T.TOXIC, "insult": T.TOXIC,
                   "threat": T.VIOLENCE, "identity_attack": T.HATE,
                   "sexual_explicit": T.SEXUAL},
    ),
    "civil_comments": dict(
        hf_dataset="google/civil_comments",
        text_col="text",
        row_labeler=_civil_comments_labeler,
    ),
    "hatexplain": dict(
        hf_dataset="Hate-speech-CNERG/hatexplain",   # script dataset: needs
        text_col="post_tokens",                      # datasets<3 / trust_remote_code
        row_text=_hatexplain_text,
        row_labeler=_hatexplain_labeler,
    ),
    "davidson": dict(
        hf_dataset="tdavidson/hate_speech_offensive",
        text_col="tweet",
        row_labeler=_davidson_labeler,
    ),
    "toxigen": dict(
        hf_dataset="toxigen/toxigen-data",   # gated: accept the access form
        hf_config="annotated",
        text_col="text",
        row_labeler=_toxigen_labeler,
    ),
    "textdetox": dict(
        hf_dataset="textdetox/multilingual_toxicity_dataset",
        text_col="text",                     # pick a language: --split en|es|ar|zh…
        row_labeler=_textdetox_labeler,
        default_split="en",
    ),
    "olid": dict(
        hf_dataset=None,   # OLID/SOLID TSV from the OffensEval release: --data olid.tsv
        text_col="tweet",
        row_labeler=_olid_labeler,
    ),
    "cyberbullying": dict(
        hf_dataset=None,   # Kaggle cyberbullying_tweets.csv: --data file.csv
        text_col="tweet_text",
        row_labeler=_cyberbullying_labeler,
    ),
    # --- benign hard negatives (reduce false positives) -------------------------
    "goemotions": dict(
        hf_dataset="google-research-datasets/go_emotions",
        hf_config="simplified",
        text_col="text",
        row_labeler=_goemotions_labeler,
    ),
    # --- misinformation ----------------------------------------------------------
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
    delimiter = "\t" if path.endswith((".tsv", ".tab")) else ","
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def load_hub_rows(dataset: str, split: str, config: str | None = None,
                  max_rows: int = 0):
    """Load a Hub dataset split. Returns the arrow-backed ``Dataset`` (it
    iterates as dicts) rather than a Python list — civil_comments has 1.8M
    rows and materializing those as dicts costs gigabytes. ``max_rows``
    shuffles and selects *before* anything touches Python objects."""
    from datasets import load_dataset

    ds = (load_dataset(dataset, config, split=split) if config
          else load_dataset(dataset, split=split))
    if max_rows and len(ds) > max_rows:
        ds = ds.shuffle(seed=13).select(range(max_rows))
    return ds


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

    # Tokenize lazily per item and pad per *batch* (dynamic padding) — encoding
    # the whole corpus up front at a fixed length OOM-kills laptop-sized RAM
    # once the mixed corpus reaches ~100K examples.
    class DS(torch.utils.data.Dataset):
        def __init__(self, ex):
            self.texts = [e.text for e in ex]
            self.labels = [e.multi_hot() for e in ex]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            enc = dict(tok(self.texts[i], truncation=True, max_length=max_length))
            enc["labels"] = self.labels[i]
            return enc

    from transformers import DataCollatorWithPadding

    pad = DataCollatorWithPadding(tok)

    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch],
                              dtype=torch.float)
        out = pad(batch)
        out["labels"] = labels
        return out

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
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size, learning_rate=lr,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="micro_f1",
            logging_steps=50, report_to=[],
        ),
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate,
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

def balance_clean(examples: list[Example], max_clean_ratio: float,
                  seed: int = 13) -> list[Example]:
    """Cap unlabeled (clean) examples at ``ratio x labeled`` so corpora with a
    huge clean majority (civil_comments, GoEmotions) don't drown the signal."""
    import random

    if max_clean_ratio <= 0:
        return examples
    labeled = [e for e in examples if e.labels]
    clean = [e for e in examples if not e.labels]
    cap = int(len(labeled) * max_clean_ratio)
    if len(clean) <= cap:
        return examples
    rng = random.Random(seed)
    rng.shuffle(clean)
    mixed = labeled + clean[:cap]
    rng.shuffle(mixed)
    return mixed


def examples_from_source(preset_name: str | None, data_path: str | None,
                         args) -> list[Example]:
    preset = PRESETS.get(preset_name or "", {})
    text_col = args.text_col or preset.get("text_col") or "text"
    labels_col = args.labels_col or preset.get("labels_col")
    label_cols = (args.label_cols.split(",") if args.label_cols
                  else preset.get("label_cols"))
    label_map = preset.get("label_map")
    if args.label_map:
        label_map = dict(p.split("=", 1)
                         for p in args.label_map.split(",") if "=" in p)

    if data_path:
        rows = load_local_rows(data_path)
    elif preset.get("hf_dataset"):
        split = args.split or preset.get("default_split") or "train"
        rows = load_hub_rows(preset["hf_dataset"], split,
                             config=preset.get("hf_config"),
                             max_rows=args.max_per_source)
    else:
        sys.exit(f"preset {preset_name!r} ships as a download — pass the "
                 f"file with --data FILE")
    examples = rows_to_examples(rows, text_col=text_col, labels_col=labels_col,
                                label_cols=label_cols, label_map=label_map,
                                row_labeler=preset.get("row_labeler"),
                                row_text=preset.get("row_text"))
    if args.max_per_source and len(examples) > args.max_per_source:
        import random

        random.Random(13).shuffle(examples)
        examples = examples[:args.max_per_source]
    return examples


def build_examples(args) -> list[Example]:
    sources: list[tuple[str | None, str | None]] = []
    if args.preset or args.data:
        sources.append((args.preset, args.data))
    for name in (args.mix.split(",") if args.mix else []):
        if name.strip():
            sources.append((name.strip(), None))
    if not sources:
        sys.exit("need --preset, --data, or --mix")

    examples: list[Example] = []
    for preset_name, data_path in sources:
        got = examples_from_source(preset_name, data_path, args)
        n_pos = sum(1 for e in got if e.labels)
        print(f"#   {preset_name or data_path}: {len(got)} examples "
              f"({n_pos} labeled)", file=sys.stderr)
        examples.extend(got)
    return balance_clean(examples, args.max_clean_ratio)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m safety.train.finetune_text",
        description="Fine-tune a multi-label safety classifier on the "
                    "12-category taxonomy.")
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="dataset preset (column mapping included)")
    p.add_argument("--data", help="local CSV/TSV/JSONL file")
    p.add_argument("--mix", metavar="P1,P2,…",
                   help="combine several Hub presets into one training corpus "
                        "(e.g. civil_comments,davidson,textdetox,goemotions)")
    p.add_argument("--split", default=None, help="Hub dataset split "
                   "(textdetox uses language codes: en, es, ar, zh, …)")
    p.add_argument("--max-per-source", type=int, default=0,
                   help="cap examples taken from each source (0 = all)")
    p.add_argument("--max-clean-ratio", type=float, default=2.0,
                   help="cap clean examples at RATIO x labeled (0 = no cap)")
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
