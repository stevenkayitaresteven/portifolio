"""Data preparation for the 12-category safety models.

A small, dependency-light pipeline that turns raw, messy text from many
sources into a clean, de-duplicated, labelled, tokenised dataset you can feed
to either the existing classifier (:mod:`safety.train.finetune_text`) or the
LoRA recipes (:mod:`safety.train.lora`).

Stages (each is its own module, usable on its own):

1. :mod:`.sources`   — ingest from local files, Hugging Face datasets, or a
   polite, ``robots.txt``-respecting web fetcher (benign text only).
2. :mod:`.clean`     — normalise unicode/whitespace, scrub PII, drop junk.
3. :mod:`.dedup`     — exact + near-duplicate removal (MinHash/LSH, no deps).
4. :mod:`.label`     — map heterogeneous source labels onto the 12 categories,
   or weak-label unlabelled text with the offline heuristics.
5. :mod:`.pipeline`  — orchestrate the above, balance, split, write a card.
6. :mod:`.tokenize`  — tokenise the splits with a Hugging Face tokenizer.

Run it end-to-end on a bundled synthetic sample (no network, no GPU)::

    python -m safety.data demo --out runs/demo_dataset

See :class:`~safety.data.config.PipelineConfig` for every knob.
"""
from __future__ import annotations

from .config import PipelineConfig
from .clean import TextCleaner
from .dedup import Deduplicator
from .label import LabelMapper, weak_label
from .pipeline import DataPipeline, Record

__all__ = [
    "PipelineConfig",
    "TextCleaner",
    "Deduplicator",
    "LabelMapper",
    "weak_label",
    "DataPipeline",
    "Record",
]
