"""Configuration for the data-prep pipeline.

One :class:`PipelineConfig` describes the whole run: which sources to ingest,
how aggressively to clean, how to balance the categories, and how to split.
It round-trips to/from JSON so a dataset build is fully reproducible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class SourceSpec:
    """One input source. ``kind`` selects the loader in :mod:`.sources`."""
    kind: str                      # "jsonl" | "csv" | "hf" | "crawl"
    path: str = ""                 # file path, Hub dataset id, or URL-list file
    text_field: str = "text"       # column holding the message text
    label_field: str = ""          # column holding the source label (optional)
    split: str = "train"           # HF split, when kind == "hf"
    label_map: dict = field(default_factory=dict)   # source-label -> category
    max_records: int = 0           # 0 = no cap for this source
    weak_label: bool = False       # heuristically label unlabelled text


@dataclass
class PipelineConfig:
    sources: list[SourceSpec] = field(default_factory=list)

    # --- cleaning ---------------------------------------------------------------
    lowercase: bool = False        # keep case by default (casing carries signal)
    scrub_pii: bool = True         # mask emails/SSNs/cards before training
    replace_urls: bool = True      # URLs -> <url>, mentions -> <user>
    min_chars: int = 3
    max_chars: int = 2000
    drop_non_text: bool = True     # drop rows that are empty after cleaning

    # --- dedup ------------------------------------------------------------------
    dedup_exact: bool = True
    dedup_near: bool = True
    near_threshold: float = 0.8    # Jaccard similarity to treat as a duplicate

    # --- balancing & splitting --------------------------------------------------
    max_per_category: int = 0      # 0 = no cap; else downsample over-represented
    max_clean_ratio: float = 2.0   # keep at most N clean rows per flagged row
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 13

    # --- tokenisation -----------------------------------------------------------
    tokenizer: str = "distilroberta-base"
    max_length: int = 256

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        sources = [SourceSpec(**s) for s in raw.pop("sources", [])]
        return cls(sources=sources, **raw)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
