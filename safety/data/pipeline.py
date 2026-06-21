"""End-to-end: raw sources -> clean, deduped, labelled, split dataset.

``DataPipeline.run()`` walks every source through clean -> dedup -> label ->
balance -> split, writes one JSONL per split, and drops a ``dataset_card.json``
recording exact provenance and per-category counts so the build is auditable
and reproducible.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

from . import sources as S
from .clean import TextCleaner
from .config import PipelineConfig, SourceSpec
from .dedup import Deduplicator
from .label import CATEGORIES, LabelMapper, to_vector, weak_label


@dataclass
class Record:
    text: str
    categories: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def flagged(self) -> bool:
        return bool(self.categories)

    def to_json(self) -> dict:
        return {"text": self.text, "categories": self.categories,
                "labels": to_vector(self.categories), "source": self.source}


class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cleaner = TextCleaner(
            lowercase=config.lowercase, scrub_pii=config.scrub_pii,
            replace_urls=config.replace_urls, min_chars=config.min_chars,
            max_chars=config.max_chars)
        self.dedup = Deduplicator(near=config.dedup_near,
                                  threshold=config.near_threshold)
        self.stats: dict = {"sources": {}, "dropped": {}}

    # --- ingest one source ----------------------------------------------------
    def _iter_source(self, spec: SourceSpec):
        if spec.kind == "jsonl":
            return S.iter_jsonl(spec.path, spec.text_field, spec.label_field)
        if spec.kind == "csv":
            return S.iter_csv(spec.path, spec.text_field, spec.label_field)
        if spec.kind == "hf":
            return S.iter_hf(spec.path, spec.split, spec.text_field,
                             spec.label_field, spec.max_records)
        if spec.kind == "crawl":
            with open(spec.path, encoding="utf-8") as fh:
                urls = [u for u in fh.read().splitlines() if u.strip()]
            return S.WebTextCrawler().crawl(urls)
        raise ValueError(f"unknown source kind: {spec.kind!r}")

    def _collect(self) -> list[Record]:
        records: list[Record] = []
        for spec in self.config.sources:
            mapper = LabelMapper(spec.label_map, source_id=spec.path)
            kept = dropped = 0
            for n, (text, raw_labels) in enumerate(self._iter_source(spec)):
                if spec.max_records and n >= spec.max_records and spec.kind != "hf":
                    break
                cleaned = self.cleaner.clean(text)
                if cleaned is None:
                    dropped += 1
                    continue
                if self.config.dedup_exact and self.dedup.is_duplicate(cleaned):
                    dropped += 1
                    continue
                if raw_labels:
                    cats = sorted({c for lab in raw_labels if (c := mapper.map(lab))})
                elif spec.weak_label:
                    # Label on the *original* text: cleaning has already masked
                    # PII/URLs, which would hide the privacy/spam signal.
                    cats = weak_label(text)
                else:
                    cats = []
                records.append(Record(cleaned, cats, source=spec.path or spec.kind))
                kept += 1
            self.stats["sources"][spec.path or spec.kind] = {
                "kept": kept, "dropped": dropped}
        return records

    # --- balance --------------------------------------------------------------
    def _balance(self, records: list[Record], rng: random.Random) -> list[Record]:
        cfg = self.config
        flagged = [r for r in records if r.flagged]
        clean = [r for r in records if not r.flagged]

        if cfg.max_per_category:
            counts = {c: 0 for c in CATEGORIES}
            kept: list[Record] = []
            rng.shuffle(flagged)
            for r in flagged:
                if any(counts[c] < cfg.max_per_category for c in r.categories):
                    kept.append(r)
                    for c in r.categories:
                        counts[c] += 1
            flagged = kept

        if cfg.max_clean_ratio and flagged:
            cap = int(len(flagged) * cfg.max_clean_ratio)
            if len(clean) > cap:
                rng.shuffle(clean)
                clean = clean[:cap]

        out = flagged + clean
        rng.shuffle(out)
        return out

    # --- split ----------------------------------------------------------------
    def run(self, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        rng = random.Random(self.config.seed)

        records = self._balance(self._collect(), rng)
        n = len(records)
        n_test = int(n * self.config.test_ratio)
        n_val = int(n * self.config.val_ratio)
        splits = {
            "test": records[:n_test],
            "val": records[n_test:n_test + n_val],
            "train": records[n_test + n_val:],
        }

        card = {"categories": CATEGORIES, "total": n, "splits": {},
                "category_counts": {}, "provenance": self.stats["sources"],
                "config": {k: v for k, v in vars(self.config).items()
                           if k != "sources"}}
        for name, rows in splits.items():
            with open(os.path.join(out_dir, f"{name}.jsonl"), "w",
                      encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r.to_json()) + "\n")
            card["splits"][name] = len(rows)
        counts = {c: 0 for c in CATEGORIES}
        for r in records:
            for c in r.categories:
                counts[c] += 1
        card["category_counts"] = counts
        card["flagged"] = sum(1 for r in records if r.flagged)
        with open(os.path.join(out_dir, "dataset_card.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(card, fh, indent=2)
        return card
