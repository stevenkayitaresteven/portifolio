"""Turn raw labels into the 12-category schema — or invent labels when none exist.

Two jobs:

* :class:`LabelMapper` — every source disagrees on what its labels mean
  (``"OFF"``, ``"identity_hate"``, ``"1"``, ``"toxic"`` …). This maps them onto
  the canonical categories using the same machinery the live detectors use
  (:func:`safety.multimodal.taxonomy.category_for_label`) plus a per-source
  override map.
* :func:`weak_label` — for *unlabelled* scraped text, bootstrap a label with the
  always-offline heuristics (lexicon, phrase lists, PII, spam signals). It's
  noisy supervision, but it's free and surprisingly useful for hard negatives
  and obvious positives. Always hold out a human-checked eval set on top.
"""
from __future__ import annotations

from ..multimodal import taxonomy as T
from ..multimodal.heuristics import (find_phrase_categories, mask_pii,
                                     spam_signals, url_risk)
from ..multimodal.wordlist import find_profanity

CATEGORIES = list(T.CATEGORIES)


class LabelMapper:
    def __init__(self, label_map: dict[str, str] | None = None,
                 source_id: str = ""):
        # Normalise keys so "Identity Hate" and "identity_hate" both match.
        self.label_map = {self._norm(k): v for k, v in (label_map or {}).items()}
        self.source_id = source_id

    @staticmethod
    def _norm(label: str) -> str:
        return str(label).strip().lower().replace(" ", "_").replace("-", "_")

    def map(self, label: str) -> str | None:
        """Map one raw label to a category, or ``None`` to drop the row."""
        raw = self._norm(label)
        if raw in self.label_map:
            mapped = self.label_map[raw]
            return mapped or None
        # Reuse the detector-side mapping (handles aliases + substring rescue).
        return T.category_for_label(self.source_id, raw)

    def vector(self, labels: list[str]) -> list[int]:
        """Multi-hot vector over the 12 categories from a list of raw labels."""
        cats = {c for lab in labels if (c := self.map(lab))}
        return [1 if c in cats else 0 for c in CATEGORIES]


def weak_label(text: str) -> list[str]:
    """Heuristically label unlabelled text. Returns canonical category names."""
    cats: set[str] = set()
    if find_profanity(text):
        cats.add(T.TOXIC)
    for cat, _phrase, _score in find_phrase_categories(text):
        cats.add(cat)
    if mask_pii(text)[1]:
        cats.add(T.PRIVACY)
    spam_score = max(spam_signals(text)[0], url_risk(text)[0])
    if spam_score >= 0.5:
        cats.add(T.SPAM)
    return sorted(cats)


def to_vector(categories: list[str]) -> list[int]:
    """Multi-hot vector over the 12 categories from canonical category names."""
    cats = set(categories)
    return [1 if c in cats else 0 for c in CATEGORIES]
