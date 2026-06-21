"""Exact and near-duplicate removal — pure Python, no dependencies.

Web-scraped and Hub corpora are full of duplicates and near-duplicates (retweets,
boilerplate, copy-paste spam). Training on them wastes compute and inflates your
metrics. This does two passes:

* **exact** — a normalised-key hash set (handles casing/punctuation noise).
* **near** — character-shingle **MinHash** signatures bucketed with **LSH**, so
  two texts that share most of their content collapse to one. O(n) in practice,
  no ``datasketch`` needed.
"""
from __future__ import annotations

import hashlib

from .clean import TextCleaner


def _shingles(text: str, k: int = 5) -> set[str]:
    """Character k-shingles — robust to small edits and word reordering."""
    text = "".join(text.lower().split())
    if len(text) <= k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _hash(s: str, seed: int) -> int:
    h = hashlib.blake2b(f"{seed}:{s}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


class Deduplicator:
    """Drop exact and near-duplicate texts, keeping the first occurrence."""

    def __init__(self, *, near: bool = True, threshold: float = 0.8,
                 num_perm: int = 64, bands: int = 16, shingle_k: int = 5):
        self.near = near
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle_k = shingle_k
        self._exact: set[str] = set()
        self._buckets: dict[tuple, list[tuple[int, ...]]] = {}

    def _signature(self, text: str) -> tuple[int, ...]:
        sh = _shingles(text, self.shingle_k) or {""}
        return tuple(min(_hash(s, p) for s in sh) for p in range(self.num_perm))

    @staticmethod
    def _similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        return sum(x == y for x, y in zip(a, b)) / len(a)

    def is_duplicate(self, text: str) -> bool:
        """Check-and-remember: ``True`` if ``text`` duplicates a prior one."""
        key = TextCleaner.dedup_key(text)
        if key in self._exact:
            return True
        self._exact.add(key)
        if not self.near:
            return False

        sig = self._signature(text)
        candidate_seen = False
        band_keys = []
        for b in range(self.bands):
            band = ("b", b) + sig[b * self.rows:(b + 1) * self.rows]
            band_keys.append(band)
            for other in self._buckets.get(band, ()):  # same band = collision
                if self._similarity(sig, other) >= self.threshold:
                    candidate_seen = True
                    break
            if candidate_seen:
                break
        if candidate_seen:
            return True
        for band in band_keys:
            self._buckets.setdefault(band, []).append(sig)
        return False
