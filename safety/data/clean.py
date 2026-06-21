"""Text cleaning and normalisation.

Cleaning moderation data has one twist most NLP pipelines get wrong: you must
*keep* the toxic content (that's the signal) while still removing things that
leak or mislead — personal data, raw URLs, control characters, encoding cruft.
:class:`TextCleaner` does exactly that and nothing destructive to the label.
"""
from __future__ import annotations

import re
import unicodedata

from ..multimodal.heuristics import mask_pii

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"(?<!\w)@\w{2,30}")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Collapse 4+ repeats ("loooove" -> "loove") — keeps emphasis, kills spam runs.
_REPEAT_RE = re.compile(r"(.)\1{3,}")


class TextCleaner:
    def __init__(self, *, lowercase: bool = False, scrub_pii: bool = True,
                 replace_urls: bool = True, min_chars: int = 3,
                 max_chars: int = 2000):
        self.lowercase = lowercase
        self.scrub_pii = scrub_pii
        self.replace_urls = replace_urls
        self.min_chars = min_chars
        self.max_chars = max_chars

    def clean(self, text: str) -> str | None:
        """Return cleaned text, or ``None`` if the row should be dropped."""
        if not text:
            return None
        # NFKC folds look-alike unicode (fullwidth, ligatures) used to dodge filters.
        text = unicodedata.normalize("NFKC", text)
        text = _CTRL_RE.sub(" ", text)
        if self.replace_urls:
            text = _URL_RE.sub("<url>", text)
            text = _MENTION_RE.sub("<user>", text)
        if self.scrub_pii:
            text, _ = mask_pii(text)
        text = _REPEAT_RE.sub(r"\1\1", text)
        text = _WS_RE.sub(" ", text).strip()
        if self.lowercase:
            text = text.lower()
        if len(text) < self.min_chars:
            return None
        if len(text) > self.max_chars:
            text = text[:self.max_chars].rsplit(" ", 1)[0]
        return text or None

    @staticmethod
    def dedup_key(text: str) -> str:
        """A normalised key for exact-duplicate detection (case/punct-insensitive)."""
        key = unicodedata.normalize("NFKC", text).lower()
        return _WS_RE.sub(" ", re.sub(r"[^\w\s]", "", key)).strip()
