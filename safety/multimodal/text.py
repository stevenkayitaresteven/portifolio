"""Text moderation: curse-word masking + transformer toxicity scoring.

Two layers, each useful without the other:

1. **Lexicon** (always on, zero deps, offline) — finds the *individual* curse
   words so they can be masked in place (``f***``). See :mod:`.wordlist`.
2. **Hugging Face model** (optional) — ``unitary/toxic-bert``, a BERT fine-tuned
   on the Jigsaw toxic-comment datasets, scores the whole message on six
   labels: toxic, severe_toxic, obscene, threat, insult, identity_hate. It
   catches hostility the wordlist can't ("go back where you came from"), and
   the wordlist catches spelled-around words the model may miss.

The model is loaded lazily on first use; if ``transformers`` is missing or the
checkpoint can't be fetched, the moderator silently runs lexicon-only.
"""
from __future__ import annotations

from .config import MultimodalConfig
from .result import (ModerationResult, ACTION_BLOCK, ACTION_FLAG, ACTION_MASK,
                     ACTION_NONE)
from .wordlist import mask_profanity


class TextModerator:
    def __init__(self, config: MultimodalConfig | None = None):
        self.config = config or MultimodalConfig.from_env()
        self._hf_pipe = None
        self._hf_failed = False

    # --- HF backend (lazy) -----------------------------------------------------
    @property
    def hf_available(self) -> bool:
        return self._load_hf() is not None

    def _load_hf(self):
        if self._hf_pipe is not None or self._hf_failed \
                or not self.config.use_hf_text:
            return self._hf_pipe
        try:
            from transformers import pipeline

            # top_k=None -> scores for every label of the multi-label head.
            self._hf_pipe = pipeline("text-classification",
                                     model=self.config.text_model,
                                     top_k=None, truncation=True)
        except Exception:
            self._hf_failed = True  # offline / missing extra: lexicon-only
        return self._hf_pipe

    def _hf_scores(self, text: str) -> dict[str, float]:
        pipe = self._load_hf()
        if pipe is None or not text.strip():
            return {}
        try:
            rows = pipe(text[:2000])  # guard against pathological inputs
            if rows and isinstance(rows[0], list):  # top_k=None nests one level
                rows = rows[0]
            return {r["label"].lower(): float(r["score"]) for r in rows}
        except Exception:
            return {}

    # --- main entry --------------------------------------------------------------
    def moderate(self, text: str) -> ModerationResult:
        res = ModerationResult(modality="text", detectors=["lexicon"])

        censored, words = mask_profanity(text)
        if words:
            res.add_category(
                "profanity", 1.0,
                f"{len(words)} curse word(s) masked: "
                + ", ".join(w[0] + "*" * (len(w) - 1) for w in words[:5]))

        scores = self._hf_scores(text)
        if scores:
            res.detectors.append(f"hf:{self.config.text_model}")
        for label, score in scores.items():
            if score >= self.config.text_threshold:
                res.add_category(label, score, f"{label} {score:.2f} (model)")
            else:
                res.scores.setdefault(label, round(score, 4))

        res.flagged = bool(res.categories)
        res.censored_text = censored if res.flagged else text
        if not res.flagged:
            res.action = ACTION_NONE
        elif not self.config.deliver_flagged_text:
            res.action = ACTION_BLOCK
        elif words:
            res.action = ACTION_MASK   # deliver with the words cut out
        else:
            res.action = ACTION_FLAG   # model-only hit: deliver but mark it
        return res
