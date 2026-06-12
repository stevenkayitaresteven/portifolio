"""Text moderation across the full 12-category taxonomy.

Three layers, strongest action wins (see :mod:`.taxonomy`):

1. **Heuristics** (always on, zero deps, offline)
   - profanity lexicon (:mod:`.wordlist`) → ``toxic``, words masked in place
   - PII regexes (:mod:`.heuristics`) → ``privacy``, the PII masked in place
   - phrase lexicons → ``self_harm`` / ``violence`` / ``criminal`` /
     ``cybersecurity`` / ``extremism`` / ``misinformation``
   - spam signals (links + promo language) → ``spam``

2. **Hugging Face ensemble** (lazy, optional) — every model in
   ``config.text_models`` runs and its labels are normalized via
   :func:`~safety.multimodal.taxonomy.category_for_label`. Defaults:
   ``KoalaAI/Text-Moderation`` (OpenAI moderation taxonomy: sexual, hate,
   violence, harassment, self-harm, child-safety) + ``unitary/toxic-bert``
   (Jigsaw labels). ``use_specialist_text_models`` adds spam, phishing and
   suicidality specialists. A local path to a checkpoint fine-tuned with
   ``safety/train/finetune_text.py`` slots in the same way.

3. **Policy** — each flagged category maps to an action
   (mask / flag / block); the message takes the most severe one.
"""
from __future__ import annotations

from . import taxonomy as T
from .config import MultimodalConfig
from .heuristics import (find_phrase_categories, mask_pii, spam_signals,
                         url_risk)
from .result import (ModerationResult, ACTION_BLOCK, ACTION_NONE)
from .wordlist import mask_profanity


class TextModerator:
    def __init__(self, config: MultimodalConfig | None = None):
        self.config = config or MultimodalConfig.from_env()
        self._pipes: dict[str, object] = {}    # model_id -> pipeline | None

    # --- HF backends (lazy, per-model) ----------------------------------------
    @property
    def model_ids(self) -> tuple[str, ...]:
        ids = tuple(self.config.text_models)
        if self.config.use_specialist_text_models:
            ids += tuple(self.config.specialist_text_models)
        return ids

    @property
    def hf_available(self) -> bool:
        return any(self._load(mid) is not None for mid in self.model_ids)

    def _load(self, model_id: str):
        if not self.config.use_hf_text:
            return None
        if model_id not in self._pipes:
            try:
                from transformers import pipeline

                # top_k=None -> scores for every label (multi-label heads too).
                self._pipes[model_id] = pipeline(
                    "text-classification", model=model_id,
                    top_k=None, truncation=True)
            except Exception:
                self._pipes[model_id] = None  # offline / missing: skip model
        return self._pipes[model_id]

    def _model_scores(self, model_id: str, text: str) -> dict[str, float]:
        """Run one model; returns {category: max_score} above threshold."""
        pipe = self._load(model_id)
        if pipe is None or not text.strip():
            return {}
        try:
            rows = pipe(text[:2000])
            if rows and isinstance(rows[0], list):  # top_k=None nests one level
                rows = rows[0]
        except Exception:
            return {}
        out: dict[str, float] = {}
        for r in rows:
            score = float(r["score"])
            if score < self.config.text_threshold:
                continue
            cat = T.category_for_label(model_id, str(r["label"]))
            if cat is not None:
                out[cat] = max(out.get(cat, 0.0), score)
        return out

    # --- main entry --------------------------------------------------------------
    def moderate(self, text: str) -> ModerationResult:
        res = ModerationResult(modality="text", detectors=["lexicon"])
        censored = text

        # 1) profanity -> toxic, masked in place
        censored, curse_words = mask_profanity(censored)
        if curse_words:
            res.add_category(
                T.TOXIC, 1.0,
                f"{len(curse_words)} curse word(s) masked: "
                + ", ".join(w[0] + "*" * (len(w) - 1) for w in curse_words[:5]))

        # 2) PII -> privacy, masked in place
        censored, pii_kinds = mask_pii(censored)
        if pii_kinds:
            res.detectors.append("pii_regex")
            res.add_category(
                T.PRIVACY, 1.0,
                "PII masked: " + ", ".join(sorted(set(pii_kinds))))

        # 3) phrase lexicons -> self_harm / violence / criminal / cyber /
        #    extremism / misinformation
        phrase_hits = find_phrase_categories(text)
        if phrase_hits:
            res.detectors.append("phrases")
        for category, phrase, score in phrase_hits:
            res.add_category(category, score, f"phrase: {phrase!r}")

        # 4) spam signals + phishing-shaped URLs
        spam_score, spam_reasons = spam_signals(text)
        if spam_score >= 0.5:
            res.detectors.append("spam_signals")
            res.add_category(T.SPAM, spam_score, "; ".join(spam_reasons))
        link_score, link_reasons = url_risk(text)
        if link_score >= 0.5:
            res.detectors.append("url_risk")
            res.add_category(T.SPAM, link_score,
                             "suspicious link: " + "; ".join(link_reasons))

        # 5) HF model ensemble
        for model_id in self.model_ids:
            scores = self._model_scores(model_id, text)
            if scores:
                res.detectors.append(f"hf:{model_id}")
            for cat, score in scores.items():
                res.add_category(cat, score, f"{cat} {score:.2f} ({model_id})")

        # 6) policy: most severe per-category action wins
        res.flagged = bool(res.categories)
        res.censored_text = censored
        if not res.flagged:
            res.action = ACTION_NONE
        elif not self.config.deliver_flagged_text:
            res.action = ACTION_BLOCK
        else:
            res.action = T.strongest_action(
                [self.config.action_for(c) for c in res.categories])
        if res.action == ACTION_BLOCK:
            res.censored_text = None  # blocked content is not delivered at all
        if T.SELF_HARM in res.categories:
            res.extra["support_note"] = (
                "If you or someone you know is struggling, help is available — "
                "call or text 988 (US) or find local support at "
                "findahelpline.com.")
        return res
