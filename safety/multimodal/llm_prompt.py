"""Shared moderation prompt + parsing for the LLM-judge and the LoRA recipes.

Keeping the prompt, the JSON contract, and the parser in one place means the
model you *train* (in :mod:`safety.train.lora`) and the model you *run* (the
:class:`~safety.multimodal.llm_judge.LLMJudgeModerator`) speak exactly the same
format — a common cause of "works in eval, fails in prod" mismatches.
"""
from __future__ import annotations

import json
import re

from . import taxonomy as T

CATEGORIES = list(T.CATEGORIES)

SYSTEM_PROMPT = (
    "You are a content-moderation classifier. Read the user message and decide "
    "which, if any, of these safety categories apply:\n"
    + ", ".join(CATEGORIES) + ".\n"
    "Respond with ONLY a compact JSON object of the form "
    '{"categories": ["..."], "reason": "..."}. '
    "Use an empty list when the message is safe. Do not add any other text."
)


def build_messages(text: str) -> list[dict]:
    """Chat-format messages for an instruction-tuned model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def format_target(categories: list[str], reason: str = "") -> str:
    """The gold assistant reply for a training example (SFT target)."""
    cats = [c for c in categories if c in CATEGORIES]
    return json.dumps({"categories": cats, "reason": reason}, separators=(",", ":"))


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_response(text: str) -> list[str]:
    """Extract canonical categories from a model's (possibly messy) reply."""
    if not text:
        return []
    match = _JSON_RE.search(text)
    raw_cats: list = []
    if match:
        try:
            obj = json.loads(match.group(0))
            raw_cats = obj.get("categories") or []
        except (json.JSONDecodeError, AttributeError):
            raw_cats = []
    if not raw_cats:
        # Fall back to scanning for bare category names in free text.
        low = text.lower()
        raw_cats = [c for c in CATEGORIES if c in low or c.replace("_", " ") in low]
    out: list[str] = []
    for c in raw_cats:
        cat = T.category_for_label("", str(c))
        if cat and cat not in out:
            out.append(cat)
    return out
