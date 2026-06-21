"""Build instruction-tuning examples from the 12-category JSONL.

Each cleaned, labelled row becomes a chat example whose assistant turn is the
exact JSON the moderator must learn to produce. Pure Python — no model needed —
so it's the thing the ``--dry-run`` validates.
"""
from __future__ import annotations

import json
import os

from ...multimodal import llm_prompt


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_sft_examples(path: str) -> list[dict]:
    """Read a ``*.jsonl`` (or a dataset dir's ``train.jsonl``) into SFT chats.

    Returns a list of ``{"messages": [...]}`` where the final assistant message
    is the gold JSON verdict.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "train.jsonl")
    examples: list[dict] = []
    for row in _iter_jsonl(path):
        cats = row.get("categories") or []
        messages = llm_prompt.build_messages(row["text"])
        messages.append({"role": "assistant",
                         "content": llm_prompt.format_target(cats)})
        examples.append({"messages": messages})
    return examples
