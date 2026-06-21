"""LoRA / QLoRA fine-tuning recipes for LLM and VLM moderators.

These turn a base instruction model into a *moderation* model that emits the
12-category JSON contract (:mod:`safety.multimodal.llm_prompt`), so it can serve
as the :class:`~safety.multimodal.llm_judge.LLMJudgeModerator` backend.

Two recipes:

* :mod:`.finetune_llm` — text LLM (e.g. Llama 3.1 8B, Mistral 7B, Gemma) with
  QLoRA (4-bit base + LoRA adapters) so it fits on a single 16 GB GPU.
* :mod:`.finetune_vlm` — vision-language model (e.g. Qwen2.5-VL) for moderating
  images with a text instruction.

Both build their training data from the JSONL produced by
:mod:`safety.data`, and both run a ``--dry-run`` that prepares everything on CPU
without loading the model — so you can validate the data and config before
renting a GPU. The actual training step needs ``pip install
"transformers>=4.45" peft trl bitsandbytes accelerate datasets`` and a GPU.
"""
from __future__ import annotations

from .data import build_sft_examples

__all__ = ["build_sft_examples"]
