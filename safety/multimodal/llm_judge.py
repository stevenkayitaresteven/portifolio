"""An optional LLM-as-judge moderation backend.

Small classifiers are fast and cheap, but they miss nuance, sarcasm, and novel
phrasing. A safety-tuned instruction model (Llama Guard, or one of the LoRA
moderators trained by :mod:`safety.train.lora`) can catch those — at the cost of
latency and compute. This backend is **off by default**; enable it with
``SAFETY_MM_USE_LLM_JUDGE=1`` and point it at either a local Hugging Face model
(``SAFETY_MM_LLM_JUDGE_MODEL``) or an OpenAI-compatible HTTP endpoint
(``SAFETY_MM_LLM_JUDGE_ENDPOINT``).

It degrades gracefully: if no backend is reachable it returns no categories and
the rest of the ensemble carries on, exactly like the other optional detectors.
The backend is injectable, so tests drive it with a stub and need no model.
"""
from __future__ import annotations

import json
import os
from typing import Callable

from . import llm_prompt
from .config import MultimodalConfig

# A backend takes chat messages and returns the model's raw text reply.
Backend = Callable[[list], str]


class LLMJudgeModerator:
    def __init__(self, config: MultimodalConfig | None = None,
                 backend: Backend | None = None):
        self.config = config or MultimodalConfig.from_env()
        self._backend = backend
        self._tried = False

    # --- backends -------------------------------------------------------------
    def _http_backend(self, endpoint: str) -> Backend:
        """OpenAI-compatible /chat/completions backend (works with vLLM, Ollama,
        TGI, OpenAI, etc.)."""
        def call(messages: list) -> str:
            import urllib.request

            body = json.dumps({
                "model": self.config.llm_judge_model or "default",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 200,
            }).encode()
            headers = {"Content-Type": "application/json"}
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "SAFETY_MM_LLM_JUDGE_KEY")
            if key:
                headers["Authorization"] = f"Bearer {key}"
            req = urllib.request.Request(endpoint, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        return call

    def _local_backend(self, model_id: str) -> Backend:
        """A local Hugging Face text-generation pipeline."""
        from transformers import pipeline

        gen = pipeline("text-generation", model=model_id)

        def call(messages: list) -> str:
            out = gen(messages, max_new_tokens=200, do_sample=False,
                      return_full_text=False)
            text = out[0]["generated_text"]
            return text if isinstance(text, str) else text[-1]["content"]
        return call

    def _resolve_backend(self) -> Backend | None:
        if self._backend is not None:
            return self._backend
        if self._tried:
            return None
        self._tried = True
        try:
            if self.config.llm_judge_endpoint:
                self._backend = self._http_backend(self.config.llm_judge_endpoint)
            elif self.config.llm_judge_model:
                self._backend = self._local_backend(self.config.llm_judge_model)
        except Exception:
            self._backend = None
        return self._backend

    @property
    def available(self) -> bool:
        return self._resolve_backend() is not None

    # --- main entry -----------------------------------------------------------
    def categories(self, text: str) -> list[str]:
        """Return the categories the judge assigns (empty if unavailable/safe)."""
        backend = self._resolve_backend()
        if backend is None or not text.strip():
            return []
        try:
            reply = backend(llm_prompt.build_messages(text))
        except Exception:
            return []
        return llm_prompt.parse_response(reply)
