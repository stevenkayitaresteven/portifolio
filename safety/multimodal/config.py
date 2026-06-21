"""Configuration for the multimodal moderators.

One :class:`MultimodalConfig` picks the Hugging Face checkpoints, the per-modality
thresholds, and the delivery policy (mask vs block, blur vs flag). The image
policy itself still lives in :class:`~safety.config.SafetyConfig` — this wraps
one. Every field is overridable from ``SAFETY_MM_<FIELD>`` environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from ..config import SafetyConfig, _coerce


@dataclass
class MultimodalConfig:
    # The image detection/blur policy (thresholds, blur style, NudeNet, ...).
    safety: SafetyConfig = field(default_factory=SafetyConfig.from_env)

    # --- Hugging Face checkpoints (fine-tuned, pretrained) --------------------
    # The text *ensemble*: every model runs, labels are normalized into the
    # 12-category taxonomy (safety/multimodal/taxonomy.py). The defaults pair
    # the OpenAI-taxonomy moderation model (sexual / hate / violence /
    # harassment / self-harm / child-safety) with Jigsaw toxic-bert. A path to
    # your own checkpoint from safety/train/finetune_text.py works here too.
    text_models: tuple[str, ...] = ("KoalaAI/Text-Moderation",
                                    "unitary/toxic-bert")
    # Specialist additions enabled by use_specialist_text_models:
    # spam, phishing, and suicidality classifiers.
    specialist_text_models: tuple[str, ...] = (
        "mshenoda/roberta-spam",
        "ealvaradob/bert-finetuned-phishing",
        "sentinet/suicidality",
    )
    use_specialist_text_models: bool = False   # 3 extra model downloads
    # Optional LLM-as-judge moderator (off by default — adds latency/compute).
    # Point at a local HF model OR an OpenAI-compatible endpoint. See
    # safety/multimodal/llm_judge.py.
    use_llm_judge: bool = False
    llm_judge_model: str = ""        # local HF model id or path (e.g. a LoRA merge)
    llm_judge_endpoint: str = ""     # OpenAI-compatible /chat/completions URL
    # ViT fine-tuned for nsfw/normal binary image classification.
    image_model: str = "Falconsai/nsfw_image_detection"
    # Whisper ASR — audio is transcribed, then the transcript is moderated.
    asr_model: str = "openai/whisper-base"

    # --- Which HF backends to enable (all degrade gracefully when the model
    #     can't be imported or downloaded) -------------------------------------
    use_hf_text: bool = True
    use_hf_image: bool = True
    use_hf_audio: bool = True

    # --- Thresholds & policy ---------------------------------------------------
    # A model label at/above this score flags the text.
    text_threshold: float = 0.50
    # Flagged text is still delivered with curse words masked (True), or
    # withheld entirely (False). Categories whose action is "block"
    # (hate, violence, self_harm, criminal, ... see taxonomy.DEFAULT_ACTIONS)
    # are withheld regardless of this switch.
    deliver_flagged_text: bool = True
    # Per-category action overrides as "category:action" pairs, e.g.
    # "spam:block,sexual:mask". Categories not listed keep their default.
    action_overrides: str = ""
    # Content we could not analyze (e.g. audio with no ASR backend) is marked
    # action="flag" so a human can review it; False delivers it silently.
    flag_unscanned: bool = True
    # Append every moderation decision (metadata only — never the content
    # itself) as JSON lines to this file. Empty = no audit log.
    audit_log: str = ""

    # OCR text embedded in images (memes, screenshots) and moderate it like a
    # message. Needs `pytesseract` + the tesseract binary; skipped when absent.
    ocr_image_text: bool = True

    # --- Video sampling ---------------------------------------------------------
    video_sample_fps: float = 1.0     # analyze ~this many frames per second
    video_max_samples: int = 32       # hard cap on analyzed frames per video
    check_video_audio: bool = True    # also scan the audio track (needs ffmpeg)

    def action_for(self, category: str) -> str:
        """The configured action for a taxonomy category."""
        from .taxonomy import DEFAULT_ACTIONS

        for pair in self.action_overrides.split(","):
            name, _, action = pair.partition(":")
            if name.strip() == category and action.strip():
                return action.strip()
        return DEFAULT_ACTIONS.get(category, "flag")

    # --- Env loading -------------------------------------------------------------
    @classmethod
    def from_env(cls, prefix: str = "SAFETY_MM_") -> "MultimodalConfig":
        cfg = cls()
        for f in fields(cls):
            if f.name == "safety":
                continue
            env_key = prefix + f.name.upper()
            if env_key not in os.environ:
                continue
            try:
                setattr(cfg, f.name, _coerce(f.name, f.type, os.environ[env_key]))
            except Exception:
                pass  # ignore malformed overrides, keep the default
        return cfg
