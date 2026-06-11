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
    # Multi-label toxicity: toxic / severe_toxic / obscene / threat / insult /
    # identity_hate. Trained on the Jigsaw toxic-comment datasets.
    text_model: str = "unitary/toxic-bert"
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
    # A toxic-bert label at/above this score flags the text.
    text_threshold: float = 0.50
    # Flagged text is still delivered with curse words masked (True), or
    # withheld entirely (False).
    deliver_flagged_text: bool = True
    # Content we could not analyze (e.g. audio with no ASR backend) is marked
    # action="flag" so a human can review it; False delivers it silently.
    flag_unscanned: bool = True

    # --- Video sampling ---------------------------------------------------------
    video_sample_fps: float = 1.0     # analyze ~this many frames per second
    video_max_samples: int = 32       # hard cap on analyzed frames per video
    check_video_audio: bool = True    # also scan the audio track (needs ffmpeg)

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
