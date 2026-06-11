"""Multimodal sensitive-content moderation: text, image, audio, and video.

This subpackage extends the image-only core into a single moderation surface
for every attachment type a chat can carry. Each modality gets its own
moderator built on a fine-tuned pretrained Hugging Face model, and every one
of them **degrades gracefully** to an offline fallback when the model (or the
network) is unavailable:

============  ======================================  =============================
 Modality      Hugging Face model (primary)            Offline fallback
============  ======================================  =============================
 text          ``unitary/toxic-bert``                  built-in profanity lexicon
 image         ``Falconsai/nsfw_image_detection``      NudeNet + wound heuristic
 audio         ``openai/whisper-base`` → text model    flagged as "unscanned"
 video         frame sampling → image model            NudeNet on sampled frames
============  ======================================  =============================

Everything funnels into one :class:`ModerationResult` so callers (the chat UI,
an API, a bot) can act uniformly: mask curse words, blur explicit images and
video, block or mute explicit audio, or just flag for review.

Quick start::

    from safety.multimodal import MultimodalModerator
    mod = MultimodalModerator()
    res = mod.moderate_text("hello world")          # -> ModerationResult
    out_path, res = mod.moderate_file("clip.mp4")   # blurred copy if explicit
"""
from __future__ import annotations

from .result import ModerationResult
from .config import MultimodalConfig
from .text import TextModerator
from .image import ImageModerator
from .audio import AudioModerator
from .video import VideoModerator
from .pipeline import MultimodalModerator, modality_for

__all__ = [
    "ModerationResult",
    "MultimodalConfig",
    "TextModerator",
    "ImageModerator",
    "AudioModerator",
    "VideoModerator",
    "MultimodalModerator",
    "modality_for",
]
