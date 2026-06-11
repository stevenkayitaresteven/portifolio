"""Audio moderation: transcribe with Whisper, then moderate the transcript.

There is no widely-adopted end-to-end "explicit audio" classifier, so the
robust pretrained route is two-stage: ``openai/whisper-base`` (multilingual
ASR, Apache-2.0) turns speech into text, and the :class:`TextModerator`
(toxic-bert + lexicon) judges it. Spoken profanity therefore gets exactly the
same taxonomy and thresholds as typed profanity.

Flagged audio is **blocked** (a chat can't blur sound) and the *censored*
transcript is returned so the receiver still sees what was said — minus the
curse words. WAV files decode via the stdlib; other formats need ``ffmpeg``
on PATH (the transformers ASR pipeline shells out to it). When no ASR backend
is available the result is marked unscanned and, by default, flagged for
human review rather than waved through.
"""
from __future__ import annotations

from .config import MultimodalConfig
from .result import (ModerationResult, ACTION_BLOCK, ACTION_FLAG, ACTION_NONE)
from .text import TextModerator


class AudioModerator:
    def __init__(self, config: MultimodalConfig | None = None,
                 text_moderator: TextModerator | None = None):
        self.config = config or MultimodalConfig.from_env()
        self.text = text_moderator or TextModerator(self.config)
        self._asr = None
        self._asr_failed = False

    # --- ASR backend (lazy) ------------------------------------------------------
    @property
    def asr_available(self) -> bool:
        return self._load_asr() is not None

    def _load_asr(self):
        if self._asr is not None or self._asr_failed \
                or not self.config.use_hf_audio:
            return self._asr
        try:
            from transformers import pipeline

            self._asr = pipeline("automatic-speech-recognition",
                                 model=self.config.asr_model)
        except Exception:
            self._asr_failed = True
        return self._asr

    @staticmethod
    def _load_wav(path: str):
        """Stdlib WAV decode -> mono float32 in [-1, 1] (no ffmpeg needed)."""
        import wave

        import numpy as np

        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            n_ch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            raise ValueError(f"unsupported WAV sample width: {width}")
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if width == 1:
            data = (data - 128.0) / 128.0
        else:
            data = data / float(2 ** (8 * width - 1))
        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)
        return {"raw": data, "sampling_rate": sr}

    def transcribe(self, path: str) -> str | None:
        """Return the transcript, or ``None`` when no backend can decode/run."""
        asr = self._load_asr()
        if asr is None:
            return None
        try:
            source = self._load_wav(path) if path.lower().endswith(".wav") else path
            out = asr(source)
            return (out.get("text") or "").strip()
        except Exception:
            return None

    # --- main entry -----------------------------------------------------------------
    def moderate_file(self, path: str) -> ModerationResult:
        transcript = self.transcribe(path)
        if transcript is None:
            return ModerationResult(
                modality="audio", scanned=False,
                flagged=self.config.flag_unscanned,
                action=ACTION_FLAG if self.config.flag_unscanned else ACTION_NONE,
                reasons=["no ASR backend (install the [hf] extra + ffmpeg) — "
                         "audio not analyzed"],
                detectors=[],
            )

        text_res = self.text.moderate(transcript)
        res = ModerationResult(
            modality="audio",
            flagged=text_res.flagged,
            action=ACTION_BLOCK if text_res.flagged else ACTION_NONE,
            categories=list(text_res.categories),
            scores=dict(text_res.scores),
            reasons=[f"transcript: {r}" for r in text_res.reasons],
            detectors=[f"hf:{self.config.asr_model}", *text_res.detectors],
            transcript=text_res.censored_text,
        )
        return res
