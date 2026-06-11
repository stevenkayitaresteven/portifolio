"""One front door for every modality: route content to the right moderator.

:class:`MultimodalModerator` lazily builds the per-modality moderators (so a
text-only caller never touches OpenCV or transformers) and dispatches files by
MIME type / extension. Plain ``.txt``/``.md`` files are read and moderated as
text, so "send a document" gets the same treatment as a typed message.
"""
from __future__ import annotations

import mimetypes
import os

from .audio import AudioModerator
from .config import MultimodalConfig
from .image import ImageModerator
from .result import ModerationResult
from .text import TextModerator
from .video import VideoModerator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac",
              ".amr", ".wma", ".weba"}
TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".json"}


def modality_for(filename: str, content_type: str | None = None) -> str | None:
    """Map a filename (and optional MIME type) to a supported modality."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TEXT_EXTS:
        return "text"
    mime = content_type or mimetypes.guess_type(filename or "")[0] or ""
    for prefix in ("image", "video", "audio", "text"):
        if mime.startswith(prefix + "/"):
            return prefix
    return None


class MultimodalModerator:
    """Lazy facade over the four modality moderators."""

    def __init__(self, config: MultimodalConfig | None = None):
        self.config = config or MultimodalConfig.from_env()
        self._text: TextModerator | None = None
        self._image: ImageModerator | None = None
        self._audio: AudioModerator | None = None
        self._video: VideoModerator | None = None

    # --- lazy accessors (also handy injection points for tests) ----------------
    @property
    def text(self) -> TextModerator:
        if self._text is None:
            self._text = TextModerator(self.config)
        return self._text

    @property
    def image(self) -> ImageModerator:
        if self._image is None:
            self._image = ImageModerator(self.config)
        return self._image

    @property
    def audio(self) -> AudioModerator:
        if self._audio is None:
            self._audio = AudioModerator(self.config, text_moderator=self.text)
        return self._audio

    @property
    def video(self) -> VideoModerator:
        if self._video is None:
            self._video = VideoModerator(self.config, image_moderator=self.image,
                                         audio_moderator=self.audio)
        return self._video

    # --- entry points ------------------------------------------------------------
    def moderate_text(self, text: str) -> ModerationResult:
        return self.text.moderate(text)

    def moderate_file(self, path: str, *, modality: str | None = None,
                      out_path: str | None = None
                      ) -> tuple[str | None, ModerationResult]:
        """Moderate a file on disk. Returns ``(delivered_path, result)``:
        the original path when clean, a redacted copy when flagged, ``None``
        when the content must be withheld or couldn't be read."""
        modality = modality or modality_for(path)
        if modality == "image":
            return self._moderate_image_file(path, out_path)
        if modality == "video":
            return self.video.moderate_file(path, out_path)
        if modality == "audio":
            res = self.audio.moderate_file(path)
            blocked = res.flagged and res.action == "block"
            return (None if blocked else path), res
        if modality == "text":
            return self._moderate_text_file(path, out_path)
        return None, ModerationResult(modality=modality or "unknown",
                                      scanned=False,
                                      reasons=[f"unsupported file type: {path}"])

    def _moderate_image_file(self, path: str, out_path: str | None):
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None, ModerationResult(modality="image", scanned=False,
                                          reasons=["could not read image"])
        out, res = self.image.moderate_bgr(img)
        if not res.flagged:
            return path, res
        if out_path is None:
            base, ext = os.path.splitext(path)
            out_path = f"{base}.blurred{ext if ext else '.jpg'}"
        cv2.imwrite(out_path, out)
        return out_path, res

    def _moderate_text_file(self, path: str, out_path: str | None):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(200_000)
        except OSError as exc:
            return None, ModerationResult(modality="text", scanned=False,
                                          reasons=[str(exc)])
        res = self.text.moderate(content)
        if not res.flagged:
            return path, res
        if res.action == "block":
            return None, res
        if out_path is None:
            base, ext = os.path.splitext(path)
            out_path = f"{base}.censored{ext or '.txt'}"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(res.censored_text or "")
        return out_path, res

    def health(self) -> dict:
        """What's actually loaded/loadable — for a /health endpoint."""
        return {
            "text": {"lexicon": True, "hf_model": self.config.text_model,
                     "hf_loaded": self._text is not None and self._text.hf_available},
            "image": {"detectors": (self._image.active_detectors
                                    if self._image else "lazy")},
            "audio": {"asr_model": self.config.asr_model,
                      "asr_loaded": self._audio is not None and self._audio.asr_available},
            "video": {"sample_fps": self.config.video_sample_fps,
                      "max_samples": self.config.video_max_samples},
        }
