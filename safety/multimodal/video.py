"""Video moderation: sampled frames through the image ensemble, plus the
audio track through the audio moderator.

There is no production-grade pretrained "NSFW video" model on the Hub, and the
standard practice (what the big moderation APIs do) is temporal sampling: pull
~1 frame/second, run each through the image classifiers, and treat any hit as
a hit for the clip. The audio track (when ``ffmpeg`` is present) is extracted
and judged by the Whisper→toxic-bert chain.

Outcomes:

* **visuals explicit** → the *entire* clip is re-encoded with every frame put
  through the solid blur (same cautious whole-image policy as photos). The
  re-encode carries no audio track, so explicit sound dies with it.
* **only the audio explicit** → the clip is re-encoded as-is, which strips
  (mutes) the sound; action = ``mute``.
* **clean** → the original file is delivered untouched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from ..blur import redact
from .audio import AudioModerator
from .config import MultimodalConfig
from .image import ImageModerator
from .result import (ModerationResult, ACTION_BLUR, ACTION_FLAG, ACTION_MUTE,
                     ACTION_NONE)


class VideoModerator:
    def __init__(self, config: MultimodalConfig | None = None,
                 image_moderator: ImageModerator | None = None,
                 audio_moderator: AudioModerator | None = None):
        self.config = config or MultimodalConfig.from_env()
        self.image = image_moderator or ImageModerator(self.config)
        self.audio = audio_moderator or AudioModerator(self.config)

    # --- audio track -----------------------------------------------------------
    def _moderate_audio_track(self, path: str) -> ModerationResult | None:
        """Extract the soundtrack with ffmpeg and moderate it; ``None`` when
        ffmpeg is absent, extraction fails, or there is no audio stream."""
        if not self.config.check_video_audio or not shutil.which("ffmpeg"):
            return None
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                 "-f", "wav", wav.name],
                capture_output=True, timeout=120)
            if proc.returncode != 0 or os.path.getsize(wav.name) < 128:
                return None
            return self.audio.moderate_file(wav.name)
        except Exception:
            return None
        finally:
            try:
                os.unlink(wav.name)
            except OSError:
                pass

    # --- main entry --------------------------------------------------------------
    def moderate_file(self, path: str,
                      out_path: str | None = None
                      ) -> tuple[str | None, ModerationResult]:
        """Analyze a video file. Returns ``(delivered_path, result)``.

        ``delivered_path`` is the original ``path`` when clean, the blurred /
        muted re-encode when flagged, or ``None`` when the file can't be opened.
        """
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, ModerationResult(modality="video", scanned=False,
                                          reasons=["could not open video"])

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, int(round(fps / max(self.config.video_sample_fps, 0.01))))
        if total > 0:  # respect the hard cap on analyzed frames
            stride = max(stride, total // max(self.config.video_max_samples, 1) + 1)

        res = ModerationResult(modality="video",
                               detectors=self.image.active_detectors,
                               scanned=bool(self.image.filter.detectors))
        sampled = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frame_res = self.image.evaluate_bgr(frame)
                sampled += 1
                if frame_res.flagged:
                    for c in frame_res.categories:
                        res.add_category(c, frame_res.scores.get(c, 0.0))
                    res.reasons.append(
                        f"frame {idx} ({idx / fps:.1f}s): "
                        + "; ".join(frame_res.reasons[:2]))
            idx += 1
        cap.release()
        res.extra["frames_sampled"] = sampled
        visual_flagged = bool(res.categories)

        audio_res = self._moderate_audio_track(path)
        audio_flagged = bool(audio_res and audio_res.flagged and audio_res.scanned)
        if audio_res is not None:
            res.transcript = audio_res.transcript
            if audio_flagged:
                for c in audio_res.categories:
                    res.add_category(c, audio_res.scores.get(c, 0.0))
                res.reasons.extend(f"audio: {r}" for r in audio_res.reasons)
                res.detectors.extend(d for d in audio_res.detectors
                                     if d not in res.detectors)

        res.flagged = visual_flagged or audio_flagged
        if not res.flagged:
            res.action = ACTION_NONE
            if not res.scanned and self.config.flag_unscanned:
                res.flagged, res.action = True, ACTION_FLAG
                res.reasons.append("no visual detector available — not analyzed")
            return path, res

        res.action = ACTION_BLUR if visual_flagged else ACTION_MUTE
        delivered = self._reencode(path, out_path, blur=visual_flagged)
        if delivered is None:  # can't rewrite -> withhold rather than leak
            res.reasons.append("re-encode failed; video withheld")
        return delivered, res

    def _reencode(self, path: str, out_path: str | None, *, blur: bool) -> str | None:
        """Rewrite the clip frame-by-frame (drops audio). ``blur=True`` puts
        every frame through the whole-image solid blur."""
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if out_path is None:
            out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        ext = os.path.splitext(out_path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            return None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if blur:
                    frame = redact(frame, [], self.config.safety, whole_image=True)
                writer.write(frame)
        finally:
            cap.release()
            writer.release()
        return self._h264_pass(out_path)

    @staticmethod
    def _h264_pass(path: str) -> str:
        """OpenCV writes MPEG-4 Part 2 (mp4v), which many browsers won't play.
        When ffmpeg is around, transcode to H.264 in place; otherwise keep."""
        if not path.endswith(".mp4") or not shutil.which("ffmpeg"):
            return path
        tmp = path + ".h264.mp4"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", tmp],
                capture_output=True, timeout=600)
            if proc.returncode == 0 and os.path.getsize(tmp) > 0:
                os.replace(tmp, path)
        except Exception:
            pass
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return path
