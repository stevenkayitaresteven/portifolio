"""The one result type every modality moderator returns.

Like :class:`~safety.types.Verdict` for images, but modality-agnostic: it says
*what* was found and *what was done about it* so a caller (chat UI, API, bot)
can render any modality the same way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


# What the moderator did (or recommends doing) with the content.
ACTION_NONE = "none"      # clean — deliver as-is
ACTION_MASK = "mask"      # text: curse words replaced with f*** style masks
ACTION_FLAG = "flag"      # deliver, but mark for review (e.g. model-only hit)
ACTION_BLUR = "blur"      # image/video: pixels redacted, blurred copy delivered
ACTION_MUTE = "mute"      # video: visuals clean but audio explicit -> sound cut
ACTION_BLOCK = "block"    # audio (or severe text): withhold the content entirely


@dataclass
class ModerationResult:
    """Aggregated decision for one piece of content (any modality)."""

    modality: str                      # "text" | "image" | "audio" | "video"
    flagged: bool = False
    action: str = ACTION_NONE          # one of the ACTION_* constants
    categories: list[str] = field(default_factory=list)   # e.g. ["profanity"]
    scores: dict[str, float] = field(default_factory=dict)  # per-category max
    reasons: list[str] = field(default_factory=list)
    detectors: list[str] = field(default_factory=list)     # what actually ran
    scanned: bool = True               # False = no backend could analyze this
    censored_text: str | None = None   # text/audio: content with words masked
    transcript: str | None = None      # audio/video: the (censored) transcript
    extra: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Confidence in the verdict: top category score when flagged,
        ``1 - max(score)`` when clean (high = confidently clean)."""
        top = max(self.scores.values(), default=0.0)
        return float(top if self.flagged else 1.0 - top)

    def add_category(self, category: str, score: float, reason: str = "") -> None:
        if category not in self.categories:
            self.categories.append(category)
        self.scores[category] = max(self.scores.get(category, 0.0), float(score))
        if reason:
            self.reasons.append(reason)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "flagged": self.flagged,
            "action": self.action,
            "categories": sorted(self.categories),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "detectors": self.detectors,
            "scanned": self.scanned,
            "censored_text": self.censored_text,
            "transcript": self.transcript,
            **({"extra": self.extra} if self.extra else {}),
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)
