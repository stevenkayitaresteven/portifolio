"""Core data types shared across the explicit-content safety package.

Everything here is plain ``dataclasses`` + ``enum`` so the types can be imported
without pulling in OpenCV, ONNX, or PyTorch. Detectors return :class:`Region`
objects, the policy turns a list of them into a :class:`Verdict`, and the blur
engine consumes both.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Iterable


class Category(str, Enum):
    """What kind of sensitive content a detection represents."""

    NUDITY = "nudity"            # exposed sexual / intimate body parts
    SUGGESTIVE = "suggestive"    # partially exposed / covered intimate areas
    GORE = "gore"               # wounds, blood, graphic injury
    SAFE = "safe"               # explicitly classified as clean

    def __str__(self) -> str:  # nicer logs / JSON
        return self.value


class Severity(int, Enum):
    """How strongly a region should be acted on. Higher = more explicit."""

    NONE = 0
    LOW = 1        # mildly suggestive (covered intimate area, bare midriff)
    MEDIUM = 2     # borderline (exposed male chest, exposed buttocks region)
    HIGH = 3       # explicit (exposed genitalia, exposed female breast, anus)

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Region:
    """A localized detection: an axis-aligned box plus its label and score.

    Coordinates are pixel values in the source image, ``x1,y1`` top-left and
    ``x2,y2`` bottom-right (inclusive of the box area, exclusive end is fine for
    slicing). ``label`` is the raw detector label (e.g. ``FEMALE_BREAST_EXPOSED``
    or ``wound``); ``category``/``severity`` are the normalized interpretation.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: float
    category: Category
    severity: Severity = Severity.MEDIUM
    source: str = "unknown"   # which detector produced it

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def expanded(self, margin: float, w: int, h: int) -> "Region":
        """Return a copy grown by ``margin`` (fraction of box size), clamped to
        the image bounds ``w`` x ``h``. Used to make the blur cover edges of the
        sensitive area rather than stopping exactly at the detected box."""
        bw, bh = self.x2 - self.x1, self.y2 - self.y1
        dx, dy = int(bw * margin), int(bh * margin)
        return Region(
            x1=max(0, self.x1 - dx), y1=max(0, self.y1 - dy),
            x2=min(w, self.x2 + dx), y2=min(h, self.y2 + dy),
            label=self.label, score=self.score, category=self.category,
            severity=self.severity, source=self.source,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.name.lower()
        return d


@dataclass
class Verdict:
    """The aggregated decision for one image."""

    explicit: bool                       # did anything cross the blur threshold?
    regions: list[Region] = field(default_factory=list)   # regions to redact
    all_regions: list[Region] = field(default_factory=list)  # every detection
    scores: dict[str, float] = field(default_factory=dict)   # per-category max score
    whole_image: bool = False            # blur the entire frame, not just boxes
    reasons: list[str] = field(default_factory=list)

    @property
    def max_severity(self) -> Severity:
        return max((r.severity for r in self.regions), default=Severity.NONE)

    def categories(self) -> set[Category]:
        return {r.category for r in self.regions}

    def to_dict(self) -> dict:
        return {
            "explicit": self.explicit,
            "whole_image": self.whole_image,
            "max_severity": str(self.max_severity),
            "categories": sorted(c.value for c in self.categories()),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "reasons": self.reasons,
            "regions": [r.to_dict() for r in self.regions],
            "num_detections": len(self.all_regions),
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)


def merge_regions(*groups: Iterable[Region]) -> list[Region]:
    """Flatten several detector outputs into one list."""
    out: list[Region] = []
    for g in groups:
        out.extend(g)
    return out
