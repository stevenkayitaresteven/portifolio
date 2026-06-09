"""Detector protocol and a tiny registry.

A detector takes a BGR ``numpy`` image (OpenCV's native layout) and returns a
list of :class:`~safety.types.Region`. Detectors are intentionally dumb: they
localize and score, they do **not** decide whether to blur — that is the
policy's job (:mod:`safety.pipeline`). This keeps thresholds in one place and
lets detectors be swapped or ensembled freely.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import Region


@runtime_checkable
class Detector(Protocol):
    """Anything with ``name`` and ``analyze`` can act as a detector."""

    name: str

    def analyze(self, image: np.ndarray) -> list[Region]:
        """Return zero or more detections for a single BGR image."""
        ...

    @property
    def available(self) -> bool:
        """Whether this detector's backend (model/weights) is importable."""
        ...


class BaseDetector:
    """Convenience base that provides a default ``available`` and ``__repr__``."""

    name: str = "base"

    def analyze(self, image: np.ndarray) -> list[Region]:  # pragma: no cover
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True

    def __repr__(self) -> str:
        status = "ready" if self.available else "unavailable"
        return f"<{type(self).__name__} name={self.name!r} {status}>"
