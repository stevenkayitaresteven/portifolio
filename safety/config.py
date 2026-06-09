"""Configuration for the explicit-content filter.

A single :class:`SafetyConfig` controls which detectors run, the score
thresholds per category, and how flagged regions are redacted. Defaults are
tuned to be *cautious* (better to over-blur a borderline photo on a public
real-estate board than to leak something explicit), but every knob is
overridable in code or from environment variables prefixed ``SAFETY_``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import os

from .types import Category, Severity


# --- NudeNet label -> (category, severity) ----------------------------------
# The detector emits 18 labels. We collapse them to our normalized taxonomy.
# Exposed sexual/intimate parts are HIGH severity (always blur); exposed
# secondary areas are MEDIUM; covered intimate areas are LOW (suggestive);
# faces / feet / armpits / belly are ignored entirely.
NUDENET_LABEL_MAP: dict[str, tuple[Category, Severity]] = {
    "FEMALE_GENITALIA_EXPOSED": (Category.NUDITY, Severity.HIGH),
    "MALE_GENITALIA_EXPOSED": (Category.NUDITY, Severity.HIGH),
    "ANUS_EXPOSED": (Category.NUDITY, Severity.HIGH),
    "FEMALE_BREAST_EXPOSED": (Category.NUDITY, Severity.HIGH),
    "BUTTOCKS_EXPOSED": (Category.NUDITY, Severity.MEDIUM),
    "MALE_BREAST_EXPOSED": (Category.SUGGESTIVE, Severity.LOW),
    "FEMALE_GENITALIA_COVERED": (Category.SUGGESTIVE, Severity.LOW),
    "ANUS_COVERED": (Category.SUGGESTIVE, Severity.LOW),
    "FEMALE_BREAST_COVERED": (Category.SUGGESTIVE, Severity.LOW),
    "BUTTOCKS_COVERED": (Category.SUGGESTIVE, Severity.LOW),
    "BELLY_EXPOSED": (Category.SUGGESTIVE, Severity.LOW),
    # Deliberately ignored (kept here so the mapping is explicit/auditable):
    # FACE_FEMALE, FACE_MALE, FEET_EXPOSED, FEET_COVERED, ARMPITS_EXPOSED,
    # ARMPITS_COVERED, BELLY_COVERED -> not sensitive, no entry == ignored.
}


@dataclass
class SafetyConfig:
    # --- Which detectors to run ---------------------------------------------
    use_nudenet: bool = True       # box-level nudity (needs `nudenet` extra)
    use_wound_heuristic: bool = True   # offline blood/wound heuristic
    use_classifier: bool = False   # fine-tuned whole-image model (needs a ckpt)
    classifier_path: str = ""      # path to a .onnx / .pt checkpoint

    # --- Per-category score thresholds (0-1). A detection at or above the
    #     threshold is eligible for redaction. ---------------------------------
    nudity_threshold: float = 0.35
    suggestive_threshold: float = 0.55
    gore_threshold: float = 0.55

    # The minimum severity that triggers a blur. Set to LOW to also blur
    # "suggestive" (covered) regions; MEDIUM to only blur clearer exposure;
    # HIGH to only redact unambiguous nudity.
    min_blur_severity: Severity = Severity.MEDIUM

    # If a whole-image classifier reports nsfw/gore above this, blur the whole
    # frame even when no box was localized.
    whole_image_threshold: float = 0.80

    # --- Redaction style -----------------------------------------------------
    blur_style: str = "gaussian"   # gaussian | pixelate | box | fill
    blur_strength: float = 1.0     # multiplier on the auto-computed blur kernel
    region_margin: float = 0.12    # grow each box by this fraction before blur
    feather: bool = True           # soft-edge the blurred region into the image
    fill_color: tuple[int, int, int] = (0, 0, 0)   # for blur_style="fill"

    # When True, a SINGLE detection anywhere in the frame blurs the WHOLE image
    # (not just the offending box). This is the safe default for a moderation
    # filter: one missed pixel of a localized box shouldn't leak the rest.
    whole_image_on_detection: bool = True

    # When True, the whole-image blur is "solid" — the frame is crushed to a
    # tiny mosaic and smeared so the original content is unrecognizable, not
    # merely softened. Set False for a lighter, still-legible blur.
    solid_blur: bool = True
    # Long-side macro-block count for the solid blur; FEWER blocks = more solid
    # (≈unreadable at 6). `blur_strength` scales this down further.
    solid_blur_blocks: int = 6

    # NudeNet raw confidence floor; below this we don't even consider a box.
    nudenet_min_confidence: float = 0.25

    # Categories that are allowed to trigger a *whole-image* blur via classifier.
    whole_image_categories: tuple[Category, ...] = (Category.NUDITY, Category.GORE)

    def threshold_for(self, category: Category) -> float:
        return {
            Category.NUDITY: self.nudity_threshold,
            Category.SUGGESTIVE: self.suggestive_threshold,
            Category.GORE: self.gore_threshold,
        }.get(category, 1.0)

    # --- Env loading ---------------------------------------------------------
    @classmethod
    def from_env(cls, prefix: str = "SAFETY_") -> "SafetyConfig":
        """Build a config, overriding any field from ``SAFETY_<FIELD>`` env vars.

        Booleans accept ``1/0/true/false/yes/no``; ``min_blur_severity`` accepts
        ``none/low/medium/high``; numbers are parsed as floats.
        """
        cfg = cls()
        for f in fields(cls):
            env_key = prefix + f.name.upper()
            if env_key not in os.environ:
                continue
            raw = os.environ[env_key]
            try:
                setattr(cfg, f.name, _coerce(f.name, f.type, raw))
            except Exception:
                pass  # ignore malformed overrides, keep the default
        return cfg


def _coerce(name: str, ftype, raw: str):
    if name == "min_blur_severity":
        return Severity[raw.strip().upper()]
    if "bool" in str(ftype):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if "float" in str(ftype):
        return float(raw)
    if "tuple" in str(ftype):  # e.g. "0,0,0"
        return tuple(int(x) for x in raw.split(","))
    return raw


DEFAULT_CONFIG = SafetyConfig()
