"""The canonical 12-category sensitive-content taxonomy.

Every detector — HF classifier, lexicon, regex, file sniffer — speaks its own
label language; this module normalizes all of them into one set of categories
with one default action each, so the policy stays in a single auditable place.

============== ========================== =====================================
 Category        Examples                   Default action (text)
============== ========================== =====================================
 toxic           insults, harassment        mask curse words, deliver
 hate            protected-group attacks    block
 sexual          NSFW content               block text / blur images
 violence        threats, harm              block
 self_harm       suicide references         block + show support note
 criminal        fraud, illegal activity    block
 cybersecurity   malware, attacks           block
 spam            promotions, phishing       flag (deliver with warning)
 privacy         PII exposure               mask the PII, deliver
 extremism       terrorist propaganda       block
 misinformation  false claims               flag (deliver with warning)
 child_safety    exploitation risks         block
============== ========================== =====================================

Actions are overridable per category via
:attr:`~safety.multimodal.config.MultimodalConfig.action_overrides`.
"""
from __future__ import annotations

from .result import (ACTION_BLOCK, ACTION_FLAG, ACTION_MASK, ACTION_NONE)

# Canonical category names (stable API — these appear in ModerationResult).
TOXIC = "toxic"
HATE = "hate"
SEXUAL = "sexual"
VIOLENCE = "violence"
SELF_HARM = "self_harm"
CRIMINAL = "criminal"
CYBERSECURITY = "cybersecurity"
SPAM = "spam"
PRIVACY = "privacy"
EXTREMISM = "extremism"
MISINFORMATION = "misinformation"
CHILD_SAFETY = "child_safety"

CATEGORIES: tuple[str, ...] = (
    TOXIC, HATE, SEXUAL, VIOLENCE, SELF_HARM, CRIMINAL, CYBERSECURITY,
    SPAM, PRIVACY, EXTREMISM, MISINFORMATION, CHILD_SAFETY,
)

# What to do when a category fires (most-severe category wins; see
# ACTION_SEVERITY). "Deliver but warn" categories stay visible because hiding
# spam/misinfo entirely teaches senders nothing and hides context from mods.
DEFAULT_ACTIONS: dict[str, str] = {
    TOXIC: ACTION_MASK,
    HATE: ACTION_BLOCK,
    SEXUAL: ACTION_BLOCK,
    VIOLENCE: ACTION_BLOCK,
    SELF_HARM: ACTION_BLOCK,
    CRIMINAL: ACTION_BLOCK,
    CYBERSECURITY: ACTION_BLOCK,
    SPAM: ACTION_FLAG,
    PRIVACY: ACTION_MASK,
    EXTREMISM: ACTION_BLOCK,
    MISINFORMATION: ACTION_FLAG,
    CHILD_SAFETY: ACTION_BLOCK,
}

# When several categories fire, the message takes the most severe action.
ACTION_SEVERITY: dict[str, int] = {
    ACTION_NONE: 0, ACTION_FLAG: 1, ACTION_MASK: 2, "mute": 3, "blur": 3,
    ACTION_BLOCK: 4,
}


def strongest_action(actions: "list[str]") -> str:
    return max(actions, key=lambda a: ACTION_SEVERITY.get(a, 0), default=ACTION_NONE)


# --- Model label -> category -------------------------------------------------------
# Per-model maps for the pretrained checkpoints we ship support for. A label
# mapped to None is an explicit "ignore" (e.g. the models' own OK/neutral
# labels). Unknown models fall back to _GENERIC.
MODEL_LABEL_MAPS: dict[str, dict[str, str | None]] = {
    # DeBERTa fine-tuned on the OpenAI moderation taxonomy.
    "KoalaAI/Text-Moderation": {
        "s": SEXUAL, "h": HATE, "v": VIOLENCE, "hr": TOXIC, "sh": SELF_HARM,
        "s3": CHILD_SAFETY, "h2": HATE, "v2": VIOLENCE, "ok": None,
    },
    # BERT fine-tuned on the Jigsaw toxic-comment challenges.
    "unitary/toxic-bert": {
        "toxic": TOXIC, "severe_toxic": TOXIC, "obscene": TOXIC,
        "insult": TOXIC, "threat": VIOLENCE, "identity_hate": HATE,
    },
    "facebook/roberta-hate-speech-dynabench-r4-target": {
        "hate": HATE, "nothate": None,
    },
    "sentinet/suicidality": {"label_1": SELF_HARM, "label_0": None},
    "mshenoda/roberta-spam": {"label_1": SPAM, "spam": SPAM,
                              "label_0": None, "ham": None},
    "ealvaradob/bert-finetuned-phishing": {"phishing": SPAM, "benign": None},
}

# Fallback for custom / fine-tuned checkpoints: label names that already use
# (or contain) our canonical category names map onto them; common aliases too.
_GENERIC: dict[str, str | None] = {
    **{c: c for c in CATEGORIES},
    "harassment": TOXIC, "insult": TOXIC, "obscene": TOXIC, "toxicity": TOXIC,
    "hate_speech": HATE, "hateful": HATE, "identity_attack": HATE,
    "identity_hate": HATE,
    "nsfw": SEXUAL, "sexual_explicit": SEXUAL, "sexual/minors": CHILD_SAFETY,
    "threat": VIOLENCE, "violence/graphic": VIOLENCE,
    "suicide": SELF_HARM, "self-harm": SELF_HARM,
    "phishing": SPAM, "scam": SPAM,
    "pii": PRIVACY, "terrorism": EXTREMISM, "radicalization": EXTREMISM,
    "fake_news": MISINFORMATION, "fake": MISINFORMATION,
    "csam": CHILD_SAFETY, "minors": CHILD_SAFETY,
    "ok": None, "safe": None, "neutral": None, "none": None, "benign": None,
    "ham": None, "real": None, "nothate": None, "not_toxic": None,
}


def category_for_label(model_id: str, label: str) -> str | None:
    """Map a raw model label to a canonical category (None = ignore)."""
    label = label.strip().lower()
    per_model = MODEL_LABEL_MAPS.get(model_id)
    if per_model is not None and label in per_model:
        return per_model[label]
    if label in _GENERIC:
        return _GENERIC[label]
    # Substring rescue for verbose labels like "LABEL_violence" / "Hate Speech".
    for key, cat in _GENERIC.items():
        if key and cat and key in label:
            return cat
    return None
