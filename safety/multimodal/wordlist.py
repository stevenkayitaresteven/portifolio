"""Offline profanity lexicon: find and mask curse words in text.

This is the zero-dependency fallback (and word-localizer) behind
:class:`~safety.multimodal.text.TextModerator`. The Hugging Face toxicity model
scores a *whole* message; this lexicon is what lets us point at — and mask —
the individual offending words, WhatsApp-style: ``f***``.

Matching is deliberately forgiving about the usual evasions: case, simple
leetspeak (``sh1t``, ``f@ck``), censor characters typed by the user (``f*ck``,
``f#ck``) and repeated letters (``fuuuck``). It is a *moderation* lexicon —
the words below exist in this file so they can be caught, not endorsed.
"""
from __future__ import annotations

import re

# Common English profanity / slurs, stored normalized (lowercase, no repeats).
# Kept to widely-flagged terms; extend via `extra_words` for your community.
PROFANITY: frozenset[str] = frozenset({
    "fuck", "fucker", "fucking", "motherfucker", "fck", "fuk",
    "shit", "shite", "bullshit", "shitty",
    "bitch", "bitches", "bastard",
    "asshole", "arsehole", "ass", "arse",
    "cunt", "dick", "dickhead", "cock", "pussy", "twat", "wanker",
    "slut", "whore", "hoe",
    "nigger", "nigga", "faggot", "fag", "retard", "retarded",
    "damn", "goddamn", "crap", "piss", "prick", "douchebag", "jackass",
    "bollocks", "bugger", "tit", "tits", "boobs",
    "porn", "porno", "blowjob", "handjob", "cum", "jizz", "dildo",
})

# Mild terms that are masked only when you opt into strict mode.
MILD: frozenset[str] = frozenset({"damn", "crap", "hell", "piss", "bugger"})

# Leetspeak / symbol substitutions applied before lookup.
_LEET = str.maketrans({
    "@": "a", "4": "a", "$": "s", "5": "s", "0": "o", "1": "i", "!": "i",
    "3": "e", "7": "t", "+": "t", "8": "b", "9": "g", "€": "e", "£": "l",
})

# A token is letters/digits plus the symbols people use to self-censor.
# (Underscore is a word char but also a common joiner, so it splits tokens.)
_TOKEN_RE = re.compile(r"[A-Za-z0-9@$!#*+€£&%]+")

# Non-letter symbols a token may carry; trailing runs of these are punctuation
# ("fuck!!!"), not letter substitutions, so they're stripped before lookup.
_SYMBOLS = "@$!#*+%&€£"

# Censor characters that may stand in for ANY letter (f*ck, s#it, f@ck).
_WILDCARDS = "*#%&@$!"


def _squeeze(word: str) -> str:
    """Collapse runs of the same character: fuuuck -> fuck."""
    return re.sub(r"(.)\1+", r"\1", word)


def _normalize(token: str) -> str:
    return token.lower().translate(_LEET)


def _variants(token: str) -> set[str]:
    """Normalized spellings a token might be hiding behind."""
    out: set[str] = set()
    for cand in {token, token.rstrip(_SYMBOLS)}:
        if not cand:
            continue
        norm = _normalize(cand)
        for base in {norm, _squeeze(norm)}:
            out.add(base)
            # Strip common suffixes so "fuckers" / "shitting" hit the stem.
            for suffix in ("s", "es", "er", "ers", "ing", "ed", "y"):
                if base.endswith(suffix) and len(base) - len(suffix) >= 3:
                    out.add(base[: -len(suffix)])
    return out


def _wildcard_match(token: str, lexicon: frozenset[str] | set[str]) -> bool:
    """True if a self-censored token (f*ck, f@ck) matches a lexicon word."""
    core = token.rstrip(_SYMBOLS)  # trailing symbols are punctuation, not letters
    if not any(c in core for c in _WILDCARDS):
        return False
    norm = _normalize("".join("." if c in _WILDCARDS else c for c in core))
    dots = norm.count(".")
    if len(norm) < 3 or dots == 0 or dots > len(norm) // 2 or norm[0] == ".":
        return False  # too short/vague to attribute, or no letter to anchor on
    pattern = re.compile(rf"^{re.escape(norm).replace(chr(92) + '.', '.')}$")
    return any(len(w) == len(norm) and pattern.match(w) for w in lexicon)


def find_profanity(text: str, *, strict: bool = False,
                   extra_words: set[str] | None = None) -> list[tuple[int, int, str]]:
    """Return ``(start, end, matched_word)`` spans of profanity in ``text``."""
    lexicon = set(PROFANITY) | (extra_words or set())
    if not strict:
        lexicon -= MILD
    hits: list[tuple[int, int, str]] = []
    for m in _TOKEN_RE.finditer(text):
        token = m.group(0)
        if any(v in lexicon for v in _variants(token)) \
                or _wildcard_match(token, lexicon):
            hits.append((m.start(), m.end(), token))
    return hits


def mask_profanity(text: str, *, strict: bool = False, keep: int = 1,
                   extra_words: set[str] | None = None) -> tuple[str, list[str]]:
    """Return ``(censored_text, matched_words)``.

    Each curse word keeps its first ``keep`` character(s) and the rest becomes
    ``*`` — e.g. ``fuck`` -> ``f***`` — so readers know *something* was cut
    without seeing what.
    """
    hits = find_profanity(text, strict=strict, extra_words=extra_words)
    if not hits:
        return text, []
    out: list[str] = []
    last = 0
    words: list[str] = []
    for start, end, token in hits:
        out.append(text[last:start])
        out.append(token[:keep] + "*" * max(1, len(token) - keep))
        words.append(token)
        last = end
    out.append(text[last:])
    return "".join(out), words
