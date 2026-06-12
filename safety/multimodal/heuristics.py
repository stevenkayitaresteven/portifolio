"""Offline heuristic detectors: PII regexes, category phrase lexicons, spam
signals. These run on every message regardless of whether the transformer
models are available — they are the floor, the models raise the ceiling.

Like the profanity lexicon, the phrases below exist so they can be *caught*.
Each list is deliberately modest and high-precision: heuristics block, so a
borderline phrase belongs in the fine-tuned model (see
``safety/train/finetune_text.py``), not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import taxonomy as T


# =============================== PII (privacy) ===============================
# Each pattern gets a masker that keeps just enough context to recognize that
# *something* was redacted (like f*** for curse words).

def _keep_edges(s: str, head: int = 2, tail: int = 0) -> str:
    body = "*" * max(1, len(s) - head - tail)
    return s[:head] + body + (s[-tail:] if tail else "")


def _mask_email(m: re.Match) -> str:
    local, _, domain = m.group(0).partition("@")
    return _keep_edges(local, 1) + "@" + domain


def _mask_digits(m: re.Match) -> str:
    s = m.group(0)
    masked = "".join("*" if c.isdigit() else c for c in s[:-4]) + s[-4:]
    return masked


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class PIIPattern:
    name: str
    regex: re.Pattern
    masker: object  # callable(re.Match) -> str
    validate: object = None  # optional callable(str) -> bool


PII_PATTERNS: tuple[PIIPattern, ...] = (
    PIIPattern("email",
               re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
               _mask_email),
    PIIPattern("credit_card",
               re.compile(r"\b(?:\d[ -]?){13,19}\b"),
               _mask_digits,
               lambda s: _luhn_ok(re.sub(r"\D", "", s))
               and 13 <= len(re.sub(r"\D", "", s)) <= 19),
    PIIPattern("ssn",
               re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
               _mask_digits),
    PIIPattern("phone",
               re.compile(r"(?:(?<=\s)|^)\+?\d[\d\s().-]{8,16}\d\b"),
               _mask_digits,
               lambda s: 9 <= len(re.sub(r"\D", "", s)) <= 15),
    PIIPattern("ipv4",
               re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
               lambda m: "*.*.*." + m.group(0).rsplit(".", 1)[1],
               lambda s: all(0 <= int(p) <= 255 for p in s.split("."))),
    PIIPattern("iban",
               re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
               lambda m: _keep_edges(m.group(0), 4)),
)


def find_pii(text: str) -> list[tuple[str, str]]:
    """Return ``(kind, matched_text)`` for every PII hit."""
    out: list[tuple[str, str]] = []
    for pat in PII_PATTERNS:
        for m in pat.regex.finditer(text):
            if pat.validate is not None and not pat.validate(m.group(0)):
                continue
            out.append((pat.name, m.group(0)))
    return out


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Redact PII in place; returns ``(masked_text, kinds_found)``."""
    kinds: list[str] = []
    for pat in PII_PATTERNS:
        def _sub(m: re.Match, _p=pat):
            if _p.validate is not None and not _p.validate(m.group(0)):
                return m.group(0)
            kinds.append(_p.name)
            return _p.masker(m)
        text = pat.regex.sub(_sub, text)
    return text, kinds


# ====================== Category phrase lexicons (block) =====================
# (category, score) per phrase; matched on lowercased, whitespace-collapsed
# text. Substring match is intentional ("kill yourself" inside a sentence).

_PHRASES: tuple[tuple[str, str, float], ...] = (
    # --- self-harm / suicide references ---
    ("kill myself", T.SELF_HARM, 0.9), ("end my life", T.SELF_HARM, 0.9),
    ("want to die", T.SELF_HARM, 0.8), ("suicide", T.SELF_HARM, 0.7),
    ("self harm", T.SELF_HARM, 0.8), ("self-harm", T.SELF_HARM, 0.8),
    ("cut myself", T.SELF_HARM, 0.85), ("kill yourself", T.SELF_HARM, 0.9),
    ("kys", T.SELF_HARM, 0.8),
    # --- violence / threats ---
    ("i will kill", T.VIOLENCE, 0.9), ("going to kill you", T.VIOLENCE, 0.9),
    ("kill you", T.VIOLENCE, 0.8), ("shoot you", T.VIOLENCE, 0.85),
    ("stab you", T.VIOLENCE, 0.85), ("beat you up", T.VIOLENCE, 0.8),
    ("hurt you badly", T.VIOLENCE, 0.8), ("burn your house", T.VIOLENCE, 0.85),
    ("break your legs", T.VIOLENCE, 0.8),
    # --- criminal: fraud / illegal trade ---
    ("stolen credit card", T.CRIMINAL, 0.9), ("stolen card numbers", T.CRIMINAL, 0.9),
    ("counterfeit money", T.CRIMINAL, 0.9), ("counterfeit cash", T.CRIMINAL, 0.9),
    ("fake id", T.CRIMINAL, 0.8), ("fake passport", T.CRIMINAL, 0.85),
    ("money laundering", T.CRIMINAL, 0.8), ("launder money", T.CRIMINAL, 0.85),
    ("buy drugs", T.CRIMINAL, 0.8), ("sell drugs", T.CRIMINAL, 0.8),
    ("hire a hitman", T.CRIMINAL, 0.95), ("hitman for hire", T.CRIMINAL, 0.95),
    ("insurance fraud", T.CRIMINAL, 0.8), ("ponzi scheme", T.CRIMINAL, 0.75),
    # --- cybersecurity: malware / attacks ---
    ("ransomware", T.CYBERSECURITY, 0.8), ("keylogger", T.CYBERSECURITY, 0.8),
    ("malware", T.CYBERSECURITY, 0.75), ("spyware", T.CYBERSECURITY, 0.75),
    ("botnet", T.CYBERSECURITY, 0.8), ("ddos attack", T.CYBERSECURITY, 0.85),
    ("ddos them", T.CYBERSECURITY, 0.85), ("sql injection", T.CYBERSECURITY, 0.75),
    ("steal passwords", T.CYBERSECURITY, 0.85),
    ("crack her password", T.CYBERSECURITY, 0.85),
    ("crack his password", T.CYBERSECURITY, 0.85),
    ("phishing kit", T.CYBERSECURITY, 0.9),
    ("remote access trojan", T.CYBERSECURITY, 0.85),
    ("exploit kit", T.CYBERSECURITY, 0.85),
    # --- extremism: terrorist propaganda / recruitment ---
    ("join isis", T.EXTREMISM, 0.95), ("join the jihad", T.EXTREMISM, 0.95),
    ("terror attack", T.EXTREMISM, 0.8), ("terrorist attack", T.EXTREMISM, 0.8),
    ("how to make a bomb", T.EXTREMISM, 0.95), ("bomb making", T.EXTREMISM, 0.9),
    ("martyrdom operation", T.EXTREMISM, 0.9),
    ("ethnic cleansing", T.EXTREMISM, 0.9), ("race war", T.EXTREMISM, 0.85),
    ("white power", T.EXTREMISM, 0.8),
    # --- misinformation: canonical debunked claims (flag only) ---
    ("vaccines cause autism", T.MISINFORMATION, 0.8),
    ("the earth is flat", T.MISINFORMATION, 0.8),
    ("earth is flat", T.MISINFORMATION, 0.75),
    ("5g causes covid", T.MISINFORMATION, 0.85),
    ("5g caused covid", T.MISINFORMATION, 0.85),
    ("miracle cure", T.MISINFORMATION, 0.6),
    ("doctors don't want you to know", T.MISINFORMATION, 0.7),
    ("doctors hate this", T.MISINFORMATION, 0.6),
)

_WS_RE = re.compile(r"\s+")


def find_phrase_categories(text: str) -> list[tuple[str, str, float]]:
    """Return ``(category, phrase, score)`` for every lexicon phrase present."""
    norm = _WS_RE.sub(" ", text.lower())
    padded = f" {norm} "
    hits: list[tuple[str, str, float]] = []
    for phrase, category, score in _PHRASES:
        needle = phrase if " " in phrase else f" {phrase} "
        if needle in (norm if " " in phrase else padded):
            hits.append((category, phrase, score))
    return hits


# ================================ Spam signals ===============================

_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
_PROMO_PHRASES = (
    "click here", "free money", "limited time offer", "act now",
    "you have won", "claim your prize", "congratulations you won",
    "double your bitcoin", "crypto giveaway", "guaranteed returns",
    "make money fast", "work from home and earn", "100% free",
    "verify your account", "account will be suspended", "urgent response",
    "wire transfer immediately", "send a gift card",
)

# --- URL risk: phishing-shaped links -----------------------------------------
_SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
               "rb.gy", "cutt.ly", "shorturl.at")
_RISKY_TLDS = (".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".zip", ".mov",
               ".click", ".loan", ".country")
_IP_URL_RE = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}", re.I)


def url_risk(text: str) -> tuple[float, list[str]]:
    """Phishing-shaped link analysis: IP-literal hosts, punycode, risky TLDs,
    shorteners combined with credential-pressure wording."""
    urls = _URL_RE.findall(text)
    if not urls:
        return 0.0, []
    score = 0.0
    reasons: list[str] = []
    lower = text.lower()
    if _IP_URL_RE.search(text):
        score += 0.6
        reasons.append("link to a raw IP address")
    if any("xn--" in u.lower() for u in urls):
        score += 0.5
        reasons.append("punycode (look-alike) domain")
    def _host(u: str) -> str:
        u = u.lower().split("://", 1)[-1]
        return u.split("/", 1)[0].split("?", 1)[0]

    if any(_host(u).endswith(tld) for u in urls for tld in _RISKY_TLDS):
        score += 0.4
        reasons.append("high-abuse TLD link")
    if any(s in u.lower() for u in urls for s in _SHORTENERS):
        score += 0.25
        reasons.append("URL shortener hides the destination")
    if any(w in lower for w in ("password", "login", "verify", "ssn",
                                "account locked", "confirm your identity")):
        score += 0.3
        reasons.append("credential-pressure wording around a link")
    return min(score, 1.0), reasons


def spam_signals(text: str) -> tuple[float, list[str]]:
    """Score 0-1 plus the reasons. URLs alone are normal; URLs + promo
    language, lots of links, or phishing-style pressure raise the score."""
    norm = _WS_RE.sub(" ", text.lower())
    urls = _URL_RE.findall(text)
    promos = [p for p in _PROMO_PHRASES if p in norm]
    reasons: list[str] = []
    score = 0.0
    if promos:
        score += 0.35 + 0.15 * (len(promos) - 1)
        reasons.append("promotional phrasing: " + ", ".join(promos[:3]))
    if urls and promos:
        score += 0.3
        reasons.append(f"{len(urls)} link(s) with promotional phrasing")
    if len(urls) >= 4:
        score += 0.4
        reasons.append(f"{len(urls)} links")
    caps = sum(1 for w in text.split() if len(w) > 3 and w.isupper())
    if caps >= 3:
        score += 0.15
        reasons.append(f"{caps} shouted words")
    return min(score, 1.0), reasons
