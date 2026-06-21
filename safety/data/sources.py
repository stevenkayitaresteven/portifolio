"""Ingest connectors — where raw records come from.

Each connector is a generator of ``(text, [raw_labels])`` tuples so the rest of
the pipeline doesn't care whether a row came from a CSV, the Hugging Face Hub,
or a web page.

A note on the web crawler, since the brief asked for "scrape and crawl":
:class:`WebTextCrawler` is deliberately tame. It only fetches a URL **allowlist
you provide**, it **obeys ``robots.txt``**, it rate-limits, and it sends an
honest User-Agent. It exists to gather *benign* everyday text (hard negatives
that cut false positives) — never to harvest harmful material. For toxicity,
hate, extremism, or anything child-safety-related, use the vetted public
datasets wired into :mod:`safety.train.finetune_text`; crawling the open web for
that content is unsafe and, for some categories, unlawful.
"""
from __future__ import annotations

import csv
import json
import time
from html.parser import HTMLParser
from urllib import robotparser
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def iter_jsonl(path: str, text_field: str = "text", label_field: str = ""):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get(text_field, "") or "")
            labels = _row_labels(row, label_field)
            if text:
                yield text, labels


def iter_csv(path: str, text_field: str = "text", label_field: str = ""):
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = str(row.get(text_field, "") or "")
            labels = _row_labels(row, label_field)
            if text:
                yield text, labels


def iter_hf(dataset_id: str, split: str = "train", text_field: str = "text",
            label_field: str = "", max_records: int = 0):
    """Stream a Hugging Face dataset (needs ``pip install datasets``)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("pip install datasets to use hf sources") from exc
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for i, row in enumerate(ds):
        if max_records and i >= max_records:
            break
        text = str(row.get(text_field, "") or "")
        labels = _row_labels(row, label_field)
        if text:
            yield text, labels


def _row_labels(row: dict, label_field: str) -> list[str]:
    if not label_field:
        return []
    val = row.get(label_field)
    if val is None or val == "":
        return []
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val]
    # Multi-label columns are sometimes "hate;violence" or "hate,violence".
    return [p for p in str(val).replace(";", ",").split(",") if p.strip()]


class _TextExtractor(HTMLParser):
    """Strip tags, scripts, and styles; keep visible text."""
    _SKIP = {"script", "style", "noscript", "template", "head"}

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self._chunks.append(data.strip())

    @property
    def text(self) -> str:
        return "\n".join(self._chunks)


class WebTextCrawler:
    """Polite, ``robots.txt``-respecting fetcher for an allowlist of URLs."""

    USER_AGENT = "SentinelDataBot/1.0 (+content-moderation-research)"

    def __init__(self, *, delay: float = 1.0, timeout: float = 10.0,
                 min_chars: int = 40):
        self.delay = delay
        self.timeout = timeout
        self.min_chars = min_chars
        self._robots: dict[str, robotparser.RobotFileParser] = {}

    def _allowed(self, url: str) -> bool:
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        rp = self._robots.get(root)
        if rp is None:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{root}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None  # no robots reachable -> be conservative below
            self._robots[root] = rp
        return rp.can_fetch(self.USER_AGENT, url) if rp else False

    def fetch(self, url: str) -> str | None:
        if not self._allowed(url):
            return None
        req = Request(url, headers={"User-Agent": self.USER_AGENT})
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                if "html" not in resp.headers.get("Content-Type", "html"):
                    return None
                raw = resp.read(2_000_000).decode("utf-8", "replace")
        except Exception:
            return None
        parser = _TextExtractor()
        parser.feed(raw)
        text = parser.text
        return text if len(text) >= self.min_chars else None

    def crawl(self, urls: list[str]):
        """Yield ``(text, [])`` for each fetchable URL, pausing between hits."""
        for i, url in enumerate(urls):
            if i:
                time.sleep(self.delay)
            text = self.fetch(url.strip())
            if text:
                yield text, []
