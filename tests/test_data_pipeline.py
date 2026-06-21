"""Tests for the data-prep pipeline (offline, no heavy deps)."""
import json

from safety.data import Deduplicator, TextCleaner, weak_label
from safety.data.config import PipelineConfig, SourceSpec
from safety.data.label import LabelMapper, to_vector
from safety.data.pipeline import DataPipeline


def test_cleaner_scrubs_pii_and_urls_but_keeps_toxicity():
    c = TextCleaner()
    out = c.clean("Email me at bob@corp.io  http://x.com you idiot")
    assert "bob@corp.io" not in out
    assert "http://x.com" not in out and "<url>" in out
    assert "idiot" in out  # the signal must survive


def test_cleaner_drops_too_short_and_normalizes_unicode():
    c = TextCleaner(min_chars=3)
    assert c.clean("  ") is None
    # fullwidth chars fold to ascii via NFKC
    assert c.clean("ｈｅｌｌｏ").lower() == "hello"


def test_dedup_exact_and_near():
    d = Deduplicator(near=True, threshold=0.7)
    assert d.is_duplicate("the quick brown fox jumps") is False
    assert d.is_duplicate("the quick brown fox jumps") is True          # exact
    assert d.is_duplicate("THE quick brown fox  jumps!") is True        # norm-exact
    assert d.is_duplicate("the quick brown fox jumps over") is True     # near
    assert d.is_duplicate("completely unrelated sentence here") is False


def test_weak_label_detects_categories():
    assert "toxic" in weak_label("you are an asshole")
    assert "violence" in weak_label("i will kill you")
    assert "privacy" in weak_label("my ssn is 123-45-6789")
    assert weak_label("let's grab lunch tomorrow") == []


def test_label_mapper_and_vector():
    m = LabelMapper({"off": "toxic", "identity_hate": "hate"})
    assert m.map("OFF") == "toxic"
    assert m.map("Identity Hate") == "hate"
    assert m.map("nonsense") is None
    vec = m.vector(["off", "identity_hate"])
    assert sum(vec) == 2
    assert to_vector(["toxic"]).count(1) == 1


def test_pipeline_end_to_end(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = ["hello there friend", "hello there friend",  # dup
            "you are an asshole", "i will kill you",
            "nice weather today", "great game last night"]
    raw.write_text("\n".join(json.dumps({"text": r}) for r in rows))

    cfg = PipelineConfig(
        sources=[SourceSpec(kind="jsonl", path=str(raw), weak_label=True)],
        max_clean_ratio=0, val_ratio=0.0, test_ratio=0.0)  # no clean capping
    card = DataPipeline(cfg).run(str(tmp_path / "out"))

    assert card["total"] == 5            # one exact duplicate removed
    assert card["flagged"] == 2          # asshole -> toxic, kill -> violence
    assert card["category_counts"]["toxic"] == 1
    assert card["category_counts"]["violence"] == 1
    # train.jsonl is valid and carries multi-hot label vectors
    lines = (tmp_path / "out" / "train.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert len(rec["labels"]) == 12 and "text" in rec
