"""Text moderation: the offline lexicon layer (no HF models, no network)."""
import pytest

from safety.multimodal import MultimodalConfig, TextModerator
from safety.multimodal.wordlist import find_profanity, mask_profanity


def offline_config() -> MultimodalConfig:
    cfg = MultimodalConfig()
    cfg.use_hf_text = cfg.use_hf_image = cfg.use_hf_audio = False
    return cfg


# --- lexicon -------------------------------------------------------------------

def test_clean_text_has_no_hits():
    assert find_profanity("good morning, the meeting is at nine") == []


@pytest.mark.parametrize("text", [
    "what the fuck",          # plain
    "what the FUCK",          # case
    "what the fuuuuck",       # stretched letters
    "what the f@ck",          # symbol leetspeak
    "what the f*ck",          # self-censored wildcard
    "sh1t happens",           # digit leetspeak
])
def test_evasions_are_caught(text):
    assert find_profanity(text), text


def test_masking_keeps_first_letter_and_length():
    censored, words = mask_profanity("oh shit, sorry")
    assert censored == "oh s***, sorry"
    assert words == ["shit"]


def test_masking_multiple_words_preserves_clean_parts():
    censored, words = mask_profanity("you bitch, give the damn book back")
    assert "b****" in censored
    assert "give the damn book back" in censored  # mild words pass by default
    assert words == ["bitch"]


def test_strict_mode_also_masks_mild_words():
    censored, words = mask_profanity("damn it", strict=True)
    assert words == ["damn"]
    assert censored.startswith("d***")


def test_innocent_words_are_not_substring_matched():
    # "class", "assistant", "Scunthorpe"-style words must not trip the filter.
    text = "the class assistant passed the assessment in Scunthorpe"
    assert find_profanity(text) == []


# --- moderator -------------------------------------------------------------------

def test_moderator_clean_text():
    res = TextModerator(offline_config()).moderate("see you at lunch!")
    assert not res.flagged
    assert res.action == "none"
    assert res.censored_text == "see you at lunch!"
    assert res.confidence == 1.0


def test_moderator_masks_and_flags():
    res = TextModerator(offline_config()).moderate("this is fucking great")
    assert res.flagged
    assert res.action == "mask"
    assert "f******" in res.censored_text
    assert "profanity" in res.categories
    assert res.scores["profanity"] == 1.0
    assert "lexicon" in res.detectors


def test_moderator_block_mode():
    cfg = offline_config()
    cfg.deliver_flagged_text = False
    res = TextModerator(cfg).moderate("you asshole")
    assert res.flagged and res.action == "block"


def test_result_round_trips_to_json():
    res = TextModerator(offline_config()).moderate("bullshit")
    d = res.to_dict()
    assert d["modality"] == "text"
    assert d["flagged"] is True
    assert isinstance(res.to_json(), str)
