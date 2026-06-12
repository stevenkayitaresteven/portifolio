"""Dataset presets for fine-tuning: label mapping per corpus, mixing,
balancing. Pure functions — no model download, fully offline."""
import pytest

from safety.multimodal import taxonomy as T
from safety.train.finetune_text import (
    Example, PRESETS, balance_clean, rows_to_examples,
    _civil_comments_labeler, _cyberbullying_labeler, _davidson_labeler,
    _goemotions_labeler, _hatexplain_labeler, _hatexplain_text,
    _olid_labeler, _textdetox_labeler, _toxigen_labeler)


def test_civil_comments_thresholds_and_category_split():
    row = {"text": "x", "toxicity": 0.9, "identity_attack": 0.7,
           "threat": 0.1, "sexual_explicit": 0.0}
    assert _civil_comments_labeler(row) == {T.TOXIC, T.HATE}
    assert _civil_comments_labeler({"text": "x", "threat": 0.8}) == {T.VIOLENCE}
    assert _civil_comments_labeler({"text": "x", "toxicity": 0.2}) == set()


def test_davidson_class_labels():
    assert _davidson_labeler({"class": 0}) == {T.HATE, T.TOXIC}
    assert _davidson_labeler({"class": 1}) == {T.TOXIC}
    assert _davidson_labeler({"class": 2}) == set()


def test_hatexplain_majority_vote_and_token_text():
    row = {"annotators": {"label": [0, 0, 2]},
           "post_tokens": ["you", "people", "disgust", "me"]}
    assert _hatexplain_labeler(row) == {T.HATE, T.TOXIC}     # majority 0 = hate
    assert _hatexplain_text(row) == "you people disgust me"
    assert _hatexplain_labeler({"annotators": {"label": [1, 1, 0]}}) == set()


def test_toxigen_human_score_threshold():
    assert _toxigen_labeler({"toxicity_human": 4.3}) == {T.HATE, T.TOXIC}
    assert _toxigen_labeler({"toxicity_human": 2.0}) == set()


def test_textdetox_binary():
    assert _textdetox_labeler({"toxic": 1}) == {T.TOXIC}
    assert _textdetox_labeler({"toxic": 0}) == set()


def test_olid_handles_olid_and_solid_schemas():
    assert _olid_labeler({"subtask_a": "OFF"}) == {T.TOXIC}
    assert _olid_labeler({"subtask_a": "NOT"}) == set()
    assert _olid_labeler({"average": "0.81"}) == {T.TOXIC}    # SOLID
    assert _olid_labeler({"average": "0.2"}) == set()


def test_cyberbullying_protected_classes_map_to_hate():
    assert _cyberbullying_labeler({"cyberbullying_type": "ethnicity"}) \
        == {T.HATE, T.TOXIC}
    assert _cyberbullying_labeler({"cyberbullying_type": "age"}) == {T.TOXIC}
    assert _cyberbullying_labeler({"cyberbullying_type": "not_cyberbullying"}) \
        == set()


def test_goemotions_drops_hostile_keeps_benign():
    assert _goemotions_labeler({"labels": [17]}) == set()     # joy -> clean
    assert _goemotions_labeler({"labels": [2, 17]}) is None   # anger -> dropped


def test_row_labeler_none_drops_row():
    rows = [{"text": "fine", "labels": [17]},
            {"text": "angry", "labels": [2]}]
    ex = rows_to_examples(rows, text_col="text",
                          row_labeler=_goemotions_labeler)
    assert [e.text for e in ex] == ["fine"]


def test_row_text_assembles_token_lists():
    rows = [{"annotators": {"label": [2, 2, 1]}, "post_tokens": ["so", "dumb"]}]
    ex = rows_to_examples(rows, text_col="post_tokens",
                          row_text=_hatexplain_text,
                          row_labeler=_hatexplain_labeler)
    assert ex[0].text == "so dumb" and ex[0].labels == {T.TOXIC}


def test_jigsaw_bias_preset_maps_identity_attack_to_hate():
    preset = PRESETS["jigsaw_bias"]
    rows = [{"comment_text": "x", "target": "0.8", "identity_attack": "0.9",
             "threat": "0", "sexual_explicit": "0"}]
    ex = rows_to_examples(rows, text_col=preset["text_col"],
                          label_cols=preset["label_cols"],
                          label_map=preset["label_map"])
    assert ex[0].labels == {T.TOXIC, T.HATE}


def test_balance_clean_caps_majority():
    labeled = [Example("bad", {T.TOXIC})] * 10
    clean = [Example(f"ok {i}") for i in range(100)]
    out = balance_clean(labeled + clean, max_clean_ratio=2.0)
    assert sum(1 for e in out if e.labels) == 10
    assert sum(1 for e in out if not e.labels) == 20
    untouched = balance_clean(labeled + clean, max_clean_ratio=0)
    assert len(untouched) == 110


def test_every_preset_declares_a_text_source():
    for name, preset in PRESETS.items():
        assert preset.get("text_col") or preset.get("row_text"), name
