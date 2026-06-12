"""The 12-category taxonomy: heuristics, label mapping, policy, file gate,
and the chat-level block/mask/flag behavior. All offline."""
import io

import pytest

from safety.multimodal import taxonomy as T
from safety.multimodal import MultimodalConfig, TextModerator, check_file_safety
from safety.multimodal.filecheck import EICAR_SIGNATURE
from safety.multimodal.heuristics import (find_pii, find_phrase_categories,
                                          mask_pii, spam_signals, url_risk)
from safety.train.finetune_text import (Example, map_source_label,
                                        multilabel_metrics, rows_to_examples,
                                        _aegis_labeler, _liar2_labeler)


def offline_config() -> MultimodalConfig:
    cfg = MultimodalConfig()
    cfg.use_hf_text = cfg.use_hf_image = cfg.use_hf_audio = False
    return cfg


def moderate(text: str, **cfg_overrides):
    cfg = offline_config()
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    return TextModerator(cfg).moderate(text)


# --- taxonomy label mapping --------------------------------------------------------

def test_koala_labels_map_to_taxonomy():
    assert T.category_for_label("KoalaAI/Text-Moderation", "S") == T.SEXUAL
    assert T.category_for_label("KoalaAI/Text-Moderation", "S3") == T.CHILD_SAFETY
    assert T.category_for_label("KoalaAI/Text-Moderation", "SH") == T.SELF_HARM
    assert T.category_for_label("KoalaAI/Text-Moderation", "OK") is None


def test_toxicbert_labels_map_to_taxonomy():
    assert T.category_for_label("unitary/toxic-bert", "identity_hate") == T.HATE
    assert T.category_for_label("unitary/toxic-bert", "threat") == T.VIOLENCE
    assert T.category_for_label("unitary/toxic-bert", "obscene") == T.TOXIC


def test_custom_model_labels_use_generic_mapping():
    assert T.category_for_label("my/finetune", "extremism") == T.EXTREMISM
    assert T.category_for_label("my/finetune", "Hate Speech") == T.HATE
    assert T.category_for_label("my/finetune", "benign") is None


def test_strongest_action_ordering():
    assert T.strongest_action(["flag", "mask", "block"]) == "block"
    assert T.strongest_action(["flag", "mask"]) == "mask"
    assert T.strongest_action([]) == "none"


# --- PII (privacy) -------------------------------------------------------------------

def test_pii_email_and_phone_are_found_and_masked():
    text = "reach me at jane.doe@example.com or +1 415 555 0199 ok?"
    kinds = {k for k, _ in find_pii(text)}
    assert {"email", "phone"} <= kinds
    masked, found = mask_pii(text)
    assert "jane.doe@example.com" not in masked
    assert "@example.com" in masked            # enough context to see it's an email
    assert "0199" in masked and "415 555" not in masked


def test_credit_card_needs_luhn():
    ok, kinds = mask_pii("card: 4111 1111 1111 1111")     # valid test number
    assert "4111 1111 1111 1111" not in ok
    assert "credit_card" in kinds
    # A Luhn-invalid run is not labeled a credit card (though a long digit run
    # may still be masked as phone-shaped — over-masking PII is the safe default).
    _, kinds_bad = mask_pii("order id 9999 8888 7777 6660")
    assert "credit_card" not in kinds_bad


def test_pii_message_is_masked_not_blocked():
    res = moderate("my email is bob@corp.io")
    assert res.flagged and res.action == "mask"
    assert T.PRIVACY in res.categories
    assert "bob@corp.io" not in res.censored_text


# --- phrase lexicons -> block categories ---------------------------------------------

@pytest.mark.parametrize("text,category", [
    ("I want to kill myself", T.SELF_HARM),
    ("i will kill you tomorrow", T.VIOLENCE),
    ("selling stolen credit card numbers", T.CRIMINAL),
    ("I can build you a ransomware kit", T.CYBERSECURITY),
    ("you should join isis brother", T.EXTREMISM),
])
def test_block_categories_are_blocked(text, category):
    hits = {c for c, _, _ in find_phrase_categories(text)}
    assert category in hits
    res = moderate(text)
    assert res.flagged and res.action == "block"
    assert category in res.categories
    assert res.censored_text is None           # blocked content is not delivered


def test_self_harm_block_carries_support_note():
    res = moderate("lately I just want to end my life")
    assert res.action == "block"
    assert "988" in res.extra["support_note"]


def test_misinformation_is_flagged_not_blocked():
    res = moderate("did you know vaccines cause autism??")
    assert res.flagged and res.action == "flag"
    assert T.MISINFORMATION in res.categories
    assert res.censored_text                    # still delivered, with a warning


def test_action_overrides_change_policy():
    res = moderate("vaccines cause autism", action_overrides="misinformation:block")
    assert res.action == "block"


# --- spam & URLs ----------------------------------------------------------------------

def test_promo_spam_is_flagged():
    text = ("CONGRATULATIONS you have won! claim your prize NOW: "
            "http://win.example.com FREE MONEY limited time offer")
    score, _ = spam_signals(text)
    assert score >= 0.5
    res = moderate(text)
    assert res.flagged and res.action == "flag"
    assert T.SPAM in res.categories


def test_phishing_shaped_url_is_flagged():
    text = "verify your password here http://192.168.4.12/login"
    score, reasons = url_risk(text)
    assert score >= 0.5 and any("IP" in r for r in reasons)
    res = moderate(text)
    assert T.SPAM in res.categories


def test_plain_link_is_not_spam():
    assert not moderate("docs are at https://docs.python.org/3/").flagged


# --- file gate ------------------------------------------------------------------------

def test_eicar_signature_is_blocked():
    res = check_file_safety("notes.txt", b"hello " + EICAR_SIGNATURE)
    assert res is not None and res.action == "block"
    assert T.CYBERSECURITY in res.categories


def test_renamed_executable_is_caught_by_magic_bytes():
    res = check_file_safety("vacation.jpg", b"MZ\x90\x00rest-of-a-pe-file")
    assert res is not None and "magic bytes" in res.reasons[0]


def test_archives_are_blocked_and_images_pass():
    assert check_file_safety("data.zip", b"PK\x03\x04") is not None
    assert check_file_safety("photo.jpg", b"\xff\xd8\xff\xe0aJFIF") is None


# --- fine-tuning helpers ---------------------------------------------------------------

def test_rows_to_examples_with_label_columns():
    rows = [{"comment_text": "you idiot", "toxic": "1", "identity_hate": "0"},
            {"comment_text": "nice day", "toxic": "0", "identity_hate": "0"}]
    ex = rows_to_examples(rows, text_col="comment_text",
                          label_cols=["toxic", "identity_hate"])
    assert ex[0].labels == {T.TOXIC}
    assert ex[1].labels == set()
    assert ex[0].multi_hot()[list(T.CATEGORIES).index(T.TOXIC)] == 1.0


def test_aegis_labeler_maps_violations():
    row = {"prompt": "x", "prompt_label": "unsafe",
           "violated_categories": "Criminal Planning/Confessions, Hate/Identity Hate"}
    assert _aegis_labeler(row) == {T.CRIMINAL, T.HATE}
    assert _aegis_labeler({"prompt_label": "safe", "violated_categories": "Hate"}) == set()


def test_liar2_labeler_binarizes():
    assert _liar2_labeler({"label": 0}) == {T.MISINFORMATION}
    assert _liar2_labeler({"label": 5}) == set()


def test_map_source_label_generic():
    assert map_source_label("severe_toxic") == T.TOXIC
    assert map_source_label("Suicide and Self Harm") == T.SELF_HARM


def test_multilabel_metrics_perfect_and_miss():
    idx = list(T.CATEGORIES).index(T.TOXIC)
    y = [[0.0] * len(T.CATEGORIES) for _ in range(4)]
    for r in (0, 1):
        y[r][idx] = 1.0
    m = multilabel_metrics(y, y)
    assert m["per_category"][T.TOXIC]["f1"] == 1.0
    assert m["micro"]["f1"] == 1.0
    miss = [[0.0] * len(T.CATEGORIES) for _ in range(4)]
    m2 = multilabel_metrics(y, miss)
    assert m2["per_category"][T.TOXIC]["recall"] == 0.0
    assert m2["per_category"][T.TOXIC]["confusion"]["fn"] == 2


# --- chat-level behavior (needs FastAPI) ------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from safety.chat import create_chat_app  # noqa: E402
from safety.multimodal import MultimodalModerator  # noqa: E402


def make_client() -> TestClient:
    cfg = offline_config()
    cfg.check_video_audio = False
    cfg.safety.use_nudenet = False
    return TestClient(create_chat_app(moderator=MultimodalModerator(cfg)))


def test_threat_message_is_blocked_in_chat():
    client = make_client()
    r = client.post("/chat/send", data={"text": "i will kill you tomorrow"})
    (msg,) = r.json()["messages"]
    assert msg["action"] == "block" and msg["text"] == ""
    assert "violence" in msg["categories"]


def test_self_harm_message_blocked_with_support_note():
    client = make_client()
    r = client.post("/chat/send", data={"text": "I want to kill myself"})
    (msg,) = r.json()["messages"]
    assert msg["action"] == "block"
    assert "988" in msg["support_note"]


def test_pii_is_redacted_in_chat():
    client = make_client()
    r = client.post("/chat/send",
                    data={"text": "her number is +1 415 555 0199, call her"})
    (msg,) = r.json()["messages"]
    assert msg["action"] == "mask"
    assert "415 555" not in msg["text"] and "privacy" in msg["categories"]


def test_eicar_upload_is_blocked_in_chat():
    client = make_client()
    r = client.post("/chat/send", files=[
        ("files", ("invoice.txt", io.BytesIO(EICAR_SIGNATURE), "text/plain")),
    ])
    (msg,) = r.json()["messages"]
    assert msg["kind"] == "file" and msg["action"] == "block"
    assert msg["media"] is None


def test_audit_endpoint_records_decisions_without_content():
    client = make_client()
    client.post("/chat/send", data={"text": "you are a fucking disaster"})
    decisions = client.get("/chat/audit").json()["decisions"]
    assert decisions and decisions[-1]["flagged"]
    assert decisions[-1]["action"] == "mask"
    # the audit log stores the decision, never the message body
    assert "text" not in decisions[-1]


def test_document_with_threats_is_blocked():
    client = make_client()
    body = b"meeting notes\ni will kill you if this ships late\n"
    r = client.post("/chat/send", files=[
        ("files", ("notes.txt", io.BytesIO(body), "text/plain")),
    ])
    (msg,) = r.json()["messages"]
    assert msg["kind"] == "text" and msg["action"] == "block"
    assert msg["media"] is None
