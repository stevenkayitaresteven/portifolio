"""Tests for the LLM-judge backend + shared prompt (stub backend, no model)."""
from safety.multimodal import llm_prompt
from safety.multimodal.config import MultimodalConfig
from safety.multimodal.llm_judge import LLMJudgeModerator
from safety.multimodal.text import TextModerator


def test_prompt_target_roundtrip():
    target = llm_prompt.format_target(["hate", "toxic", "bogus"], "insult")
    assert "bogus" not in target           # unknown categories dropped
    assert llm_prompt.parse_response(target) == ["hate", "toxic"]


def test_parse_response_is_robust():
    assert llm_prompt.parse_response('noise {"categories": ["violence"]} tail') \
        == ["violence"]
    assert llm_prompt.parse_response("Hate Speech and self harm here") \
        == ["hate", "self_harm"]
    assert llm_prompt.parse_response("totally safe message") == []
    assert llm_prompt.parse_response("") == []


def _stub(messages):
    text = messages[-1]["content"].lower()
    if "ugly" in text:
        return '{"categories": ["hate"], "reason": "insult"}'
    return '{"categories": []}'


def test_judge_categories_with_stub():
    judge = LLMJudgeModerator(MultimodalConfig(), backend=_stub)
    assert judge.available is True
    assert judge.categories("you are ugly") == ["hate"]
    assert judge.categories("good morning") == []


def test_judge_disabled_when_no_backend():
    judge = LLMJudgeModerator(MultimodalConfig())   # no model/endpoint configured
    assert judge.available is False
    assert judge.categories("anything") == []


def test_text_moderator_uses_judge():
    cfg = MultimodalConfig()
    cfg.use_hf_text = False
    cfg.use_llm_judge = True
    mod = TextModerator(cfg)
    mod._judge = LLMJudgeModerator(cfg, backend=_stub)   # inject stub

    res = mod.moderate("you are ugly")
    assert "hate" in res.categories
    assert "llm-judge" in res.detectors
    assert res.action == "block"            # hate -> block

    safe = mod.moderate("lovely day for a walk")
    assert safe.categories == [] and safe.action == "none"
