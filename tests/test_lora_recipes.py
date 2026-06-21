"""Tests for the LoRA recipe data builders (CPU, no model)."""
import json

from safety.train.lora.config import LoraConfig
from safety.train.lora.data import build_sft_examples
from safety.train.lora.finetune_vlm import build_vlm_examples


def test_build_sft_examples(tmp_path):
    jl = tmp_path / "train.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in [
        {"text": "you are an idiot", "categories": ["toxic"]},
        {"text": "have a great day", "categories": []},
    ]))
    examples = build_sft_examples(str(jl))
    assert len(examples) == 2
    msgs = examples[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
    assert json.loads(msgs[-1]["content"])["categories"] == ["toxic"]
    # safe row -> empty categories
    assert json.loads(examples[1]["messages"][-1]["content"])["categories"] == []


def test_build_sft_examples_accepts_directory(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"text": "hi", "categories": []}) + "\n")
    assert len(build_sft_examples(str(tmp_path))) == 1


def test_lora_config_roundtrip(tmp_path):
    cfg = LoraConfig(base_model="mistralai/Mistral-7B-Instruct-v0.3", lora_r=8)
    path = tmp_path / "c.json"
    cfg.to_json(str(path))
    loaded = LoraConfig.from_json(str(path))
    assert loaded.base_model == cfg.base_model and loaded.lora_r == 8


def test_build_vlm_examples(tmp_path):
    for cls in ("safe", "nudity", "gore"):
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        (d / "x.jpg").write_bytes(b"")     # dry path-only build, pixels unused
    examples = build_vlm_examples(str(tmp_path))
    assert len(examples) == 3
    targets = {json.loads(e["messages"][1]["content"][0]["text"])["categories"][0]
               if json.loads(e["messages"][1]["content"][0]["text"])["categories"]
               else "safe" for e in examples}
    assert targets == {"safe", "sexual", "violence"}
