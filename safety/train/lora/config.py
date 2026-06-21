"""LoRA / QLoRA hyper-parameters, loaded from a JSON config."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class LoraConfig:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"])
    load_in_4bit: bool = True          # QLoRA
    # optimisation
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    max_seq_len: int = 1024

    @classmethod
    def from_json(cls, path: str) -> "LoraConfig":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
