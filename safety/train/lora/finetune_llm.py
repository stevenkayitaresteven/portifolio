"""QLoRA fine-tune a text LLM into a 12-category moderation judge.

    # validate data + config on CPU, no model download:
    python -m safety.train.lora.finetune_llm --config configs/llama31_8b_moderator.json \
        --data runs/corpus --dry-run

    # the real run (needs a GPU):
    python -m safety.train.lora.finetune_llm --config configs/llama31_8b_moderator.json \
        --data runs/corpus --out runs/llama31-moderator

Then serve it as the live judge::

    SAFETY_MM_USE_LLM_JUDGE=1 SAFETY_MM_LLM_JUDGE_MODEL=runs/llama31-moderator \
        python -m safety chat

Design notes:
* **QLoRA** — the base model loads in 4-bit (bitsandbytes) and only small LoRA
  adapters train, so a 7-8B model fits on one 16 GB GPU.
* Training masks the prompt and learns only the assistant JSON (TRL's
  completion-only collator), so the model learns to *produce verdicts*, not to
  parrot the instructions.
"""
from __future__ import annotations

import argparse
import json

from .config import LoraConfig
from .data import build_sft_examples


def _print_dry_run(cfg: LoraConfig, examples: list[dict]) -> None:
    print("=== DRY RUN (no model loaded) ===")
    print(f"base model      : {cfg.base_model}")
    print(f"quantization    : {'4-bit QLoRA' if cfg.load_in_4bit else 'bf16 LoRA'}")
    print(f"LoRA r/alpha    : {cfg.lora_r} / {cfg.lora_alpha}  "
          f"(dropout {cfg.lora_dropout})")
    print(f"target modules  : {', '.join(cfg.target_modules)}")
    print(f"epochs/lr/bsz   : {cfg.epochs} / {cfg.learning_rate} / "
          f"{cfg.batch_size}x{cfg.grad_accum} (accum)")
    print(f"max seq length  : {cfg.max_seq_len}")
    print(f"training examples: {len(examples)}")
    if examples:
        print("--- first example ---")
        print(json.dumps(examples[0]["messages"], indent=2)[:1200])


def train(cfg: LoraConfig, data_path: str, out_dir: str) -> None:  # pragma: no cover - needs GPU
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    examples = build_sft_examples(data_path)
    dataset = Dataset.from_list(examples)

    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16)

    peft_cfg = PeftLoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM")

    sft_cfg = SFTConfig(
        output_dir=out_dir, num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate, max_seq_length=cfg.max_seq_len,
        logging_steps=10, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True)

    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset,
                         peft_config=peft_cfg, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved LoRA moderator to {out_dir}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="LoRA config JSON")
    p.add_argument("--data", required=True, help="dataset dir or train.jsonl")
    p.add_argument("--out", default="runs/llm-moderator")
    p.add_argument("--base-model", help="override base_model in the config")
    p.add_argument("--dry-run", action="store_true",
                   help="prepare data + config on CPU, don't load the model")
    args = p.parse_args(argv)

    cfg = LoraConfig.from_json(args.config)
    if args.base_model:
        cfg.base_model = args.base_model

    examples = build_sft_examples(args.data)
    if args.dry_run:
        _print_dry_run(cfg, examples)
        return 0
    if not examples:
        print("no training examples found — build a dataset first "
              "(python -m safety.data ...)")
        return 1
    train(cfg, args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
