"""LoRA fine-tune a vision-language model (Qwen2.5-VL) to moderate images.

Same JSON contract as the text judge, but the user turn carries an image plus
the moderation instruction, and the assistant learns the verdict. Training data
is the ImageFolder layout the image classifier already uses::

    data/train/{safe,nudity,gore,...}/*.jpg

Folder name -> category (safe -> [], nudity -> sexual, gore -> violence; extend
in ``FOLDER_TO_CATEGORY``).

    # CPU validation (builds the conversations, no model):
    python -m safety.train.lora.finetune_vlm --data ./img_data --dry-run

    # real run (GPU):
    python -m safety.train.lora.finetune_vlm --data ./img_data \
        --out runs/qwen25vl-moderator

We ship **no imagery**; bring your own labelled set.
"""
from __future__ import annotations

import argparse
import json
import os

from ...multimodal import llm_prompt
from ..dataset import MultiLabelImageFolder, discover_classes
from .config import LoraConfig

# Map ImageFolder class names onto canonical categories.
FOLDER_TO_CATEGORY = {
    "safe": None, "clean": None, "neutral": None,
    "nudity": "sexual", "nsfw": "sexual", "sexual": "sexual",
    "gore": "violence", "violence": "violence", "wound": "violence",
}

VLM_INSTRUCTION = ("Moderate this image. " + llm_prompt.SYSTEM_PROMPT)


def build_vlm_examples(data_dir: str) -> list[dict]:
    """Build (image, instruction) -> verdict conversations from an ImageFolder."""
    classes = discover_classes(data_dir)
    ds = MultiLabelImageFolder(os.path.join(data_dir, "train"), classes,
                               transform=lambda x: x)
    examples: list[dict] = []
    for path, idx in ds.samples:
        cat = FOLDER_TO_CATEGORY.get(classes[idx].lower(), classes[idx].lower())
        cats = [cat] if cat else []
        examples.append({"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": VLM_INSTRUCTION}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": llm_prompt.format_target(cats)}]},
        ]})
    return examples


def _print_dry_run(cfg: LoraConfig, examples: list[dict]) -> None:
    print("=== VLM DRY RUN (no model loaded) ===")
    print(f"base model       : {cfg.base_model}")
    print(f"LoRA r/alpha     : {cfg.lora_r} / {cfg.lora_alpha}")
    print(f"training examples: {len(examples)}")
    if examples:
        sample = examples[0]
        img = sample["messages"][0]["content"][0]["image"]
        target = sample["messages"][1]["content"][0]["text"]
        print(f"first image      : {img}")
        print(f"first target     : {target}")


def train(cfg: LoraConfig, examples: list[dict], out_dir: str) -> None:  # pragma: no cover - needs GPU
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import (AutoProcessor,
                              Qwen2_5_VLForConditionalGeneration)
    from trl import SFTConfig, SFTTrainer

    processor = AutoProcessor.from_pretrained(cfg.base_model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto")

    peft_cfg = PeftLoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM")
    sft_cfg = SFTConfig(
        output_dir=out_dir, num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate, bf16=True,
        gradient_checkpointing=True, remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True})

    trainer = SFTTrainer(model=model, args=sft_cfg,
                         train_dataset=Dataset.from_list(examples),
                         peft_config=peft_cfg, processing_class=processor)
    trainer.train()
    trainer.save_model(out_dir)
    print(f"saved Qwen2.5-VL moderator to {out_dir}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="ImageFolder dataset root")
    p.add_argument("--config", help="LoRA config JSON (defaults to Qwen2.5-VL)")
    p.add_argument("--out", default="runs/vlm-moderator")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = (LoraConfig.from_json(args.config) if args.config
           else LoraConfig(base_model="Qwen/Qwen2.5-VL-7B-Instruct"))
    examples = build_vlm_examples(args.data)
    if args.dry_run:
        _print_dry_run(cfg, examples)
        return 0
    if not examples:
        print("no images found under data/train/<class>/")
        return 1
    train(cfg, examples, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
