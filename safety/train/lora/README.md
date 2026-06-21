# LoRA / QLoRA moderation recipes

Turn a general instruction model into a **12-category moderation judge** that
emits the JSON contract in [`safety/multimodal/llm_prompt.py`](../../multimodal/llm_prompt.py),
then plug it straight back into the chat as the
[LLM-judge backend](../../multimodal/llm_judge.py).

## The honest scope on "fine-tune all 11 models"

You don't train eleven separate models — that's expensive and pointless for one
classification job. You pick the **right base for the job** and run **one of two
recipes**. Both swap the base model with a single config field, so all eleven
from the brief are covered:

| Requested model | Recipe | How |
|---|---|---|
| Llama 3.1 (8B) | `finetune_llm` | `configs/llama31_8b_moderator.json` |
| Mistral 7B | `finetune_llm` | `configs/mistral7b_moderator.json` |
| Gemma 3 | `finetune_llm` | `configs/gemma3_moderator.json` |
| Qwen3 MoE | `finetune_llm` | set `base_model: Qwen/Qwen3-30B-A3B` (needs more VRAM) |
| Qwen2.5-Coder | `finetune_llm` | set `base_model: Qwen/Qwen2.5-Coder-7B-Instruct` |
| DeepSeek-R1 (distill) | `finetune_llm` | `base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| OpenAI OSS-20B | `finetune_llm` | `base_model: openai/gpt-oss-20b` (24 GB+ GPU) |
| Qwen2.5-VL | `finetune_vlm` | `configs/qwen25_vl_moderator.json` |
| InternVL | `finetune_vlm` | `base_model: OpenGVLab/InternVL3-8B` + its processor |
| LLaVA | `finetune_vlm` | `base_model: llava-hf/llava-1.5-7b-hf` + its processor |
| MiniCPM-V | `finetune_vlm` | `base_model: openbmb/MiniCPM-V-2_6` + its processor |

> VLMs other than Qwen2.5-VL use different model/processor classes; swap the two
> `from_pretrained` lines in `finetune_vlm.py` accordingly (noted in the file).

## Workflow

```bash
# 1) Build a clean, labelled dataset (CPU, see ../../data/)
python -m safety.data build --config my_build.json --out runs/corpus

# 2) Validate data + config on CPU — no GPU, no model download
python -m safety.train.lora.finetune_llm \
    --config safety/train/lora/configs/llama31_8b_moderator.json \
    --data runs/corpus --dry-run

# 3) Train on a GPU (Colab T4/A100, or your own 16 GB+ card)
pip install "transformers>=4.45" peft trl bitsandbytes accelerate datasets
python -m safety.train.lora.finetune_llm \
    --config safety/train/lora/configs/llama31_8b_moderator.json \
    --data runs/corpus --out runs/llama31-moderator

# 4) Serve it as the live judge
SAFETY_MM_USE_LLM_JUDGE=1 \
SAFETY_MM_LLM_JUDGE_MODEL=runs/llama31-moderator \
    python -m safety chat
```

A ready-to-run Colab notebook is at
[`notebooks/finetune_moderator_colab.ipynb`](../../../notebooks/finetune_moderator_colab.ipynb).

## Why QLoRA

The base model loads in 4-bit (bitsandbytes nf4) and only the small LoRA
adapters train — a 7–8B model fits on a single 16 GB GPU (a free Colab T4).
Training masks the prompt and learns only the assistant JSON, so the model
learns to *produce verdicts*, not echo the instructions.
