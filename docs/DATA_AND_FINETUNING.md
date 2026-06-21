# Data pipeline, LLM-judge & LoRA fine-tuning

How Sentinel goes from raw text on the internet to a trained moderator running
in the chat. Three pieces fit together:

```
  sources ──► safety.data ──► clean / dedup / label / split ──► JSONL + card
                                                                   │
                          ┌────────────────────────────────────────┤
                          ▼                                         ▼
   safety.train.finetune_text                       safety.train.lora.finetune_llm
   (small multi-label encoder)                      (QLoRA LLM moderation judge)
                          │                                         │
                          ▼                                         ▼
        SAFETY_MM_TEXT_MODELS=...                  SAFETY_MM_USE_LLM_JUDGE=1
                                                   SAFETY_MM_LLM_JUDGE_MODEL=...
                          └──────────► python -m safety chat ◄──────┘
```

## 1. Data preparation — `safety/data/`

A dependency-light pipeline (stdlib core) that does the unglamorous 80%:

| Stage | Module | What it does |
|---|---|---|
| ingest | `sources.py` | local JSONL/CSV, Hugging Face datasets, or a polite `robots.txt`-respecting web fetcher (benign text only) |
| clean | `clean.py` | NFKC-normalise, strip control chars, replace URLs/mentions, **scrub PII while keeping the toxic signal**, length-filter |
| dedup | `dedup.py` | exact (normalised hash) + near-dup (**MinHash + LSH**, no deps) |
| label | `label.py` | map heterogeneous source labels onto the 12 categories, or **weak-label** unlabelled text with the offline heuristics |
| balance/split | `pipeline.py` | cap over-represented categories, hold clean:flagged ratio, train/val/test split, write a `dataset_card.json` |
| tokenize | `tokenize.py` | tokenise the splits with any HF tokenizer |

```bash
python -m safety.data demo --out runs/demo            # synthetic, fully offline
python -m safety.data build --config build.json --out runs/corpus
python -m safety.data tokenize --data runs/corpus --tokenizer distilroberta-base
```

> **On scraping:** the crawler only fetches a URL allowlist you provide, obeys
> `robots.txt`, and is meant for *benign* hard negatives. Never crawl the open
> web for toxic/hate/extremist/child-safety material — use the vetted public
> datasets (Jigsaw, Civil Comments, Aegis 2.0, …) already wired into
> `safety.train.finetune_text`.

A `build.json` is just a list of sources:

```json
{
  "sources": [
    {"kind": "hf", "path": "google/civil_comments", "text_field": "text",
     "label_field": "toxicity", "max_records": 50000},
    {"kind": "jsonl", "path": "my_chat_logs.jsonl", "weak_label": true}
  ],
  "max_per_category": 20000, "near_threshold": 0.85
}
```

## 2. LLM-as-judge backend — `safety/multimodal/llm_judge.py`

An optional moderator that asks a safety-tuned LLM to classify a message and
returns the 12-category JSON. **Off by default**; degrades gracefully if no
backend is reachable. Point it at a local model or any OpenAI-compatible
endpoint (vLLM, Ollama, TGI, OpenAI):

```bash
SAFETY_MM_USE_LLM_JUDGE=1 SAFETY_MM_LLM_JUDGE_MODEL=runs/llama31-moderator \
    python -m safety chat
# or a hosted endpoint:
SAFETY_MM_USE_LLM_JUDGE=1 \
SAFETY_MM_LLM_JUDGE_ENDPOINT=http://localhost:8001/v1/chat/completions \
    python -m safety chat
```

The prompt, the JSON contract, and the parser live in one place
(`llm_prompt.py`) so the model you **train** and the model you **run** speak the
same format.

## 3. LoRA / QLoRA recipes — `safety/train/lora/`

Fine-tune a general model into a moderation judge. One recipe per modality;
swap the base model to cover every model in the brief (Llama 3.1, Mistral 7B,
Gemma 3, Qwen2.5-Coder, Qwen3-MoE, DeepSeek-R1-distill, GPT-OSS-20B → text;
Qwen2.5-VL, LLaVA, InternVL, MiniCPM-V → vision). Full mapping table and the
Colab notebook: [`safety/train/lora/README.md`](../safety/train/lora/README.md).

```bash
# validate data + config on CPU (no GPU, no download)
python -m safety.train.lora.finetune_llm \
    --config safety/train/lora/configs/llama31_8b_moderator.json \
    --data runs/corpus --dry-run

# train on a GPU, then serve as the judge
python -m safety.train.lora.finetune_llm --config ...llama31... --data runs/corpus \
    --out runs/llama31-moderator
SAFETY_MM_USE_LLM_JUDGE=1 SAFETY_MM_LLM_JUDGE_MODEL=runs/llama31-moderator \
    python -m safety chat
```

Everything except the GPU training step runs and is unit-tested on CPU.
