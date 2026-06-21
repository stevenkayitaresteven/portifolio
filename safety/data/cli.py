"""Command line for the data pipeline.

    python -m safety.data demo  --out runs/demo            # synthetic, offline
    python -m safety.data build --config my_build.json --out runs/corpus
    python -m safety.data crawl --urls urls.txt --out raw.jsonl
    python -m safety.data tokenize --data runs/corpus --tokenizer distilroberta-base
"""
from __future__ import annotations

import argparse
import json
import os

from .config import PipelineConfig, SourceSpec
from .pipeline import DataPipeline

# A tiny synthetic corpus so `demo` runs with no network and no data of your
# own — enough to exercise clean -> dedup -> weak-label -> balance -> split.
_DEMO_ROWS = [
    "I really enjoyed the game last night, great match!",
    "Thanks so much for your help, you're the best.",
    "Let's meet at the cafe around noon tomorrow.",
    "The weather has been lovely this whole week.",
    "you are such an idiot and a moron",
    "i will kill you if you show up here again",
    "my ssn is 123-45-6789 and email me at bob@corp.io",
    "CLAIM your FREE PRIZE now http://totally-legit.tk win cash",
    "happy to review your pull request this afternoon",
    "what the f@ck is wrong with you people",
    "the recipe needs two cups of flour and one egg",
    "great work on the release, congrats to the team",
    "I really enjoyed the game last night, great match!",   # exact dup
    "I really enjoyed the game last  night,  great match!",  # near dup
]


def _demo_config(out: str) -> tuple[PipelineConfig, str]:
    raw = os.path.join(out, "_demo_raw.jsonl")
    os.makedirs(out, exist_ok=True)
    with open(raw, "w", encoding="utf-8") as fh:
        for row in _DEMO_ROWS:
            fh.write(json.dumps({"text": row}) + "\n")
    cfg = PipelineConfig(
        sources=[SourceSpec(kind="jsonl", path=raw, weak_label=True)],
        max_clean_ratio=3.0, val_ratio=0.2, test_ratio=0.2)
    return cfg, raw


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m safety.data")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the pipeline on a synthetic sample")
    d.add_argument("--out", default="runs/demo_dataset")

    b = sub.add_parser("build", help="run the pipeline from a JSON config")
    b.add_argument("--config", required=True)
    b.add_argument("--out", required=True)

    c = sub.add_parser("crawl", help="fetch an allowlist of URLs to JSONL")
    c.add_argument("--urls", required=True, help="file with one URL per line")
    c.add_argument("--out", required=True)
    c.add_argument("--delay", type=float, default=1.0)

    t = sub.add_parser("tokenize", help="tokenise built splits")
    t.add_argument("--data", required=True)
    t.add_argument("--tokenizer", default="distilroberta-base")
    t.add_argument("--max-length", type=int, default=256)

    args = p.parse_args(argv)

    if args.cmd == "demo":
        cfg, _ = _demo_config(args.out)
        card = DataPipeline(cfg).run(args.out)
        print(json.dumps(card, indent=2))
        return 0

    if args.cmd == "build":
        cfg = PipelineConfig.from_json(args.config)
        card = DataPipeline(cfg).run(args.out)
        print(json.dumps({k: card[k] for k in
                          ("total", "flagged", "splits", "category_counts")},
                         indent=2))
        return 0

    if args.cmd == "crawl":
        from .sources import WebTextCrawler
        with open(args.urls, encoding="utf-8") as fh:
            urls = [u for u in fh.read().splitlines() if u.strip()]
        n = 0
        with open(args.out, "w", encoding="utf-8") as out:
            for text, _ in WebTextCrawler(delay=args.delay).crawl(urls):
                out.write(json.dumps({"text": text}) + "\n")
                n += 1
        print(f"wrote {n} pages to {args.out}")
        return 0

    if args.cmd == "tokenize":
        from .tokenize import tokenize_dataset
        summary = tokenize_dataset(args.data, args.tokenizer, args.max_length)
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
