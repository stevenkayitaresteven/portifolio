#!/usr/bin/env python3
"""One command to publish the Sentinel chat demo to Hugging Face Spaces.

This gives you a public URL anyone can open and use — no install required:

    https://huggingface.co/spaces/<your-username>/sentinel-chat

Prerequisites
-------------
    pip install huggingface_hub
    export HF_TOKEN=hf_xxx        # a "write" token from hf.co/settings/tokens

Run it from the repo root
-------------------------
    python deploy/deploy_to_hf.py                 # -> <you>/sentinel-chat
    python deploy/deploy_to_hf.py my-space-name   # custom Space name

It creates a Docker Space, uploads the repo, and swaps in the Space README
(deploy/space_readme.md) which carries the `sdk: docker` / `app_port: 7860`
metadata Spaces needs. Re-running just pushes an update.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub first", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("huggingface_token")
    if not token:
        print("set HF_TOKEN to a write token (hf.co/settings/tokens)",
              file=sys.stderr)
        return 1

    api = HfApi(token=token)
    user = api.whoami()["name"]
    name = sys.argv[1] if len(sys.argv) > 1 else "sentinel-chat"
    repo_id = f"{user}/{name}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker",
                    exist_ok=True)

    # Push everything except local/dev cruft; the Space card replaces README.md.
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(REPO_ROOT),
        ignore_patterns=[".git*", ".venv/*", "**/__pycache__/*", "runs/*",
                         "*.onnx", "*.pt", ".pytest_cache/*", "README.md"],
    )
    api.upload_file(
        path_or_fileobj=str(REPO_ROOT / "deploy" / "space_readme.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )

    print(f"\n  Deployed -> https://huggingface.co/spaces/{repo_id}")
    print("  First build takes a few minutes; the URL is live once it finishes.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
