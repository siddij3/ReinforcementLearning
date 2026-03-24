#!/usr/bin/env python3
"""
Pre-download every Hugging Face model used by this project into the local cache.

Default cache: <repo>/hf_home/ (see hf_token.ensure_hf_environment).

Usage (from repository root):
    python download_hf_models.py

Requires network + optional HF_TOKEN for rate limits. After this completes,
normal runs load from disk; set HF_OFFLINE=1 in .env to forbid any Hub access.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root when executed as python download_hf_models.py
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hf_token

hf_token.ensure_hf_environment()

from huggingface_hub import snapshot_download


def main() -> int:
    repos = list(hf_token.ALL_PROJECT_HF_REPOS)
    home = hf_token.hf_home()
    print(f"HF_HOME={home}")
    print(f"Downloading {len(repos)} model repos...\n")

    for i, repo_id in enumerate(repos, 1):
        print(f"[{i}/{len(repos)}] {repo_id} ...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_files_only=False,
                resume_download=True,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            return 1
        print("  OK")

    print("\nDone. Run training/features as usual; models load from this cache.")
    print("Optional: set HF_OFFLINE=1 in .env to use disk only (no Hub).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
