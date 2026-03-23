"""
Hugging Face Hub token setup for higher rate limits and reliable downloads.

Reads HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) from the process environment and
optionally from a repo-root ``.env`` file (requires ``python-dotenv``).

The Hub libraries (``huggingface_hub``, ``transformers``) read these env vars
automatically; we mirror both names so either spelling works.

Do not commit real tokens. Copy ``.env.example`` → ``.env`` and paste your token.
"""

from __future__ import annotations

import os
from pathlib import Path

_done = False


def project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_hf_environment() -> None:
    """
    Load ``.env`` from the repository root (if present) and sync token env vars.

    Safe to call multiple times (no-op after the first successful run).
    """
    global _done
    if _done:
        return
    _done = True

    root = project_root()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if token:
        # Libraries accept either name; set both for consistency.
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_TOKEN", token)
