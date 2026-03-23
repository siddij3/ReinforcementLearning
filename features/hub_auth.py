"""
Ensure repo root is on sys.path and Hugging Face token env is loaded before Hub downloads.

Used by lazy ``_load`` helpers in feature modules so ``python features/foo.py`` still works.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_hf_token_for_downloads() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import hf_token

    hf_token.ensure_hf_environment()
