"""
Hugging Face Hub: local cache directory, token, optional offline mode.

- **Disk cache**: By default all models are stored under ``<repo>/hf_home/``
  (``HF_HOME``). Run ``python download_hf_models.py`` once to fetch snapshots;
  later loads use the cache and do not re-download.

- **Token**: ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` from env or ``.env``.

- **Offline** (optional): Set ``HF_OFFLINE=1`` in ``.env`` after downloading
  to force loads from disk only (no Hub network calls).

Do not commit real tokens. Copy ``.env.example`` → ``.env`` and paste your token.
"""

from __future__ import annotations

import os
from pathlib import Path

_done = False

# All Hub repo IDs used by ``features/`` (keep in sync when adding models).
ALL_PROJECT_HF_REPOS: tuple[str, ...] = (
    "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
    "Jean-Baptiste/roberta-large-ner-english",
    "sentence-transformers/all-MiniLM-L6-v2",
    "tanfiona/unicausal-tok-cls-baseline",
    "cross-encoder/nli-deberta-v3-small",
    "Nashhz/SBERT_KFOLD_JobDescriptions_Skills_UserPortfolios",
    "distilgpt2",
)


def project_root() -> Path:
    return Path(__file__).resolve().parent


def hf_home() -> Path:
    """Resolved Hugging Face cache root (``HF_HOME``)."""
    return Path(os.environ.get("HF_HOME", str(project_root() / "hf_home"))).resolve()


def _configure_local_hf_home(root: Path) -> None:
    """Point ``HF_HOME`` at ``<repo>/hf_home`` unless already set (e.g. in ``.env``)."""
    default = (root / "hf_home").resolve()
    default.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(default))


def _configure_offline_from_env() -> None:
    v = os.environ.get("HF_OFFLINE", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ensure_hf_environment() -> None:
    """
    1) Load ``.env`` from repo root (optional).
    2) Set ``HF_HOME`` to local ``hf_home/`` if unset.
    3) If ``HF_OFFLINE=1``, enable Hub/transformers offline mode.
    4) Once: mirror HF token env vars.

    Call before any ``transformers`` / ``sentence_transformers`` Hub download.
    Safe to call multiple times.
    """
    root = project_root()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    _configure_local_hf_home(root)
    _configure_offline_from_env()

    global _done
    if _done:
        return
    _done = True

    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_TOKEN", token)
