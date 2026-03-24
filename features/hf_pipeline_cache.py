"""
Process-wide cache for Hugging Face ``pipeline`` and SentenceTransformer models.

Several feature modules used identical models with separate module-level dicts, which
loaded duplicate weights into RAM (e.g. four copies of RoBERTa-large NER) and could
crash the process on memory pressure (Windows exit -1073741819).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

_PIPELINES: Dict[Tuple[Any, ...], Any] = {}
_SENTENCE_TRANSFORMERS: Dict[str, Any] = {}
_CAUSAL_LMS: Dict[str, Any] = {}


def _ensure_hf() -> None:
    try:
        from .hub_auth import ensure_hf_token_for_downloads
    except ImportError:
        from hub_auth import ensure_hf_token_for_downloads
    ensure_hf_token_for_downloads()


def get_transformers_pipeline(task: str, model: str, **kwargs: Any) -> Any:
    key = (task, model, tuple(sorted(kwargs.items())))
    if key not in _PIPELINES:
        _ensure_hf()
        from transformers import pipeline

        _PIPELINES[key] = pipeline(task, model=model, **kwargs)
    return _PIPELINES[key]


def get_sentence_transformer(model_name: str) -> Any:
    if model_name not in _SENTENCE_TRANSFORMERS:
        _ensure_hf()
        from sentence_transformers import SentenceTransformer

        _SENTENCE_TRANSFORMERS[model_name] = SentenceTransformer(model_name)
    return _SENTENCE_TRANSFORMERS[model_name]


def get_causal_lm(model_name: str) -> Tuple[Any, Any]:
    """Returns ``(tokenizer, model)`` with ``model.eval()`` set."""
    if model_name not in _CAUSAL_LMS:
        _ensure_hf()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        _CAUSAL_LMS[model_name] = (tokenizer, model)
    return _CAUSAL_LMS[model_name]
