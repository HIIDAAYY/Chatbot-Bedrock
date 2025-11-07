"""Thin wrapper around Pinecone integrated embeddings for FAQ retrieval."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Sequence

from pinecone import Pinecone

import config

logger = logging.getLogger(__name__)


class PineconeNotConfigured(RuntimeError):
    """Raised when Pinecone settings are incomplete."""


def _require_pinecone_settings():
    settings = config.get_settings()
    if not settings.pinecone_enabled:
        raise PineconeNotConfigured("Pinecone configuration is incomplete")
    if not settings.pinecone_environment:
        raise PineconeNotConfigured("PINECONE_ENV must be set when Pinecone is enabled")
    return settings


@lru_cache(maxsize=1)
def _pinecone_client() -> Pinecone:
    settings = _require_pinecone_settings()
    return Pinecone(api_key=settings.pinecone_api_key or "", environment=settings.pinecone_environment)


@lru_cache(maxsize=1)
def _pinecone_index():
    settings = _require_pinecone_settings()
    return _pinecone_client().Index(settings.pinecone_index or "")


def _extract_values(response) -> Sequence[float]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        return []
    first = data[0]
    values = None
    if isinstance(first, dict):
        values = first.get("values")
    else:
        values = getattr(first, "values", None)
    if callable(values):  # handle dict.values method accidentally retrieved
        values = None
    return values or []


def _embed_text(text: str, *, input_type: str) -> List[float]:
    settings = config.get_settings()
    if not settings.pinecone_embedding_model:
        raise PineconeNotConfigured("PINECONE_EMBEDDING_MODEL must be configured for Pinecone usage")

    try:
        response = _pinecone_client().inference.embed(
            model=settings.pinecone_embedding_model,
            inputs=[{"text": text}],
            parameters={"input_type": input_type, "truncate": "END"},
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("pinecone_embed_error", extra={"error": str(exc)})
        return []

    values = _extract_values(response)
    return [float(v) for v in values]


def search_chunks(query: str) -> List[Dict[str, float | str]]:
    """Return matching FAQ chunks ordered by similarity."""
    settings = config.get_settings()
    if not settings.pinecone_enabled:
        return []

    vector = _embed_text(query, input_type="query")
    if not vector:
        return []

    try:
        index = _pinecone_index()
    except PineconeNotConfigured:
        return []
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("pinecone_index_error", extra={"error": str(exc)})
        return []

    try:
        response = index.query(
            vector=vector,
            top_k=settings.pinecone_top_k,
            include_values=False,
            include_metadata=True,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("pinecone_query_error", extra={"error": str(exc)})
        return []

    matches: List[Dict[str, float | str]] = []
    threshold = settings.pinecone_score_threshold
    for match in response.get("matches", []):
        metadata = match.get("metadata") or {}
        text = metadata.get("text")
        score_raw = match.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            score = 0.0
        if not text or score < threshold:
            continue
        matches.append({"text": str(text), "score": score})

    return matches


def clear_caches() -> None:
    """Reset cached clients (useful for tests)."""
    _pinecone_client.cache_clear()
    _pinecone_index.cache_clear()
