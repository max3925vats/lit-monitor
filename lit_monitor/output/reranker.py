"""
Cross-encoder reranker for improving top-K retrieval precision.

Uses ``sentence-transformers`` (hard dep) with a HuggingFace CrossEncoder.
The default model is ``mixedbread-ai/mxbai-rerank-large-v2`` (~670MB) and
is automatically accelerated via PyTorch MPS on Apple Silicon.

Usage (via EmbeddingsDB — callers don't instantiate directly):
    ``find_similar_chunks(query, rerank_with_query=query)``
    ``find_similar_to_text(query, rerank_with_query=query)``

The singleton is lazy-loaded on first call; the model is NOT downloaded at
import time or at EmbeddingsDB construction — only when a reranked query
is actually made.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton — shared across all EmbeddingsDB instances.
_reranker_instance: Reranker | None = None
_reranker_model_loaded: str | None = None  # tracks which model is loaded


def get_reranker(
    model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
    device: str | None = None,
) -> Reranker:
    """
    Return the module-level Reranker singleton, loading the model on first call.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the CrossEncoder.
    device:
        PyTorch device string.  ``None`` → auto-select (MPS > CUDA > CPU).
    """
    global _reranker_instance, _reranker_model_loaded
    if _reranker_instance is None or _reranker_model_loaded != model_name:
        logger.info("Loading cross-encoder reranker: %s", model_name)
        _reranker_instance = Reranker(model_name=model_name, device=device)
        _reranker_model_loaded = model_name
    return _reranker_instance


class Reranker:
    """
    Thin wrapper around ``sentence_transformers.CrossEncoder``.

    Instantiate via ``get_reranker()`` to get the singleton.
    """

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install it with: pip install sentence-transformers>=3.0"
            ) from exc

        # device=None → CrossEncoder auto-selects (MPS > CUDA > CPU)
        self._model = CrossEncoder(model_name, device=device)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_key: str = "document",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using the cross-encoder.

        Parameters
        ----------
        query:
            The query string.
        candidates:
            List of result dicts (each must have ``text_key`` populated).
        text_key:
            Key in each candidate dict holding the passage text.
            Default: ``"document"`` (ChromaDB result format).
        top_k:
            Truncate output to top_k after reranking.  ``None`` = return all.

        Returns
        -------
        list[dict]
            Sorted by cross-encoder score descending.  Each dict has all
            original keys plus ``rerank_score: float``.
        """
        if not candidates:
            return []

        pairs = [(query, c.get(text_key, "")) for c in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        return [{**c, "rerank_score": float(s)} for c, s in ranked]
