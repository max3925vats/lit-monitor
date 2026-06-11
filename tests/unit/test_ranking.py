"""CU-1: unit tests for the shared interest-vector cosine ranker.

Exercises ``scripts/learning/ranking.rank_papers_by_interest`` — the single
shared implementation that both the web ``/insights`` page (via
``scripts/api/insights._rank_top_papers``) and the CLI ``learning view`` command
call. The tests cover the happy path (sort-desc + cap + title resolution) and
the defensive paths (degenerate interest vector, bad candidate embedding, no
embeddings store, missing title) the production callers rely on.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from lit_monitor.learning.ranking import rank_papers_by_interest


class _FakeEmbeddings:
    """Minimal stand-in for ``EmbeddingsDB`` exposing ``get_paper_embeddings``."""

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def get_paper_embeddings(self) -> dict[str, np.ndarray]:
        return self._mapping


def _state_db(titles: dict[str, Any]) -> MagicMock:
    """A fake state_db whose ``get_paper(doi)`` returns ``{"title": ...}`` or None."""
    sdb = MagicMock()
    sdb.get_paper.side_effect = lambda doi: titles.get(doi)
    return sdb


def test_ranks_by_cosine_desc_and_caps() -> None:
    """top_k=2 → the two highest-cosine papers, sorted descending, titles resolved."""
    # Interest direction points along +x.
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb = _FakeEmbeddings(
        {
            "doi/aligned": np.array([1.0, 0.0, 0.0], dtype=np.float32),   # cos = 1.0
            "doi/partial": np.array([1.0, 1.0, 0.0], dtype=np.float32),   # cos ≈ 0.707
            "doi/orth": np.array([0.0, 1.0, 0.0], dtype=np.float32),      # cos = 0.0
            "doi/anti": np.array([-1.0, 0.0, 0.0], dtype=np.float32),     # cos = -1.0
        }
    )
    state_db = _state_db(
        {
            "doi/aligned": {"title": "Aligned"},
            "doi/partial": {"title": "Partial"},
        }
    )

    out = rank_papers_by_interest(vec, emb, state_db, top_k=2)

    assert [r["doi"] for r in out] == ["doi/aligned", "doi/partial"]
    assert all({"doi", "title", "cosine"} <= set(r) for r in out)
    assert out[0]["cosine"] >= out[1]["cosine"]
    assert out[0]["title"] == "Aligned"


def test_degenerate_interest_vector_returns_empty() -> None:
    """A zero (or NaN) interest vector ⇒ no ranking, empty list (no crash)."""
    emb = _FakeEmbeddings({"doi/a": np.array([1.0, 0.0], dtype=np.float32)})
    state_db = _state_db({})

    zeros = np.zeros(2, dtype=np.float32)
    assert rank_papers_by_interest(zeros, emb, state_db, top_k=5) == []

    nans = np.array([np.nan, np.nan], dtype=np.float32)
    assert rank_papers_by_interest(nans, emb, state_db, top_k=5) == []


def test_skips_bad_paper_embedding() -> None:
    """A zero/NaN-norm candidate embedding is skipped; the rest still rank."""
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb = _FakeEmbeddings(
        {
            "doi/good": np.array([1.0, 0.0, 0.0], dtype=np.float32),     # cos = 1.0
            "doi/zero": np.zeros(3, dtype=np.float32),                   # zero norm → skip
            "doi/nan": np.array([np.nan, 0.0, 0.0], dtype=np.float32),   # NaN norm → skip
        }
    )
    state_db = _state_db({"doi/good": {"title": "Good"}})

    out = rank_papers_by_interest(vec, emb, state_db, top_k=5)

    # Only the good candidate survives; no exception raised.
    assert [r["doi"] for r in out] == ["doi/good"]


def test_no_embeddings_store_returns_empty() -> None:
    """A None embeddings store ⇒ empty list (the /insights page passes None to skip)."""
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    state_db = _state_db({})
    assert rank_papers_by_interest(vec, None, state_db, top_k=5) == []


def test_title_falls_back_to_doi() -> None:
    """When state_db.get_paper(doi) → None, the title falls back to the bare DOI."""
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb = _FakeEmbeddings({"doi/unknown": np.array([1.0, 0.0, 0.0], dtype=np.float32)})
    state_db = _state_db({})  # get_paper → None for every doi

    out = rank_papers_by_interest(vec, emb, state_db, top_k=5)

    assert len(out) == 1
    assert out[0]["doi"] == "doi/unknown"
    assert out[0]["title"] == "doi/unknown"
