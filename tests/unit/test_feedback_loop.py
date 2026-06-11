"""
Bundle J2 integration: the active-learning loop demonstrably learns.

This is the end-to-end "does the loop actually learn" proof, kept fully offline
and deterministic with SYNTHETIC embeddings (a fake embed_lookup). It exercises
the real pure core (build_interest_inputs → compute_interest_vector) followed by
the real ranker leg (rank_papers with interest_weight > 0), and asserts that a
candidate aligned with the SAVED direction out-ranks one aligned with the
DISMISSED direction.

No Ollama, no ChromaDB, no DB — every vector is hand-built.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np

from lit_monitor.learning.rocchio import (
    build_interest_inputs,
    compute_interest_vector,
    soft_gate,
)
from lit_monitor.llm.ranker import rank_papers

# Two orthogonal "taste directions" in a tiny embedding space.
_LIKED_DIR = np.array([1.0, 0.0, 0.0], dtype=np.float32)       # saved toward this
_DISLIKED_DIR = np.array([0.0, 1.0, 0.0], dtype=np.float32)    # dismissed toward this
_NEUTRAL_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)     # library filler


def _event(doi: str, signal_type: str, weight: float) -> dict:
    return {
        "doi": doi,
        "signal_type": signal_type,
        "weight": weight,
        "rating": None,
        "source": None,
        "created_at": "2026-06-01 12:00:00",
    }


def _make_db(score: float = 0.5) -> MagicMock:
    db = MagicMock()
    db.find_similar_to_text.return_value = [
        {"score": score, "id": "y", "document": "", "metadata": {}}
    ]
    return db


def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = "{}"
    return llm


def test_loop_learns_saved_direction_ranks_above_dismissed_direction():
    """Log ≥10 positive feedback events clustered toward _LIKED_DIR plus a few
    negatives toward _DISLIKED_DIR, recompute the interest vector, then rank a
    liked-aligned vs a disliked-aligned candidate with feedback weight on. The
    liked-aligned candidate must rank ABOVE the disliked-aligned one."""

    # --- 1. Synthetic feedback + embedding library (the embed_lookup source). ---
    embeddings: dict[str, np.ndarray] = {}
    events: list[dict] = []

    # 12 SAVED papers, each near _LIKED_DIR (small jitter so they're distinct).
    rng = np.random.default_rng(42)
    for i in range(12):
        doi = f"10.1/like{i}"
        jitter = rng.normal(0.0, 0.02, size=3).astype(np.float32)
        embeddings[doi] = (_LIKED_DIR + jitter).astype(np.float32)
        events.append(_event(doi, "saved", 1.0))

    # 3 DISMISSED papers, each near _DISLIKED_DIR.
    for i in range(3):
        doi = f"10.1/dis{i}"
        jitter = rng.normal(0.0, 0.02, size=3).astype(np.float32)
        embeddings[doi] = (_DISLIKED_DIR + jitter).astype(np.float32)
        events.append(_event(doi, "dismissed", 1.0))

    # A few neutral library papers so the centroid is not degenerate.
    for i in range(5):
        embeddings[f"10.1/neutral{i}"] = _NEUTRAL_DIR.copy()

    def embed_lookup(doi: str) -> np.ndarray | None:
        return embeddings.get(doi)

    # --- 2. Recompute the interest vector (real pure core). ---
    library_centroid = np.mean(
        np.stack(list(embeddings.values())), axis=0
    ).astype(np.float32)
    positive, negative, n_events = build_interest_inputs(
        events, embed_lookup, now=datetime(2026, 6, 5)
    )
    interest_vec = compute_interest_vector(library_centroid, positive, negative)

    # Sanity: enough events to open the cold-start gate, and a finite direction.
    assert n_events == 15  # 12 saved + 3 dismissed
    assert soft_gate(n_events) > 0.0
    assert np.isfinite(interest_vec).all()
    # The learned direction should lean toward LIKED and away from DISLIKED.
    assert float(np.dot(interest_vec, _LIKED_DIR)) > float(
        np.dot(interest_vec, _DISLIKED_DIR)
    )

    # --- 3. Rank two fresh candidates with the feedback weight ON. ---
    liked_candidate = {
        "doi": "10.1/cand-liked",
        "title": "Candidate aligned with what you SAVED",
        "abstract": "",
        "_embedding": _LIKED_DIR.copy(),
    }
    disliked_candidate = {
        "doi": "10.1/cand-disliked",
        "title": "Candidate aligned with what you DISMISSED",
        "abstract": "",
        "_embedding": _DISLIKED_DIR.copy(),
    }

    # Same base similarity for both → only the interest leg can separate them.
    ranked = rank_papers(
        [disliked_candidate, liked_candidate],  # input order favors the disliked one
        _make_db(0.5),
        _make_llm(),
        interest_vec=interest_vec,
        interest_vec_n_events=n_events,
        interest_weight=0.5,  # opt-in: feedback signal ON
    )

    by_doi = {p["doi"]: p for p in ranked}
    liked_score = by_doi["10.1/cand-liked"]["similarity_score"]
    disliked_score = by_doi["10.1/cand-disliked"]["similarity_score"]

    # The loop learned: the saved-direction candidate now scores higher AND
    # sorts first, despite being passed in second.
    assert liked_score > disliked_score, (
        f"feedback loop failed to lift the liked candidate: "
        f"liked={liked_score} vs disliked={disliked_score}"
    )
    assert ranked[0]["doi"] == "10.1/cand-liked"
    # And the breakdown attributes the lift to the feedback signal.
    assert by_doi["10.1/cand-liked"]["score_breakdown"]["feedback"] > 0.0


def test_loop_inert_without_weight_even_after_learning():
    """Same learned vector, but feedback weight 0 → ranking is unchanged
    (the saved-direction candidate gets no lift). Proves the opt-in contract."""
    embeddings = {
        f"10.1/like{i}": _LIKED_DIR.copy() for i in range(12)
    }
    events = [_event(f"10.1/like{i}", "saved", 1.0) for i in range(12)]

    def embed_lookup(doi: str) -> np.ndarray | None:
        return embeddings.get(doi)

    centroid = np.mean(np.stack(list(embeddings.values())), axis=0).astype(np.float32)
    pos, neg, n = build_interest_inputs(events, embed_lookup, now=datetime(2026, 6, 5))
    vec = compute_interest_vector(centroid, pos, neg)

    cand = {
        "doi": "10.1/cand",
        "title": "Aligned candidate",
        "abstract": "",
        "_embedding": _LIKED_DIR.copy(),
    }
    ranked = rank_papers(
        [cand],
        _make_db(0.5),
        _make_llm(),
        interest_vec=vec,
        interest_vec_n_events=n,
        interest_weight=0.0,  # OFF
    )
    # No lift: score equals the base similarity, no feedback contribution.
    assert ranked[0]["similarity_score"] == 0.5
    assert ranked[0]["score_breakdown"]["feedback"] == 0.0
    assert "_interest_score" not in ranked[0]
