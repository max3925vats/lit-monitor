"""G8: Reciprocal Rank Fusion combiner for hybrid retrieval.

Used by G9 to merge vector + graph ranked DOI lists into one hybrid ordering.
Pure stateless helper — no DB, no IO.

Reference: Cormack, Clarke, Buettcher (2009) "Reciprocal Rank Fusion outperforms
Condorcet and individual rank learning methods", SIGIR 2009.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists into one via RRF: score(d) = Σ 1/(k + rank_i(d)).

    Docs absent from a ranking contribute 0 (not 1/k). Ties are broken by
    first-appearance order across the input rankings (deterministic, stable).

    Args:
        rankings: list of ranked DOI lists. Order within each list is the rank
            (position 0 = rank 1, position 1 = rank 2, ...).
        k: smoothing constant, standard default 60.

    Returns:
        List of (doi, fused_score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}

    for ranking in rankings:
        for position, doc in enumerate(ranking):
            rank = position + 1  # rank is 1-indexed in the RRF formula
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            if doc not in first_seen:
                first_seen[doc] = len(first_seen)

    # Sort: primary key = -score (desc); secondary key = first_seen (asc, stable)
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_seen[kv[0]]),
    )
