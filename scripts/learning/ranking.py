"""CU-1: shared interest-vector cosine ranker.

Single source of truth for "rank library papers by cosine alignment with the
Rocchio interest vector". Both call sites use it:

- the web ``/insights`` page, via ``scripts/api/insights._rank_top_papers``
  (a thin wrapper preserved as the test/patch seam), and
- the CLI ``learning view`` command in ``scripts/cli.py``.

The two call sites previously carried byte-identical copies of this logic (see
``CLAUDE.md`` → "Known gaps → Duplicate cosine ranker"); this module collapses
them so the guards can never silently drift. The CLI keeps its own cosmetic
rendering (signed ``+.3f`` cosine, 60-char title truncation) at the render site
— only the *computation* is shared here.

The ranker is DEFENSIVE: any read/compute failure degrades to ``[]`` rather than
raising, so neither the Insights page nor the CLI can be taken down by a
degenerate vector, a missing embeddings store, or a transient DB error.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["rank_papers_by_interest"]


def rank_papers_by_interest(
    interest_vec: np.ndarray,
    embeddings_db: Any,
    state_db: Any,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Rank library papers by cosine alignment with the interest vector.

    Finite- and zero-norm-guards both the interest vector and each library
    embedding; titles are resolved via ``state_db.get_paper(doi)`` and fall back
    to the bare DOI.

    Args:
        interest_vec:  The Rocchio interest vector (``np.ndarray``-like).
        embeddings_db: An ``EmbeddingsDB`` exposing ``get_paper_embeddings()``
                       (a ``{doi: vector}`` dict), or ``None`` to skip ranking.
        state_db:      A ``StateDB`` (duck-typed) exposing ``get_paper(doi)``.
        top_k:         Max number of aligned library papers to return.

    Returns:
        The top-``top_k`` ``{"doi", "title", "cosine"}`` dicts (``cosine``
        rounded to 4 dp), sorted by cosine descending. ``[]`` on any failure
        (degenerate vector, no embeddings store, cosine error).
    """
    try:
        i_norm = float(np.linalg.norm(interest_vec))
        if i_norm < 1e-9 or not np.isfinite(i_norm):
            return []  # degenerate interest vector ⇒ no ranking

        all_embeddings = embeddings_db.get_paper_embeddings() if embeddings_db else None
        if not all_embeddings:
            return []

        scores: list[tuple[str, float]] = []
        for doi, emb in all_embeddings.items():
            emb_arr = np.asarray(emb, dtype=np.float32)
            c_norm = float(np.linalg.norm(emb_arr))
            if c_norm < 1e-9 or not np.isfinite(c_norm):
                continue
            cos = float(np.dot(emb_arr, interest_vec) / (c_norm * i_norm))
            if np.isfinite(cos):
                scores.append((doi, cos))

        scores.sort(key=lambda t: t[1], reverse=True)

        top: list[dict] = []
        for doi, cos in scores[:top_k]:
            title = ""
            try:
                rec = state_db.get_paper(doi)
                title = (rec.get("title") or "") if rec else ""
            except Exception:  # noqa: BLE001 — title is cosmetic
                title = ""
            top.append({"doi": doi, "title": title or doi, "cosine": round(cos, 4)})
        return top
    except Exception as exc:  # noqa: BLE001 — ranking is a nice-to-have, never fatal
        logger.warning("interest-alignment ranking failed: %s", exc)
        return []
