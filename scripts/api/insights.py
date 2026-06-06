"""FI-1: read-only assemblers for the Insights page.

Two functions sit on top of the P5 active-learning engine and shape its state
into JSON-friendly dicts for the HTTP/UI layer:

- ``get_learning_state`` — the global interest vector summary + (optionally) the
  top library papers most aligned with it.
- ``get_cluster_weights`` — per-cluster atrophy feedback weights, floor-aware.

Both are strictly READ-ONLY (no writes, no engine recompute) and DEFENSIVE: any
read/compute failure degrades to an empty/absent result rather than raising to
the caller, so the Insights page can never be taken down by a degenerate vector,
a missing embeddings store, or a transient DB error.

The cosine-ranking helper here is a clean standalone port of the ranking logic
in ``scripts/cli.py`` (the ``learning view`` command). It is NOT a refactor of
the CLI — the CLI still owns its own copy. If this ranking is ever generalised,
both call sites should be unified to avoid silent divergence.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scripts.clustering.atrophy import compute_cluster_feedback_weights_from_db
from scripts.learning.rocchio import soft_gate

logger = logging.getLogger(__name__)

__all__ = ["get_learning_state", "get_cluster_weights"]


def _read_interest_computed_at(state_db: Any, scope: str = "global") -> str:
    """Best-effort read of ``interest_vectors.computed_at`` for ``scope``.

    Mirrors the raw-SQL read the CLI uses. Degrades to ``"unknown"`` on any
    failure — the timestamp is cosmetic and must never crash the summary.
    """
    try:
        with state_db._connect() as conn:
            row = conn.execute(
                "SELECT computed_at FROM interest_vectors WHERE scope = ?",
                (scope,),
            ).fetchone()
        return (row["computed_at"] if row else "") or "unknown"
    except Exception as exc:  # noqa: BLE001 — cosmetic field, never fatal
        logger.warning("interest computed_at read failed: %s", exc)
        return "unknown"


def _rank_top_papers(
    state_db: Any,
    embeddings_db: Any,
    interest_vec: np.ndarray,
    top_k: int,
) -> list[dict]:
    """Rank library papers by cosine alignment with the interest vector.

    Standalone port of the ``learning view`` ranking in ``cli.py``. Finite- and
    zero-norm-guards both the interest vector and each library embedding; titles
    are resolved via ``state_db.get_paper(doi)`` and fall back to the bare DOI.

    Returns the top-``top_k`` ``{"doi", "title", "cosine"}`` dicts, or ``[]`` on
    any failure (degenerate vector, no embeddings store, cosine error).
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


def get_learning_state(
    state_db: Any,
    embeddings_db: Any,
    *,
    top_k: int = 10,
) -> dict:
    """Assemble the global interest-vector summary for the Insights page.

    Args:
        state_db:      A ``StateDB`` (duck-typed). Must expose
                       ``get_interest_vector("global")``, ``_connect()`` and
                       ``get_paper(doi)``.
        embeddings_db: An ``EmbeddingsDB`` exposing ``get_paper_embeddings()``,
                       or ``None`` to skip the alignment ranking.
        top_k:         Max number of aligned library papers to surface.

    Returns:
        When no interest vector is stored: ``{"available": False}``.
        Otherwise a dict with keys: ``available`` (True), ``n_events``,
        ``soft_gate`` (in ``[0, 1)``), ``inert`` (True iff the gate is 0.0 —
        below the cold-start floor), ``computed_at``, ``dim``, and
        ``top_papers`` (list of ``{"doi", "title", "cosine"}``; ``[]`` when the
        ranking cannot be computed).
    """
    try:
        stored = state_db.get_interest_vector("global")
    except Exception as exc:  # noqa: BLE001 — degrade to "absent"
        logger.warning("get_interest_vector failed: %s", exc)
        return {"available": False}

    if stored is None:
        return {"available": False}

    interest_vec, n_events = stored
    gate = soft_gate(n_events)

    # Finite-guard the vector before computing dim / ranking.
    vec = np.asarray(interest_vec, dtype=np.float32)
    finite = bool(vec.size) and bool(np.all(np.isfinite(vec)))

    summary: dict[str, Any] = {
        "available": True,
        "n_events": int(n_events),
        "soft_gate": float(gate),
        "inert": gate <= 0.0,
        "computed_at": _read_interest_computed_at(state_db, "global"),
        "dim": int(vec.shape[0]) if vec.ndim >= 1 else 0,
        "top_papers": [],
    }

    if finite:
        summary["top_papers"] = _rank_top_papers(
            state_db, embeddings_db, vec, top_k
        )
    return summary


def get_cluster_weights(state_db: Any, *, floor: float) -> dict:
    """Assemble per-cluster atrophy feedback weights for the Insights page.

    Args:
        state_db: A ``StateDB`` exposing ``list_active_clusters()`` and the
                  surface ``compute_cluster_feedback_weights_from_db`` needs.
        floor:    The atrophy floor (config ``feedback.minimum_cluster_floor``);
                  a weight never decays below it.

    Returns:
        ``{"floor": floor, "clusters": [...]}`` where each cluster is
        ``{"id", "name", "weight", "at_floor"}`` (``at_floor`` iff
        ``weight <= floor``), sorted by ``weight`` ascending (most-atrophied
        first). Degrades to ``{"floor": floor, "clusters": []}`` on any error.
    """
    try:
        weights = compute_cluster_feedback_weights_from_db(state_db, floor=floor)
        clusters = state_db.list_active_clusters()
    except Exception as exc:  # noqa: BLE001 — page must never crash
        logger.warning("get_cluster_weights failed: %s", exc)
        return {"floor": floor, "clusters": []}

    out: list[dict] = []
    for cluster in clusters:
        try:
            cid = int(cluster["id"])
        except (KeyError, TypeError, ValueError):
            continue
        # A cluster with no assignments has no entry in `weights` ⇒ full weight.
        weight = float(weights.get(cid, 1.0))
        name = cluster.get("display_name") or f"Cluster {cid}"
        out.append(
            {
                "id": cid,
                "name": name,
                "weight": weight,
                "at_floor": weight <= floor,
            }
        )

    out.sort(key=lambda c: c["weight"])
    return {"floor": floor, "clusters": out}
