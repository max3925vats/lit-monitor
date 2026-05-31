"""H4 + H5: paper-related HTTP endpoints.

H4: GET /api/papers/{doi}            — full paper snapshot.
H5: GET /api/papers/{doi}/related    — related papers (vector / graph / hybrid).

The graph backend is constructed lazily via safe_graph_db() so the server
can start without a kuzu installation; requests hit 503 instead of an
import error at boot time.

Route registration order:
  1. /api/papers/{doi:path}/related   ← more-specific suffix registered FIRST
  2. /api/papers/{doi:path}           ← catch-all registered SECOND
FastAPI resolves /api/papers/10.x/y/related against route 1 before 2, so
the suffix wins without ambiguity.
"""
from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from scripts.api.queries import get_paper_snapshot, get_related_papers

# Module-level names so tests can monkeypatch them without reaching into
# the original modules (patch the binding at the route's own namespace).
from scripts.graph.import_citations import safe_graph_db

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

router = APIRouter(tags=["papers"])


# ---------------------------------------------------------------------------
# H5: GET /api/papers/{doi:path}/related
# Registered BEFORE the bare {doi:path} route so the /related suffix is
# matched first by FastAPI's router.
# ---------------------------------------------------------------------------


@router.get("/api/papers/{doi:path}/related")
def related_papers(
    doi: str,
    mode: Literal["vector", "graph", "hybrid"] = Query("vector"),
    k: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Return papers related to the given seed DOI.

    Path parameters:
        doi:  Seed paper DOI (may contain slashes).

    Query parameters:
        mode: Retrieval strategy — "vector" (default), "graph", or "hybrid".
        k:    Number of results, 1–100 (default 20).

    Returns:
        200 — list of {doi, score} dicts, ranked descending.
        404 — seed DOI not found (get_related_papers returned None).
        422 — invalid DOI format, bad mode value, or k out of range.
        503 — graph backend unavailable.
    """
    # Strip trailing /related from path (FastAPI includes it in {doi:path}).
    # FastAPI captures the suffix literally, so doi will be e.g. "10.x/y/related"
    # — we need to strip the trailing "/related" segment that was part of the match.
    # Because FastAPI registers this route as /api/papers/{doi:path}/related, the
    # {doi:path} capture does NOT include "related" — it stops before the literal
    # "/related" suffix.  No stripping needed.

    if not _DOI_RE.match(doi):
        raise HTTPException(status_code=422, detail=f"invalid DOI: {doi!r}")

    graph_db = safe_graph_db()
    if graph_db is None:
        raise HTTPException(status_code=503, detail="graph backend unavailable")

    results = get_related_papers(doi, mode=mode, k=k, graph_db=graph_db)
    if results is None:
        raise HTTPException(status_code=404, detail=f"paper {doi!r} not found")
    return results


# ---------------------------------------------------------------------------
# H4: GET /api/papers/{doi:path} — paper snapshot (catch-all; must be LAST)
# ---------------------------------------------------------------------------


@router.get("/api/papers/{doi:path}")
def paper_snapshot(doi: str) -> dict:
    """Return a full snapshot of one paper.

    Path parameters:
        doi: DOI string (may contain slashes — captured by {doi:path}).

    Returns:
        200 — {metadata, entities_by_type, relationships_in, relationships_out}
        404 — paper not found in graph (metadata key is empty)
        422 — doi does not match ^10.\\d{4,9}/\\S+$ (rejected before hitting DB)
        503 — graph backend unavailable (kuzu not installed or DB unreachable)
    """
    if not _DOI_RE.match(doi):
        raise HTTPException(status_code=422, detail=f"invalid DOI: {doi!r}")

    # safe_graph_db returns None when the [graph] extra is not installed.
    # Tests monkeypatch this name to return a MagicMock or None as needed.
    graph_db = safe_graph_db()
    if graph_db is None:
        raise HTTPException(status_code=503, detail="graph backend unavailable")

    snapshot = get_paper_snapshot(doi, graph_db)

    # H1 convention: empty metadata dict means the DOI was not found.
    if not snapshot.get("metadata"):
        raise HTTPException(status_code=404, detail=f"paper {doi!r} not found")

    return snapshot


__all__ = ["router"]
