"""H4: GET /api/papers/{doi} — paper snapshot.

Wraps H1's get_paper_snapshot so the web UI and MCP clients share one
contract.  Returns the same shape that MCP get_paper_details returns:
{metadata, entities_by_type, relationships_in, relationships_out}.

The graph backend is constructed lazily via safe_graph_db() so the server
can start without a kuzu installation; requests to this endpoint will get
503 instead of an import error at boot time.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from scripts.api.queries import get_paper_snapshot

# safe_graph_db is a module-level name so tests can monkeypatch it.
from scripts.graph.import_citations import safe_graph_db

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

router = APIRouter(tags=["papers"])


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
