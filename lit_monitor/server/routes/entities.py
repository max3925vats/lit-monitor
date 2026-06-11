"""H6: Entity neighborhood + listing HTTP endpoints.

Routes:
    GET /api/entities                        — list entities by type.
    GET /api/entities/{canonical_id:path}    — entity neighborhood.

Registration order matters: the bare /api/entities route (no path param) is
registered BEFORE /api/entities/{canonical_id:path} so that a request without
a path segment hits the listing handler, not the neighborhood handler with an
empty canonical_id.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from lit_monitor.api.queries import get_entity_neighborhood, list_entities
from lit_monitor.core.config import get_config  # noqa: F401 — kept for future cfg use

# Module-level name so tests can monkeypatch without reaching the original module.
from lit_monitor.graph.import_citations import safe_graph_db

router = APIRouter(tags=["entities"])

# Closed vocabulary of entity types (H6 spec).
EntityType = Literal["topic", "method", "material", "author", "journal", "keyword"]


# ---------------------------------------------------------------------------
# GET /api/entities — listing route (MUST be registered before /{canonical_id:path})
# ---------------------------------------------------------------------------


@router.get("/api/entities")
def list_entities_route(
    type: EntityType = Query(..., description="Entity type to list"),
    top_k: int = Query(20, ge=1, le=200, description="Maximum number of results"),
) -> list[dict]:
    """List entities of a given type, ranked by mention count.

    Query parameters:
        type:   Entity type — one of topic, method, material, author, journal, keyword.
        top_k:  Maximum results, 1–200 (default 20).

    Returns:
        200 — list of {canonical_id, type, surface, mention_count} dicts.
        422 — missing/invalid type or top_k out of range.
        503 — graph backend unavailable.
    """
    graph_db = safe_graph_db()
    if graph_db is None:
        raise HTTPException(status_code=503, detail="graph backend unavailable")
    # list_entities signature: (entity_type, top_k, graph_db)
    return list_entities(type, top_k, graph_db)


# ---------------------------------------------------------------------------
# GET /api/entities/{canonical_id:path} — neighborhood (catch-all; registered LAST)
# ---------------------------------------------------------------------------


@router.get("/api/entities/{canonical_id:path}")
def entity_neighborhood(canonical_id: str) -> dict:
    """Return papers and co-entities for a given canonical entity.

    Path parameters:
        canonical_id: Normalized canonical entity identifier (may contain slashes).

    Returns:
        200 — {canonical_id, papers, co_entities}
        404 — entity not found (empty papers AND co_entities).
        422 — canonical_id is empty.
        503 — graph backend unavailable.
    """
    if not canonical_id or not canonical_id.strip():
        raise HTTPException(status_code=422, detail="canonical_id is required")

    graph_db = safe_graph_db()
    if graph_db is None:
        raise HTTPException(status_code=503, detail="graph backend unavailable")

    # get_entity_neighborhood signature: (canonical_id, graph_db) — both positional.
    result = get_entity_neighborhood(canonical_id, graph_db)

    # If both lists are empty the entity doesn't exist in the graph → 404.
    if not result.get("papers") and not result.get("co_entities"):
        raise HTTPException(
            status_code=404, detail=f"entity {canonical_id!r} not found"
        )

    return result


__all__ = ["router"]
