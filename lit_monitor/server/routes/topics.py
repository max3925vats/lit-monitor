"""Bundle E (v0.9): HTTP endpoints for topic-level graph query expansion.

Endpoints
---------
GET  /api/topics/{topic_name}/expansion-suggestions  — return top-k co-entity suggestions

Results are suggestion-only — never auto-applied (per locked decision #4).
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException

from lit_monitor.server.runtime import get_runtime

logger = logging.getLogger(__name__)

router = APIRouter()


def _safe_graph_db() -> Any | None:
    """Return the graph_db instance, or None on any failure."""
    try:
        rt = get_runtime()
        return getattr(rt, "graph_db", None)
    except Exception:
        return None


@router.get("/api/topics/{topic_name}/expansion-suggestions")
def get_expansion_suggestions(topic_name: str, top_k: int = 3) -> dict:
    """Return top-k co-occurring entity suggestions for a topic's primary entity.

    Args:
        topic_name: URL-encoded topic name (matched against Entity.surface).
        top_k:      Maximum number of suggestions to return (default 3).

    Returns:
        {topic_name, suggestions: [str], top_k_requested}

    Note: Suggestion-only. The caller is responsible for presenting these to
    the user and applying them only after explicit opt-in.
    """
    from lit_monitor.graph.query_expansion import suggest_expansions

    name = unquote(topic_name).strip()
    if not name:
        raise HTTPException(status_code=422, detail="topic_name must not be empty")

    graph_db = _safe_graph_db()
    if graph_db is None:
        logger.info("E: graph DB unavailable for expansion suggestions")
        return {"topic_name": name, "suggestions": [], "top_k_requested": top_k}

    suggestions = suggest_expansions(name, graph_db, top_k=top_k)
    return {
        "topic_name": name,
        "suggestions": suggestions,
        "top_k_requested": top_k,
    }
