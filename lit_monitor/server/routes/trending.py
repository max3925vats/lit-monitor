"""Bundle E (v0.9): HTTP endpoints for trending-concept suggestion queue.

Endpoints
---------
GET    /api/trending                 — list pending trending suggestions
POST   /api/trending/{id}/accept     — accept suggestion; add to topics.yaml
POST   /api/trending/{id}/dismiss    — dismiss suggestion (respects cooldown)
GET    /trending                     — render suggestion queue UI (HTMX)

All persistent state lives in state.db::trending_concepts_suggested.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from lit_monitor.server.config_io import safe_save_topics
from lit_monitor.server.runtime import get_runtime

logger = logging.getLogger(__name__)

router = APIRouter()


def _safe_db() -> Any | None:
    """Return the state_db instance, or None on any failure."""
    try:
        return get_runtime().state_db
    except Exception:
        return None


@router.get("/api/trending")
def get_trending_suggestions() -> list[dict]:
    """Return all pending trending-concept suggestions, newest first.

    Returns an empty list when the state DB is unavailable or no suggestions
    exist yet.
    """
    db = _safe_db()
    if db is None:
        return []
    try:
        return db.get_pending_trending_suggestions()
    except Exception as exc:
        logger.warning("E: get_trending_suggestions failed: %s", exc)
        return []


@router.post("/api/trending/{suggestion_id}/accept")
def accept_trending_suggestion(suggestion_id: int) -> dict:
    """Accept a trending suggestion; add the concept to topics.yaml.

    Marks the row accepted in state.db and calls safe_save_topics() to
    append the concept to config/topics.yaml. Never raises on save failure —
    logs and returns the error message instead.

    Raises:
        HTTPException(404): If suggestion_id does not exist in state.db.
    """
    db = _safe_db()
    if db is None:
        raise HTTPException(status_code=503, detail="State DB unavailable")

    row = db.get_trending_suggestion_by_id(suggestion_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")

    # Persist the user action first.
    db.update_trending_action(suggestion_id, "accepted")

    # Add concept to topics.yaml as a new search entry.
    new_topic = {
        "name": row["concept_text"],
        "query": row["concept_text"],
        "databases": ["pubmed", "arxiv"],
        # Track that this was auto-added from trending suggestions.
        "source": f"trending_suggestion:{suggestion_id}",
    }
    try:
        safe_save_topics(new_topic)
        logger.info(
            "E: trending concept %r accepted and added to topics.yaml",
            row["concept_text"],
        )
    except Exception as exc:
        logger.warning(
            "E: accepted trending concept %r but safe_save_topics failed: %s",
            row["concept_text"],
            exc,
        )
        return {"status": "accepted", "saved_to_topics": False, "error": str(exc)}

    return {"status": "accepted", "saved_to_topics": True}


@router.post("/api/trending/{suggestion_id}/dismiss")
def dismiss_trending_suggestion(suggestion_id: int) -> dict:
    """Dismiss a trending suggestion; cooldown will prevent re-suggestion.

    The cooldown window is configured via trending_concepts.cooldown_days_after_dismiss
    (default 60 days). The action_at timestamp set here is checked by
    find_trending_concepts() on the next brain-build run.

    Raises:
        HTTPException(404): If suggestion_id does not exist in state.db.
    """
    db = _safe_db()
    if db is None:
        raise HTTPException(status_code=503, detail="State DB unavailable")

    row = db.get_trending_suggestion_by_id(suggestion_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")

    db.update_trending_action(suggestion_id, "dismissed")
    logger.info(
        "E: trending concept %r dismissed (id=%d)",
        row["concept_text"],
        suggestion_id,
    )
    return {"status": "dismissed"}


@router.get("/trending", response_class=HTMLResponse)
def trending_page(request: Request) -> HTMLResponse:
    """Render the trending-concept suggestion queue as an HTMX-driven HTML page."""
    from lit_monitor.server.app import templates

    db = _safe_db()
    suggestions: list[dict] = []
    if db is not None:
        try:
            suggestions = db.get_pending_trending_suggestions()
        except Exception as exc:
            logger.warning("E: trending page: failed to load suggestions: %s", exc)

    # Starlette 1.1+ rejects the legacy 2-arg ("name", {"request": ...}) form;
    # use the modern (request, name, context) signature (A3-2).
    return templates.TemplateResponse(
        request,
        "trending/index.html",
        {"suggestions": suggestions},
    )
