"""Bundle H (v0.9): feedback event HTTP endpoints.

Routes:
    POST /api/feedback                       — record a feedback event
    GET  /api/feedback/recent                — recent events (paginated)
    GET  /api/feedback/summary               — counts per signal_type
    GET  /feedback                           — admin/debug HTML page
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_db() -> Any | None:
    """Return state_db or None on failure."""
    try:
        return get_runtime().state_db
    except Exception:
        return None


def _get_templates():
    """Lazy import to avoid circular dependency at module load."""
    from scripts.server.app import templates
    return templates


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class FeedbackBody(BaseModel):
    """Request body for POST /api/feedback."""

    doi: str
    signal_type: Literal["opened", "saved", "dismissed", "rated", "thumbs_up", "thumbs_down"]
    rating: int | None = None
    source: str | None = None


# ---------------------------------------------------------------------------
# POST /api/feedback — record event
# ---------------------------------------------------------------------------

@router.post("/api/feedback")
def post_feedback(body: FeedbackBody) -> dict:
    """Record one feedback event.

    Validates signal_type against the closed vocabulary (enforced by the
    Literal type annotation) and applies additional cross-field rules:
      - signal_type='rated' requires rating in [1, 5].
      - rating is rejected for any other signal_type (422).

    Returns:
        {"ok": True} on success.

    Raises:
        HTTPException(422): On cross-field validation failure.
        HTTPException(503): When state DB is unavailable.
    """
    # Cross-field validation: rating semantics.
    if body.signal_type == "rated":
        if body.rating is None or not (1 <= body.rating <= 5):
            raise HTTPException(
                status_code=422,
                detail="rating required (1-5) when signal_type='rated'",
            )
    elif body.rating is not None:
        raise HTTPException(
            status_code=422,
            detail="rating only valid when signal_type='rated'",
        )

    db = _safe_db()
    if db is None:
        raise HTTPException(status_code=503, detail="State DB unavailable")

    try:
        db.record_feedback_event(
            body.doi,
            body.signal_type,
            rating=body.rating,
            source=body.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"ok": True}


# ---------------------------------------------------------------------------
# GET /api/feedback/recent — paginated events
# ---------------------------------------------------------------------------

@router.get("/api/feedback/recent")
def get_feedback_recent(
    limit: int = Query(default=50, ge=1, le=500),
    since: str | None = Query(default=None),
) -> list[dict]:
    """Return recent feedback events, newest first.

    Query params:
        limit: Max rows (default 50, max 500).
        since: ISO date/datetime lower bound (optional).
    """
    db = _safe_db()
    if db is None:
        return []
    try:
        return db.list_feedback_events(limit=limit, since=since)
    except Exception as exc:
        logger.warning("feedback/recent failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# GET /api/feedback/summary — counts per signal_type
# ---------------------------------------------------------------------------

@router.get("/api/feedback/summary")
def get_feedback_summary() -> dict[str, int]:
    """Return total counts per signal_type over all time."""
    db = _safe_db()
    if db is None:
        return {}
    try:
        return db.feedback_summary()
    except Exception as exc:
        logger.warning("feedback/summary failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# GET /feedback — HTML admin/debug page
# ---------------------------------------------------------------------------

@router.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request) -> HTMLResponse:
    """Render the feedback admin page — recent events + Chart.js summary."""
    db = _safe_db()
    recent: list[dict] = []
    summary: dict[str, int] = {}
    if db is not None:
        try:
            recent = db.list_feedback_events(limit=100)
            summary = db.feedback_summary()
        except Exception as exc:
            logger.warning("feedback page: DB read failed: %s", exc)

    templates = _get_templates()
    return templates.TemplateResponse(
        request,
        "feedback/index.html",
        {"recent": recent, "summary": summary},
    )
