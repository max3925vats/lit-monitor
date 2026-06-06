"""Bundle H (v0.9): feedback event HTTP endpoints.

Routes:
    POST /api/feedback                       — record a feedback event
    GET  /api/feedback/recent                — recent events (paginated)
    GET  /api/feedback/summary               — counts per signal_type
    GET  /insights                           — read-only insights HTML page
    GET  /feedback                           — redirect → /insights (FI-2 rename)
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from scripts.api import insights as insights_api
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
# FI-2: read-only Insights page seams
# ---------------------------------------------------------------------------

def _cluster_floor() -> float:
    """Resolve the atrophy floor (``feedback.minimum_cluster_floor``).

    Module-level seam; defensive → defaults to 0.1 when config is unreadable.
    """
    try:
        return float(get_runtime().config.feedback.minimum_cluster_floor)
    except Exception:  # noqa: BLE001 — config absent ⇒ documented default
        return 0.1


def _learning_state() -> dict:
    """Assemble the global learning-state summary (read-only).

    Wraps ``insights.get_learning_state`` over the runtime's state + embeddings
    DBs. Module-level seam so the page can be tested without a live engine.
    Defensive: any failure degrades to ``{"available": False}``.
    """
    try:
        rt = get_runtime()
        return insights_api.get_learning_state(rt.state_db, rt.embeddings_db)
    except Exception as exc:  # noqa: BLE001 — page must never 500 on a bad vector
        logger.warning("insights learning-state read failed: %s", exc)
        return {"available": False}


def _cluster_weights() -> dict:
    """Assemble per-cluster atrophy weights (read-only).

    Wraps ``insights.get_cluster_weights`` with the configured floor. Module-level
    seam. Defensive: degrades to ``{"floor": …, "clusters": []}`` on any failure.
    """
    floor = _cluster_floor()
    try:
        return insights_api.get_cluster_weights(get_runtime().state_db, floor=floor)
    except Exception as exc:  # noqa: BLE001 — page must never 500 on a cluster read
        logger.warning("insights cluster-weights read failed: %s", exc)
        return {"floor": floor, "clusters": []}


# ---------------------------------------------------------------------------
# GET /insights — read-only insights HTML page (FI-2)
# ---------------------------------------------------------------------------

@router.get("/insights", response_class=HTMLResponse)
def insights_page(request: Request) -> HTMLResponse:
    """Render the read-only Insights page.

    Sections: learning-state card, cluster-weights table, server-rendered
    inline-SVG signal-mix chart, and the recent-events table. All reads are
    defensive — a degenerate engine state degrades to graceful empties rather
    than 500-ing.
    """
    db = _safe_db()
    recent: list[dict] = []
    summary: dict[str, int] = {}
    if db is not None:
        try:
            recent = db.list_feedback_events(limit=100)
            summary = db.feedback_summary()
        except Exception as exc:  # noqa: BLE001 — DB read is non-fatal for the page
            logger.warning("insights page: DB read failed: %s", exc)

    learning = _learning_state()
    clusters = _cluster_weights()

    templates = _get_templates()
    return templates.TemplateResponse(
        request,
        "insights/index.html",
        {
            "recent": recent,
            "summary": summary,
            "learning": learning,
            "clusters": clusters,
        },
    )


# ---------------------------------------------------------------------------
# GET /feedback — redirect to /insights (page renamed in FI-2)
# ---------------------------------------------------------------------------

@router.get("/feedback")
def feedback_page() -> RedirectResponse:
    """Permanent in-app redirect: the old /feedback page is now /insights."""
    return RedirectResponse("/insights", status_code=307)
