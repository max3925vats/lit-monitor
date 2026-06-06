"""Bundle H (v0.9): feedback event HTTP endpoints.

Routes:
    POST /api/feedback                       — record a feedback event
    GET  /api/feedback/recent                — recent events (paginated)
    GET  /api/feedback/summary               — counts per signal_type
    GET  /insights                           — read-only insights HTML page
    GET  /feedback                           — redirect → /insights (FI-2 rename)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from scripts.api import insights as insights_api
from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])

#: FI-3: default lookback window (in weeks) for the feedback timeline. Bounds the
#: SVG so the chart stays readable and uses FI-1's ``feedback_timeline(weeks=…)``
#: branch rather than scanning all history.
_TIMELINE_DEFAULT_WEEKS: int = 12

#: FI-3: accepted timeline granularities. Any other value falls back to "week".
_TIMELINE_GRANULARITIES: frozenset[str] = frozenset({"week", "day"})


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


def _timeline(granularity: str, weeks: int | None) -> list[dict]:
    """FI-3: feedback-timeline rows for *granularity* over the last *weeks*.

    Thin module-level seam wrapping ``StateDB.feedback_timeline`` so the
    timeline fragment can be tested without a live state DB (route tests patch
    this single function). Unlike the other seams here, this one does NOT swallow
    exceptions — the fragment route owns the 500-leak guard so a failed read is
    logged once at the route boundary with a generic notice returned.
    """
    return get_runtime().state_db.feedback_timeline(granularity, weeks=weeks)


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
# FI-3: GET /insights/timeline — lazy inline-SVG feedback timeline fragment
# ---------------------------------------------------------------------------

@router.get("/insights/timeline", response_class=HTMLResponse)
async def insights_timeline(
    request: Request,
    granularity: str = Query("week"),
) -> HTMLResponse:
    """Lazy fragment: feedback-event counts over the last lookback window.

    HTMX-loaded into ``#insights-timeline`` on the Insights page. The bounded
    lookback (``_TIMELINE_DEFAULT_WEEKS``) both keeps the SVG readable and
    exercises FI-1's ``feedback_timeline(weeks=…)`` branch. Off-thread (the read
    touches SQLite). Any failure → a generic notice fragment with a full
    server-side log — the exception text never reaches the browser.
    """
    # Defensive: an unknown granularity from a hand-crafted URL falls back to the
    # default rather than raising, so the fragment always renders.
    if granularity not in _TIMELINE_GRANULARITIES:
        granularity = "week"
    weeks = _TIMELINE_DEFAULT_WEEKS
    try:
        rows = await asyncio.to_thread(_timeline, granularity, weeks)
    except Exception as exc:  # noqa: BLE001 — advisory fragment, never 500-leak
        # exc is in the message (not just the traceback) so a log scrape catches
        # the cause; the browser only ever sees the generic notice in-template.
        logger.error("insights timeline fragment failed: %s", exc, exc_info=True)
        rows, weeks = [], weeks
        error = True
    else:
        error = False

    return _get_templates().TemplateResponse(
        request,
        "insights/_timeline.html",
        {
            "rows": rows,
            "granularity": granularity,
            "weeks": weeks,
            "error": error,
        },
    )


# ---------------------------------------------------------------------------
# FI-3: GET /insights/top-papers — lazy top-aligned-papers fragment
# ---------------------------------------------------------------------------

@router.get("/insights/top-papers", response_class=HTMLResponse)
async def insights_top_papers(request: Request) -> HTMLResponse:
    """Lazy fragment: library papers most aligned with the interest vector.

    HTMX-loaded into ``#insights-top-papers`` on the Insights page. Reuses
    ``_learning_state`` (which already degrades to ``{"available": False}`` on a
    bad vector) so this route is itself 500-leak-safe: the only extra guard is a
    generic notice if the seam unexpectedly raises.
    """
    try:
        learning = await asyncio.to_thread(_learning_state)
    except Exception as exc:  # noqa: BLE001 — advisory fragment, never 500-leak
        logger.error("insights top-papers fragment failed: %s", exc, exc_info=True)
        learning = {"available": False}

    return _get_templates().TemplateResponse(
        request,
        "insights/_top_papers.html",
        {"learning": learning},
    )


# ---------------------------------------------------------------------------
# GET /feedback — redirect to /insights (page renamed in FI-2)
# ---------------------------------------------------------------------------

@router.get("/feedback")
def feedback_page() -> RedirectResponse:
    """Permanent in-app redirect: the old /feedback page is now /insights."""
    return RedirectResponse("/insights", status_code=307)
