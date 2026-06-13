"""Brain-build dashboard read-only view (F3.1).

GET /brain-build — overall progress bar + per-paper table + last 10 runs +
failed-papers expander. Pure read access to the state DB; never mutates.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from lit_monitor.server.app import templates
from lit_monitor.server.runtime import get_runtime

logger = logging.getLogger(__name__)
router = APIRouter(tags=["brain-build"])

# Per-paper table page size (Task C — 10 rows per page).
_PAPERS_PAGE_SIZE = 10


def _safe_runtime():
    """Return the per-process runtime or None on any failure."""
    try:
        return get_runtime()
    except Exception:  # pragma: no cover - get_runtime itself does not raise
        logger.debug("get_runtime failed", exc_info=True)
        return None


def _safe_db():
    """Return the state_db, or None if creds/config unavailable.

    The runtime properties lazily build their clients; a missing config file
    or invalid path will raise here, so we swallow and return None so the
    dashboard can degrade to the "DB unavailable" panel.
    """
    r = _safe_runtime()
    if r is None:
        return None
    try:
        return r.state_db
    except Exception:
        logger.debug("state_db unavailable", exc_info=True)
        return None


def _safe_collection_name() -> str | None:
    """Read the configured Zotero collection name without raising."""
    r = _safe_runtime()
    if r is None:
        return None
    try:
        return getattr(r.config.zotero, "collection_name", None)
    except Exception:
        return None


def _extract_error(row: dict) -> str | None:
    """Pull an error message from extraction_json if present, else failure_reason.

    Brain-build writes failures into ``papers.extraction_json`` as
    ``{"error": "..."}`` for the simple/complex phases; the older path stores a
    short reason directly on ``brain_build_progress.failure_reason``. Either
    one is enough to flag the row as failed.
    """
    raw = row.get("paper_extraction_json")
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            data = None
        if isinstance(data, dict):
            err = data.get("error")
            if err:
                return str(err)
    return row.get("failure_reason")


def _authors_short(authors_json: str | None) -> str:
    """Render a short authors string from the JSON-encoded authors list."""
    if not authors_json:
        return ""
    try:
        authors = json.loads(authors_json)
    except Exception:
        return ""
    if not isinstance(authors, list) or not authors:
        return ""
    first = str(authors[0])
    if len(authors) > 1:
        return f"{first} et al."
    return first


@router.get("/brain-build", response_class=HTMLResponse)
def dashboard(
    request: Request,
    bb_offset: int = Query(0, ge=0),
) -> HTMLResponse:
    """Render the read-only brain-build dashboard.

    The per-paper table is paginated (``bb_offset``, 10 rows/page). The
    progress bar must reflect ALL papers, so we fetch the full set once, derive
    progress/failed-rows from it, then slice a 10-row window for the table.
    """
    db = _safe_db()
    if db is None:
        return templates.TemplateResponse(
            request,
            "brain_build/index.html",
            {
                "collection_name": None,
                "progress": None,
                "rows": [],
                "failed_rows": [],
                "recent_runs": [],
                "db_unavailable": True,
            },
        )

    collection = _safe_collection_name()
    try:
        # Fetch the full set so progress + failed_rows stay accurate across ALL
        # papers (not just the current page). Pagination slices a window below.
        full_rows = db.get_all_brain_build_progress(limit=100000)
    except Exception:
        logger.warning("get_all_brain_build_progress failed", exc_info=True)
        full_rows = []

    # Decorate the FULL set: failed_rows needs error_message across all papers,
    # and the table window reuses the same decorated dicts.
    for r in full_rows:
        r["error_message"] = _extract_error(r)
        r["authors_short"] = _authors_short(r.get("paper_authors"))

    total = len(full_rows)
    done = sum(1 for r in full_rows if r.get("fully_complete"))
    pct = round(100 * done / total) if total else 0

    failed_rows = [r for r in full_rows if r.get("error_message")]

    # Clamp the offset to a valid page start; slice the 10-row window.
    offset = max(bb_offset, 0)
    rows = full_rows[offset : offset + _PAPERS_PAGE_SIZE]

    try:
        recent_runs = db.get_recent_runs_by_type("brain_build", limit=10)
    except Exception:
        logger.debug("get_recent_runs_by_type failed", exc_info=True)
        recent_runs = []

    return templates.TemplateResponse(
        request,
        "brain_build/index.html",
        {
            "collection_name": collection,
            "progress": {"done": done, "total": total, "pct": pct},
            "rows": rows,
            "failed_rows": failed_rows,
            "recent_runs": recent_runs,
            # Per-paper table pagination context (Task C).
            "bb_offset": offset,
            "bb_total": total,
            "bb_window_start": (offset + 1) if rows else 0,
            "bb_window_end": offset + len(rows),
            "bb_has_prev": offset > 0,
            "bb_has_next": offset + _PAPERS_PAGE_SIZE < total,
            "bb_prev_offset": max(offset - _PAPERS_PAGE_SIZE, 0),
            "bb_next_offset": offset + _PAPERS_PAGE_SIZE,
            "db_unavailable": False,
        },
    )


__all__ = ["router"]
