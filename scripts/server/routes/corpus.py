"""FE2-2: the /corpus list page — a read lens over the processed-papers corpus.

GET /corpus renders a searchable / filterable / sortable / paginated table over
``StateDB.list_papers`` (FE2-1). When the request carries an ``HX-Request``
header (the filter form's hx-get), only the ``corpus/_table.html`` fragment is
returned so HTMX can swap it into ``#corpus-table`` in place; otherwise the full
``corpus/index.html`` page is rendered.

``_list_papers`` is a module-level wrapper around the real
``StateDB.list_papers`` so route tests can patch the single seam
``scripts.server.routes.corpus._list_papers`` without a live SQLite DB.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)

corpus_router = APIRouter(tags=["corpus"])

# status_gap select options surfaced in the UI. "" = no gap filter ("(any)").
_STATUS_GAP_OPTIONS = (
    "",
    "missing_graph",
    "missing_notes",
    "missing_embeddings",
    "low_confidence",
)


def _get_templates():
    """Lazy import to avoid a circular dependency at module load.

    Mirrors the accessor used by ask.py / themes.py: the ``templates`` object is
    created in app.py, which imports this router inside create_app().
    """
    from scripts.server.app import templates  # noqa: PLC0415

    return templates


def _list_papers(**kwargs: Any) -> tuple[list[dict], int]:
    """Resolve the real StateDB and delegate to ``list_papers``.

    Single patch point for the route tests — they replace this function so no
    live state DB is required. Returns ``(rows, total)``.
    """
    return get_runtime().state_db.list_papers(**kwargs)


@corpus_router.get("/corpus", response_class=HTMLResponse)
def corpus_index(
    request: Request,
    search: str = Query(""),
    source_type: str = Query(""),
    status_gap: str = Query(""),
    theme: str = Query(""),
    sort: str = Query("last_updated"),
    order: str = Query("desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> HTMLResponse:
    """Render the corpus list page (full page) or its table fragment (HTMX)."""
    try:
        rows, total = _list_papers(
            search=search or None,
            source_type=source_type or None,
            status_gap=status_gap or None,
            theme=theme or None,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
    except Exception:  # noqa: BLE001 — advisory page surface, never 500-leak
        # Full detail (message + traceback) stays server-side only; the browser
        # gets an empty result set so the page still renders.
        logger.error("corpus listing failed", exc_info=True)
        rows, total = [], 0

    ctx = {
        "rows": rows,
        "total": total,
        "search": search,
        "source_type": source_type,
        "status_gap": status_gap,
        "theme": theme,
        "sort": sort,
        "order": order,
        "limit": limit,
        "offset": offset,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        "prev_offset": max(offset - limit, 0),
        "next_offset": offset + limit,
        "status_gap_options": _STATUS_GAP_OPTIONS,
    }

    templates = _get_templates()
    # HTMX filter/sort/paginate requests want only the table region back.
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "corpus/_table.html", ctx)
    return templates.TemplateResponse(request, "corpus/index.html", ctx)


__all__ = ["corpus_router"]
