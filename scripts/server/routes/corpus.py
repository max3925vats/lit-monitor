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

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from scripts.graph.import_citations import safe_graph_db  # patched in tests
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


# ---------------------------------------------------------------------------
# FE2-3: GET /corpus/{doi:path} — per-paper detail page.
# Registered AFTER GET /corpus so the bare list route is not swallowed by the
# {doi:path} param (FastAPI matches in declaration order).
# ---------------------------------------------------------------------------


def _get_paper_row(doi: str) -> dict | None:
    """Fetch one paper's display record from state.db, or ``None`` if absent.

    Module-level seam the detail route tests monkeypatch (no live SQLite). The
    raw ``extraction_json`` column is parsed into an ``extraction`` dict; the
    ``authors`` column (a JSON array string) is parsed into a list. Both parses
    are defensive: a bad/empty blob yields ``{}`` / ``[]`` rather than raising.
    """
    record = get_runtime().state_db.get_paper(doi)
    if record is None:
        return None

    raw = record.get("extraction_json")
    try:
        extraction = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        extraction = {}
    if not isinstance(extraction, dict):
        extraction = {}

    raw_authors = record.get("authors")
    try:
        authors = json.loads(raw_authors) if raw_authors else []
    except (ValueError, TypeError):
        authors = []
    if not isinstance(authors, list):
        authors = []

    return {
        "doi": record.get("doi"),
        "title": record.get("title"),
        "authors": authors,
        "year": record.get("year"),
        "journal": record.get("journal"),
        "source_type": record.get("source_type"),
        "zotero_key": record.get("zotero_key"),
        "note_path": record.get("note_path"),
        "extraction": extraction,
    }


def _get_score_breakdown(doi: str) -> dict | None:
    """Return the most recent per-signal score decomposition for *doi*, or None.

    Reuses the Bundle B query from ``scripts/server/routes/papers.py`` — the
    latest non-null ``score_breakdown_json`` row in ``discovery_paper_results``.
    Returns ``{"doi", "run_id", "breakdown": {...}}`` or ``None`` when the paper
    has no stored breakdown (e.g. never surfaced by discovery). Module-level so
    the detail route tests can patch it.
    """
    state_db = get_runtime().state_db
    with state_db._connect() as conn:
        row = conn.execute(
            "SELECT score_breakdown_json, run_id, doi "
            "FROM discovery_paper_results "
            "WHERE doi = ? AND score_breakdown_json IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (doi,),
        ).fetchone()
    if row is None:
        return None
    try:
        breakdown = json.loads(row[0])
    except (ValueError, TypeError):
        return None
    return {"doi": row[2], "run_id": row[1], "breakdown": breakdown}


# ---------------------------------------------------------------------------
# FE2-4: lazy HTMX fragments — related work + local knowledge graph.
#
# These two suffix routes (/related, /graph) MUST be registered BEFORE the bare
# GET /corpus/{doi:path} detail route. {doi:path} is greedy, but FastAPI matches
# routes in declaration order and prefers the more specific literal-suffix path,
# so declaring them first guarantees "/corpus/10.x/y/related" resolves here and
# not into the detail route with doi="10.x/y/related".
#
# Both seams resolve the graph via ``safe_graph_db`` and return ``None`` when the
# graph isn't built (extra not installed / DB absent) so the routes can render a
# "graph not built" notice instead of 500-ing. The route tests monkeypatch
# ``_get_related`` / ``_get_paper_snapshot`` for the happy paths, and
# ``safe_graph_db`` (imported above) for the no-graph branch.
# ---------------------------------------------------------------------------

# Related-work fragment fetches a small fixed number of neighbours; the mode
# toggle (vector/graph/hybrid) re-requests with the same cap.
_RELATED_K = 10


def _get_related(doi: str, mode: str, k: int) -> list[dict] | None:
    """Return related papers for *doi* under *mode*, or ``None`` if no graph.

    Wraps ``safe_graph_db`` + ``queries.get_related_papers``. Returns ``None``
    (the graph-not-built sentinel) when ``safe_graph_db`` hands back no handle —
    the related-work view here is graph-backed for every mode, so without a graph
    there is nothing to show. The GraphDB handle we acquire is always closed.

    Module-level so the route tests can patch this single seam without a live
    KuzuDB. Mirrors the lazy-acquire / always-close idiom from ask.py.
    """
    from scripts.api.queries import get_related_papers  # noqa: PLC0415

    db = safe_graph_db()
    if db is None:
        return None
    try:
        return get_related_papers(doi, mode=mode, k=k, graph_db=db)
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001 — defensive close, never fatal
            pass


def _get_paper_snapshot(doi: str) -> dict | None:
    """Return the graph snapshot for *doi*, or ``None`` if the graph isn't built.

    Wraps ``safe_graph_db`` + ``queries.get_paper_snapshot``. Returns ``None``
    when no graph handle is available so the route renders a notice rather than
    500-ing. The acquired GraphDB handle is always closed.

    Module-level patch point for the graph-fragment route tests.
    """
    from scripts.api.queries import get_paper_snapshot  # noqa: PLC0415

    db = safe_graph_db()
    if db is None:
        return None
    try:
        return get_paper_snapshot(doi, db)
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001 — defensive close, never fatal
            pass


@corpus_router.get("/corpus/{doi:path}/related", response_class=HTMLResponse)
async def corpus_related(
    request: Request,
    doi: str,
    mode: str = Query("vector"),
) -> HTMLResponse:
    """Lazy fragment: papers related to *doi* (vector/graph/hybrid mode toggle).

    Off-thread (the retrieval touches the graph/vector store). No graph → a
    "graph not built" notice (200). Any failure → a generic notice fragment with
    a full server-side log — the exception text never reaches the browser.
    """
    # Defensive: an unknown mode value from a hand-crafted URL falls back to the
    # default rather than raising, so the fragment always renders.
    if mode not in ("vector", "graph", "hybrid"):
        mode = "vector"
    try:
        related = await asyncio.to_thread(_get_related, doi, mode, _RELATED_K)
    except Exception as exc:  # noqa: BLE001 — advisory fragment, never 500-leak
        # exc is in the message (not just the traceback) so a log scrape catches
        # the failure cause; the browser only ever sees the generic notice below.
        logger.error(
            "corpus related fragment failed doi=%s: %s", doi, exc, exc_info=True
        )
        return _get_templates().TemplateResponse(
            request,
            "corpus/_related.html",
            {"doi": doi, "mode": mode, "related": None, "error": True},
        )
    return _get_templates().TemplateResponse(
        request,
        "corpus/_related.html",
        {"doi": doi, "mode": mode, "related": related, "error": False},
    )


@corpus_router.get("/corpus/{doi:path}/graph", response_class=HTMLResponse)
async def corpus_graph(request: Request, doi: str) -> HTMLResponse:
    """Lazy fragment: the local knowledge-graph view for *doi*.

    Renders entities grouped by type + incoming/outgoing relationships. Off-thread
    (snapshot runs several Cypher queries). No graph → a "graph not built" notice
    (200). Any failure → a generic notice fragment + full server-side log.
    """
    try:
        snapshot = await asyncio.to_thread(_get_paper_snapshot, doi)
    except Exception as exc:  # noqa: BLE001 — advisory fragment, never 500-leak
        logger.error(
            "corpus graph fragment failed doi=%s: %s", doi, exc, exc_info=True
        )
        return _get_templates().TemplateResponse(
            request,
            "corpus/_graph.html",
            {"doi": doi, "snapshot": None, "error": True},
        )
    return _get_templates().TemplateResponse(
        request,
        "corpus/_graph.html",
        {"doi": doi, "snapshot": snapshot, "error": False},
    )


@corpus_router.get("/corpus/{doi:path}", response_class=HTMLResponse)
def corpus_detail(request: Request, doi: str) -> HTMLResponse:
    """Render the per-paper detail page: extraction, score, links, actions."""
    try:
        row = _get_paper_row(doi)
        if row is None:
            raise HTTPException(status_code=404, detail=f"paper {doi!r} not found")
        breakdown = _get_score_breakdown(doi)
    except HTTPException:
        # 404 (and any other intentional HTTP status) must propagate intact —
        # only unexpected errors are masked by the generic 500 guard below.
        raise
    except Exception:  # noqa: BLE001 — 500-leak guard, never expose internals
        logger.error("corpus detail failed doi=%s", doi, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error") from None

    ctx = {"paper": row, "breakdown": breakdown}
    return _get_templates().TemplateResponse(request, "corpus/detail.html", ctx)


__all__ = ["corpus_router"]
