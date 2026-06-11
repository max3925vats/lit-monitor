"""FG-4: the /graph "Knowledge Graph" overview page — a read-only corpus lens.

``GET /graph`` renders instantly (a near-static shell). The heavy corpus-wide
Cypher counts are deferred to ``GET /graph/overview``, which the page lazy-loads
via HTMX (``hx-get`` + ``hx-trigger="load"``) into ``#graph-overview``. This
keeps the first paint fast and lets the overview degrade gracefully when the
knowledge graph is absent (no [graph] data built yet) — a 200 notice, never a
500.

``_get_graph_overview`` is the single module-level seam the route tests
monkeypatch, so no live KuzuDB / SQLite is required. It mirrors the
graph-absent + always-close idioms from ``scripts/server/routes/corpus.py``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

graph_router = APIRouter(tags=["graph"])


def _get_templates():
    """Lazy import to avoid a circular dependency at module load.

    Mirrors the accessor used by corpus.py / ask.py: the ``templates`` object is
    created in app.py, which imports this router inside create_app().
    """
    from lit_monitor.server.app import templates  # noqa: PLC0415

    return templates


def _last_indexed() -> str | None:
    """Best-effort ``MAX(last_updated)`` over graph-indexed papers, or ``None``.

    Reads the state DB directly (the graph itself does not store an index
    timestamp). Defensive: any failure (missing config, unbuilt DB, schema
    mismatch) returns ``None`` so the overview still renders — this is an
    advisory "last indexed" line, never load-bearing.
    """
    try:
        from lit_monitor.core.config import get_config  # noqa: PLC0415
        from lit_monitor.core.state_db import StateDB  # noqa: PLC0415

        state_db = StateDB(get_config().state_db.path)
        with state_db._connect() as conn:
            row = conn.execute(
                "SELECT MAX(last_updated) FROM papers WHERE graph_indexed = 1"
            ).fetchone()
        return row[0] if row and row[0] else None
    except Exception as exc:  # noqa: BLE001 — advisory field, never raise
        logger.debug("graph._last_indexed failed: %s", exc)
        return None


def _get_graph_overview() -> dict[str, Any]:
    """Assemble the corpus-wide graph-health overview, or a not-built sentinel.

    Acquires the graph via ``safe_graph_db`` (``None`` when the [graph] data is
    absent). When no handle is available, returns ``{"available": False}`` so the
    route renders a "no knowledge graph yet" notice instead of 500-ing. On the
    happy path, fans the read across the FG-1 corpus-stats queries plus the FG-2
    ``(indexed, total)`` count idiom, and a best-effort ``last_indexed`` line.

    The acquired GraphDB handle is always closed (try/except, mirroring
    ``check_graph``). Module-level so the route tests can patch this single seam.
    """
    from lit_monitor.api.queries import (  # noqa: PLC0415
        get_corpus_stats,
        get_entity_counts_by_source,
        get_entity_counts_by_type,
    )
    from lit_monitor.graph import safe_graph_db  # noqa: PLC0415 — patch point in tests
    from lit_monitor.setup.check_graph import _indexed_total  # noqa: PLC0415

    db = safe_graph_db()
    if db is None:
        return {"available": False}

    try:
        stats = get_corpus_stats(db)
        by_type = get_entity_counts_by_type(db)
        by_source = get_entity_counts_by_source(db)
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001 — defensive close, never fatal
            pass

    indexed, total = _indexed_total()

    return {
        "available": True,
        "paper_count": stats.get("paper_count", 0),
        "entity_count": stats.get("entity_count", 0),
        "by_predicate": stats.get("edge_counts_by_predicate", {}),
        "by_type": by_type,
        "by_source": by_source,
        "indexed": indexed,
        "total": total,
        "last_indexed": _last_indexed(),
    }


@graph_router.get("/graph", response_class=HTMLResponse)
def graph_index(request: Request) -> HTMLResponse:
    """Render the Knowledge Graph overview shell (instant; counts lazy-load)."""
    return _get_templates().TemplateResponse(request, "graph/index.html", {})


@graph_router.get("/graph/overview", response_class=HTMLResponse)
async def graph_overview(request: Request) -> HTMLResponse:
    """Lazy fragment: corpus-wide graph-health counts (off-thread Cypher).

    No graph → a "no knowledge graph yet" notice (200). Any failure → a generic
    notice fragment with a full server-side log — the exception text (which may
    embed a DB path) never reaches the browser.
    """
    try:
        overview = await asyncio.to_thread(_get_graph_overview)
    except Exception as exc:  # noqa: BLE001 — advisory fragment, never 500-leak
        # exc is in the message (not just the traceback) so a log scrape catches
        # the cause; the browser only ever sees the generic error branch below.
        logger.error("graph overview fragment failed: %s", exc, exc_info=True)
        return _get_templates().TemplateResponse(
            request,
            "graph/_overview.html",
            {"available": False, "error": True},
        )
    return _get_templates().TemplateResponse(
        request, "graph/_overview.html", {**overview, "error": False}
    )


__all__ = ["graph_router"]
