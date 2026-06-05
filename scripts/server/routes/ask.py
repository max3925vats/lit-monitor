"""WA1 (FE-1): the /ask page — browser surface over the graph run_pipeline.

GET /ask renders the page (or a no-graph notice). POST /ask/answer runs the
NL→Cypher→execute→summarize pipeline off-thread and returns an HTML fragment.
The JSON POST /api/ask (scripts/server/routes/query.py) is unchanged — this is
the page-facing, fragment-rendering sibling.

`run_pipeline` and `safe_graph_db` are imported at module top so the route tests
can patch ``scripts.server.routes.ask.run_pipeline`` / ``.safe_graph_db``.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from scripts.core.config import get_config  # patched in tests
from scripts.graph.ask import run_pipeline  # patched in tests
from scripts.graph.import_citations import safe_graph_db  # patched in tests
from scripts.mcp.cypher_guard import CypherSafetyError, guard
from scripts.obsidian_tools.save_answer import save_ask_answer

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_templates():
    """Lazy import to avoid a circular dependency at module load.

    Mirrors the accessor used by themes.py / feedback.py: the ``templates``
    object is created in app.py, which imports this router inside create_app().
    """
    from scripts.server.app import templates  # noqa: PLC0415

    return templates


@router.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request) -> HTMLResponse:
    """Render the /ask page, or a no-graph notice if the graph is unavailable."""
    graph = safe_graph_db()
    has_graph = graph is not None
    if graph is not None:
        # safe_graph_db hands back a live handle; close it — the page render
        # doesn't query the graph, it only needs the availability signal.
        try:
            graph.close()
        except Exception:  # noqa: BLE001 — defensive close, never fatal
            pass
    return _get_templates().TemplateResponse(
        request, "ask/index.html", {"has_graph": has_graph}
    )


@router.post("/ask/answer", response_class=HTMLResponse)
async def ask_answer(request: Request, question: str = Form("")) -> HTMLResponse:
    """Run the ask pipeline off-thread; return the _answer.html fragment.

    Empty/whitespace question → friendly inline message. Any pipeline failure →
    a generic error fragment (no exception text in the body) + a full server-side
    log with traceback, mirroring the route layer's 500-leak discipline.
    """
    q = (question or "").strip()
    if not q:
        return _get_templates().TemplateResponse(
            request,
            "ask/_answer.html",
            {"error": "Please enter a question.", "result": None},
        )
    try:
        result = await asyncio.to_thread(run_pipeline, q, summarize=True)
    except Exception as exc:  # noqa: BLE001 — advisory page surface, never 500-leak
        # Full detail (message + traceback) stays server-side only; the browser
        # gets a generic fragment below. The exc value is in the message so a
        # log scrape catches the failure cause, not just the question.
        logger.error(
            "ask pipeline failed for question=%r: %s", q, exc, exc_info=True
        )
        return _get_templates().TemplateResponse(
            request,
            "ask/_answer.html",
            {
                "error": "The query engine failed — check the server logs.",
                "result": None,
            },
        )
    return _get_templates().TemplateResponse(
        request,
        "ask/_answer.html",
        {"result": result, "question": q, "error": None},
    )


def _execute_guarded_cypher(safe_query: str) -> list[dict]:
    """Execute an ALREADY-GUARDED read-only Cypher; return jsonable rows.

    Mirrors query.py::_execute_cypher (scripts/server/routes/query.py:91): open
    the graph via ``safe_graph_db``, run via ``execute_cypher`` (which does the
    JSON-coercion of each row), and surface unavailability/failure as a raised
    error so the route's generic-error handler catches it — the browser never
    sees backend detail.
    """
    from scripts.graph.ask import execute_cypher  # noqa: PLC0415

    db = safe_graph_db()
    if db is None:
        raise RuntimeError("graph backend unavailable")
    try:
        rows = execute_cypher(db, safe_query)
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001 — defensive close, never fatal
            pass
    if rows is None:
        raise RuntimeError("cypher execution failed")
    return rows


@router.post("/ask/cypher", response_class=HTMLResponse)
async def ask_cypher(request: Request, cypher: str = Form("")) -> HTMLResponse:
    """B3-guarded, read-only Cypher re-run; returns the _cypher_result fragment.

    The guard runs BEFORE any execution: mutations / CALL / empty raise
    ``CypherSafetyError`` and we return a read-only notice WITHOUT touching the
    graph. Only a guard-approved query reaches ``_execute_guarded_cypher`` (run
    off-thread). Execution failure → generic fragment + a full server-side log,
    mirroring /ask/answer's 500-leak discipline.
    """
    try:
        safe_q = guard(cypher or "")
    except CypherSafetyError as exc:
        # Fixed string only — never echo the rejected query back to the browser.
        logger.warning("ask/cypher rejected: %s", exc)
        return _get_templates().TemplateResponse(
            request,
            "ask/_cypher_result.html",
            {"error": "Read-only queries only (mutations are blocked).", "rows": None},
        )
    try:
        rows = await asyncio.to_thread(_execute_guarded_cypher, safe_q)
    except Exception:  # noqa: BLE001 — advisory page surface, never 500-leak
        logger.error("ask/cypher execution failed", exc_info=True)
        return _get_templates().TemplateResponse(
            request,
            "ask/_cypher_result.html",
            {"error": "Query execution failed — check the server logs.", "rows": None},
        )
    return _get_templates().TemplateResponse(
        request,
        "ask/_cypher_result.html",
        {"rows": rows, "error": None},
    )


@router.post("/ask/save", response_class=HTMLResponse)
async def ask_save(request: Request, payload: str = Form("")) -> HTMLResponse:
    """Save the answer (passed verbatim in the hidden ``payload`` field).

    The fragment embeds ``{question, prose, rows, cypher}`` as one JSON string so
    the save does NOT re-run the pipeline (re-running would burn another LLM call
    and could return a different answer). Any failure — malformed JSON, missing
    config, I/O — yields a generic error fragment with a full server-side log;
    the exception text / vault path never reaches the browser (500-leak
    discipline).
    """
    try:
        data = json.loads(payload or "")
        config = get_config()
        note_path = await asyncio.to_thread(
            save_ask_answer,
            data.get("question", ""),
            data.get("prose"),
            data.get("rows") or [],
            data.get("cypher"),
            config,
        )
    except Exception:  # noqa: BLE001 — advisory page surface, never 500-leak
        logger.error("ask/save failed", exc_info=True)
        return _get_templates().TemplateResponse(
            request,
            "ask/_saved.html",
            {"error": "Could not save the answer — check the server logs.", "path": None},
        )
    return _get_templates().TemplateResponse(
        request,
        "ask/_saved.html",
        {"path": note_path, "error": None},
    )


__all__ = ["router"]
