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
import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from scripts.graph.ask import run_pipeline  # patched in tests
from scripts.graph.import_citations import safe_graph_db  # patched in tests

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


__all__ = ["router"]
