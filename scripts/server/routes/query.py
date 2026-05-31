"""H8: POST /api/ask + POST /api/cypher.

/api/ask wraps the same pipeline the A5 CLI runs (now via shared
run_pipeline in scripts/graph/ask.py) so CLI and HTTP return the same
shape.

/api/cypher reuses B3's cypher_guard for read-only enforcement +
LIMIT 100 injection. SECURITY-CRITICAL: a guard bypass = arbitrary
write Cypher from any HTTP client (bind is loopback-only, but a
local-network-exposed config or a malicious browser extension could
still hit it). The guard is defense-in-depth.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    model: str | None = None
    summarize: bool = True

    @field_validator("question")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class CypherRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v


# ---------------------------------------------------------------------------
# /api/ask
# ---------------------------------------------------------------------------


@router.post("/api/ask")
async def ask(body: AskRequest) -> dict:
    """H8: NL → Cypher → execute → summarize.

    Returns {prose, rows, cypher}. Partial state (e.g. None cypher, empty
    rows) is returned as 200 — the endpoint is best-effort; clients should
    check the individual fields for None / empty-list signals.

    The pipeline contains blocking LLM calls so it runs in a thread via
    asyncio.to_thread to avoid stalling the event loop.
    """
    from scripts.graph.ask import run_pipeline  # noqa: PLC0415

    # LLM calls inside run_pipeline are blocking → off-thread.
    result = await asyncio.to_thread(
        run_pipeline,
        body.question,
        model=body.model,
        summarize=body.summarize,
    )
    return {
        "prose": result.prose,
        "rows": result.rows,
        "cypher": result.cypher,
    }


# ---------------------------------------------------------------------------
# /api/cypher
# ---------------------------------------------------------------------------


def _execute_cypher(query: str) -> list[dict]:
    """Execute a guard-approved Cypher query against the graph backend.

    Module-level function (not a closure) so tests can monkeypatch it
    without needing to reach inside the route closure.

    Args:
        query: A read-only Cypher string already approved by
            ``cypher_guard.guard()`` (LIMIT injected if absent).

    Returns:
        List of result row dicts.

    Raises:
        HTTPException 503: Graph backend unavailable (safe_graph_db returned None).
        HTTPException 500: Cypher execution failed (execute_cypher returned None).
    """
    from scripts.graph import safe_graph_db  # noqa: PLC0415
    from scripts.graph.ask import execute_cypher  # noqa: PLC0415

    db = safe_graph_db()
    if db is None:
        raise HTTPException(status_code=503, detail="graph backend unavailable")

    rows = execute_cypher(db, query)
    if rows is None:
        raise HTTPException(status_code=500, detail="cypher execution failed")

    return rows


@router.post("/api/cypher")
def run_cypher_endpoint(body: CypherRequest) -> dict:
    """H8: read-only Cypher escape hatch guarded by B3's safety filter.

    The guard (scripts/mcp/cypher_guard.py) handles:
    - Mutation keyword rejection (CREATE/MERGE/DELETE/SET/DROP/ALTER/REMOVE/LOAD CSV)
    - LIMIT injection when absent (appends LIMIT 100)
    - Comment stripping before keyword matching (prevents comment-buried bypass)
    - Case-insensitive matching

    Security note: detail on mutation rejection is a fixed string — the
    user's query is never echoed back to prevent info-leak of e.g. injection
    attempts.
    """
    from scripts.mcp.cypher_guard import CypherSafetyError, guard  # noqa: PLC0415

    try:
        safe_query = guard(body.query, hard_limit=100)
    except CypherSafetyError:
        # Fixed string only — do NOT include body.query or exc details.
        # Echoing the user's query back in a 400 would leak the content of
        # rejected injection payloads (Opus info-leak check).
        raise HTTPException(status_code=400, detail="mutation rejected")

    rows = _execute_cypher(safe_query)
    return {"rows": rows}
