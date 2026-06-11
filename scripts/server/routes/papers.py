"""H4 + H5 + H7 + Bundle B: paper-related HTTP endpoints.

H4: GET  /api/papers/{doi}            — full paper snapshot.
H5: GET  /api/papers/{doi}/related    — related papers (vector / graph / hybrid).
H7: POST /api/papers/{doi}/relink     — re-run Obsidian note relink.
H7: POST /api/papers/{doi}/re-extract — re-run LLM extraction + rerender.
B:  GET  /api/papers/{doi}/score-breakdown — per-signal score decomposition.

The graph backend is constructed lazily via safe_graph_db() so the server
can start without a kuzu installation; requests hit 503 instead of an
import error at boot time.

Route registration order (suffixed routes registered BEFORE the catch-all):
  1. /api/papers/{doi:path}/related          — GET
  2. /api/papers/{doi:path}/score-breakdown  — GET   (Bundle B)
  3. /api/papers/{doi:path}/relink           — POST  (H7)
  4. /api/papers/{doi:path}/re-extract       — POST  (H7)
  5. /api/papers/{doi:path}                  — GET   (catch-all; must be LAST)
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from scripts.api.queries import get_paper_snapshot, get_related_papers

# Module-level names so tests can monkeypatch them without reaching into
# the original modules (patch the binding at the route's own namespace).
from scripts.graph.import_citations import safe_graph_db

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["papers"])


# ---------------------------------------------------------------------------
# H5: GET /api/papers/{doi:path}/related
# Registered BEFORE the bare {doi:path} route so the /related suffix is
# matched first by FastAPI's router.
# ---------------------------------------------------------------------------


@router.get("/api/papers/{doi:path}/related")
def related_papers(
    doi: str,
    mode: Literal["vector", "graph", "hybrid"] = Query("vector"),
    k: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Return papers related to the given seed DOI.

    Path parameters:
        doi:  Seed paper DOI (may contain slashes).

    Query parameters:
        mode: Retrieval strategy — "vector" (default), "graph", or "hybrid".
        k:    Number of results, 1–100 (default 20).

    Returns:
        200 — list of {doi, score} dicts, ranked descending.
        404 — seed DOI not found (get_related_papers returned None).
        422 — invalid DOI format, bad mode value, or k out of range.
        503 — graph backend unavailable.
    """
    # Strip trailing /related from path (FastAPI includes it in {doi:path}).
    # FastAPI captures the suffix literally, so doi will be e.g. "10.x/y/related"
    # — we need to strip the trailing "/related" segment that was part of the match.
    # Because FastAPI registers this route as /api/papers/{doi:path}/related, the
    # {doi:path} capture does NOT include "related" — it stops before the literal
    # "/related" suffix.  No stripping needed.

    if not _DOI_RE.match(doi):
        raise HTTPException(status_code=422, detail=f"invalid DOI: {doi!r}")

    graph_db = safe_graph_db()
    if graph_db is None:
        raise HTTPException(status_code=503, detail="graph backend unavailable")

    # AR-2: always release the KuzuDB handle. Without this finally the handle
    # leaked a file descriptor and held the Kuzu write lock for the lifetime of
    # the process, contending with discovery/backfill. Mirrors corpus.py's
    # _get_related_graph defensive-close pattern.
    try:
        results = get_related_papers(doi, mode=mode, k=k, graph_db=graph_db)
    finally:
        try:
            graph_db.close()
        except Exception:  # noqa: BLE001 — defensive close, never fatal
            pass
    if results is None:
        raise HTTPException(status_code=404, detail=f"paper {doi!r} not found")
    return results


# ---------------------------------------------------------------------------
# Bundle B: GET /api/papers/{doi:path}/score-breakdown
# Returns the most recent per-signal score decomposition for a paper.
# Registered BEFORE the bare {doi:path} catch-all so the suffix matches first.
# ---------------------------------------------------------------------------


def _get_score_breakdown_db():
    """Return the StateDB instance for score-breakdown lookups.

    Thin lazy-import wrapper so tests can monkeypatch this name to inject a
    temporary StateDB without touching the full server runtime.
    """
    from scripts.server.runtime import get_runtime

    return get_runtime().state_db


@router.get("/api/papers/{doi:path}/score-breakdown")
def get_score_breakdown(doi: str) -> dict:
    """Bundle B: return the most recent per-signal score decomposition for a paper.

    Path parameters:
        doi: DOI string (may contain slashes — captured by {doi:path}).

    Returns:
        200 — {doi, run_id, breakdown: {signal: float, ...}}
        404 — no breakdown stored for this DOI (paper not in discovery results,
              or run predates Bundle B deployment).
        422 — doi does not match ^10.\\d{4,9}/\\S+$ (rejected before hitting DB).
    """
    if not _DOI_RE.match(doi):
        raise HTTPException(status_code=422, detail=f"invalid DOI: {doi!r}")

    import json as _json

    state_db = _get_score_breakdown_db()
    with state_db._connect() as conn:
        row = conn.execute(
            "SELECT score_breakdown_json, run_id, doi "
            "FROM discovery_paper_results "
            "WHERE doi = ? AND score_breakdown_json IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (doi,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No score breakdown for {doi!r}",
        )

    return {
        "doi": row[2],
        "run_id": row[1],
        "breakdown": _json.loads(row[0]),
    }


# ---------------------------------------------------------------------------
# H4: GET /api/papers/{doi:path} — paper snapshot (catch-all; must be LAST)
# ---------------------------------------------------------------------------


@router.get("/api/papers/{doi:path}")
def paper_snapshot(doi: str) -> dict:
    """Return a full snapshot of one paper.

    Path parameters:
        doi: DOI string (may contain slashes — captured by {doi:path}).

    Returns:
        200 — {metadata, entities_by_type, relationships_in, relationships_out}
        404 — paper not found in graph (metadata key is empty)
        422 — doi does not match ^10.\\d{4,9}/\\S+$ (rejected before hitting DB)
        503 — graph backend unavailable (kuzu not installed or DB unreachable)
    """
    if not _DOI_RE.match(doi):
        raise HTTPException(status_code=422, detail=f"invalid DOI: {doi!r}")

    # safe_graph_db returns None when the [graph] extra is not installed.
    # Tests monkeypatch this name to return a MagicMock or None as needed.
    graph_db = safe_graph_db()
    if graph_db is None:
        raise HTTPException(status_code=503, detail="graph backend unavailable")

    # AR-2: same leak class as /related — always release the KuzuDB handle.
    try:
        snapshot = get_paper_snapshot(doi, graph_db)
    finally:
        try:
            graph_db.close()
        except Exception:  # noqa: BLE001 — defensive close, never fatal
            pass

    # H1 convention: empty metadata dict means the DOI was not found.
    if not snapshot.get("metadata"):
        raise HTTPException(status_code=404, detail=f"paper {doi!r} not found")

    return snapshot


# ---------------------------------------------------------------------------
# H7: POST /api/papers/{doi:path}/relink + /re-extract trigger endpoints
#
# Design notes:
#   - The HTTP layer NEVER touches .md files directly.  Persist-zone safety
#     is enforced inside obsidian_tools.{relink,re_extract}.
#   - _doi_exists / _invoke_relink / _invoke_re_extract are thin helpers so
#     tests can monkeypatch each concern in isolation.
#   - 500 detail = str(exc) only — never a traceback (info-leak guard).
#   - No concurrent-trigger guard: two simultaneous POSTs for the same DOI
#     race inside the tool layer; this is a pre-existing concern, not
#     introduced here.
# ---------------------------------------------------------------------------


def _doi_exists(doi: str) -> bool:
    """Return True when *doi* exists in the state.db papers table.

    Uses the public StateDB.get_paper() helper (which opens its own
    connection) so we don't need _connect() directly.
    """
    from scripts.server.runtime import get_runtime

    runtime = get_runtime()
    return runtime.state_db.get_paper(doi) is not None


def _invoke_relink(doi: str) -> dict:
    """Wire up runtime dependencies and call relink_note for *doi*.

    Thin lazy-import wrapper — monkeypatched in tests so the Ollama /
    Obsidian stack is never touched during unit tests.

    Note: DOI normalization (case, whitespace) is the caller's
    responsibility; this function passes *doi* verbatim to the tool.
    """
    from scripts.obsidian_tools.relink import relink_note
    from scripts.server.runtime import get_runtime

    runtime = get_runtime()
    record = runtime.state_db.get_paper(doi)
    # note_path must be set; if missing the tool will raise — that becomes a 500.
    note_path = record.get("note_path") if record else None
    result = relink_note(
        note_path=note_path,
        embeddings_db=runtime.embeddings_db,
        state_db=runtime.state_db,
        config=runtime.config,
    )
    # relink_note returns None; return a minimal summary dict so the JSON
    # body is always a dict (consistent with re_extract which returns a dict).
    return result if isinstance(result, dict) else {}


def _invoke_re_extract(doi: str) -> dict:
    """Wire up runtime dependencies and call re_extract for *doi*.

    Thin lazy-import wrapper — monkeypatched in tests.
    """
    from scripts.llm.llm_client import get_client
    from scripts.obsidian_tools.re_extract import re_extract
    from scripts.server.runtime import get_runtime

    runtime = get_runtime()
    cfg = runtime.config
    llm = get_client(cfg, mode="brain_build")
    return re_extract(
        doi=doi,
        config=cfg,
        state_db=runtime.state_db,
        llm=llm,
        rerender=True,
    )


@router.post("/api/papers/{doi:path}/relink")
def trigger_relink(doi: str) -> JSONResponse:
    """H7: re-link an Obsidian note's Related Work + reverse-citation block.

    Path parameters:
        doi: DOI string (may contain slashes — captured by {doi:path}).

    Returns:
        200 — {doi, status: "ok", summary: <tool return>}
        404 — DOI not in state.db papers table.
        422 — malformed DOI (fails ^10.\\d{4,9}/\\S+$ check).
        500 — tool raised; detail = generic "Internal error" (full exception
              logged server-side only, never returned to the client).
    """
    if not _DOI_RE.match(doi):
        raise HTTPException(status_code=422, detail=f"invalid DOI: {doi!r}")
    if not _doi_exists(doi):
        raise HTTPException(status_code=404, detail=f"paper {doi!r} not found")
    try:
        summary = _invoke_relink(doi)
    except Exception:
        # P2.2 info-leak guard: the client gets a generic detail only. The
        # full exception (which can embed filesystem paths, e.g. from a
        # FileNotFoundError) is logged server-side with the traceback so
        # operators can still diagnose it, but it never crosses the wire.
        _logger.error("H7 relink failed doi=%s", doi, exc_info=True)
        return JSONResponse(
            {"doi": doi, "status": "error", "detail": "Internal error"},
            status_code=500,
        )
    return JSONResponse({"doi": doi, "status": "ok", "summary": summary})


@router.post("/api/papers/{doi:path}/re-extract")
def trigger_re_extract(doi: str) -> JSONResponse:
    """H7: re-extract structured fields and re-render the Obsidian note.

    Path parameters:
        doi: DOI string (may contain slashes — captured by {doi:path}).

    Returns:
        200 — {doi, status: "ok", summary: <tool return>}
        404 — DOI not in state.db papers table.
        422 — malformed DOI.
        500 — tool raised; detail = generic "Internal error" (full exception
              logged server-side only, never returned to the client).
    """
    if not _DOI_RE.match(doi):
        raise HTTPException(status_code=422, detail=f"invalid DOI: {doi!r}")
    if not _doi_exists(doi):
        raise HTTPException(status_code=404, detail=f"paper {doi!r} not found")
    try:
        summary = _invoke_re_extract(doi)
    except Exception:
        # P2.2 info-leak guard: generic detail to the client; full exception
        # + traceback logged server-side only (see trigger_relink above).
        _logger.error("H7 re-extract failed doi=%s", doi, exc_info=True)
        return JSONResponse(
            {"doi": doi, "status": "error", "detail": "Internal error"},
            status_code=500,
        )
    return JSONResponse({"doi": doi, "status": "ok", "summary": summary})


__all__ = ["router"]
