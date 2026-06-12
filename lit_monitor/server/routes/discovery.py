"""Discovery dashboard + process controls + SSE progress stream + read API.

Routes:
    GET  /discovery                           — dashboard (last run, history, today's digest)
    GET  /discovery/{run_id}                  — per-run HTML detail page with paper cards (P8)
    POST /api/discovery/start                 — spawn ``lit-monitor run [--dry-run]``
    POST /api/discovery/stop                  — SIGTERM the running subprocess
    GET  /api/discovery/status                — JSON status for polling
    GET  /api/discovery/controls              — HTMX HTML fragment (5-second self-refresh)
    GET  /api/discovery/stream                — SSE tail of newest discovery JSONL log
    GET  /api/discovery/runs                  — paginated list of discovery_runs (P5)
    GET  /api/discovery/runs/{run_id}         — full run dict + papers (P5)
    GET  /api/discovery/runs/{run_id}/papers  — papers for run sorted by score (P5)
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from lit_monitor.api.queries import (
    get_discovery_run,
    get_discovery_run_papers,
    get_discovery_runs,
)
from lit_monitor.server.app import templates
from lit_monitor.server.routes.sse import stream_log
from lit_monitor.server.runtime import get_runtime

logger = logging.getLogger(__name__)
router = APIRouter(tags=["discovery"])

# Repo root: scripts/server/routes/discovery.py → 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# FG-3: valid retrieval modes for the discovery start form's rag_mode select.
_VALID_RAG_MODES = ("vector", "graph", "hybrid")

# AR-7: page size for the paginated discovery-runs table on the dashboard.
_RUNS_PAGE_SIZE = 20


def _graph_extra_available() -> bool:
    """True when the optional [graph] extra (kuzu) is importable.

    A lightweight availability probe — does NOT open the production graph DB
    (which would run migrations + take the write lock and mask a 'DB busy'
    error as 'extra missing').
    """
    return importlib.util.find_spec("kuzu") is not None


def _default_rag_mode() -> str:
    """Resolve the default retrieval mode from ``retrieval.default_mode`` config.

    Mirrors the CLI's resolution (``cli.py`` G9): the config accessor already
    validates the value to one of {vector,graph,hybrid} and falls back to
    'vector' when unset/invalid. Best-effort — any failure yields 'vector' so
    the form still renders (500-leak-safe).
    """
    try:
        mode = get_runtime().config.retrieval.default_mode
    except Exception:
        return "vector"
    return mode if mode in _VALID_RAG_MODES else "vector"


def _safe_runtime():
    try:
        return get_runtime()
    except Exception:
        return None


def _safe_db():
    r = _safe_runtime()
    if r is None:
        return None
    try:
        return r.state_db
    except Exception:
        return None


def _vault_root() -> Path | None:
    """Resolve the configured Obsidian vault path."""
    r = _safe_runtime()
    if r is None:
        return None
    try:
        vault = r.config.obsidian.vault_path
    except Exception:
        return None
    if not vault:
        return None
    return Path(vault).expanduser()


def _digests_folder_name() -> str:
    r = _safe_runtime()
    if r is None:
        return "Literature/Digests"
    try:
        return getattr(r.config.obsidian, "digests_folder", "Literature/Digests")
    except Exception:
        return "Literature/Digests"


def _todays_digest() -> tuple[Path | None, str | None]:
    """Locate today's digest file, if any. Returns (path, content_or_None).

    The discovery pipeline writes to ``{vault}/{digests_folder}/Discovery_{YYYY-MM-DD}.md``
    using ``date.today()`` (local timezone). Falls back to the most recent
    ``Discovery_*.md`` if today's exact filename is missing.
    """
    vault = _vault_root()
    if vault is None:
        return None, None
    folder = vault / _digests_folder_name()
    if not folder.exists() or not folder.is_dir():
        return None, None
    today = date.today().strftime("%Y-%m-%d")
    exact = folder / f"Discovery_{today}.md"
    if exact.exists():
        try:
            return exact, exact.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Cannot read digest %s", exact, exc_info=True)
            return exact, None
    # Fallback: newest Discovery_*.md in the folder.
    candidates = sorted(
        folder.glob("Discovery_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, None
    newest = candidates[0]
    try:
        return newest, newest.read_text(encoding="utf-8")
    except OSError:
        return newest, None


@router.get("/discovery", response_class=HTMLResponse)
def dashboard(
    request: Request,
    runs_offset: int = Query(0, ge=0),
) -> HTMLResponse:
    db = _safe_db()
    recent_runs: list[dict] = []
    last_run: dict | None = None
    if db is not None:
        try:
            recent_runs = db.get_recent_runs_by_type("discovery", limit=10)
            if recent_runs:
                last_run = recent_runs[0]
        except Exception:
            logger.warning("get_recent_runs_by_type failed", exc_info=True)

    # P8: also fetch structured discovery_runs (have id, total_found, total_ingested)
    # to populate the run-history table with links to /discovery/{run_id}.
    # AR-7: this table is now paginated (corpus-style offset/limit) so a long run
    # history no longer renders unbounded. The "latest run" summary always shows
    # the newest run (offset 0) independent of the table's current page.
    discovery_runs: list[dict] = []
    latest_discovery_run: dict | None = None
    runs_total = 0
    # AR-7: clamp the requested offset to >= 0. A hand-crafted ?runs_offset= past
    # the end yields an empty table, but the runs pager renders unconditionally so
    # Prev (shown whenever offset > 0) still navigates back to a valid page.
    offset = max(runs_offset, 0)
    if db is not None:
        try:
            state_db = get_runtime().state_db
            result = get_discovery_runs(
                state_db, limit=_RUNS_PAGE_SIZE, offset=offset
            )
            discovery_runs = result.get("runs", [])
            # Coerce defensively: a non-int total (e.g. a partial test double)
            # would break the has_next arithmetic at render time.
            try:
                runs_total = int(result.get("total", 0))
            except (TypeError, ValueError):
                runs_total = len(discovery_runs)
            # The latest run is always the newest row (offset 0), shown in the
            # summary card regardless of which page the table is on.
            if offset == 0:
                latest_discovery_run = discovery_runs[0] if discovery_runs else None
            else:
                head = get_discovery_runs(state_db, limit=1, offset=0).get("runs", [])
                latest_discovery_run = head[0] if head else None
        except Exception:
            logger.warning("get_discovery_runs failed", exc_info=True)

    digest_path, digest_content = _todays_digest()
    return templates.TemplateResponse(
        request,
        "discovery/index.html",
        {
            "last_run": last_run,
            "recent_runs": recent_runs,
            # P8: structured discovery_runs with id field for linking
            "discovery_runs": discovery_runs,
            "latest_discovery_run": latest_discovery_run,
            # AR-7: pagination context for the discovery-runs table.
            "runs_total": runs_total,
            "runs_offset": offset,
            "runs_has_prev": offset > 0,
            "runs_has_next": offset + _RUNS_PAGE_SIZE < runs_total,
            "runs_prev_offset": max(offset - _RUNS_PAGE_SIZE, 0),
            "runs_next_offset": offset + _RUNS_PAGE_SIZE,
            "digest_path": str(digest_path) if digest_path else None,
            "digest_content": digest_content,
            "db_unavailable": db is None,
        },
    )


# ---------------------------------------------------------------------------
# P8: per-run HTML detail page
# ---------------------------------------------------------------------------


@router.get("/discovery/{run_id}", response_class=HTMLResponse)
def discovery_run_detail(request: Request, run_id: int) -> HTMLResponse:
    """P8: per-run HTML detail page with paper cards + Relink / Re-extract buttons.

    Route ordering note: the ``run_id: int`` type annotation causes FastAPI to
    reject non-numeric path segments with 422, so this route does NOT swallow
    P3's ``/discovery/notify-handler`` path — that falls through to the
    separately registered ``discovery_notify_router``.

    Args:
        request: FastAPI request object.
        run_id:  Primary key of the discovery_runs row.

    Returns:
        HTML page with paper cards, or 404 when run_id does not exist.
    """
    state_db = get_runtime().state_db
    run = get_discovery_run(state_db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    papers = get_discovery_run_papers(state_db, run_id, top_k=100)

    # Bundle H / P4: read web_ui.show_feedback_buttons config flag.
    # P4 flipped the default True so feedback is collected without explicit
    # config (mirrors WebUiSettings.show_feedback_buttons default). An explicit
    # `show_feedback_buttons = false` in extraction config still turns it off.
    show_feedback_buttons = True
    try:
        from lit_monitor.server.config_io import load_config
        cfg = load_config("extraction")
        show_feedback_buttons = bool(
            cfg.get("web_ui", {}).get("show_feedback_buttons", True)
        )
    except Exception:
        pass

    return templates.TemplateResponse(
        request,
        "discovery/run_detail.html",
        {
            "run": run,
            "papers": papers,
            "show_feedback_buttons": show_feedback_buttons,
            "detail_crumb": f"Run {run_id}",
        },
    )


# ---------------------------------------------------------------------------
# Process control endpoints (F4.2)
# ---------------------------------------------------------------------------


async def _spawn_discovery(
    *, dry_run: bool, rag_mode: str
) -> asyncio.subprocess.Process:
    """Build the argv and spawn ``lit-monitor run [--dry-run] --rag-mode <mode>``."""
    argv = ["uv", "run", "lit-monitor", "run"]
    if dry_run:
        argv.append("--dry-run")
    # FG-3: thread the chosen retrieval mode through to the CLI.
    argv += ["--rag-mode", rag_mode]
    logger.info("Spawning discovery: %s", " ".join(argv))
    return await asyncio.create_subprocess_exec(*argv, cwd=str(_REPO_ROOT))


@router.post("/api/discovery/start")
async def discovery_start(
    request: Request,
    dry_run: bool = Form(False),
    rag_mode: str = Form(""),
) -> JSONResponse:
    """Spawn a discovery run subprocess. Refuses if a run is already in flight.

    FG-3: reads the ``rag_mode`` form field. Invalid/missing values fall back to
    the config default (``retrieval.default_mode``). Selecting ``graph``/``hybrid``
    without the ``[graph]`` extra installed is refused up-front with a visible
    warning (mirrors the CLI W4 hard error) rather than spawning a run that would
    crash mid-flight.
    """
    runtime = get_runtime()
    slot = runtime.processes["discovery"]
    # FG-3: validate the requested mode; fall back to config default otherwise.
    mode = rag_mode if rag_mode in _VALID_RAG_MODES else _default_rag_mode()
    # FG-3 / W4-mirror: graph|hybrid require the [graph] extra. Probe via a
    # lightweight import check (does NOT open the production graph DB) and
    # refuse to start before any search begins, so the user gets a clear
    # warning instead of a subprocess that dies partway through. We do not
    # pre-check whether the graph is *built* here — only whether the extra is
    # *installed* — matching the CLI's W4 guard.
    if mode in ("graph", "hybrid") and not _graph_extra_available():
        return JSONResponse(
            {
                "started": False,
                "reason": (
                    f"--rag-mode {mode} requires the [graph] extra. "
                    "Install with: uv sync --extra graph"
                ),
            },
            status_code=400,
        )
    async with slot.lock:
        if slot.is_running():
            return JSONResponse(
                {
                    "started": False,
                    "reason": "already running",
                    "pid": slot.process.pid if slot.process else None,
                },
                status_code=409,
            )
        try:
            slot.process = await _spawn_discovery(dry_run=dry_run, rag_mode=mode)
            slot.started_at = datetime.now(UTC).isoformat(timespec="seconds")
            slot.dry_run = dry_run
        except Exception:  # noqa: BLE001 — surface any spawn failure
            # A3-5 info-leak guard: full exception logged server-side; client
            # gets a generic reason (raw spawn error can embed argv / paths).
            logger.exception("Failed to spawn discovery")
            return JSONResponse(
                {"started": False, "reason": "spawn failed"},
                status_code=500,
            )
    return JSONResponse(
        {
            "started": True,
            "pid": slot.process.pid,
            "started_at": slot.started_at,
            "dry_run": dry_run,
        }
    )


@router.post("/api/discovery/stop")
async def discovery_stop() -> JSONResponse:
    """SIGTERM the discovery subprocess; wait up to 30s, then SIGKILL."""
    runtime = get_runtime()
    slot = runtime.processes["discovery"]
    stopped = await slot.stop(timeout=30.0)
    return JSONResponse({"stopped": stopped})


@router.get("/api/discovery/status")
async def discovery_status() -> JSONResponse:
    """Polled by the controls fragment to drive the button state."""
    runtime = get_runtime()
    slot = runtime.processes["discovery"]
    running = slot.is_running()
    return JSONResponse(
        {
            "running": running,
            "pid": slot.process.pid if running and slot.process else None,
            "started_at": slot.started_at if running else None,
            "dry_run": slot.dry_run if running else None,
        }
    )


@router.get("/api/discovery/controls", response_class=HTMLResponse)
async def discovery_controls(request: Request) -> HTMLResponse:
    """HTMX-polled HTML fragment that renders Run/Dry-run or Stop buttons."""
    runtime = get_runtime()
    slot = runtime.processes["discovery"]
    running = slot.is_running()
    return templates.TemplateResponse(
        request,
        "discovery/_controls.html",
        {
            "is_running": running,
            "pid": slot.process.pid if running and slot.process else None,
            "started_at": slot.started_at if running else None,
            "dry_run": slot.dry_run if running else False,
            # FG-3: default retrieval mode for the rag_mode select (config-driven).
            "default_rag_mode": _default_rag_mode(),
            "rag_modes": _VALID_RAG_MODES,
        },
    )


@router.get("/api/discovery/stream")
async def discovery_stream(request: Request) -> EventSourceResponse:
    """SSE stream of the newest discovery JSONL log."""
    return stream_log(request, "discovery")


# ---------------------------------------------------------------------------
# P5: read endpoints for discovery_runs + discovery_paper_results
# ---------------------------------------------------------------------------


@router.get("/api/discovery/runs")
def list_discovery_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    """P5: paginated list of discovery runs.

    Returns ``{runs: [...], total: N}``.  Each run contains id, started_at,
    finished_at, status, total_found, total_ingested.
    """
    return get_discovery_runs(get_runtime().state_db, limit=limit, offset=offset)


@router.get("/api/discovery/runs/{run_id}")
def get_discovery_run_detail(run_id: int) -> dict:
    """P5: full detail for a single discovery run including its papers.

    Returns run dict + ``papers`` list sorted by score DESC.
    404 when run_id does not exist.
    """
    state_db = get_runtime().state_db
    run = get_discovery_run(state_db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    # Attach papers (no top_k cap on detail view — callers use /papers for capped)
    run["papers"] = get_discovery_run_papers(state_db, run_id)
    return run


@router.get("/api/discovery/runs/{run_id}/papers")
def list_discovery_run_papers(
    run_id: int,
    top_k: int = Query(20, ge=1, le=100),
) -> dict:
    """P5: papers for a discovery run, sorted by score DESC.

    Returns ``{papers: [...]}``.  Empty list when run_id has no results.
    """
    papers = get_discovery_run_papers(get_runtime().state_db, run_id, top_k=top_k)
    return {"papers": papers}
