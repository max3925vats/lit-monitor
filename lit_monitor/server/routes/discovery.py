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
import os
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import markdown
import nh3
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from lit_monitor.api.queries import (
    discovery_run_same_day_ordinal,
    get_discovery_run,
    get_discovery_run_conversion,
    get_discovery_run_history,
    get_discovery_run_papers,
    get_discovery_runs,
)
from lit_monitor.output.digest_renderer import digest_filename, render_digest
from lit_monitor.server.app import templates
from lit_monitor.server.routes.sse import stream_log
from lit_monitor.server.runtime import get_runtime

logger = logging.getLogger(__name__)
router = APIRouter(tags=["discovery"])

# Repo root: scripts/server/routes/discovery.py → 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# FG-3: valid retrieval modes for the discovery start form's rag_mode select.
_VALID_RAG_MODES = ("vector", "graph", "hybrid")

# P4: page size for the paginated Run History card on the dashboard.
_RUNS_PAGE_SIZE = 10

# Task 8: fallback window when no coverage_until frontier exists yet.
_DEFAULT_WINDOW_DAYS = 14


def _window_defaults() -> dict:
    """Compute the at-load search-window defaults for the time-range card.

    Mirrors the adaptive frontier an untouched run would use: N = (today -
    coverage_until).days, since = coverage_until. When no frontier is set yet
    (fresh install), fall back to a 14-day rolling window (matching the
    pipeline's first-run behaviour). Best-effort and 500-leak-safe — any failure
    yields the 14-day fallback so the card still renders. The returned ``days``
    is clamped to >= 0 (a future-dated frontier would otherwise go negative).
    """
    today = date.today()
    frontier: date | None = None
    db = _safe_db()
    if db is not None:
        try:
            from lit_monitor.pipelines.discovery import read_coverage_until

            frontier = read_coverage_until(db)
        except Exception:
            logger.warning("read_coverage_until failed", exc_info=True)
            frontier = None
    if frontier is None:
        since = today - timedelta(days=_DEFAULT_WINDOW_DAYS)
        return {"window_default_days": _DEFAULT_WINDOW_DAYS,
                "window_default_since": since.isoformat()}
    days = max((today - frontier).days, 0)
    return {"window_default_days": days,
            "window_default_since": frontier.isoformat()}


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


def _digest_options(db) -> list[dict]:
    """Newest-first ``[{id, date, ordinal}, ...]`` for the digest-viewer dropdown.

    One entry per discovery run. The label is ``Discovery_{date}`` for the day's
    first run and ``Discovery_{date}_{ordinal}`` for later same-day runs, so two
    runs on the same date get DISTINCT labels (the bug was two identical
    ``Discovery_{date}`` options pointing at the same file). Derived from
    ``get_discovery_run_history`` so the dropdown mirrors the Run History card.
    Defensive: any failure (or no db) yields an empty list, so the viewer shows
    a friendly empty state instead of 500-ing.
    """
    if db is None:
        return []
    try:
        result = get_discovery_run_history(db, limit=50, offset=0)
        runs = result.get("runs", []) or []
    except Exception:
        logger.warning("digest_options: history query failed", exc_info=True)
        return []
    options: list[dict] = []
    for r in runs:
        rid = r.get("id")
        if rid is None:
            continue
        try:
            ordinal = discovery_run_same_day_ordinal(db, rid)
        except Exception:
            logger.warning("digest_options: ordinal query failed", exc_info=True)
            ordinal = 1
        options.append(
            {"id": rid, "date": (r.get("started_at") or "")[:10], "ordinal": ordinal}
        )
    return options


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

    # P4: unified Run History — discovery_runs LEFT JOIN run_log so a single
    # paginated card carries both structured run metadata (id, found, ingested)
    # AND the per-paper processed/skipped/failed counts. Historical rows (no
    # run_log FK) return None for the three count columns; the template renders
    # those as em-dashes. Paginated by 10 (offset-link prev/next).
    run_history: list[dict] = []
    runs_total = 0
    # Clamp the requested offset to >= 0. A hand-crafted ?runs_offset= past the
    # end yields an empty card, but the pager renders whenever rows exist so Prev
    # (shown whenever offset > 0) still navigates back to a valid page.
    offset = max(runs_offset, 0)
    if db is not None:
        try:
            state_db = get_runtime().state_db
            result = get_discovery_run_history(
                state_db, limit=_RUNS_PAGE_SIZE, offset=offset
            )
            run_history = result.get("runs", [])
            # Coerce defensively: a non-int total (e.g. a partial test double)
            # would break the has_next arithmetic at render time.
            try:
                runs_total = int(result.get("total", 0))
            except (TypeError, ValueError):
                runs_total = len(run_history)
        except Exception:
            logger.warning("get_discovery_run_history failed", exc_info=True)

    # Digest viewer: newest-first dropdown options ({id, date}); the template
    # auto-loads the newest into #digest-view via /api/discovery/digest.
    digest_options = _digest_options(db)
    return templates.TemplateResponse(
        request,
        "discovery/index.html",
        {
            "last_run": last_run,
            "recent_runs": recent_runs,
            # P4: unified run-history rows for the single Run History card.
            "run_history": run_history,
            # P4: pagination context for the Run History card (page size 10).
            "runs_total": runs_total,
            "runs_offset": offset,
            "runs_has_prev": offset > 0,
            "runs_has_next": offset + _RUNS_PAGE_SIZE < runs_total,
            "runs_prev_offset": max(offset - _RUNS_PAGE_SIZE, 0),
            "runs_next_offset": offset + _RUNS_PAGE_SIZE,
            "digest_options": digest_options,
            "db_unavailable": db is None,
            # Task 8: at-load defaults for the search time-range card (summary +
            # pre-populated rolling inputs). Computed from the coverage_until
            # frontier; falls back to a 14-day window when unset.
            **_window_defaults(),
        },
    )


# ---------------------------------------------------------------------------
# Digest viewer fragment — render a stored (or freshly-created) digest as HTML
# ---------------------------------------------------------------------------

# markdown.markdown extensions: GFM-ish tables, fenced code, and sane (PEP-style)
# list nesting so the rendered digest reads like the Obsidian note does.
_MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists"]


@router.get("/api/discovery/digest", response_class=HTMLResponse)
def discovery_digest(run_id: int = Query(...)) -> HTMLResponse:
    """Render the ``Discovery_{date}.md`` digest for a run as sanitized HTML.

    Exists path → read the file. Missing → render it from the run's papers via
    ``render_digest`` and write it atomically into the vault, then render.

    Every body is an HTMX-swappable fragment (never a 500): missing run →
    ``Run not found`` (404), no vault → a field-note, IO/render failure → a
    generic card (the absolute path is logged server-side, NOT leaked — A3-5).
    """
    state_db = get_runtime().state_db
    run = get_discovery_run(state_db, run_id)
    if run is None:
        return HTMLResponse(
            '<div class="card danger">Run not found.</div>', status_code=404
        )

    vault = _vault_root()
    if vault is None:
        return HTMLResponse(
            '<p class="field-note">Vault path not configured.</p>'
        )

    date_str = (run.get("started_at") or "")[:10]
    # Same-day runs each get their own file via a per-day ordinal suffix, so the
    # day's 2nd run reads/creates Discovery_{date}_2.md (its own papers) instead
    # of colliding on the bare file (which holds the 1st run's content).
    ordinal = discovery_run_same_day_ordinal(state_db, run_id)
    path = vault / _digests_folder_name() / digest_filename(date_str, ordinal)

    try:
        if path.exists():
            md = path.read_text(encoding="utf-8")
        else:
            papers = get_discovery_run_papers(state_db, run_id, top_k=1000)
            md = render_digest(run, papers)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: temp file in the same dir, then os.replace.
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=".digest-", suffix=".md.tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(md)
                os.replace(tmp_name, path)
            except OSError:
                # Best-effort cleanup of the stray temp file before re-raising.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

        html_body = markdown.markdown(md, extensions=_MD_EXTENSIONS)
        # nh3's default allowlist permits headings/p/ul/ol/li/a/code/pre/em/
        # strong/blockquote and full tables — and strips <script>/<style>/event
        # handlers. This is the XSS boundary for user/LLM-authored digest text.
        safe = nh3.clean(html_body)
    except OSError:
        # A3-5 info-leak guard: the OSError embeds the absolute vault path. Log
        # it (with traceback) server-side; the client gets a generic card only.
        logger.error("discovery_digest: digest IO failed", exc_info=True)
        return HTMLResponse(
            '<div class="card danger">Could not render digest — '
            "check server logs.</div>",
            status_code=500,
        )
    except Exception:  # noqa: BLE001 — render failure must not 500-leak either
        logger.error("discovery_digest: render failed", exc_info=True)
        return HTMLResponse(
            '<div class="card danger">Could not render digest — '
            "check server logs.</div>",
            status_code=500,
        )

    return HTMLResponse(f'<div class="digest-rendered">{safe}</div>')


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

    conversion = get_discovery_run_conversion(state_db, run_id)

    return templates.TemplateResponse(
        request,
        "discovery/run_detail.html",
        {
            "run": run,
            "papers": papers,
            "conversion": conversion,
            "show_feedback_buttons": show_feedback_buttons,
            "detail_crumb": f"Run {run_id}",
        },
    )


# ---------------------------------------------------------------------------
# Process control endpoints (F4.2)
# ---------------------------------------------------------------------------


def _window_argv(
    *,
    window_mode: str,
    since_days: str,
    since: str,
    from_date: str,
    to_date: str,
) -> list[str]:
    """Map the time-range card's selection to the CLI's window flags.

    There is ONE window-resolution path (the CLI's ``resolve_window`` + its
    mutual-exclusivity validation); the web form only chooses which flags to
    pass. ``default`` (or any unrecognized/empty mode, or a mode whose inputs are
    blank) appends NOTHING → the run uses the adaptive frontier. Values are
    passed through verbatim; the CLI validates date format and ranges and will
    refuse a malformed spawn loudly rather than silently mis-running.
    """
    if window_mode == "rolling":
        # Prefer the explicit day count; fall back to the date if only that was
        # written. (site.js keeps them in sync, so usually since_days is set.)
        if since_days:
            return ["--since-days", since_days]
        if since:
            return ["--since", since]
        return []
    if window_mode == "range":
        if from_date and to_date:
            return ["--from", from_date, "--to", to_date]
        return []
    # 'default' / unknown → adaptive frontier (no flags).
    return []


async def _spawn_discovery(
    *,
    dry_run: bool,
    rag_mode: str,
    window_mode: str = "default",
    since_days: str = "",
    since: str = "",
    from_date: str = "",
    to_date: str = "",
) -> asyncio.subprocess.Process:
    """Build the argv and spawn ``lit-monitor run`` with the chosen flags.

    Appends ``[--dry-run]``, ``--rag-mode <mode>`` (FG-3), and the window flags
    matching ``window_mode`` (Task 8): ``--since-days N`` / ``--since D`` /
    ``--from D --to D``; ``default`` appends no window flags (adaptive frontier).
    """
    argv = ["uv", "run", "lit-monitor", "run"]
    if dry_run:
        argv.append("--dry-run")
    # FG-3: thread the chosen retrieval mode through to the CLI.
    argv += ["--rag-mode", rag_mode]
    # Task 8: thread the search-window selection through to the CLI flags.
    argv += _window_argv(
        window_mode=window_mode,
        since_days=since_days,
        since=since,
        from_date=from_date,
        to_date=to_date,
    )
    logger.info("Spawning discovery: %s", " ".join(argv))
    return await asyncio.create_subprocess_exec(*argv, cwd=str(_REPO_ROOT))


@router.post("/api/discovery/start")
async def discovery_start(
    request: Request,
    dry_run: bool = Form(False),
    rag_mode: str = Form(""),
    window_mode: str = Form("default"),
    since_days: str = Form(""),
    since: str = Form(""),
    from_date: str = Form(""),
    to_date: str = Form(""),
) -> JSONResponse:
    """Spawn a discovery run subprocess. Refuses if a run is already in flight.

    FG-3: reads the ``rag_mode`` form field. Invalid/missing values fall back to
    the config default (``retrieval.default_mode``). Selecting ``graph``/``hybrid``
    without the ``[graph]`` extra installed is refused up-front with a visible
    warning (mirrors the CLI W4 hard error) rather than spawning a run that would
    crash mid-flight.

    Task 8: reads the search-window selection (``window_mode`` +
    ``since_days``/``since``/``from_date``/``to_date``) and threads it into the
    spawned argv as the matching CLI flags. ``default`` (untouched) appends no
    window flags → the adaptive frontier is used (unchanged behaviour). The CLI
    owns window validation (mutual exclusivity, date format/ranges).
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
            slot.process = await _spawn_discovery(
                dry_run=dry_run,
                rag_mode=mode,
                window_mode=window_mode,
                since_days=since_days,
                since=since,
                from_date=from_date,
                to_date=to_date,
            )
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
            # Task 8: defaults for the collapsed-summary line inside controls.
            **_window_defaults(),
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
