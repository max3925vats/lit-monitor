"""Discovery-run schedule management endpoints.

Front-end for scripts.server.scheduler. Supports macOS (launchd) and Linux
(systemd user timer). Unsupported platforms (Windows, BSD, etc.) get a
banner + disabled form.
"""
from __future__ import annotations

import logging
import subprocess

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse

from lit_monitor.api.queries import get_discovery_run_history
from lit_monitor.server.app import templates
from lit_monitor.server.runtime import get_runtime
from lit_monitor.server.scheduler import (
    ScheduleSpec,
    detect_platform,
    read_schedule,
    remove_schedule,
    write_schedule,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["schedule"])

# P4: page size for the Run History card (mirrors discovery's _RUNS_PAGE_SIZE).
_RUNS_PAGE_SIZE = 10


def _safe_db():
    """Return the runtime state DB, or None when it is unavailable.

    Mirrors discovery's _safe_db(): a missing/broken runtime yields None so the
    schedule dashboard renders a friendly empty Run History card instead of 500.
    """
    try:
        return get_runtime().state_db
    except Exception:
        logger.debug("state_db unavailable for schedule dashboard", exc_info=True)
        return None


def _safe_current_schedule() -> ScheduleSpec | None:
    """Read current schedule, returning None on any error or unsupported OS."""
    plat = detect_platform()
    if plat == "unsupported":
        return None
    try:
        return read_schedule()
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("read_schedule failed", exc_info=True)
        return None


@router.get("/schedule", response_class=HTMLResponse)
def dashboard(
    request: Request,
    runs_offset: int = Query(0, ge=0),
) -> HTMLResponse:
    plat = detect_platform()
    current = _safe_current_schedule()

    # P4: SAME consolidated Run History card as /discovery, but filtered to
    # scheduled runs only (trigger="scheduled") — manual "Run now"/CLI runs never
    # appear here. Run numbers are the GLOBAL discovery_runs.id, so a run shown as
    # #3 on Discovery is also #3 here. Paginated by 10 (offset-link prev/next),
    # pager base /schedule. Defensive: any failure → empty card, never a 500.
    db = _safe_db()
    run_history: list[dict] = []
    runs_total = 0
    offset = max(runs_offset, 0)
    if db is not None:
        try:
            result = get_discovery_run_history(
                db, limit=_RUNS_PAGE_SIZE, offset=offset, trigger="scheduled"
            )
            run_history = result.get("runs", [])
            try:
                runs_total = int(result.get("total", 0))
            except (TypeError, ValueError):
                runs_total = len(run_history)
        except Exception:
            logger.warning("get_discovery_run_history (scheduled) failed", exc_info=True)

    return templates.TemplateResponse(
        request,
        "schedule/index.html",
        {
            "platform": plat,
            "is_supported": plat != "unsupported",
            "current": current,
            # P4: unified run-history rows + page-size-10 pager context. pager_base
            # and empty_text are set in the template (passed to the shared partial).
            "run_history": run_history,
            "runs_total": runs_total,
            "runs_offset": offset,
            "runs_has_prev": offset > 0,
            "runs_has_next": offset + _RUNS_PAGE_SIZE < runs_total,
            "runs_prev_offset": max(offset - _RUNS_PAGE_SIZE, 0),
            "runs_next_offset": offset + _RUNS_PAGE_SIZE,
        },
    )


@router.post("/api/schedule", response_class=HTMLResponse)
def create_schedule(
    request: Request,
    day_of_week: str = Form(...),
    time: str = Form(...),
) -> HTMLResponse:
    plat = detect_platform()
    if plat == "unsupported":
        return HTMLResponse(
            f'<div class="card danger">Scheduling is not supported on this platform ({plat}).</div>',
            status_code=400,
        )
    try:
        spec = ScheduleSpec.parse(day_of_week, time)
    except ValueError as exc:
        # Info-leak guard: log the parse detail server-side; return a static
        # guidance message (the only useful, non-leaky thing for the user).
        logger.info("schedule parse rejected: %s", exc)
        return HTMLResponse(
            '<div class="card danger">Invalid schedule — day must be a weekday '
            'name and time must be HH:MM (24h).</div>',
            status_code=400,
        )
    try:
        path = write_schedule(spec)
    except subprocess.CalledProcessError as exc:
        # stderr may be bytes (from capture_output=True) or str.
        if isinstance(exc.stderr, (bytes, bytearray)):
            stderr = exc.stderr.decode("utf-8", errors="replace")
        else:
            stderr = exc.stderr or ""
        # A3-5 info-leak guard: the subprocess stderr can embed absolute paths
        # and command lines. Log it server-side; the client gets a generic
        # message that points the operator at the server logs.
        logger.error(
            "%s schedule install failed: %s", plat, stderr or exc, exc_info=True
        )
        return HTMLResponse(
            f'<div class="card danger">{plat} schedule install failed — '
            "check server logs for details.</div>",
            status_code=500,
        )
    except NotImplementedError as exc:
        # Info-leak guard: the message embeds the platform string; log it and
        # return a static notice.
        logger.info("schedule install not implemented: %s", exc)
        return HTMLResponse(
            '<div class="card danger">Scheduling is not supported on this '
            'platform.</div>',
            status_code=400,
        )
    return HTMLResponse(
        f'<div class="card success">Schedule installed at <code>{path}</code>. '
        f'Reloading…</div>',
        headers={"HX-Refresh": "true"},
    )


@router.delete("/api/schedule", response_class=HTMLResponse)
def delete_schedule(request: Request) -> HTMLResponse:
    plat = detect_platform()
    if plat == "unsupported":
        return HTMLResponse(
            f'<div class="card danger">Not supported on this platform ({plat}).</div>',
            status_code=400,
        )
    try:
        remove_schedule()
    except Exception:
        # A3-5 info-leak guard: full exception logged server-side; generic
        # client message (the raw error can embed launchd/systemd paths).
        logger.exception("remove_schedule failed")
        return HTMLResponse(
            '<div class="card danger">Could not remove schedule — '
            "check server logs for details.</div>",
            status_code=500,
        )
    return HTMLResponse(
        '<div class="card success">Schedule removed. Reloading…</div>',
        headers={"HX-Refresh": "true"},
    )
