"""Discovery-run schedule management endpoints.

Front-end for scripts.server.scheduler. Supports macOS (launchd) and Linux
(systemd user timer). Unsupported platforms (Windows, BSD, etc.) get a
banner + disabled form.
"""
from __future__ import annotations

import logging
import subprocess

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

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


def _safe_recent_runs() -> list[dict]:
    """Last 10 discovery runs, or [] if state DB is unavailable."""
    try:
        r = get_runtime()
        return r.state_db.get_recent_runs_by_type("discovery", limit=10)
    except Exception:
        logger.debug("get_recent_runs_by_type failed", exc_info=True)
        return []


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
def dashboard(request: Request) -> HTMLResponse:
    plat = detect_platform()
    current = _safe_current_schedule()
    recent_runs = _safe_recent_runs()
    return templates.TemplateResponse(
        request,
        "schedule/index.html",
        {
            "platform": plat,
            "is_supported": plat != "unsupported",
            "current": current,
            "recent_runs": recent_runs,
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
        return HTMLResponse(
            f'<div class="card danger">{exc}</div>',
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
        return HTMLResponse(
            f'<div class="card danger">{exc}</div>',
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
