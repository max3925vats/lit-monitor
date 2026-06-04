"""Process control endpoints for brain-build + discovery.

Spawns and supervises CLI subprocesses via the runtime process registry.
SIGTERM-then-SIGKILL stop pattern; the existing rate-limit-abort precedent
in ``scripts/pipelines/brain_build.py`` handles graceful within-pipeline
cleanup, so the 30-second grace window gives the pipeline time to finish
the current paper before being forced down.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from scripts.server.app import templates
from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)
router = APIRouter(tags=["control"])

# Repo root: scripts/server/routes/control.py → 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]


async def _spawn_brain_build(
    *,
    resume: bool,
    max_papers: int | None,
    all_library: bool,
) -> asyncio.subprocess.Process:
    """Build the argv and spawn ``lit-monitor brain-build``."""
    argv = ["uv", "run", "lit-monitor", "brain-build"]
    if resume:
        argv.append("--resume")
    else:
        argv.append("--no-resume")
    if max_papers is not None:
        argv.extend(["--max-papers", str(max_papers)])
    if all_library:
        argv.append("--all-library")
    logger.info("Spawning brain-build: %s", " ".join(argv))
    return await asyncio.create_subprocess_exec(
        *argv, cwd=str(_REPO_ROOT),
    )


@router.post("/api/brain-build/start")
async def brain_build_start(
    request: Request,
    resume: bool = Form(True),
    max_papers: int | None = Form(None),
    all_library: bool = Form(False),
) -> JSONResponse:
    """Spawn a brain-build subprocess. Refuses if a build is already running."""
    runtime = get_runtime()
    slot = runtime.processes["brain_build"]
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
            slot.process = await _spawn_brain_build(
                resume=resume,
                max_papers=max_papers,
                all_library=all_library,
            )
            slot.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        except Exception:  # noqa: BLE001 — surface any spawn failure
            # A3-5 info-leak guard: full exception (with traceback) logged
            # server-side; client gets a generic reason only — the raw spawn
            # error can embed argv / filesystem paths.
            logger.exception("Failed to spawn brain-build")
            return JSONResponse(
                {"started": False, "reason": "spawn failed"},
                status_code=500,
            )
    return JSONResponse(
        {"started": True, "pid": slot.process.pid, "started_at": slot.started_at}
    )


@router.post("/api/brain-build/stop")
async def brain_build_stop() -> JSONResponse:
    """SIGTERM the brain-build subprocess; wait up to 30s, then SIGKILL."""
    runtime = get_runtime()
    slot = runtime.processes["brain_build"]
    stopped = await slot.stop(timeout=30.0)
    return JSONResponse({"stopped": stopped})


@router.get("/api/brain-build/status")
async def brain_build_status() -> JSONResponse:
    """Polled by the controls fragment to drive the button state."""
    runtime = get_runtime()
    slot = runtime.processes["brain_build"]
    running = slot.is_running()
    return JSONResponse(
        {
            "running": running,
            "pid": slot.process.pid if running and slot.process else None,
            "started_at": slot.started_at if running else None,
        }
    )


@router.get("/api/brain-build/controls", response_class=HTMLResponse)
async def brain_build_controls(request: Request) -> HTMLResponse:
    """HTMX-polled HTML fragment that renders Start/Resume or Stop buttons."""
    runtime = get_runtime()
    slot = runtime.processes["brain_build"]
    running = slot.is_running()
    return templates.TemplateResponse(
        request,
        "brain_build/_controls.html",
        {
            "is_running": running,
            "pid": slot.process.pid if running and slot.process else None,
            "started_at": slot.started_at if running else None,
        },
    )


__all__ = ["router"]
