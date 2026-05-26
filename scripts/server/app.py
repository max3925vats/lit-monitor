"""FastAPI application factory for the lit-monitor web UI.

Only the landing page and a health endpoint are wired here. Sub-app
routers (Setup, Brain-Build, Discovery, Schedule) are introduced in later
F-series tasks and will be mounted alongside these via their own
include_router calls.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.server.routes.fs import router as fs_router
from scripts.server.routes.zotero import router as zotero_router

# NOTE: setup router imports `templates` from this module, so its import
# must come AFTER the `templates = Jinja2Templates(...)` line below to
# avoid an AttributeError at import time. We import it inside create_app().
from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _read_version() -> str:
    """Resolve the installed package version, falling back to ``dev``."""

    try:
        from importlib.metadata import version

        return version("lit-monitor")
    except Exception:
        # importlib.metadata.PackageNotFoundError when running from a fresh
        # source tree without an editable install. Anything else: degrade quietly.
        return "dev"


__version__ = _read_version()


def _safe_last_run() -> dict | None:
    """Return the most recent run_log entry, or ``None`` on any failure.

    Wrapped in try/except so a missing state DB or unconfigured env does
    not break the health endpoint.
    """

    try:
        runtime = get_runtime()
        rows = runtime.state_db.get_recent_runs(limit=1)
        return rows[0] if rows else None
    except Exception as exc:
        logger.debug("last_run lookup failed: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan hook.

    The ingest worker startup will be wired in here in F6.0. For now,
    nothing to do on enter or exit.
    """

    yield


def create_app() -> FastAPI:
    """Build and return the FastAPI application instance."""

    app = FastAPI(title="lit-monitor", version=__version__, lifespan=lifespan)

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "last_run": _safe_last_run(),
        }

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        last_run = _safe_last_run()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "version": __version__,
                "last_run": last_run,
            },
        )

    app.include_router(fs_router)
    app.include_router(zotero_router)

    # Imported lazily so the setup module can safely `from scripts.server.app
    # import templates` — by the time create_app() runs, `templates` is bound.
    from scripts.server.routes.brain_build import router as brain_build_router
    from scripts.server.routes.control import router as control_router
    from scripts.server.routes.discovery import router as discovery_router
    from scripts.server.routes.schedule import router as schedule_router
    from scripts.server.routes.setup import router as setup_router
    from scripts.server.routes.sse import router as sse_router

    app.include_router(setup_router)
    app.include_router(brain_build_router)
    app.include_router(control_router)
    app.include_router(sse_router)
    app.include_router(discovery_router)
    app.include_router(schedule_router)

    return app


__all__ = ["create_app", "__version__"]
