"""FastAPI application factory for the lit-monitor web UI.

Only the landing page and a health endpoint are wired here. Sub-app
routers (Setup, Brain-Build, Discovery, Schedule) are introduced in later
F-series tasks and will be mounted alongside these via their own
include_router calls.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lit_monitor.server.csrf import _ALLOWED_HOSTS, LocalOriginMiddleware
from lit_monitor.server.routes.fs import router as fs_router
from lit_monitor.server.routes.zotero import router as zotero_router

# NOTE: setup router imports `templates` from this module, so its import
# must come AFTER the `templates = Jinja2Templates(...)` line below to
# avoid an AttributeError at import time. We import it inside create_app().
from lit_monitor.server.runtime import get_runtime

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def configure_templates(t):
    """Register the shared Jinja globals + filters on a Jinja2Templates instance.

    Centralised so production AND test-local templates stay in sync (the app-shell
    sidebar needs nav_groups / active_group_for_path on every render)."""
    import json as _json

    from lit_monitor.server.nav import NAV_GROUPS, active_group_for_path
    t.env.globals["nav_groups"] = NAV_GROUPS
    t.env.globals["active_group_for_path"] = active_group_for_path
    t.env.filters["fromjson"] = _json.loads
    return t


configure_templates(templates)


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


def _resolve_allowed_hosts() -> frozenset[str]:
    """Return the local-host set for the CSRF guard, plus the configured bind host.

    The defaults cover loopback (and the TestClient ``testserver`` Host). If the
    user has persisted a ``[server].host`` in the secrets TOML and bound the
    server to a non-localhost address, that address must also count as "local"
    so the DNS-rebinding Host check doesn't 403 legitimate requests. Reading the
    config is best-effort: any failure degrades to the loopback-only defaults
    rather than breaking app creation.

    A wildcard bind address (``0.0.0.0`` / ``::``) is a *bind* address, not a
    reachable ``Host`` value, so it is never added to the allow-list — doing so
    would be inert at best and is sloppy. Real hostnames are still added.
    """
    try:
        from lit_monitor.server.config_io import load_server_config

        host = str(load_server_config().get("host", "")).strip().lower()
    except Exception as exc:  # noqa: BLE001 — never let config IO break boot
        logger.debug("Could not read [server].host for CSRF allow-list: %s", exc)
        host = ""

    # Skip wildcard bind addresses: they bind every interface but are not a
    # value any client sends as Host, so allow-listing them protects nothing.
    if host and host not in ("0.0.0.0", "::") and host not in _ALLOWED_HOSTS:
        return frozenset(_ALLOWED_HOSTS | {host})
    return _ALLOWED_HOSTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan hook.

    The ingest worker startup will be wired in here in F6.0. For now,
    nothing to do on enter or exit.
    """

    yield


def create_app() -> FastAPI:
    """Build and return the FastAPI application instance."""

    # Read the dev flag from the environment (set by `serve --dev`).
    # Anything other than "1" is treated as off, including unset.
    dev_mode = os.environ.get("LIT_MONITOR_DEV") == "1"

    app = FastAPI(title="lit-monitor", version=__version__, lifespan=lifespan)
    app.state.dev_mode = dev_mode
    # Exposed on app.state so base.html footer can render the version on every
    # route, not just routes that explicitly pass `version` into their context.
    app.state.version = __version__

    # AR-1: CSRF + DNS-rebinding guard. Installed before any router so it runs
    # on every request. The allowed-host set is the loopback defaults plus the
    # configured bind host -- so if the user ever binds non-localhost via
    # [server].host, their own bind address still counts as "local" and isn't
    # 403'd by the DNS-rebinding check. Reading the host is wrapped defensively:
    # any failure (no/unreadable secrets file) falls back to loopback-only,
    # which is the safe default and never breaks app boot.
    app.add_middleware(LocalOriginMiddleware, allowed_hosts=_resolve_allowed_hosts())

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

    # Imported lazily so the setup module can safely `from lit_monitor.server.app
    # import templates` — by the time create_app() runs, `templates` is bound.
    from lit_monitor.server.routes.brain_build import router as brain_build_router
    from lit_monitor.server.routes.control import router as control_router
    from lit_monitor.server.routes.discovery import router as discovery_router
    from lit_monitor.server.routes.discovery_notify import (  # noqa: PLC0415
        router as discovery_notify_router,
    )
    from lit_monitor.server.routes.health import router as health_router
    from lit_monitor.server.routes.ingest import router as ingest_router
    from lit_monitor.server.routes.schedule import router as schedule_router
    from lit_monitor.server.routes.setup import router as setup_router
    from lit_monitor.server.routes.sse import router as sse_router

    app.include_router(setup_router)
    app.include_router(brain_build_router)
    app.include_router(control_router)
    app.include_router(sse_router)
    # P3 notify-handler MUST register BEFORE discovery_router so that
    # /discovery/notify-handler is matched first. Otherwise P8's
    # /discovery/{run_id:int} route is checked first and FastAPI returns
    # 422 (can't coerce "notify-handler" to int) without falling through.
    app.include_router(discovery_notify_router)
    app.include_router(discovery_router)
    app.include_router(schedule_router)
    app.include_router(health_router)
    # H2/A1: POST /api/ingest — production async ingest endpoint for the Zotero
    # plugin / web ingestors. Validates + queues + schedules a BackgroundTask
    # that runs the real brain_build ingestion, then returns 202 immediately.
    # Clients poll GET /api/ingest/{doi}/status for the terminal state.
    app.include_router(ingest_router)

    # H4: GET /api/papers/{doi} — paper snapshot (wraps H1 get_paper_snapshot).
    # H5: GET /api/papers/{doi}/related — related papers (vector/graph/hybrid).
    from lit_monitor.server.routes.papers import router as papers_router  # noqa: PLC0415

    app.include_router(papers_router)

    # H6: GET /api/entities — listing; GET /api/entities/{id} — neighborhood.
    from lit_monitor.server.routes.entities import router as entities_router  # noqa: PLC0415

    app.include_router(entities_router)

    # H8: POST /api/ask — NL→Cypher→execute→summarize pipeline.
    #     POST /api/cypher — read-only Cypher escape hatch with B3's guard.
    from lit_monitor.server.routes.query import router as query_router  # noqa: PLC0415

    app.include_router(query_router)

    # FE-1: /ask page (browser surface over run_pipeline; JSON /api/ask unchanged).
    from lit_monitor.server.routes.ask import router as ask_router  # noqa: PLC0415

    app.include_router(ask_router)

    # H10: POST /api/search — free-text paper retrieval (vector/graph/hybrid).
    from lit_monitor.server.routes.search import router as search_router  # noqa: PLC0415

    app.include_router(search_router)

    # Bundle G (v0.9): /api/domain — LLM extraction of structured focus
    # areas from config/domain_context.yaml. Also /domain page (HTMX UI).
    from lit_monitor.server.routes.domain import router as domain_router  # noqa: PLC0415

    app.include_router(domain_router)

    # Bundle E (v0.9): /api/trending + /trending — trending-concept suggestion queue.
    from lit_monitor.server.routes.trending import router as trending_router  # noqa: PLC0415

    app.include_router(trending_router)

    # Bundle E (v0.9): /api/topics/{name}/expansion-suggestions — query expansion.
    from lit_monitor.server.routes.topics import router as topics_router  # noqa: PLC0415

    app.include_router(topics_router)

    # Bundle H (v0.9): /themes + /themes/{id} — cluster review surface.
    # /themes (static) must be registered BEFORE /themes/{cluster_id} (typed int param)
    # so the index route is matched first. Both live in the same router; FastAPI
    # evaluates routes in registration order so the static GET /themes endpoint
    # (no param) will never be shadowed by the /{cluster_id} typed param.
    from lit_monitor.server.routes.themes import router as themes_router  # noqa: PLC0415

    app.include_router(themes_router)

    # Bundle H (v0.9): /feedback + /api/feedback — feedback events admin view.
    from lit_monitor.server.routes.feedback import router as feedback_router  # noqa: PLC0415

    app.include_router(feedback_router)

    # FE2-2: /corpus — list lens over the processed-papers corpus (state DB).
    from lit_monitor.server.routes.corpus import corpus_router  # noqa: PLC0415

    app.include_router(corpus_router)

    # FG-4: /graph — read-only corpus-wide Knowledge Graph overview page.
    from lit_monitor.server.routes.graph import graph_router  # noqa: PLC0415

    app.include_router(graph_router)

    # Bundle H (v0.9): /settings + /api/settings/{section} — Advanced Settings.
    from lit_monitor.server.routes.settings import router as settings_router  # noqa: PLC0415

    app.include_router(settings_router)

    # Dev-only /dev test surface. Wrapped in try/except so a syntax error or
    # import-time failure inside routes/dev.py downgrades to a warning rather
    # than killing the whole server.
    if dev_mode:
        try:
            from lit_monitor.server.routes import dev as dev_routes

            app.include_router(dev_routes.router)
            logger.info("Dev mode enabled — /dev router mounted.")
        except Exception as exc:  # pragma: no cover — defensive boot path
            logger.warning("Failed to mount /dev router: %s", exc)

    return app


__all__ = ["create_app", "__version__"]
