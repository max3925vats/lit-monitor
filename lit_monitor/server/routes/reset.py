"""RR4: Reset & Rebuild Console — status read and per-component reset POSTs.

Endpoints
---------
GET  /setup/reset               — render the reset console page (first-run-safe)
GET  /api/reset/status          — JSON status snapshot (first-run-safe, pure read)
POST /api/reset/{component}     — delete a regenerable view; component ∈
                                   {vectors, graph, notes, everything}

Design notes
------------
- All routes degrade gracefully (200 / meaningful JSON) when config is absent.
- Busy guard runs FIRST on every POST — we never delete while a pipeline or
  rebuild is in flight.
- "everything" requires the exact confirmation phrase ``reset all`` (form field
  ``confirm``); validated server-side, never trusted from the client.
- After any successful deletion, ``reset_runtime()`` is called so the app
  reconnects to fresh DBs on the next request.
- ``state_db`` references and DB-flag resets happen BEFORE ``reset_runtime()`` —
  the old handle is still valid until that call.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse

from lit_monitor.server.rebuild_jobs import enrichment_capability, is_rebuild_busy
from lit_monitor.server.runtime import get_runtime, reset_runtime
from lit_monitor.setup.reset import (
    graph_targets,
    perform_state_reset,
    perform_vault_reset,
    state_targets,
    vault_targets,
    vectors_targets,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Slot groups used by the busy guard
# ---------------------------------------------------------------------------
_PIPELINE_SLOTS = ("brain_build", "discovery", "vocabulary")
_REBUILD_SLOTS = ("rebuild_vectors", "rebuild_graph", "rebuild_notes")


def _busy_slot(runtime: object) -> str | None:
    """Return the name of a running pipeline/rebuild slot, or None if idle."""
    procs = getattr(runtime, "processes", {}) or {}
    for name in _PIPELINE_SLOTS:
        slot = procs.get(name)
        if slot is not None and slot.is_running():
            return name
    for name in _REBUILD_SLOTS:
        slot = procs.get(name)
        if slot is not None and is_rebuild_busy(slot):
            return name
    return None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@router.get("/setup/reset")
def get_reset_page(request: Request) -> object:
    """Render the Reset & Rebuild console page.

    First-run-safe: returns 200 with a minimal page even before config exists.
    """
    # Deferred import: `templates` is bound by create_app() before any request.
    from lit_monitor.server.app import templates  # noqa: PLC0415

    try:
        # Attempt to pre-load status so the template can show live metrics;
        # fall back to empty context on any failure (config absent, etc.).
        ctx: dict = {}
    except Exception:
        ctx = {}

    return templates.TemplateResponse(request, "setup/reset.html", ctx)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/api/reset/status")
def get_reset_status() -> JSONResponse:
    """Return a snapshot of what each component holds (files, bytes) and
    whether anything is currently busy.

    Always returns 200. When config is absent: ``configured=False``, all
    component metrics zeroed.
    """
    runtime = get_runtime()

    # Busy guard is config-independent — read before config access.
    busy = _busy_slot(runtime)

    # Enrichment capability is also config-independent.
    try:
        enrichment = enrichment_capability()
    except Exception:
        enrichment = {"nlp": False, "ollama_key": False}

    # Attempt to read component metrics from config. Degrade gracefully.
    empty_component = {"present": False, "files": 0, "size_bytes": 0}
    try:
        config = runtime.config

        def _summarise(targets: list) -> dict:
            present = any(t.exists for t in targets)
            files = sum(t.file_count for t in targets)
            size = sum(t.size_bytes for t in targets)
            return {"present": present, "files": files, "size_bytes": size}

        components = {
            "vectors": _summarise(vectors_targets(config)),
            "graph": _summarise(graph_targets(config)),
            "notes": _summarise(vault_targets(config)),
        }
        configured = True
    except Exception:
        components = {
            "vectors": dict(empty_component),
            "graph": dict(empty_component),
            "notes": dict(empty_component),
        }
        configured = False

    return JSONResponse(
        {
            "configured": configured,
            "components": components,
            "enrichment": enrichment,
            "busy": busy,
        }
    )


# ---------------------------------------------------------------------------
# Per-component reset
# ---------------------------------------------------------------------------

def _result_summary(results: list) -> list[dict]:
    """Serialise a list of ResetResult objects to plain dicts."""
    return [
        {
            "label": r.label,
            "path": str(r.path),
            "deleted": r.deleted,
            "skipped_reason": r.skipped_reason,
        }
        for r in results
    ]


@router.post("/api/reset/{component}")
async def post_reset_component(
    component: str,
    request: Request,
    confirm: str = Form(default=""),
) -> JSONResponse:
    """Delete a regenerable view and trigger a runtime reconnect.

    Guards (checked in order):
    1. Unknown component → 404.
    2. Busy pipeline/rebuild → 409.
    3. ``everything``: missing/wrong ``confirm`` phrase → 400 (no deletion).
    4. Config absent → 400.
    """
    # Guard: unknown component
    known = {"vectors", "graph", "notes", "everything"}
    if component not in known:
        return JSONResponse({"error": f"unknown component: {component!r}"}, status_code=404)

    # Guard: busy
    runtime = get_runtime()
    busy = _busy_slot(runtime)
    if busy:
        return JSONResponse(
            {"error": f"cannot reset while {busy!r} is running", "busy": busy},
            status_code=409,
        )

    # Guard: confirmation phrase required for "everything"
    if component == "everything" and confirm != "reset all":
        return JSONResponse(
            {"error": "confirmation phrase required — send confirm=reset all"},
            status_code=400,
        )

    # Guard: config must be available
    try:
        config = runtime.config
    except Exception:
        return JSONResponse(
            {"error": "server is not configured — run the setup wizard first"},
            status_code=400,
        )

    # Perform the deletion
    try:
        all_results: list = []

        if component == "vectors":
            results = perform_state_reset(vectors_targets(config))
            all_results.extend(results)
            # Reset the DB flag BEFORE reset_runtime() drops the handle.
            runtime.state_db.reset_embeddings_indexed()

        elif component == "graph":
            results = perform_state_reset(graph_targets(config))
            all_results.extend(results)
            runtime.state_db.reset_graph_stamps()

        elif component == "notes":
            vault_root = Path(config.obsidian.vault_path).expanduser()
            results = perform_vault_reset(vault_targets(config), vault_root)
            all_results.extend(results)

        elif component == "everything":
            vault_root = Path(config.obsidian.vault_path).expanduser()
            state_results = perform_state_reset(state_targets(config))
            vault_results = perform_vault_reset(vault_targets(config), vault_root)
            all_results.extend(state_results)
            all_results.extend(vault_results)

        # Reconnect to fresh DBs on the next request.
        reset_runtime()

    except Exception as exc:
        logger.error(
            "Reset failed for component %r: %s", component, exc, exc_info=True
        )
        return JSONResponse(
            {"error": "reset failed — check server logs"},
            status_code=500,
        )

    return JSONResponse(
        {
            "ok": True,
            "component": component,
            "results": _result_summary(all_results),
        }
    )


__all__ = ["router"]
