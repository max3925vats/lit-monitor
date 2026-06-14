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

from lit_monitor.server.rebuild_jobs import (
    enrichment_capability,
    is_rebuild_busy,
    rebuild_argvs,
    run_rebuild_sequence,
    slot_name,
)
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

# Valid component names for the rebuild endpoints.
_REBUILD_COMPONENTS = ("vectors", "graph", "notes")


def _schedule_rebuild(slot: object, argvs: list[list[str]]) -> None:
    """Schedule ``run_rebuild_sequence(slot, argvs)`` as a background asyncio task.

    Defined as a module-level function so tests can patch it without touching
    the asyncio event loop.
    """
    import asyncio  # noqa: PLC0415

    asyncio.create_task(run_rebuild_sequence(slot, argvs))


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
    try:
        config = runtime.config

        def _sum_size(targets: list) -> int:
            return sum(t.size_bytes for t in targets)

        def _sum_md_files(targets: list) -> int:
            """Count *.md files across vault target directories."""
            total = 0
            for tgt in targets:
                if tgt.exists:
                    total += sum(1 for _ in tgt.path.rglob("*.md"))
            return total

        # vectors: meaningful count = papers with embeddings in ChromaDB.
        vtgts = vectors_targets(config)
        try:
            v_count = runtime.state_db.count_embeddings_indexed()
        except Exception:
            v_count = 0

        # graph: meaningful count = papers indexed into KuzuDB.
        gtgts = graph_targets(config)
        try:
            g_count = runtime.state_db.count_graph_indexed()
        except Exception:
            g_count = 0

        # notes: meaningful count = *.md files in vault target dirs.
        ntgts = vault_targets(config)
        try:
            n_count = _sum_md_files(ntgts)
        except Exception:
            n_count = 0

        components = {
            "vectors": {
                "present": any(t.exists for t in vtgts),
                "count": v_count,
                "unit": "paper",
                "size_bytes": _sum_size(vtgts),
            },
            "graph": {
                "present": any(t.exists for t in gtgts),
                "count": g_count,
                "unit": "paper",
                "size_bytes": _sum_size(gtgts),
            },
            "notes": {
                "present": any(t.exists for t in ntgts),
                "count": n_count,
                "unit": "note",
                "size_bytes": _sum_size(ntgts),
            },
        }
        configured = True
    except Exception:
        components = {
            "vectors": {"present": False, "count": 0, "unit": "paper", "size_bytes": 0},
            "graph": {"present": False, "count": 0, "unit": "paper", "size_bytes": 0},
            "notes": {"present": False, "count": 0, "unit": "note", "size_bytes": 0},
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


# ---------------------------------------------------------------------------
# Rebuild: spawn
# ---------------------------------------------------------------------------

@router.post("/api/rebuild/{component}")
async def post_rebuild_component(component: str, request: Request) -> JSONResponse:
    """Spawn a background rebuild job for the given component.

    Guards (in order):
    1. Unknown component → 404.
    2. Any pipeline/rebuild slot busy → 409.
    3. graph + enrich requested but NLP/Ollama-key unavailable → 400.
    4. Schedules ``run_rebuild_sequence`` via ``_schedule_rebuild`` and returns 200.
    """
    # Guard: unknown component
    if component not in _REBUILD_COMPONENTS:
        return JSONResponse(
            {"error": f"unknown rebuild component: {component!r}"},
            status_code=404,
        )

    # Guard: busy
    runtime = get_runtime()
    busy = _busy_slot(runtime)
    if busy:
        return JSONResponse(
            {"error": f"cannot rebuild while {busy!r} is running", "busy": busy},
            status_code=409,
        )

    # Parse form data to read the optional `enrich` field.
    try:
        form = await request.form()
        enrich_raw = form.get("enrich", "")
    except Exception:
        enrich_raw = ""
    enrich = str(enrich_raw).lower() in ("true", "1", "on")

    # Guard: graph enrichment requires NLP extra + Ollama key.
    if component == "graph" and enrich:
        try:
            cap = enrichment_capability()
        except Exception:
            cap = {"nlp": False, "ollama_key": False}
        if not (cap["nlp"] and cap["ollama_key"]):
            return JSONResponse(
                {"error": "enrichment unavailable", "capability": cap},
                status_code=400,
            )

    # Build the argv sequence and schedule.
    try:
        argvs = rebuild_argvs(component, enrich=enrich)
        slot = runtime.processes[slot_name(component)]
        _schedule_rebuild(slot, argvs)
    except Exception as exc:
        logger.error(
            "Rebuild schedule failed for component %r: %s", component, exc, exc_info=True
        )
        return JSONResponse(
            {"error": "rebuild schedule failed — check server logs"},
            status_code=500,
        )

    return JSONResponse({"started": True, "component": component, "enrich": enrich})


# ---------------------------------------------------------------------------
# Rebuild: SSE buffer-follower
# ---------------------------------------------------------------------------

def _new_lines(
    out: list[str],
    total: int,
    idx: int,
) -> tuple[list[str], int]:
    """Return lines not yet seen by the stream and the updated absolute index.

    ``out``   — current snapshot of ``slot.output`` (bounded buffer).
    ``total`` — ``slot.total_appended`` (monotonic; never decremented).
    ``idx``   — absolute count of lines this stream has already emitted.

    When the buffer drops lines from its front (a front-drop), ``idx`` may
    point before ``out[0]``.  We detect that via ``base = total - len(out)``
    and clamp ``idx`` forward to ``base`` so no lines are silently skipped.
    """
    base = total - len(out)  # absolute index of out[0]
    if idx < base:
        # We fell behind a front-drop; resume from the earliest buffered line.
        idx = base
    lines = out[idx - base : total - base]  # out[idx-base .. len(out)-1]
    return lines, idx + len(lines)


@router.get("/api/rebuild/{component}/stream")
async def rebuild_stream(request: Request, component: str) -> object:
    """SSE stream that follows ``slot.output`` as the rebuild job appends to it.

    This endpoint reads from the in-memory ``slot.output`` buffer, NOT from the
    subprocess stdout pipe.  ``run_rebuild_sequence`` is the sole reader of
    the pipe; it drains each line into ``slot.output`` via ``slot.append_line``.
    A second reader on the same pipe would cause corruption — hence the buffer
    model here.

    Front-drop safety: ``slot.total_appended`` is a monotonic counter that
    never decrements even when ``append_line`` drops lines from the buffer's
    front.  ``_new_lines`` uses it to compute the correct slice offset so no
    buffered lines are silently skipped after a front-drop.
    """
    import asyncio  # noqa: PLC0415
    from html import escape  # noqa: PLC0415

    from sse_starlette.sse import EventSourceResponse  # noqa: PLC0415

    if component not in _REBUILD_COMPONENTS:
        # Emit a single SSE error event then close.
        async def _err_gen():
            yield {"event": "error", "data": f"unknown component: {component!r}"}

        return EventSourceResponse(_err_gen(), ping=15)

    runtime = get_runtime()
    slot = runtime.processes[slot_name(component)]

    async def _gen():
        idx = 0  # absolute count of lines this stream has emitted
        while True:
            if await request.is_disconnected():
                return

            lines, idx = _new_lines(slot.output, slot.total_appended, idx)
            for line in lines:
                yield {
                    "event": "progress",
                    "data": f'<div class="log-line">{escape(line)}</div>',
                }

            # Done when the sequence is no longer active AND the buffer is drained.
            if not is_rebuild_busy(slot) and idx >= slot.total_appended:
                yield {"event": "done", "data": "rebuild finished"}
                return

            await asyncio.sleep(0.5)

    return EventSourceResponse(_gen(), ping=15)


__all__ = ["router"]
