"""Server-Sent Events streams for live pipeline progress.

GET /api/brain-build/stream — tails the newest brain_build JSONL log
under the canonical logs dir (``<state_db parent>/logs``, default
``~/.config/lit-monitor/logs/``; see ``_resolve_logs_dir`` below), forwards
each new line as a ``progress`` event.
GET /api/discovery/stream — same machinery, but tails the newest discovery
log instead. Both routes are thin wrappers around ``stream_log``.

15-second heartbeats are emitted by sse-starlette via the ``ping`` kwarg.
Closes on client disconnect.

The JSONL log layout is defined by ``scripts.cli._setup_logging`` /
``_JsonlFileHandler`` — one JSON object per line, file named
``{YYYY-MM-DD}_{mode}.jsonl``. The filename suffix is the only mode
indicator, so "filter by mode" reduces to "tail the newest file whose
name ends in ``_{mode}.jsonl``".
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sse"])

# Canonical logs directory — mirrors ``scripts.cli._resolve_log_dir`` so the
# SSE streams tail the same files the CLI writes. Logs live under the user
# config dir (~/.config/lit-monitor/logs/), co-located with state.db. Falls back
# to that default when config is unavailable.
_DEFAULT_LOGS_DIR = Path("~/.config/lit-monitor/logs").expanduser()


def _resolve_logs_dir() -> Path:
    """Resolve ``<state_db parent>/logs`` from config; fall back to the default.

    Kept in lockstep with ``scripts.cli._resolve_log_dir`` — both derive the log
    dir from ``config.state_db.path`` so a user override of the state-db path
    co-locates logs, and both fall back to ``~/.config/lit-monitor/logs``.
    """
    try:
        from lit_monitor.core.config import get_config
        cfg = get_config()
        state_path = getattr(getattr(cfg, "state_db", None), "path", None)
        if state_path:
            return Path(state_path).expanduser().parent / "logs"
    except Exception:  # noqa: BLE001 — SSE must degrade gracefully without config
        pass
    return _DEFAULT_LOGS_DIR

# Poll interval between readline attempts when at EOF.
_POLL_INTERVAL_S = 0.5


def _newest_log(mode: str) -> Path | None:
    """Return the most-recently-modified ``*_{mode}.jsonl`` under logs/.

    Returns ``None`` when the logs directory or any matching file is
    absent.
    """

    logs_dir = _resolve_logs_dir()
    if not logs_dir.exists():
        return None
    candidates = sorted(
        logs_dir.glob(f"*_{mode}.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_line(fh: IO[str]) -> str:
    """Blocking readline — run via ``asyncio.to_thread`` from the loop."""

    return fh.readline()


def _prettify_jsonl_line(stripped: str) -> str:
    """Convert one JSONL log record into a single-line HTML fragment.

    Output shape (consumed by HTMX ``sse-swap="progress"``+
    ``hx-swap="beforeend"``)::

        <div class="log-line log-INFO">
          <span class="log-ts">02:15:01</span>
          <span class="log-level">INFO</span>
          <span class="log-logger">search.search_runner</span>
          <span class="log-msg">Searching topic: …</span>
        </div>

    Non-JSON or malformed lines fall back to ``<div class="log-line log-raw">``
    so the stream stays usable when a non-JSON message slips into the log.
    All interpolated fields go through ``html.escape``.
    """
    import json
    from html import escape

    try:
        rec = json.loads(stripped)
    except (ValueError, TypeError):
        return f'<div class="log-line log-raw">{escape(stripped)}</div>'

    ts = rec.get("ts", "") if isinstance(rec, dict) else ""
    # ISO timestamps are "YYYY-MM-DDTHH:MM:SS+00:00" — slice to HH:MM:SS.
    ts_short = ts[11:19] if len(ts) >= 19 and ts[10:11] == "T" else ts

    level = (rec.get("level") or "INFO").upper() if isinstance(rec, dict) else "INFO"

    lg = rec.get("logger", "") if isinstance(rec, dict) else ""
    # Drop the "lit_monitor." prefix so the logger column stays narrow.
    if lg.startswith("lit_monitor."):
        lg = lg[len("lit_monitor."):]

    msg = rec.get("msg", "") if isinstance(rec, dict) else ""

    return (
        f'<div class="log-line log-{escape(level)}">'
        f'<span class="log-ts">{escape(ts_short)}</span>'
        f'<span class="log-level">{escape(level)}</span>'
        f'<span class="log-logger">{escape(lg)}</span>'
        f'<span class="log-msg">{escape(msg)}</span>'
        f'</div>'
    )


async def _tail(log_path: Path, request: Request) -> AsyncIterator[dict]:
    """Async generator yielding sse-starlette event dicts.

    Opens ``log_path`` and seeks to EOF so only NEW lines are streamed —
    the historical content is not replayed. Polls for new data every
    ``_POLL_INTERVAL_S`` seconds. Exits cleanly when the client
    disconnects.

    Each new JSONL line is prettified into a one-line HTML fragment via
    ``_prettify_jsonl_line`` before being emitted as the ``progress`` event
    data, so HTMX consumers (Panel 5, Discovery, brain-build) can render
    columnar, level-coloured rows instead of raw JSON.
    """

    try:
        fh = open(log_path, encoding="utf-8", errors="replace")
        fh.seek(0, 2)  # seek to EOF — only stream new lines
    except OSError as exc:
        logger.warning("SSE: could not open %s: %s", log_path, exc)
        yield {"event": "error", "data": f"could not open log: {exc}"}
        return

    try:
        while True:
            if await request.is_disconnected():
                return
            line = await asyncio.to_thread(_read_line, fh)
            if line:
                stripped = line.rstrip()
                if stripped:
                    yield {"event": "progress", "data": _prettify_jsonl_line(stripped)}
            else:
                await asyncio.sleep(_POLL_INTERVAL_S)
    finally:
        fh.close()


def stream_log(request: Request, mode: str) -> EventSourceResponse:
    """Return an SSE response tailing the newest ``*_{mode}.jsonl`` log.

    If no log file exists yet, emits a single ``error`` event and closes.
    Otherwise tails the file indefinitely until the client disconnects.

    Used by both ``/api/brain-build/stream`` and ``/api/discovery/stream``.
    """

    log_path = _newest_log(mode)
    if log_path is None:
        async def _empty() -> AsyncIterator[dict]:
            yield {"event": "error", "data": f"no {mode} log file found"}

        return EventSourceResponse(_empty(), ping=15)
    return EventSourceResponse(_tail(log_path, request), ping=15)


@router.get("/api/brain-build/stream")
async def stream_brain_build(request: Request) -> EventSourceResponse:
    """SSE stream of the newest brain_build JSONL log."""

    return stream_log(request, "brain_build")


__all__ = ["router", "stream_log"]
