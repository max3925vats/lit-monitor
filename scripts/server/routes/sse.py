"""Server-Sent Events streams for live pipeline progress.

GET /api/brain-build/stream — tails the newest brain_build JSONL log
under ``<repo>/logs/``, forwards each new line as a ``progress`` event.
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

# Repo-root logs directory — mirrors ``scripts.cli._LOG_DIR``.
# scripts/server/routes/sse.py  -> parents[3] is the repo root.
_LOGS_DIR = Path(__file__).resolve().parents[3] / "logs"

# Poll interval between readline attempts when at EOF.
_POLL_INTERVAL_S = 0.5


def _newest_log(mode: str) -> Path | None:
    """Return the most-recently-modified ``*_{mode}.jsonl`` under logs/.

    Returns ``None`` when the logs directory or any matching file is
    absent.
    """

    if not _LOGS_DIR.exists():
        return None
    candidates = sorted(
        _LOGS_DIR.glob(f"*_{mode}.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_line(fh: IO[str]) -> str:
    """Blocking readline — run via ``asyncio.to_thread`` from the loop."""

    return fh.readline()


async def _tail(log_path: Path, request: Request) -> AsyncIterator[dict]:
    """Async generator yielding sse-starlette event dicts.

    Opens ``log_path`` and seeks to EOF so only NEW lines are streamed —
    the historical content is not replayed. Polls for new data every
    ``_POLL_INTERVAL_S`` seconds. Exits cleanly when the client
    disconnects.
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
                    yield {"event": "progress", "data": stripped}
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
