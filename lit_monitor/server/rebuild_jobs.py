"""Rebuild-job helpers: argv builders + capability detection + spawn-into-slot.

Reset wipes a regenerable view; rebuild regenerates it from state.db by spawning
the matching CLI command as a tracked subprocess (ProcessSlot). The rebuild CLI
commands print to the rich console (not logs/*.jsonl), so the SSE layer (RR5)
tails the slot's stdout buffer — NOT sse.stream_log.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SLOT_FOR = {
    "vectors": "rebuild_vectors",
    "graph": "rebuild_graph",
    "notes": "rebuild_notes",
}


def slot_name(component: str) -> str:
    """Map a rebuild component to its ProcessSlot key. Raises KeyError if unknown."""
    return _SLOT_FOR[component]


def rebuild_argvs(component: str, *, enrich: bool = False) -> list[list[str]]:
    """The CLI command(s) to run, in order, for a component rebuild."""
    if component == "vectors":
        return [["lit-monitor", "embeddings", "rebuild", "--confirm"]]
    if component == "notes":
        return [["lit-monitor", "obsidian", "rerender"]]
    if component == "graph":
        seq = [["lit-monitor", "graph", "backfill", "--all"]]
        if enrich:
            seq.append(["lit-monitor", "graph", "backfill", "--ner-with-llm"])
            seq.append(["lit-monitor", "graph", "backfill", "--relationships-with-llm"])
        return seq
    raise ValueError(f"unknown rebuild component: {component!r}")


def enrichment_capability() -> dict[str, bool]:
    """Whether the server can run graph LLM enrichment right now.

    nlp: the [nlp] extra (BioBERT NER) is importable. ollama_key: OLLAMA_API_KEY set.
    """
    try:
        nlp_ok = (
            importlib.util.find_spec("lit_monitor.graph.ner") is not None
            and importlib.util.find_spec("transformers") is not None
        )
    except (ImportError, ValueError):
        nlp_ok = False
    return {"nlp": nlp_ok, "ollama_key": bool(os.environ.get("OLLAMA_API_KEY"))}


async def run_rebuild_sequence(slot: Any, argvs: list[list[str]]) -> int:
    """Run each argv in ``argvs`` sequentially inside ``slot``, streaming stdout
    into ``slot.append_line(...)``. Aborts the chain on the first non-zero exit.
    Returns the exit code of the last command run (0 = all succeeded).

    Locking mirrors ``dev_graph_backfill_start``: ``slot.lock`` is held ONLY
    around the spawn (setting ``slot.process``/``slot.started_at``), then released
    before draining stdout — the start handler holds the lock for the spawn and
    the SSE generator drains the pipe outside it. Holding the lock for the whole
    drain would block ``slot.stop()`` (which also acquires ``slot.lock``) for the
    entire run.
    """
    from datetime import UTC, datetime

    last_rc = 0
    for argv in argvs:
        # Spawn under the lock so a concurrent stop()/status sees a consistent
        # (process, started_at) pair, exactly like dev_graph_backfill_start.
        async with slot.lock:
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except Exception:  # noqa: BLE001 — surface any spawn failure as a line
                logger.exception("rebuild spawn failed: %s", argv)
                slot.append_line("✗ rebuild step failed (spawn error)")
                return 1
            slot.process = process
            slot.started_at = datetime.now(UTC).isoformat(timespec="seconds")

        # Drain stdout OUTSIDE the lock — the lock guards spawn/stop atomicity,
        # not the (potentially long) read loop.
        if process.stdout is not None:
            async for raw in process.stdout:
                slot.append_line(raw.decode("utf-8", errors="replace").rstrip())

        rc = await process.wait()
        last_rc = rc
        if rc != 0:
            slot.append_line(f"✗ rebuild step failed (exit {rc})")
            return rc

    return last_rc
