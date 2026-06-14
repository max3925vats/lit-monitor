"""Unit tests for RR3 rebuild-job helpers + spawn-into-slot coroutine.

These tests never spawn a real subprocess: ``asyncio.create_subprocess_exec`` is
monkeypatched with a stub that returns a fake process whose ``.stdout`` yields a
few canned lines and whose ``.returncode``/``.wait()`` are controllable.
"""

from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------------------
# argv builders
# ---------------------------------------------------------------------------
def test_graph_argv_base_is_single_backfill():
    from lit_monitor.server.rebuild_jobs import rebuild_argvs

    assert rebuild_argvs("graph", enrich=False) == [
        ["lit-monitor", "graph", "backfill", "--all"]
    ]


def test_graph_argv_enrich_chains_three():
    from lit_monitor.server.rebuild_jobs import rebuild_argvs

    assert rebuild_argvs("graph", enrich=True) == [
        ["lit-monitor", "graph", "backfill", "--all"],
        ["lit-monitor", "graph", "backfill", "--ner-with-llm"],
        ["lit-monitor", "graph", "backfill", "--relationships-with-llm"],
    ]


def test_vectors_and_notes_argv():
    from lit_monitor.server.rebuild_jobs import rebuild_argvs

    assert rebuild_argvs("vectors") == [
        ["lit-monitor", "embeddings", "rebuild", "--confirm"]
    ]
    assert rebuild_argvs("notes") == [["lit-monitor", "obsidian", "rerender"]]


def test_unknown_component_raises():
    from lit_monitor.server.rebuild_jobs import rebuild_argvs

    with pytest.raises(ValueError):
        rebuild_argvs("bogus")


# ---------------------------------------------------------------------------
# slot_name mapping
# ---------------------------------------------------------------------------
def test_slot_name_maps_each_component():
    from lit_monitor.server.rebuild_jobs import slot_name

    assert slot_name("vectors") == "rebuild_vectors"
    assert slot_name("graph") == "rebuild_graph"
    assert slot_name("notes") == "rebuild_notes"


def test_slot_name_unknown_raises():
    from lit_monitor.server.rebuild_jobs import slot_name

    with pytest.raises(KeyError):
        slot_name("bogus")


# ---------------------------------------------------------------------------
# capability detection
# ---------------------------------------------------------------------------
def test_enrichment_capability_no_key(monkeypatch):
    from lit_monitor.server import rebuild_jobs

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    cap = rebuild_jobs.enrichment_capability()
    assert cap["ollama_key"] is False
    assert set(cap) == {"nlp", "ollama_key"}


def test_enrichment_capability_with_key(monkeypatch):
    from lit_monitor.server import rebuild_jobs

    monkeypatch.setenv("OLLAMA_API_KEY", "sk-test")
    cap = rebuild_jobs.enrichment_capability()
    assert cap["ollama_key"] is True


# ---------------------------------------------------------------------------
# runtime exposes the new slots
# ---------------------------------------------------------------------------
def test_runtime_has_rebuild_slots():
    from lit_monitor.server.runtime import ServerRuntime

    rt = ServerRuntime()
    for key in ("rebuild_vectors", "rebuild_graph", "rebuild_notes"):
        assert key in rt.processes


# ---------------------------------------------------------------------------
# run_rebuild_sequence — fake-subprocess harness
# ---------------------------------------------------------------------------
class _FakeStdout:
    """Async-iterable stdout yielding pre-canned byte lines."""

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeProcess:
    def __init__(self, lines: list[bytes], returncode: int):
        self.stdout = _FakeStdout(lines)
        self.returncode = returncode
        self.pid = 4242

    async def wait(self) -> int:
        return self.returncode


class _FakeSlot:
    """Minimal stand-in for ProcessSlot exposing only what the coroutine uses."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.output: list[str] = []
        self.append_line = self.output.append
        self.process = None
        self.started_at = None


def _patch_exec(monkeypatch, processes: list[_FakeProcess], recorded: list[list[str]]):
    """Patch create_subprocess_exec to hand out ``processes`` in order, recording argvs."""
    it = iter(processes)

    async def _fake_exec(*argv, **kwargs):
        recorded.append(list(argv))
        return next(it)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


def test_run_rebuild_sequence_streams_and_chains(monkeypatch):
    from lit_monitor.server.rebuild_jobs import run_rebuild_sequence

    recorded: list[list[str]] = []
    procs = [
        _FakeProcess([b"line1\n", b"line2\n"], returncode=0),
        _FakeProcess([b"line3\n"], returncode=0),
    ]
    _patch_exec(monkeypatch, procs, recorded)

    slot = _FakeSlot()
    argvs = [
        ["lit-monitor", "graph", "backfill", "--all"],
        ["lit-monitor", "graph", "backfill", "--ner-with-llm"],
    ]
    rc = asyncio.run(run_rebuild_sequence(slot, argvs))

    assert rc == 0
    # Both commands spawned, in order.
    assert recorded == argvs
    # Streamed stdout captured into the slot's buffer.
    assert "line1" in slot.output
    assert "line2" in slot.output
    assert "line3" in slot.output


def test_run_rebuild_sequence_aborts_on_first_failure(monkeypatch):
    from lit_monitor.server.rebuild_jobs import run_rebuild_sequence

    recorded: list[list[str]] = []
    procs = [
        _FakeProcess([b"boom\n"], returncode=3),
        _FakeProcess([b"never\n"], returncode=0),
    ]
    _patch_exec(monkeypatch, procs, recorded)

    slot = _FakeSlot()
    argvs = [
        ["lit-monitor", "graph", "backfill", "--all"],
        ["lit-monitor", "graph", "backfill", "--ner-with-llm"],
    ]
    rc = asyncio.run(run_rebuild_sequence(slot, argvs))

    assert rc == 3
    # Only the first command was spawned — chain aborted.
    assert recorded == [argvs[0]]
    # A clear failure line was appended.
    assert any("3" in ln and ("fail" in ln.lower() or "✗" in ln) for ln in slot.output)
