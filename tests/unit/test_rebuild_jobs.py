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
        self.sequence_active: bool = False

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None


def _patch_exec(
    monkeypatch,
    processes: list[_FakeProcess],
    recorded: list[list[str]],
    recorded_kwargs: list[dict] | None = None,
):
    """Patch create_subprocess_exec to hand out ``processes`` in order.

    Appends the positional argv to ``recorded`` and, when ``recorded_kwargs``
    is provided, appends the kwargs dict there too.
    """
    it = iter(processes)

    async def _fake_exec(*argv, **kwargs):
        recorded.append(list(argv))
        if recorded_kwargs is not None:
            recorded_kwargs.append(dict(kwargs))
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
    # Both commands spawned with uv run prefix, in order.
    assert recorded == [["uv", "run", *a] for a in argvs]
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
    assert recorded == [["uv", "run", *argvs[0]]]
    # A clear failure line was appended.
    assert any("3" in ln and ("fail" in ln.lower() or "✗" in ln) for ln in slot.output)


# ---------------------------------------------------------------------------
# New tests for RR3 reviewer fixes
# ---------------------------------------------------------------------------

def test_repo_root_is_repo():
    """_REPO_ROOT must point at the project root (confirmed by pyproject.toml)."""
    from lit_monitor.server.rebuild_jobs import _REPO_ROOT

    assert (_REPO_ROOT / "pyproject.toml").exists(), (
        f"_REPO_ROOT={_REPO_ROOT!r} does not contain pyproject.toml"
    )


def test_run_rebuild_sequence_uses_uv_run_and_cwd(monkeypatch):
    """Every subprocess spawn must start with ('uv', 'run', 'lit-monitor', ...)
    and pass cwd=str(_REPO_ROOT)."""
    from lit_monitor.server.rebuild_jobs import _REPO_ROOT, run_rebuild_sequence

    recorded: list[list[str]] = []
    recorded_kwargs: list[dict] = []
    procs = [_FakeProcess([b"ok\n"], returncode=0)]
    _patch_exec(monkeypatch, procs, recorded, recorded_kwargs)

    slot = _FakeSlot()
    argvs = [["lit-monitor", "graph", "backfill", "--all"]]
    asyncio.run(run_rebuild_sequence(slot, argvs))

    assert len(recorded) == 1
    # The spawn argv must start with "uv run lit-monitor".
    assert recorded[0][:3] == ["uv", "run", "lit-monitor"], (
        f"Expected ['uv', 'run', 'lit-monitor', ...], got {recorded[0]!r}"
    )
    # The cwd must be the repo root.
    assert recorded_kwargs[0]["cwd"] == str(_REPO_ROOT), (
        f"Expected cwd={str(_REPO_ROOT)!r}, got {recorded_kwargs[0].get('cwd')!r}"
    )


def test_is_rebuild_busy_true_during_sequence(monkeypatch):
    """sequence_active is True while the sequence runs; False + process=None after."""
    from lit_monitor.server.rebuild_jobs import run_rebuild_sequence

    slot = _FakeSlot()
    observed_active_during: list[bool] = []

    async def _fake_exec(*argv, **kwargs):
        # At spawn time the flag must already be set.
        observed_active_during.append(slot.sequence_active)
        return _FakeProcess([b"line\n"], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    asyncio.run(run_rebuild_sequence(slot, [["lit-monitor", "embeddings", "rebuild", "--confirm"]]))

    # Flag was True when the subprocess was spawned.
    assert observed_active_during == [True]
    # After the sequence, both flag and stale process reference are cleared.
    assert slot.sequence_active is False
    assert slot.process is None


def test_run_rebuild_sequence_spawn_error_returns_1(monkeypatch):
    """A FileNotFoundError from create_subprocess_exec must: return non-zero,
    append a failure line, and always clear sequence_active and process."""
    from lit_monitor.server.rebuild_jobs import run_rebuild_sequence

    async def _boom(*argv, **kwargs):
        raise FileNotFoundError("uv not found")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)

    slot = _FakeSlot()
    rc = asyncio.run(
        run_rebuild_sequence(slot, [["lit-monitor", "embeddings", "rebuild", "--confirm"]])
    )

    assert rc != 0, "Expected non-zero exit on spawn failure"
    assert any("fail" in ln.lower() or "✗" in ln for ln in slot.output), (
        f"Expected a failure line in output, got {slot.output!r}"
    )
    # finally block must fire even on spawn error.
    assert slot.sequence_active is False
    assert slot.process is None
