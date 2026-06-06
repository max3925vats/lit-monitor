"""Unit tests for FG-5 — /dev Panel 8 graph-backfill SSE trigger.

Panel 8 mirrors the Panel-5 dry-run SSE mechanics (single background slot,
spawn via ``asyncio.create_subprocess_exec``, SSE-stream stdout, stop kills it)
but targets ``lit-monitor graph backfill`` instead of ``lit-monitor run``.

Radio + modifier → CLI-flag mapping under test (PF-4; confirmed against
``scripts/cli.py:2604-2647``). The form now sends a single mutually-exclusive
``backfill_kind`` radio (``schema`` | ``ner`` | ``relationships``) plus optional
``with_llm`` and ``all`` modifiers::

    backfill_kind=schema  + all          → --all
    backfill_kind=ner     + with_llm     → --ner-with-llm
    backfill_kind=ner                    → --ner
    backfill_kind=relationships + with_llm → --relationships-with-llm
    backfill_kind=relationships          → --relationships
    + ``all`` is an independent SCOPE flag that prepends ``--all`` to any kind.

Coverage:
  - backfill_kind=ner + with_llm + all → --ner-with-llm --all (no --relationships).
  - backfill_kind=relationships (plain) → --relationships, no --ner / -with-llm.
  - backfill_kind=schema + all → --all only (no --ner / --relationships).
  - start refuses (no spawn) when the slot is already busy.
  - Panel 8 markup is present in /dev (dev mode) and /dev is 404 without the flag.

All spawn-tests stub ``asyncio.create_subprocess_exec`` so nothing real runs.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient against a freshly-created app with dev mode on."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")
    from scripts.server.app import create_app
    from scripts.server.runtime import reset_runtime

    reset_runtime()
    return TestClient(create_app())


def _make_plain_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with dev mode OFF (no /dev router mounted)."""
    monkeypatch.delenv("LIT_MONITOR_DEV", raising=False)
    from scripts.server.app import create_app
    from scripts.server.runtime import reset_runtime

    reset_runtime()
    return TestClient(create_app())


def _spawn_mock_for(pid: int) -> AsyncMock:
    """Build an ``asyncio.create_subprocess_exec`` stand-in returning a dummy proc."""
    fake_proc = MagicMock()
    fake_proc.pid = pid
    fake_proc.returncode = None
    fake_proc.stdout = MagicMock()
    return AsyncMock(return_value=fake_proc)


@pytest.mark.unit
def test_backfill_kind_ner_with_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """backfill_kind=ner + with_llm + all → --ner-with-llm --all (radio excludes relationships)."""
    client = _make_dev_client(monkeypatch)
    spawn_mock = _spawn_mock_for(77777)

    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post(
            "/api/dev/graph/backfill/start",
            data={"backfill_kind": "ner", "with_llm": "on", "all": "on"},
        )

    assert resp.status_code == 200
    assert spawn_mock.await_count == 1
    argv = list(spawn_mock.await_args.args)
    flat = " ".join(argv)
    assert "graph" in flat and "backfill" in flat
    assert "--ner-with-llm" in flat
    # radio: relationships NOT emitted with ner
    assert "--relationships" not in flat
    assert "--all" in flat
    assert argv[:5] == ["uv", "run", "lit-monitor", "graph", "backfill"]


@pytest.mark.unit
def test_backfill_kind_relationships_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """backfill_kind=relationships (no with_llm, no all) → --relationships only."""
    client = _make_dev_client(monkeypatch)
    spawn_mock = _spawn_mock_for(88888)

    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post(
            "/api/dev/graph/backfill/start",
            data={"backfill_kind": "relationships"},
        )

    assert resp.status_code == 200
    argv = list(spawn_mock.await_args.args)
    flat = " ".join(argv)
    assert "--relationships" in flat and "--relationships-with-llm" not in flat
    assert "--ner" not in flat


@pytest.mark.unit
def test_backfill_kind_schema_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """backfill_kind=schema + all → --all only (no kind flags)."""
    client = _make_dev_client(monkeypatch)
    spawn_mock = _spawn_mock_for(99999)

    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post(
            "/api/dev/graph/backfill/start",
            data={"backfill_kind": "schema", "all": "on"},
        )

    assert resp.status_code == 200
    argv = list(spawn_mock.await_args.args)
    flat = " ".join(argv)
    assert "--all" in flat and "--ner" not in flat and "--relationships" not in flat


@pytest.mark.unit
def test_backfill_kind_schema_no_scope_defaults_to_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-only with no --all scope still emits --all (CLI needs a scope to be valid)."""
    client = _make_dev_client(monkeypatch)
    spawn_mock = _spawn_mock_for(11111)

    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post(
            "/api/dev/graph/backfill/start",
            data={"backfill_kind": "schema"},
        )

    assert resp.status_code == 200
    argv = list(spawn_mock.await_args.args)
    assert "--all" in argv
    assert "--ner" not in argv and "--relationships" not in argv


@pytest.mark.unit
def test_dev_graph_backfill_blocks_if_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the graph-backfill slot is busy, start returns an 'already running' pill, no spawn."""
    client = _make_dev_client(monkeypatch)

    from scripts.server.runtime import get_runtime

    runtime = get_runtime()
    fake_proc = MagicMock()
    fake_proc.returncode = None  # still alive
    fake_proc.pid = 5151
    runtime.processes["dev_graph_backfill"].process = fake_proc

    spawn_mock = AsyncMock()
    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post("/api/dev/graph/backfill/start", data={"all": "on"})

    assert resp.status_code == 200
    assert spawn_mock.await_count == 0
    assert 'class="pill danger"' in resp.text
    assert "already running" in resp.text.lower()


@pytest.mark.unit
def test_dev_graph_backfill_stop_noop_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop with nothing running returns a benign pill, never 500s."""
    client = _make_dev_client(monkeypatch)
    resp = client.post("/api/dev/graph/backfill/stop")
    assert resp.status_code == 200
    assert "nothing to stop" in resp.text.lower()


@pytest.mark.unit
def test_dev_graph_panel_present_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    """The /dev page (dev mode) advertises Panel 8 — 'Knowledge Graph backfill'."""
    client = _make_dev_client(monkeypatch)
    resp = client.get("/dev")
    assert resp.status_code == 200
    assert "Knowledge Graph backfill" in resp.text
    # The SSE wiring + flag checkboxes must be present.
    assert "/api/dev/graph/backfill/start" in resp.text
    assert "/api/dev/graph/backfill/stream" in resp.text


@pytest.mark.unit
def test_dev_graph_panel_absent_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without LIT_MONITOR_DEV=1, neither /dev nor the backfill route is reachable."""
    client = _make_plain_client(monkeypatch)
    assert client.get("/dev").status_code == 404
    assert client.post(
        "/api/dev/graph/backfill/start", data={"all": "on"}
    ).status_code == 404
