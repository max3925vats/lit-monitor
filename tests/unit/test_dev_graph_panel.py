"""Unit tests for FG-5 — /dev Panel 8 graph-backfill SSE trigger.

Panel 8 mirrors the Panel-5 dry-run SSE mechanics (single background slot,
spawn via ``asyncio.create_subprocess_exec``, SSE-stream stdout, stop kills it)
but targets ``lit-monitor graph backfill`` instead of ``lit-monitor run``.

Checkbox → CLI-flag mapping under test (confirmed against
``scripts/cli.py:2604-2647``)::

    all                       → --all
    ner + with_llm            → --ner-with-llm
    ner (alone)               → --ner
    relationships + with_llm  → --relationships-with-llm
    relationships (alone)     → --relationships

Coverage:
  - start spawns the subprocess with the expected argv (combined --*-with-llm).
  - start with plain flags (no with_llm) maps to --ner / --relationships.
  - start defaults to --all when no scope/flags are checked.
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


@pytest.mark.unit
def test_dev_graph_backfill_start_spawns_expected_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """all + ner + relationships + with_llm → --all --ner-with-llm --relationships-with-llm."""
    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.pid = 77777
    fake_proc.returncode = None
    fake_proc.stdout = MagicMock()
    spawn_mock = AsyncMock(return_value=fake_proc)

    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post(
            "/api/dev/graph/backfill/start",
            data={"all": "on", "ner": "on", "relationships": "on", "with_llm": "on"},
        )

    assert resp.status_code == 200
    assert spawn_mock.await_count == 1
    argv = list(spawn_mock.await_args.args)
    flat = " ".join(argv)
    assert "graph" in flat and "backfill" in flat
    assert "--all" in flat
    assert "--ner-with-llm" in flat
    assert "--relationships-with-llm" in flat
    # combined flags only — the plain forms must NOT also appear as bare argv items
    assert "--ner" not in argv
    assert "--relationships" not in argv
    assert argv[:5] == ["uv", "run", "lit-monitor", "graph", "backfill"]


@pytest.mark.unit
def test_dev_graph_backfill_plain_flags_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ner + relationships (no with_llm) → --ner / --relationships, not the combined forms."""
    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.pid = 88888
    fake_proc.returncode = None
    fake_proc.stdout = MagicMock()
    spawn_mock = AsyncMock(return_value=fake_proc)

    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post(
            "/api/dev/graph/backfill/start",
            data={"ner": "on", "relationships": "on"},
        )

    assert resp.status_code == 200
    argv = list(spawn_mock.await_args.args)
    assert "--ner" in argv and "--ner-with-llm" not in argv
    assert "--relationships" in argv and "--relationships-with-llm" not in argv
    # No scope checkbox checked, but plain flags were → --all must NOT be forced.
    assert "--all" not in argv


@pytest.mark.unit
def test_dev_graph_backfill_defaults_to_all_when_nothing_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No checkboxes at all → default to --all (dev convenience, valid CLI invocation)."""
    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.pid = 99999
    fake_proc.returncode = None
    fake_proc.stdout = MagicMock()
    spawn_mock = AsyncMock(return_value=fake_proc)

    with patch(
        "scripts.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post("/api/dev/graph/backfill/start", data={})

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
