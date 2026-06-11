"""Unit tests for the Panel-5 /api/dev/dryrun/{start,stream,result} endpoints (Task #70).

Coverage:
  - start spawns the subprocess via asyncio.create_subprocess_exec with the
    expected argv.
  - start refuses (no spawn) when the dev_dryrun slot is already busy.
  - result detects an unchanged state.db mtime → green "read-only" pill.
  - result detects a CHANGED state.db mtime → red "leak detected" pill.

All tests enable LIT_MONITOR_DEV=1 before importing create_app so the /dev
router is mounted.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient against a freshly-created app with dev mode on."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")
    from lit_monitor.server.app import create_app
    from lit_monitor.server.runtime import reset_runtime

    reset_runtime()
    return TestClient(create_app())


@pytest.mark.unit
def test_dryrun_start_spawns_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """POST /api/dev/dryrun/start spawns `uv run lit-monitor run --dry-run`."""
    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.pid = 99999
    fake_proc.returncode = None
    spawn_mock = AsyncMock(return_value=fake_proc)

    # Stub the signature snapshot so the route doesn't depend on a real config.
    with (
        patch(
            "lit_monitor.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
        ),
        patch(
            "lit_monitor.server.routes.dev._snapshot_state_db_signature",
            return_value={"state.db": (12345, 4096)},
        ),
        patch("lit_monitor.server.routes.dev._newest_digest_path", return_value=None),
    ):
        resp = client.post("/api/dev/dryrun/start")

    assert resp.status_code == 200
    assert spawn_mock.await_count == 1
    args = spawn_mock.await_args.args
    assert list(args) == ["uv", "run", "lit-monitor", "run", "--dry-run"]
    assert 'class="pill success"' in resp.text
    assert "dry-run started" in resp.text


@pytest.mark.unit
def test_dryrun_start_blocks_if_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the dev_dryrun slot is already busy, start returns a 'already running' pill."""
    client = _make_dev_client(monkeypatch)

    from lit_monitor.server.runtime import get_runtime

    runtime = get_runtime()
    fake_proc = MagicMock()
    fake_proc.returncode = None  # still alive
    fake_proc.pid = 4242
    runtime.processes["dev_dryrun"].process = fake_proc

    spawn_mock = AsyncMock()
    with patch(
        "lit_monitor.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post("/api/dev/dryrun/start")

    assert resp.status_code == 200
    assert spawn_mock.await_count == 0  # NO new spawn
    assert 'class="pill danger"' in resp.text
    assert "already running" in resp.text.lower()


@pytest.mark.unit
def test_dryrun_result_detects_no_state_db_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchanged signature → green 'Read-only' pill."""
    client = _make_dev_client(monkeypatch)

    from lit_monitor.server.runtime import get_runtime

    runtime = get_runtime()
    slot = runtime.processes["dev_dryrun"]
    sig = {
        "state.db": (1717000000_000000000, 4096),
        "state.db-wal": (-1, -1),
        "state.db-shm": (-1, -1),
    }
    slot.metadata["state_db_signature_before"] = sig
    slot.started_at = "2026-05-26T00:00:00+00:00"

    with (
        patch(
            "lit_monitor.server.routes.dev._snapshot_state_db_signature",
            return_value=dict(sig),
        ),
        patch("lit_monitor.server.routes.dev._newest_digest_path", return_value=None),
    ):
        resp = client.get("/api/dev/dryrun/result")

    assert resp.status_code == 200
    assert 'class="pill success"' in resp.text
    assert "Read-only" in resp.text


@pytest.mark.unit
def test_dryrun_result_detects_state_db_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed signature → red 'LEAK detected' pill."""
    client = _make_dev_client(monkeypatch)

    from lit_monitor.server.runtime import get_runtime

    runtime = get_runtime()
    slot = runtime.processes["dev_dryrun"]
    before = {
        "state.db": (1717000000_000000000, 4096),
        "state.db-wal": (-1, -1),
        "state.db-shm": (-1, -1),
    }
    after = dict(before)
    after["state.db"] = (1717000050_000000000, 8192)  # mtime + size both moved
    slot.metadata["state_db_signature_before"] = before
    slot.started_at = "2026-05-26T00:00:00+00:00"

    with (
        patch(
            "lit_monitor.server.routes.dev._snapshot_state_db_signature",
            return_value=after,
        ),
        patch("lit_monitor.server.routes.dev._newest_digest_path", return_value=None),
    ):
        resp = client.get("/api/dev/dryrun/result")

    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    assert "LEAK detected" in resp.text


@pytest.mark.unit
def test_dryrun_refuses_when_discovery_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start refuses (no spawn) when prod discovery slot is busy."""
    client = _make_dev_client(monkeypatch)

    from lit_monitor.server.runtime import get_runtime

    runtime = get_runtime()
    # Mark the prod discovery slot as running.
    fake_proc = MagicMock()
    fake_proc.returncode = None
    fake_proc.pid = 1234
    runtime.processes["discovery"].process = fake_proc

    spawn_mock = AsyncMock()
    with patch(
        "lit_monitor.server.routes.dev.asyncio.create_subprocess_exec", spawn_mock
    ):
        resp = client.post("/api/dev/dryrun/start")

    assert resp.status_code == 200
    assert spawn_mock.await_count == 0
    assert 'class="pill danger"' in resp.text
    assert "discovery run is in progress" in resp.text


@pytest.mark.unit
def test_dryrun_result_detects_wal_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A WAL sibling change between before/after surfaces as a LEAK pill."""
    client = _make_dev_client(monkeypatch)

    from lit_monitor.server.runtime import get_runtime

    runtime = get_runtime()
    slot = runtime.processes["dev_dryrun"]
    before = {
        "state.db": (1717000000_000000000, 4096),
        "state.db-wal": (-1, -1),  # WAL didn't exist
        "state.db-shm": (-1, -1),
    }
    after = dict(before)
    after["state.db-wal"] = (1717000050_000000000, 32768)  # WAL appeared
    slot.metadata["state_db_signature_before"] = before
    slot.started_at = "2026-05-26T00:00:00+00:00"

    with (
        patch(
            "lit_monitor.server.routes.dev._snapshot_state_db_signature",
            return_value=after,
        ),
        patch("lit_monitor.server.routes.dev._newest_digest_path", return_value=None),
    ):
        resp = client.get("/api/dev/dryrun/result")

    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    assert "LEAK detected" in resp.text
    assert "state.db-wal" in resp.text


@pytest.mark.unit
def test_dryrun_status_idle_when_no_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no prior spawn, /api/dev/dryrun/status returns 200 + the idle pill."""
    client = _make_dev_client(monkeypatch)
    resp = client.get("/api/dev/dryrun/status")
    assert resp.status_code == 200
    assert "idle" in resp.text
    assert 'class="pill"' in resp.text


@pytest.mark.unit
def test_dryrun_status_running_parses_iso_started_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: slot.started_at is an ISO string per runtime.py:52 convention.

    The status endpoint previously assumed it was a float (from time.time())
    and crashed with `TypeError: unsupported operand type(s) for -: 'float' and 'str'`
    every 2 seconds during a running dry-run. This test wires a fake running
    process with a valid ISO timestamp and asserts the endpoint returns 200
    with the elapsed time rendered.
    """
    from datetime import UTC, datetime, timedelta

    from lit_monitor.server.runtime import get_runtime

    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.returncode = None  # still running
    runtime = get_runtime()
    slot = runtime.processes["dev_dryrun"]
    slot.process = fake_proc
    # ISO-formatted ~7 seconds ago — exactly the shape ProcessSlot stores.
    started = (datetime.now(UTC) - timedelta(seconds=7)).isoformat(timespec="seconds")
    slot.started_at = started

    resp = client.get("/api/dev/dryrun/status")
    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}. Body: {resp.text[:300]!r}"
    )
    assert "running" in resp.text
    # Elapsed should be ~7s; allow 5-10s window for slow CI.
    elapsed_match = "5s" in resp.text or "6s" in resp.text or "7s" in resp.text \
                    or "8s" in resp.text or "9s" in resp.text or "10s" in resp.text
    assert elapsed_match, f"Expected elapsed ~7s in: {resp.text}"


@pytest.mark.unit
def test_dryrun_status_running_tolerates_malformed_started_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If slot.started_at is somehow malformed, the endpoint must still return
    200 (degrading to "no elapsed shown") rather than 500'ing.
    """
    from lit_monitor.server.runtime import get_runtime

    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.returncode = None
    runtime = get_runtime()
    slot = runtime.processes["dev_dryrun"]
    slot.process = fake_proc
    slot.started_at = "not-an-iso-timestamp"

    resp = client.get("/api/dev/dryrun/status")
    assert resp.status_code == 200
    assert "running" in resp.text


@pytest.mark.unit
def test_dryrun_status_completed_when_exited_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that exited with returncode=0 renders the green completed pill."""
    from lit_monitor.server.runtime import get_runtime

    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.returncode = 0  # exited cleanly
    runtime = get_runtime()
    slot = runtime.processes["dev_dryrun"]
    slot.process = fake_proc

    resp = client.get("/api/dev/dryrun/status")
    assert resp.status_code == 200
    assert "completed" in resp.text
    assert 'class="pill success"' in resp.text


@pytest.mark.unit
def test_dryrun_status_failed_when_exited_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that exited with non-zero returncode renders the red failed pill."""
    from lit_monitor.server.runtime import get_runtime

    client = _make_dev_client(monkeypatch)

    fake_proc = MagicMock()
    fake_proc.returncode = -15  # SIGTERM
    runtime = get_runtime()
    slot = runtime.processes["dev_dryrun"]
    slot.process = fake_proc

    resp = client.get("/api/dev/dryrun/status")
    assert resp.status_code == 200
    assert "-15" in resp.text
    assert 'class="pill danger"' in resp.text
