"""Unit tests for /api/weekly/{start,stop,status,controls} (F4.2)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app
from scripts.server.runtime import get_runtime, reset_runtime


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.mark.unit
def test_weekly_status_idle(client):
    """Fresh runtime: status should say running=false."""
    r = client.get("/api/weekly/status")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False
    assert body["pid"] is None


@pytest.mark.unit
def test_weekly_start_refuses_when_already_running(client):
    """If slot.process is alive, POST /start returns 409."""
    fake = MagicMock()
    fake.returncode = None
    fake.pid = 999
    runtime = get_runtime()
    runtime.processes["weekly"].process = fake
    runtime.processes["weekly"].started_at = "2026-05-26T00:00:00+00:00"

    r = client.post("/api/weekly/start", data={"dry_run": "false"})
    assert r.status_code == 409
    body = r.json()
    assert body["started"] is False
    assert body["pid"] == 999


@pytest.mark.unit
def test_weekly_stop_when_idle_returns_false(client):
    """POST /stop with no running process returns {stopped: false}."""
    r = client.post("/api/weekly/stop")
    assert r.status_code == 200
    assert r.json() == {"stopped": False}


@pytest.mark.unit
def test_weekly_controls_renders_start_buttons_when_idle(client):
    """Idle controls fragment exposes both Run-now and Dry-run buttons."""
    r = client.get("/api/weekly/controls")
    assert r.status_code == 200
    assert b"Run now" in r.content
    assert b"Dry-run (preview only)" in r.content
