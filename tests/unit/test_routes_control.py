"""Unit tests for /api/brain-build/{start,stop,status} (F3.3)."""
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
def test_status_returns_not_running_initially(client):
    """Fresh runtime: status should say running=false, pid=None."""
    r = client.get("/api/brain-build/status")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False
    assert body["pid"] is None
    assert body["started_at"] is None


@pytest.mark.unit
def test_start_refuses_when_already_running(client):
    """If slot.process is alive (returncode=None), POST /start returns 409."""
    fake = MagicMock()
    fake.returncode = None  # still running
    fake.pid = 12345
    runtime = get_runtime()
    runtime.processes["brain_build"].process = fake
    runtime.processes["brain_build"].started_at = "2026-05-26T00:00:00+00:00"

    r = client.post("/api/brain-build/start", data={"resume": "true"})
    assert r.status_code == 409
    body = r.json()
    assert body["started"] is False
    assert "already" in body["reason"].lower()
    assert body["pid"] == 12345


@pytest.mark.unit
def test_stop_returns_false_when_nothing_running(client):
    """POST /stop with no running process returns {stopped: false}."""
    r = client.post("/api/brain-build/stop")
    assert r.status_code == 200
    assert r.json() == {"stopped": False}


# Q3.5: max_papers must be bounded (ge=1, le=100000) so absurd/negative
# values are rejected with 422 before reaching the CLI subprocess.


@pytest.mark.unit
def test_start_rejects_negative_max_papers(client):
    """POST /start with max_papers=-1 → 422 (below ge=1)."""
    r = client.post(
        "/api/brain-build/start", data={"resume": "true", "max_papers": "-1"}
    )
    assert r.status_code == 422


@pytest.mark.unit
def test_start_rejects_absurd_max_papers(client):
    """POST /start with a huge max_papers → 422 (above le=100000)."""
    r = client.post(
        "/api/brain-build/start",
        data={"resume": "true", "max_papers": "999999999"},
    )
    assert r.status_code == 422
