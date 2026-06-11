"""Unit tests for /schedule + /api/schedule (F5.2)."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app
from scripts.server.runtime import reset_runtime
from scripts.server.scheduler import ScheduleSpec


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.mark.unit
def test_schedule_page_unsupported_platform(client):
    with patch("scripts.server.routes.schedule.detect_platform", return_value="unsupported"):
        resp = client.get("/schedule")
    assert resp.status_code == 200
    assert b"Scheduling not supported" in resp.content


@pytest.mark.unit
def test_schedule_page_shows_current(client):
    spec = ScheduleSpec.parse("wed", "09:15")
    with patch("scripts.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("scripts.server.routes.schedule.read_schedule", return_value=spec):
        resp = client.get("/schedule")
    assert resp.status_code == 200
    assert b"wed" in resp.content
    assert b"09:15" in resp.content


@pytest.mark.unit
def test_schedule_current_renders_as_definition_list(client):
    # AR-7: the 'Current schedule' key/value block is a <dl class="kv"> with
    # <dt>/<dd>, not a <th>-in-<tbody> table. The day/time values must survive.
    spec = ScheduleSpec.parse("wed", "09:15")
    with patch("scripts.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("scripts.server.routes.schedule.read_schedule", return_value=spec):
        resp = client.get("/schedule")
    assert resp.status_code == 200
    body = resp.text
    assert '<dl class="kv">' in body
    assert "<dt>Day of week</dt>" in body
    assert "<dt>Time</dt>" in body
    # Values preserved verbatim (same as test_schedule_page_shows_current).
    assert "wed" in body and "09:15" in body


@pytest.mark.unit
def test_create_schedule_invokes_writer(client):
    with patch("scripts.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("scripts.server.routes.schedule.write_schedule") as mock_write:
        mock_write.return_value = "/tmp/fake.plist"
        resp = client.post("/api/schedule", data={"day_of_week": "mon", "time": "08:00"})
    assert resp.status_code == 200
    assert resp.headers.get("HX-Refresh") == "true"
    mock_write.assert_called_once()
    spec_arg = mock_write.call_args[0][0]
    assert spec_arg.day_of_week == "mon"
    assert spec_arg.time == "08:00"


@pytest.mark.unit
def test_create_schedule_rejects_bad_time(client):
    with patch("scripts.server.routes.schedule.detect_platform", return_value="macos"):
        resp = client.post("/api/schedule", data={"day_of_week": "mon", "time": "25:99"})
    assert resp.status_code == 400


@pytest.mark.unit
def test_create_schedule_rejected_on_unsupported_platform(client):
    with patch("scripts.server.routes.schedule.detect_platform", return_value="unsupported"):
        resp = client.post("/api/schedule", data={"day_of_week": "mon", "time": "08:00"})
    assert resp.status_code == 400


@pytest.mark.unit
def test_delete_schedule_invokes_remover(client):
    with patch("scripts.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("scripts.server.routes.schedule.remove_schedule") as mock_rm:
        resp = client.delete("/api/schedule")
    assert resp.status_code == 200
    mock_rm.assert_called_once()
