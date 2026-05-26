from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app
from scripts.server.runtime import reset_runtime


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.mark.unit
def test_dashboard_renders_db_unavailable_when_runtime_fails(client):
    with patch("scripts.server.routes.weekly.get_runtime", side_effect=RuntimeError):
        resp = client.get("/weekly-lit-run")
    assert resp.status_code == 200
    assert b"State DB unavailable" in resp.content


@pytest.mark.unit
def test_dashboard_shows_no_runs_message_when_db_empty(client):
    fake_db = MagicMock()
    fake_db.get_recent_runs_by_type.return_value = []
    fake_runtime = MagicMock()
    fake_runtime.state_db = fake_db
    fake_runtime.config.obsidian.vault_path = ""
    with patch("scripts.server.routes.weekly.get_runtime", return_value=fake_runtime):
        resp = client.get("/weekly-lit-run")
    assert resp.status_code == 200
    assert b"No weekly runs in the log yet." in resp.content


@pytest.mark.unit
def test_dashboard_filters_by_run_type_weekly(client):
    fake_db = MagicMock()
    fake_db.get_recent_runs_by_type.return_value = [
        {"run_id": "abc", "started_at": "2026-05-26", "status": "complete",
         "papers_processed": 3, "papers_skipped": 0, "papers_failed": 0,
         "finished_at": "2026-05-26 12:00:00"},
    ]
    fake_runtime = MagicMock()
    fake_runtime.state_db = fake_db
    fake_runtime.config.obsidian.vault_path = ""
    with patch("scripts.server.routes.weekly.get_runtime", return_value=fake_runtime):
        resp = client.get("/weekly-lit-run")
    assert resp.status_code == 200
    fake_db.get_recent_runs_by_type.assert_called_once_with("weekly", limit=10)
    assert b"abc" in resp.content
