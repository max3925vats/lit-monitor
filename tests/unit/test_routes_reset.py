"""Unit tests for RR4: reset routes (status + per-component reset POSTs).

TDD: tests are written first. Stash-and-rerun verification is documented in
the bundle definition-of-done section.

Config-isolated by conftest.py autouse fixtures (config absent by default).
Routes that need config/runtime inject fake runtimes via patch.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app
from lit_monitor.server.runtime import reset_runtime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_runtime():
    """Reset the runtime singleton before and after each test."""
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# 1. First-run-safe status
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_status_first_run_safe(client):
    """GET /api/reset/status returns 200 with configured=False when config absent."""
    resp = client.get("/api/reset/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is False
    # All component metrics should be empty/zeroed
    for comp in ("vectors", "graph", "notes"):
        assert comp in data["components"]
        assert data["components"][comp]["present"] is False
        assert data["components"][comp]["files"] == 0
        assert data["components"][comp]["size_bytes"] == 0
    # enrichment and busy must always be present
    assert "enrichment" in data
    assert "busy" in data


# ---------------------------------------------------------------------------
# 2. Page endpoint first-run safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_page_first_run_safe(client):
    """GET /setup/reset returns 200 even when no config is present."""
    resp = client.get("/setup/reset")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. "everything" reset rejects bad confirmation phrase
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_everything_reset_rejects_bad_phrase(client):
    """POST /api/reset/everything with wrong phrase → 400; perform_state_reset not called."""
    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        resp = client.post("/api/reset/everything", data={"confirm": "wrong phrase"})
    assert resp.status_code == 400
    mock_reset.assert_not_called()


@pytest.mark.unit
def test_everything_reset_rejects_missing_phrase(client):
    """POST /api/reset/everything with no confirm field → 400; nothing deleted."""
    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        resp = client.post("/api/reset/everything", data={})
    assert resp.status_code == 400
    mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Busy-guard blocks reset when pipeline running
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_blocked_when_pipeline_running(client):
    """POST /api/reset/vectors → 409 when a pipeline slot is running."""
    # Build a fake runtime where brain_build is running
    fake_runtime = MagicMock()
    brain_slot = MagicMock()
    brain_slot.is_running.return_value = True
    # Other slots idle
    idle_slot = MagicMock()
    idle_slot.is_running.return_value = False
    idle_slot.sequence_active = False
    fake_runtime.processes = {
        "brain_build": brain_slot,
        "discovery": idle_slot,
        "vocabulary": idle_slot,
        "rebuild_vectors": idle_slot,
        "rebuild_graph": idle_slot,
        "rebuild_notes": idle_slot,
    }

    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
            resp = client.post("/api/reset/vectors")

    assert resp.status_code == 409
    data = resp.json()
    assert "busy" in data
    mock_reset.assert_not_called()


@pytest.mark.unit
def test_reset_blocked_when_rebuild_running(client):
    """POST /api/reset/graph → 409 when a rebuild slot has sequence_active."""
    fake_runtime = MagicMock()
    idle_proc = MagicMock()
    idle_proc.is_running.return_value = False
    busy_rebuild = MagicMock()
    busy_rebuild.is_running.return_value = False
    busy_rebuild.sequence_active = True  # between commands in a sequence

    fake_runtime.processes = {
        "brain_build": idle_proc,
        "discovery": idle_proc,
        "vocabulary": idle_proc,
        "rebuild_vectors": idle_proc,
        "rebuild_graph": busy_rebuild,
        "rebuild_notes": idle_proc,
    }

    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
            resp = client.post("/api/reset/graph")

    assert resp.status_code == 409
    mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# 5. vectors reset happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vectors_reset_happy_path(tmp_path):
    """POST /api/reset/vectors deletes chroma dir, calls reset_embeddings_indexed,
    and calls reset_runtime.
    """
    # Create a fake chroma directory to simulate an existing target
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "dummy.bin").write_bytes(b"data")

    # Build a fake config pointing at this temp chroma dir
    fake_config = MagicMock()
    state_db_path = tmp_path / "state.db"
    fake_config.state_db.path = str(state_db_path)
    # obsidian — needed for vault_targets not to blow up but not used in vectors reset
    fake_config.obsidian.vault_path = str(tmp_path / "vault")

    # Build a fake state_db with reset_embeddings_indexed spy
    fake_state_db = MagicMock()

    # Build a fake runtime with all slots idle
    idle_slot = MagicMock()
    idle_slot.is_running.return_value = False
    idle_slot.sequence_active = False

    fake_runtime = MagicMock()
    fake_runtime.config = fake_config
    fake_runtime.state_db = fake_state_db
    fake_runtime.processes = {
        "brain_build": idle_slot,
        "discovery": idle_slot,
        "vocabulary": idle_slot,
        "rebuild_vectors": idle_slot,
        "rebuild_graph": idle_slot,
        "rebuild_notes": idle_slot,
    }

    client = TestClient(create_app())

    with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
        with patch("lit_monitor.server.routes.reset.reset_runtime") as mock_reset_runtime:
            resp = client.post("/api/reset/vectors")

    assert resp.status_code == 200, resp.text

    # Chroma dir should be gone
    assert not chroma_dir.exists()

    # reset_embeddings_indexed should have been called
    fake_state_db.reset_embeddings_indexed.assert_called_once()

    # reset_runtime must be called to reconnect fresh DBs
    mock_reset_runtime.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Unknown component → 404
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_unknown_component_returns_404(client):
    """POST /api/reset/bogus → 404."""
    resp = client.post("/api/reset/bogus")
    assert resp.status_code == 404
