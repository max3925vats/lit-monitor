"""Unit tests for the /dev page panels (Task #67).

Covers the new panel scaffold and the Panel-1/Panel-2 endpoints:
  - GET  /dev               — page renders with 7 panel placeholders.
  - POST /api/dev/ruff      — clean + dirty exit paths render pill + output.
  - POST /api/dev/diagnose  — table rendered from mocked run_diagnose().
  - POST /api/dev/check     — table rendered from mocked run_health_check().
  - GET  /api/dev/status    — three count rows from a mocked state_db.

All tests set LIT_MONITOR_DEV=1 BEFORE importing create_app so the /dev
router is mounted (gated by app.py at import-of-routes time).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient against a freshly-created app with dev mode on."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")
    from lit_monitor.server.app import create_app

    return TestClient(create_app())


@pytest.mark.unit
def test_dev_landing_renders_panel_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /dev returns 200 and the body advertises all seven panels."""
    client = _make_dev_client(monkeypatch)
    resp = client.get("/dev")

    assert resp.status_code == 200
    body = resp.text
    # Loose substring checks — wording can evolve so long as the seven
    # panel headings are still present.
    for label in ("Panel 1", "Panel 2", "Panel 3", "Panel 4",
                  "Panel 5", "Panel 6", "Panel 7"):
        assert label in body, f"missing {label} from /dev body"


@pytest.mark.unit
def test_dev_ruff_endpoint_returns_pill(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean ruff run yields a success pill in the rendered fragment."""
    client = _make_dev_client(monkeypatch)

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "All checks passed!\n"
    fake_result.stderr = ""

    with patch("scripts.server.routes.dev.subprocess.run", return_value=fake_result):
        resp = client.post("/api/dev/ruff")

    assert resp.status_code == 200
    assert 'class="pill success"' in resp.text
    assert "All checks passed!" in resp.text


@pytest.mark.unit
def test_dev_ruff_endpoint_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero ruff exit yields a danger pill and the error text appears (escaped)."""
    client = _make_dev_client(monkeypatch)

    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = "scripts/foo.py:12:5: E501 line too long\n"
    fake_result.stderr = ""

    with patch("scripts.server.routes.dev.subprocess.run", return_value=fake_result):
        resp = client.post("/api/dev/ruff")

    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    # The ":" in the lint output should appear inside the <pre> tag.
    assert "E501 line too long" in resp.text
    assert "<pre" in resp.text


@pytest.mark.unit
def test_dev_diagnose_endpoint_returns_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/dev/diagnose renders the per-check table from run_diagnose()."""
    client = _make_dev_client(monkeypatch)

    fake_results = {
        "paths.yaml": (True, "/etc/lit/paths.yaml"),
        "extraction.yaml": (False, "not found"),
    }
    with patch("scripts.setup.diagnose.run_diagnose", return_value=fake_results):
        resp = client.post("/api/dev/diagnose")

    assert resp.status_code == 200
    assert '<table class="dev-checks-table">' in resp.text
    assert "paths.yaml" in resp.text
    assert "extraction.yaml" in resp.text


@pytest.mark.unit
def test_dev_check_endpoint_returns_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/dev/check renders the per-check table from run_health_check()."""
    client = _make_dev_client(monkeypatch)

    # Nested shape: {section: {check_name: (ok, msg)}} — the renderer must
    # handle both flat and nested results.
    fake_results = {
        "ollama": {"reachable": (True, "200 OK")},
        "zotero": {"library": (False, "401 unauthorized")},
    }
    with patch("scripts.setup.health_check.run_health_check", return_value=fake_results):
        resp = client.post("/api/dev/check")

    assert resp.status_code == 200
    assert '<table class="dev-checks-table">' in resp.text
    assert "ollama.reachable" in resp.text
    assert "zotero.library" in resp.text


@pytest.mark.unit
def test_dev_status_endpoint_returns_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/dev/status returns the three count rows from the state DB."""
    client = _make_dev_client(monkeypatch)

    # Build a fake conn whose execute().fetchone()[0] returns canned counts
    # in the order the route queries them: papers, run_log, embedded_papers.
    fake_conn = MagicMock()
    canned = iter([(42,), (7,), (5,)])

    def _fake_execute(_sql: str) -> MagicMock:
        cur = MagicMock()
        cur.fetchone.return_value = next(canned)
        return cur

    fake_conn.execute.side_effect = _fake_execute

    fake_db = MagicMock()
    # _connect() is a contextmanager — make __enter__ return our conn.
    fake_db._connect.return_value.__enter__.return_value = fake_conn
    fake_db._connect.return_value.__exit__.return_value = False

    fake_runtime = MagicMock()
    fake_runtime.state_db = fake_db

    with patch("scripts.server.runtime.get_runtime", return_value=fake_runtime):
        resp = client.get("/api/dev/status")

    assert resp.status_code == 200
    body = resp.text
    assert "papers" in body
    assert "embedded papers" in body
    assert "recorded runs" in body
    assert "42" in body
    assert "7" in body
    assert "5" in body
