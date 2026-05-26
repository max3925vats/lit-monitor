"""Unit tests for Panel 7 — sandbox status + clear (Task #72).

Covers:
  - GET  /api/dev/sandbox/status — counts table rendered from mocked sandbox_status().
  - POST /api/dev/sandbox/clear  — confirm gate, success path, partial path.

All tests set LIT_MONITOR_DEV=1 BEFORE importing create_app so the /dev
router is mounted (gated by app.py at import-of-routes time).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _make_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient against a freshly-created app with dev mode on."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")
    from scripts.server.app import create_app

    return TestClient(create_app())


@pytest.mark.unit
def test_sandbox_status_panel_renders_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five count rows render from a mocked sandbox_status() dict."""
    client = _make_dev_client(monkeypatch)
    fake_status = {
        "state_db_rows": 3,
        "papers_collection_count": 2,
        "chunks_collection_count": 17,
        "vault_file_count": 4,
        "last_modified": "2026-05-26T10:00:00",
    }
    # Patch where the route function imports it from (the sandbox module).
    with patch(
        "scripts.server.dev_sandbox.sandbox_status", return_value=fake_status
    ):
        resp = client.get("/api/dev/sandbox/status")

    assert resp.status_code == 200
    body = resp.text
    # Each labelled count row appears with its value.
    assert "state_dev.db rows" in body
    assert "<td>3</td>" in body
    assert "dev-papers embeddings" in body
    assert "<td>2</td>" in body
    assert "dev-chunks embeddings" in body
    assert "<td>17</td>" in body
    assert "Literature/_Dev/*.md files" in body
    assert "<td>4</td>" in body
    assert "state_dev.db last modified" in body
    assert "2026-05-26T10:00:00" in body


@pytest.mark.unit
def test_sandbox_clear_without_confirm_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST without confirm=yes yields the warning pill + safety message."""
    client = _make_dev_client(monkeypatch)
    # Confirm field present but empty/wrong — must NOT call clear_sandbox.
    with patch("scripts.server.dev_sandbox.clear_sandbox") as mock_clear:
        resp = client.post("/api/dev/sandbox/clear", data={"confirm": ""})

    assert resp.status_code == 200
    body = resp.text
    assert "pill warning" in body
    assert "Type" in body and "yes" in body
    mock_clear.assert_not_called()


@pytest.mark.unit
def test_sandbox_clear_with_confirm_calls_clear_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """confirm=yes invokes clear_sandbox and renders success pill + action list."""
    client = _make_dev_client(monkeypatch)
    fake_result = {
        "status": "cleared",
        "actions": "unlinked state_dev.db;deleted dev-papers",
    }
    with patch(
        "scripts.server.dev_sandbox.clear_sandbox", return_value=fake_result
    ) as mock_clear:
        resp = client.post("/api/dev/sandbox/clear", data={"confirm": "yes"})

    assert resp.status_code == 200
    body = resp.text
    mock_clear.assert_called_once_with(confirm=True)
    assert "pill success" in body
    assert "✓ cleared" in body
    # Each action from the semicolon-joined string becomes its own <li>.
    assert "<li>unlinked state_dev.db</li>" in body
    assert "<li>deleted dev-papers</li>" in body


@pytest.mark.unit
def test_sandbox_clear_partial_renders_warning_pill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial-wipe status renders a warning pill (not success, not danger)."""
    client = _make_dev_client(monkeypatch)
    fake_result = {
        "status": "partial",
        "actions": "FAILED chroma cleanup: X",
    }
    with patch(
        "scripts.server.dev_sandbox.clear_sandbox", return_value=fake_result
    ):
        resp = client.post("/api/dev/sandbox/clear", data={"confirm": "yes"})

    assert resp.status_code == 200
    body = resp.text
    assert "pill warning" in body
    assert "⚠ partial wipe" in body
    assert "FAILED chroma cleanup: X" in body
