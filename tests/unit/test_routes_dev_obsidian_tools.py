"""Unit tests for /api/dev Panel 4 endpoints (Task #69).

Panel 4 surfaces four Obsidian tools — relink, re-extract, rerender, synthesize —
on a DOI already in the sandbox state DB. Tests verify:

- the sandbox DOI dropdown endpoint (empty + populated)
- input validation (empty DOI → danger pill)
- happy paths for the four POST endpoints with the tool entry-points mocked
- the multi-checkbox ``phases`` form field is parsed correctly for re-extract

All tool entry points already accept ``state_db`` / ``embeddings_db`` as plain
kwargs (Path A in the task notes), so the route layer just passes the sandbox
instances through.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient against a freshly-created app with /dev mounted."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")
    from lit_monitor.server.app import create_app

    return TestClient(create_app())


@pytest.mark.unit
def test_sandbox_dois_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Empty sandbox state DB → placeholder option, no <option value="DOI">."""
    client = _make_dev_client(monkeypatch)
    # Redirect SANDBOX_STATE_DB_PATH at a non-existent file so the endpoint
    # takes its short-circuit empty path.
    fake_path = tmp_path / "state_dev.db"
    monkeypatch.setattr(
        "lit_monitor.server.dev_sandbox.SANDBOX_STATE_DB_PATH", fake_path
    )
    resp = client.get("/api/dev/sandbox-dois")
    assert resp.status_code == 200
    assert "sandbox empty" in resp.text
    # No DOI options should be present.
    assert 'value="10.' not in resp.text


@pytest.mark.unit
def test_sandbox_dois_lists_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Two rows in the sandbox DB → two <option> tags with DOI + title text."""
    client = _make_dev_client(monkeypatch)

    # Build a real sqlite file so the endpoint's _connect path works end-to-end.
    from lit_monitor.core.state_db import StateDB

    fake_db_path = tmp_path / "state_dev.db"
    db = StateDB(fake_db_path)
    db.upsert_paper(
        {"doi": "10.1/aaa", "title": "First paper", "source_type": "paper"}
    )
    db.upsert_paper(
        {"doi": "10.2/bbb", "title": "Second paper", "source_type": "paper"}
    )

    monkeypatch.setattr(
        "lit_monitor.server.dev_sandbox.SANDBOX_STATE_DB_PATH", fake_db_path
    )
    resp = client.get("/api/dev/sandbox-dois")
    assert resp.status_code == 200
    body = resp.text
    assert 'value="10.1/aaa"' in body
    assert "First paper" in body
    assert 'value="10.2/bbb"' in body
    assert "Second paper" in body


@pytest.mark.unit
def test_relink_endpoint_with_no_doi_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/dev/relink with empty DOI → danger pill, no tool call."""
    client = _make_dev_client(monkeypatch)
    resp = client.post("/api/dev/relink", data={"doi": ""})
    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    assert "DOI required" in resp.text


@pytest.mark.unit
def test_relink_endpoint_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Mock relink_note + sandbox deps → success pill in response."""
    client = _make_dev_client(monkeypatch)

    # Make a real note file so the existence check passes.
    note_file = tmp_path / "Literature" / "_Dev" / "sandbox_10_0_test.md"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    note_file.write_text("# stub\n", encoding="utf-8")

    fake_sdb = MagicMock()
    fake_sdb.get_paper.return_value = {
        "doi": "10.0/test",
        "note_path": str(note_file),
    }
    fake_edb = MagicMock()

    with patch(
        "lit_monitor.server.dev_sandbox.sandbox_state_db", return_value=fake_sdb
    ), patch(
        "lit_monitor.server.dev_sandbox.sandbox_embeddings_db", return_value=fake_edb
    ), patch(
        "lit_monitor.obsidian_tools.relink.relink_note"
    ) as mock_relink, patch(
        "lit_monitor.core.config.get_config", return_value=MagicMock()
    ):
        resp = client.post("/api/dev/relink", data={"doi": "10.0/test"})

    assert resp.status_code == 200
    assert 'class="pill success"' in resp.text
    assert "relinked" in resp.text
    # Confirm the sandbox deps were threaded through.
    mock_relink.assert_called_once()
    call_kwargs = mock_relink.call_args.kwargs
    assert call_kwargs["state_db"] is fake_sdb
    assert call_kwargs["embeddings_db"] is fake_edb
    assert call_kwargs["note_path"] == str(note_file)


@pytest.mark.unit
def test_re_extract_phases_parsed_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """phases=simple&phases=complex → re_extract called with phases=['simple','complex']."""
    client = _make_dev_client(monkeypatch)

    fake_sdb = MagicMock()
    fake_sdb.get_paper.return_value = {
        "doi": "10.0/test",
        "note_path": "/tmp/x.md",
    }

    with patch(
        "lit_monitor.server.dev_sandbox.sandbox_state_db", return_value=fake_sdb
    ), patch(
        "lit_monitor.obsidian_tools.re_extract.re_extract"
    ) as mock_re_extract, patch(
        "lit_monitor.llm.llm_client.get_client", return_value=MagicMock()
    ), patch(
        "lit_monitor.core.config.get_config", return_value=MagicMock()
    ):
        mock_re_extract.return_value = {
            "title": "x",
            "authors": ["a"],
            "year": 2025,
            "_overall_confidence": 0.7,
        }
        # httpx form encoding: pass a list for a repeated key. Starlette's
        # form parser exposes both entries via getlist("phases").
        resp = client.post(
            "/api/dev/re-extract",
            data={"doi": "10.0/test", "phases": ["simple", "complex"]},
        )

    assert resp.status_code == 200
    assert 'class="pill success"' in resp.text
    mock_re_extract.assert_called_once()
    kwargs = mock_re_extract.call_args.kwargs
    assert kwargs["phases"] == ["simple", "complex"]
    assert kwargs["doi"] == "10.0/test"
    assert kwargs["state_db"] is fake_sdb


@pytest.mark.unit
def test_synthesize_with_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/dev/synthesize with topic="UFDF" → synthesize called + success pill."""
    client = _make_dev_client(monkeypatch)

    fake_sdb = MagicMock()
    fake_edb = MagicMock()

    with patch(
        "lit_monitor.server.dev_sandbox.sandbox_state_db", return_value=fake_sdb
    ), patch(
        "lit_monitor.server.dev_sandbox.sandbox_embeddings_db", return_value=fake_edb
    ), patch(
        "lit_monitor.obsidian_tools.synthesize.synthesize",
        return_value="/tmp/vault/Literature/Connections/Synthesis_UFDF.md",
    ) as mock_syn, patch(
        "lit_monitor.llm.llm_client.get_client", return_value=MagicMock()
    ), patch(
        "lit_monitor.core.config.get_config", return_value=MagicMock()
    ):
        resp = client.post("/api/dev/synthesize", data={"topic": "UFDF"})

    assert resp.status_code == 200
    assert 'class="pill success"' in resp.text
    assert "synthesis written" in resp.text
    mock_syn.assert_called_once()
    kwargs = mock_syn.call_args.kwargs
    assert kwargs["topic"] == "UFDF"
    assert kwargs["state_db"] is fake_sdb
    assert kwargs["embeddings_db"] is fake_edb
