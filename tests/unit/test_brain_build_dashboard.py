"""Unit tests for the brain-build dashboard (F3.1).

Covers:
1. StateDB.get_all_brain_build_progress joins papers metadata via DOI.
2. GET /brain-build renders the "DB unavailable" warning when the runtime
   cannot produce a state_db.
3. GET /brain-build aggregates progress (done/total/pct) correctly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server import runtime as runtime_mod
from lit_monitor.server.app import create_app


# ---------------------------------------------------------------------------
# 1) StateDB-level test: the join populates paper_title
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_get_all_brain_build_progress_joins_papers(tmp_path) -> None:
    """A papers row matching the progress row's DOI surfaces as paper_title."""
    from lit_monitor.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    doi = "10.1234/example.001"
    db.upsert_paper(
        {
            "doi": doi,
            "title": "Joined Title",
            "authors": ["Alice", "Bob"],
            "year": 2024,
            "source_type": "paper",
            "status": "extraction_complete",
        }
    )
    db.upsert_brain_build_progress(zotero_key="ZK_001", doi=doi)

    rows = db.get_all_brain_build_progress(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["zotero_key"] == "ZK_001"
    assert row["doi"] == doi
    assert row["paper_title"] == "Joined Title"
    assert row["paper_year"] == 2024
    # authors round-trips as JSON; the route layer decodes it for display.
    assert "Alice" in (row["paper_authors"] or "")


# ---------------------------------------------------------------------------
# 2) Route test: warning card when state_db is unreachable
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dashboard_renders_db_unavailable_when_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If get_runtime raises, the dashboard shows the unavailable card."""
    # Force a fresh runtime singleton, then patch get_runtime to raise.
    runtime_mod.reset_runtime()

    def _boom() -> None:
        raise RuntimeError("no runtime in test")

    # Patch on the route module (where it's imported) and the source module.
    from lit_monitor.server.routes import brain_build as bb_route

    monkeypatch.setattr(bb_route, "get_runtime", _boom)

    client = TestClient(create_app())
    resp = client.get("/brain-build")
    assert resp.status_code == 200
    assert "State DB unavailable" in resp.text
    # Sanity: the per-paper table heading should NOT render in the unavail branch.
    assert "Per-paper status" not in resp.text


# ---------------------------------------------------------------------------
# 3) Route test: progress aggregation (2 of 3 fully complete → 67%)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dashboard_aggregates_progress_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three rows, two fully_complete → done=2, total=3, pct=67."""
    runtime_mod.reset_runtime()

    class _FakeDB:
        def get_all_brain_build_progress(self, limit: int = 100) -> list[dict]:
            return [
                {
                    "zotero_key": "K1",
                    "doi": "10.1/a",
                    "fully_complete": 1,
                    "simple_complete": 1,
                    "complex_complete": 1,
                    "failure_reason": None,
                    "paper_title": "Alpha",
                    "paper_year": 2023,
                    "paper_authors": None,
                    "paper_extraction_json": None,
                },
                {
                    "zotero_key": "K2",
                    "doi": "10.1/b",
                    "fully_complete": 1,
                    "simple_complete": 1,
                    "complex_complete": 1,
                    "failure_reason": None,
                    "paper_title": "Beta",
                    "paper_year": 2024,
                    "paper_authors": None,
                    "paper_extraction_json": None,
                },
                {
                    "zotero_key": "K3",
                    "doi": "10.1/c",
                    "fully_complete": 0,
                    "simple_complete": 0,
                    "complex_complete": 0,
                    "failure_reason": None,
                    "paper_title": "Gamma",
                    "paper_year": 2024,
                    "paper_authors": None,
                    "paper_extraction_json": None,
                },
            ]

        def get_recent_runs(self, limit: int = 10) -> list[dict]:
            return []

    class _FakeZotero:
        collection_name = "Test Collection"

    class _FakeConfig:
        zotero = _FakeZotero()

    class _FakeRuntime:
        state_db = _FakeDB()
        config = _FakeConfig()

    from lit_monitor.server.routes import brain_build as bb_route

    monkeypatch.setattr(bb_route, "get_runtime", lambda: _FakeRuntime())

    client = TestClient(create_app())
    resp = client.get("/brain-build")
    assert resp.status_code == 200
    body = resp.text
    # Progress: sl-progress-bar renders done/total/pct (B4 — exact label relaxed)
    assert "2 / 3" in body
    assert "67%" in body
    # Collection name surfaced.
    assert "Test Collection" in body
    # Per-paper table renders with one row per fake.
    assert "Alpha" in body and "Beta" in body and "Gamma" in body


# ---------------------------------------------------------------------------
# B7 — Shoelace controls + sl-details collection switcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_brain_build_controls_use_shoelace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /api/brain-build/controls returns sl-switch + sl-button with correct names (B1)."""
    runtime_mod.reset_runtime()

    import unittest.mock as mock

    fake_slot = mock.MagicMock()
    fake_slot.is_running.return_value = False
    fake_slot.process = None
    fake_slot.started_at = None

    class _FakeRuntime:
        processes = {"brain_build": fake_slot}

    from lit_monitor.server.routes import control as ctrl_route

    monkeypatch.setattr(ctrl_route, "get_runtime", lambda: _FakeRuntime())

    client = TestClient(create_app())
    resp = client.get("/api/brain-build/controls")
    assert resp.status_code == 200
    html = resp.text
    # Shoelace components present
    assert "<sl-switch" in html
    assert "<sl-button" in html
    # Form field names preserved for server-side Form() parsing
    assert 'name="resume"' in html
    assert 'name="max_papers"' in html


@pytest.mark.unit
def test_brain_build_collection_switcher_is_inline_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /brain-build renders the collection switcher as a plain inline control.

    User review (2026-06-13) flagged three things to fix on this control:
      1. "Current collection" must NOT be in a ``<code>`` font — plain text.
      2. The switcher must NOT be collapsible — drop the ``<sl-details>`` wrapper.
      3. The HTMX wrapper (#collection-field) + the switch form must remain so
         the dropdown still loads and the switch still works.
    """
    runtime_mod.reset_runtime()

    class _FakeDB:
        def get_all_brain_build_progress(self, limit: int = 100) -> list[dict]:
            return []

        def get_recent_runs(self, limit: int = 10) -> list[dict]:
            return []

    class _FakeZotero:
        collection_name = "TestCol"

    class _FakeConfig:
        zotero = _FakeZotero()

    class _FakeRuntime:
        state_db = _FakeDB()
        config = _FakeConfig()

    from lit_monitor.server.routes import brain_build as bb_route

    monkeypatch.setattr(bb_route, "get_runtime", lambda: _FakeRuntime())

    client = TestClient(create_app())
    resp = client.get("/brain-build")
    assert resp.status_code == 200
    html = resp.text
    # (2) No collapsible wrapper around the collection switcher.
    assert 'summary="Change collection' not in html
    # (1) The current collection name is plain text, not wrapped in <code>.
    #     The whole .collection-switcher section must contain no <code>…
    start = html.index("collection-switcher")
    end = html.index("</section>", start)
    switcher = html[start:end]
    assert "<code>" not in switcher, "current collection must be plain text, not <code>"
    assert "TestCol" in switcher
    # (3) The switch form + HTMX wrapper survive so the control still works.
    assert 'id="collection-field"' in switcher
    assert 'id="collection-switch-form"' in switcher
