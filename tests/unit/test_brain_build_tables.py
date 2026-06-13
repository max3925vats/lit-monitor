"""Unit tests for the Brain-build dashboard table redesign (Tasks C + D).

Task C — Per-paper status table:
  - wrapped in a collapsed-by-default ``<sl-details summary="Per-paper status">``
  - phase headers renamed Core fields / Deep fields / Indexed + helper text
  - DOI rendered as a truncating link (``.doi-cell``) to ``https://doi.org/…``
    with the full DOI in ``title=``; ``(no DOI)`` stays plain text
  - ``.status-dot`` (green/red/grey) replaces ✓/· per phase
  - 10-row pagination via ``?bb_offset=`` that does NOT regress the progress bar

Task D — Recent runs table + discovery run-id truncation:
  - route fetches brain_build runs only (``get_recent_runs_by_type``)
  - Type column dropped, Completed column added, Run ID truncated w/ title
  - wrapped in a collapsed ``<sl-details summary="Recent runs">``
  - discovery "Last run" block omits the run_id (review 2026-06-13)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server import runtime as runtime_mod
from lit_monitor.server.app import create_app


def _make_rows(n: int, *, fully: int = 0, failed_idx: set[int] | None = None) -> list[dict]:
    """Build ``n`` fake brain_build_progress rows.

    ``fully`` rows (the first ``fully``) are fully_complete; rows in
    ``failed_idx`` carry an extraction-json error so error_message is truthy.
    """
    failed_idx = failed_idx or set()
    rows: list[dict] = []
    for i in range(n):
        rows.append(
            {
                "zotero_key": f"K{i:03d}",
                "doi": f"10.1/{i:03d}",
                "fully_complete": 1 if i < fully else 0,
                "simple_complete": 1 if i < fully else 0,
                "complex_complete": 1 if i < fully else 0,
                "failure_reason": None,
                "paper_title": f"Paper {i}",
                "paper_year": 2024,
                "paper_authors": None,
                "paper_extraction_json": (
                    '{"error": "boom"}' if i in failed_idx else None
                ),
            }
        )
    return rows


def _fake_runtime(rows: list[dict], runs: list[dict] | None = None):
    """A runtime double whose state_db returns ``rows`` and ``runs``."""
    runs = runs or []
    calls: dict[str, object] = {}

    class _FakeDB:
        def get_all_brain_build_progress(
            self, limit: int = 100, offset: int = 0
        ) -> list[dict]:
            calls["bb_limit"] = limit
            return list(rows)

        def get_recent_runs(self, limit: int = 10) -> list[dict]:
            calls["called_recent_runs"] = True
            return list(runs)

        def get_recent_runs_by_type(
            self, run_type: str, limit: int = 10
        ) -> list[dict]:
            calls["by_type"] = (run_type, limit)
            return list(runs)

    class _FakeZotero:
        collection_name = "TestCol"

    class _FakeConfig:
        zotero = _FakeZotero()

    class _FakeRuntime:
        state_db = _FakeDB()
        config = _FakeConfig()

    return _FakeRuntime, calls


def _get(monkeypatch, rows, runs=None, query: str = "") -> str:
    runtime_mod.reset_runtime()
    fake_rt, _calls = _fake_runtime(rows, runs)
    from lit_monitor.server.routes import brain_build as bb_route

    monkeypatch.setattr(bb_route, "get_runtime", lambda: fake_rt())
    client = TestClient(create_app())
    resp = client.get(f"/brain-build{query}")
    assert resp.status_code == 200
    return resp.text


# ---------------------------------------------------------------------------
# Task C — per-paper table card
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_per_paper_table_in_collapsed_sl_details(monkeypatch) -> None:
    html = _get(monkeypatch, _make_rows(3))
    assert '<sl-details summary="Per-paper status"' in html
    # Collapsed by default: the per-paper details must not carry `open`.
    idx = html.index('summary="Per-paper status"')
    # look at the opening tag only (up to its closing '>')
    tag = html[html.rindex("<sl-details", 0, idx) : html.index(">", idx) + 1]
    assert " open" not in tag


@pytest.mark.unit
def test_phase_headers_renamed_and_helper_present(monkeypatch) -> None:
    html = _get(monkeypatch, _make_rows(2))
    assert "Core fields" in html
    assert "Deep fields" in html
    assert "Indexed" in html
    # old labels gone
    assert "<th>Simple</th>" not in html
    assert "<th>Complex</th>" not in html
    assert "<th>Fully</th>" not in html
    # helper text describing the three phases
    assert "basic extraction" in html
    assert "deep extraction" in html
    assert "embedded into the brain" in html


@pytest.mark.unit
def test_doi_rendered_as_truncating_link(monkeypatch) -> None:
    html = _get(monkeypatch, _make_rows(1))
    assert 'class="doi-cell"' in html
    assert 'href="https://doi.org/10.1/000"' in html
    assert 'title="10.1/000"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener"' in html


@pytest.mark.unit
def test_missing_doi_is_plain_text(monkeypatch) -> None:
    rows = _make_rows(1)
    rows[0]["doi"] = None
    html = _get(monkeypatch, rows)
    assert "(no DOI)" in html
    # the no-DOI row must not produce a doi.org link
    assert "https://doi.org/None" not in html


@pytest.mark.unit
def test_status_dots_all_three_states_render(monkeypatch) -> None:
    """A green (done), a red (failed), and a grey (pending) row each appear."""
    # row0 fully complete (all green); row1 failed (red); row2 pending (grey)
    rows = _make_rows(3, fully=1, failed_idx={1})
    html = _get(monkeypatch, rows)
    assert "status-dot ok" in html
    assert "status-dot fail" in html
    assert "status-dot pending" in html
    # accessibility labels present
    assert "Complete" in html
    assert "Failed" in html
    assert "Not started" in html


@pytest.mark.unit
def test_pagination_windows_to_ten_rows(monkeypatch) -> None:
    """25 papers → first page shows 10 rows; Next link present, Prev absent."""
    rows = _make_rows(25)
    html = _get(monkeypatch, rows, query="?bb_offset=0")
    # Exactly 10 of 25 titles render on page one.
    shown = [f"Paper {i}" for i in range(25) if f"Paper {i}" in html]
    assert len(shown) == 10
    assert "bb_offset=10" in html  # Next link
    assert "bb_offset=-" not in html  # never negative


@pytest.mark.unit
def test_progress_bar_not_regressed_by_offset(monkeypatch) -> None:
    """Progress totals reflect ALL papers, not the 10-row window.

    20 of 25 papers are fully_complete (the first 20). Viewing the LAST page
    (offset=20) shows only rows 20–24 — all incomplete. If the progress bar
    were (wrongly) computed from the visible window it would read 0/5 / 0%.
    It must still read 20/25 / 80% across the whole corpus.
    """
    rows = _make_rows(25, fully=20)
    html = _get(monkeypatch, rows, query="?bb_offset=20")
    assert "20 / 25" in html
    assert "80%" in html
    # The window's own done-count (0) must NOT leak into the bar.
    assert "0 / 5" not in html


# ---------------------------------------------------------------------------
# Task D — recent runs table
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_recent_runs_uses_by_type_brain_build(monkeypatch) -> None:
    runtime_mod.reset_runtime()
    fake_rt, calls = _fake_runtime(_make_rows(0), runs=[])
    from lit_monitor.server.routes import brain_build as bb_route

    monkeypatch.setattr(bb_route, "get_runtime", lambda: fake_rt())
    client = TestClient(create_app())
    resp = client.get("/brain-build")
    assert resp.status_code == 200
    assert calls.get("by_type") == ("brain_build", 10)
    assert "called_recent_runs" not in calls


@pytest.mark.unit
def test_recent_runs_table_columns_and_card(monkeypatch) -> None:
    runs = [
        {
            "run_id": "abcdef1234567890",
            "run_type": "brain_build",
            "started_at": "2026-06-01 10:00",
            "finished_at": "2026-06-01 10:30",
            "status": "complete",
            "papers_processed": 5,
            "papers_failed": 1,
        }
    ]
    html = _get(monkeypatch, _make_rows(0), runs=runs)
    assert '<sl-details summary="Recent runs"' in html
    # Type column dropped; Completed column added.
    assert "<th>Type</th>" not in html
    assert "<th>Completed</th>" in html
    # Run ID truncated to 8 chars with full id in title.
    assert "abcdef12" in html
    assert 'title="abcdef1234567890"' in html
    assert "2026-06-01 10:30" in html  # finished_at


# ---------------------------------------------------------------------------
# Review 2026-06-13 — the discovery Last-run block dropped Run ID entirely
# (BB-D's truncation is superseded). The run_id must no longer be rendered there.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_discovery_last_run_omits_run_id(monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    runtime_mod.reset_runtime()
    full_id = "11111111-2222-3333-4444-555555555555"
    fake_db = MagicMock()
    fake_db.get_recent_runs_by_type.return_value = [
        {
            "run_id": full_id,
            "started_at": "2026-06-01",
            "finished_at": "2026-06-01",
            "status": "complete",
            "papers_processed": 1,
            "papers_skipped": 0,
            "papers_failed": 0,
        }
    ]
    # Empty Run History so the run_id can only come from the Last-run block.
    fake_db.get_discovery_run_history.return_value = {"runs": [], "total": 0}
    fake_rt = MagicMock()
    fake_rt.state_db = fake_db
    fake_rt.config.obsidian.vault_path = ""
    with patch(
        "lit_monitor.server.routes.discovery.get_runtime", return_value=fake_rt
    ):
        client = TestClient(create_app())
        resp = client.get("/discovery")
    assert resp.status_code == 200
    html = resp.text
    # The Last-run block no longer shows the run id (truncated or full).
    assert "Run ID" not in html
    assert full_id not in html
    assert "11111111…" not in html
    # The one-line summary + its other fields still render.
    assert "last-run-line" in html
    assert "complete" in html
