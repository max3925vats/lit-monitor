"""Unit tests for /schedule + /api/schedule (F5.2)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app
from lit_monitor.server.runtime import reset_runtime
from lit_monitor.server.scheduler import ScheduleSpec


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
    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="unsupported"):
        resp = client.get("/schedule")
    assert resp.status_code == 200
    assert b"Scheduling not supported" in resp.content


@pytest.mark.unit
def test_schedule_page_shows_current(client):
    spec = ScheduleSpec.parse("wed", "09:15")
    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("lit_monitor.server.routes.schedule.read_schedule", return_value=spec):
        resp = client.get("/schedule")
    assert resp.status_code == 200
    assert b"wed" in resp.content
    assert b"09:15" in resp.content


@pytest.mark.unit
def test_schedule_current_renders_as_definition_list(client):
    # AR-7: the 'Current schedule' key/value block is a <dl class="kv"> with
    # <dt>/<dd>, not a <th>-in-<tbody> table. The day/time values must survive.
    spec = ScheduleSpec.parse("wed", "09:15")
    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("lit_monitor.server.routes.schedule.read_schedule", return_value=spec):
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
    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("lit_monitor.server.routes.schedule.write_schedule") as mock_write:
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
    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="macos"):
        resp = client.post("/api/schedule", data={"day_of_week": "mon", "time": "25:99"})
    assert resp.status_code == 400


@pytest.mark.unit
def test_create_schedule_rejected_on_unsupported_platform(client):
    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="unsupported"):
        resp = client.post("/api/schedule", data={"day_of_week": "mon", "time": "08:00"})
    assert resp.status_code == 400


@pytest.mark.unit
def test_delete_schedule_invokes_remover(client):
    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("lit_monitor.server.routes.schedule.remove_schedule") as mock_rm:
        resp = client.delete("/api/schedule")
    assert resp.status_code == 200
    mock_rm.assert_called_once()


@pytest.mark.unit
def test_schedule_form_uses_shoelace(client):
    """C4: schedule page form uses Shoelace components; day_of_week pre-selection
    is on the sl-select HOST (value= attribute), not on an sl-option (gotcha)."""
    import re

    with patch("lit_monitor.server.routes.schedule.detect_platform", return_value="macos"), \
         patch("lit_monitor.server.routes.schedule.read_schedule", return_value=None):
        resp = client.get("/schedule")

    assert resp.status_code == 200
    html = resp.text

    # Shoelace components are present
    assert "<sl-select" in html
    assert "<sl-button" in html

    # Time field is an sl-input with type="time"
    assert 'type="time"' in html

    # Day-of-week pre-selection is on the HOST (value= attr on <sl-select>),
    # NOT via `selected` on an individual <sl-option> — Shoelace gotcha.
    assert re.search(r'<sl-select\b[^>]*\bname="day_of_week"[^>]*\bvalue="', html), (
        "sl-select for day_of_week must carry value= on the host element"
    )

    # Sanity: no native <select> or <input type="time"> remain in the form block
    assert "<select" not in html
    assert '<input type="time"' not in html


# ---------------------------------------------------------------------------
# P4: consolidated "Run History" card on /schedule — scheduled runs only.
# Mirrors the Discovery Run History card (shared partial) but filters the
# unified history query to trigger="scheduled" so manual runs never appear.
# ---------------------------------------------------------------------------


def _scheduled_rows() -> list[dict]:
    """Two scheduled runs (global ids 3 and 5 — manual runs #1/#2/#4 absent)."""
    return [
        {
            "id": 5, "started_at": "2026-06-12 08:00:00",
            "finished_at": "2026-06-12 08:04:00", "status": "success",
            "total_found": 9, "total_ingested": 4,
            "papers_processed": 4, "papers_skipped": 1, "papers_failed": 0,
            "trigger": "scheduled",
        },
        {
            "id": 3, "started_at": "2026-06-05 08:00:00",
            "finished_at": "2026-06-05 08:03:00", "status": "success",
            "total_found": 2, "total_ingested": 1,
            "papers_processed": None, "papers_skipped": None, "papers_failed": None,
            "trigger": "scheduled",
        },
    ]


def _runtime_with_db() -> MagicMock:
    """Fake runtime exposing a non-None state_db so _safe_db() succeeds.

    The unified history query itself is patched at the route's import site
    (get_discovery_run_history); this just makes _safe_db() return a sentinel.
    """
    rt = MagicMock()
    rt.state_db.get_recent_runs_by_type.return_value = []
    return rt


def _patch_schedule(rt: MagicMock, history_fn):
    """Context-manager bundle patching detect_platform/read_schedule/runtime/query."""
    return (
        patch("lit_monitor.server.routes.schedule.detect_platform", return_value="macos"),
        patch("lit_monitor.server.routes.schedule.read_schedule", return_value=None),
        patch("lit_monitor.server.routes.schedule.get_runtime", return_value=rt),
        patch(
            "lit_monitor.server.routes.schedule.get_discovery_run_history",
            history_fn,
        ),
    )


@pytest.mark.unit
def test_schedule_renders_run_history_card(client):
    """The schedule page shows the shared 'Run History' sl-details card."""
    rt = _runtime_with_db()
    hist = MagicMock(return_value={"runs": _scheduled_rows(), "total": 2})
    p1, p2, p3, p4 = _patch_schedule(rt, hist)
    with p1, p2, p3, p4:
        resp = client.get("/schedule")
    assert resp.status_code == 200
    assert '<sl-details summary="Run History"' in resp.text
    # Old plain "Recent discovery runs" table is gone.
    assert "Recent discovery runs" not in resp.text


@pytest.mark.unit
def test_schedule_history_query_filters_scheduled(client):
    """The route MUST call get_discovery_run_history with trigger='scheduled'."""
    rt = _runtime_with_db()
    hist = MagicMock(return_value={"runs": _scheduled_rows(), "total": 2})
    p1, p2, p3, p4 = _patch_schedule(rt, hist)
    with p1, p2, p3, p4:
        resp = client.get("/schedule")
    assert resp.status_code == 200
    # Assert via the recorded kwargs on the patched query fn.
    _, kwargs = hist.call_args
    assert kwargs.get("trigger") == "scheduled"
    assert kwargs.get("limit") == 10
    assert kwargs.get("offset") == 0


@pytest.mark.unit
def test_schedule_renders_global_run_id(client):
    """A scheduled run with global id=3 renders as #3 (manual #1/#2 don't appear)."""
    rt = _runtime_with_db()
    hist = MagicMock(return_value={"runs": _scheduled_rows(), "total": 2})
    p1, p2, p3, p4 = _patch_schedule(rt, hist)
    with p1, p2, p3, p4:
        resp = client.get("/schedule")
    body = resp.text
    assert "#3" in body
    assert "#5" in body
    # Per-row link still points at the discovery detail page (global id).
    assert "/discovery/3" in body


@pytest.mark.unit
def test_schedule_pager_links_point_at_schedule(client):
    """Pager links on /schedule must target /schedule?runs_offset=, NOT /discovery."""
    # 10 rows returned with total=12 so the page-size-10 pager renders a Next link.
    rows = [
        {
            "id": 100 + i, "started_at": "2026-06-01 08:00:00",
            "finished_at": "2026-06-01 08:01:00", "status": "success",
            "total_found": 1, "total_ingested": 1,
            "papers_processed": 1, "papers_skipped": 0, "papers_failed": 0,
            "trigger": "scheduled",
        }
        for i in range(10)
    ]
    rt = _runtime_with_db()
    hist = MagicMock(return_value={"runs": rows, "total": 12})
    p1, p2, p3, p4 = _patch_schedule(rt, hist)
    with p1, p2, p3, p4:
        resp = client.get("/schedule")
    body = resp.text
    assert "/schedule?runs_offset=10" in body   # Next → page 2, schedule-scoped
    assert "/discovery?runs_offset=" not in body  # never the discovery pager base


@pytest.mark.unit
def test_schedule_pager_does_inplace_htmx_swap(client):
    """The /schedule Run History pager paginates IN PLACE via HTMX, targeting
    a ``#run-history-body`` wrapper inside the sl-details, with hx-get pointed
    at /schedule and the plain href kept as a no-JS fallback."""
    rows = [
        {
            "id": 100 + i, "started_at": "2026-06-01 08:00:00",
            "finished_at": "2026-06-01 08:01:00", "status": "success",
            "total_found": 1, "total_ingested": 1,
            "papers_processed": 1, "papers_skipped": 0, "papers_failed": 0,
            "trigger": "scheduled",
        }
        for i in range(10)
    ]
    rt = _runtime_with_db()
    hist = MagicMock(return_value={"runs": rows, "total": 12})
    p1, p2, p3, p4 = _patch_schedule(rt, hist)
    with p1, p2, p3, p4:
        resp = client.get("/schedule")
    body = resp.text
    assert 'id="run-history-body"' in body
    body_idx = body.index('id="run-history-body"')
    details_idx = body.index('summary="Run History"')
    assert details_idx < body_idx  # wrapper inside the card
    assert 'hx-get="/schedule?runs_offset=10"' in body
    assert 'hx-target="#run-history-body"' in body
    assert 'hx-select="#run-history-body"' in body
    assert 'href="/schedule?runs_offset=10"' in body  # fallback


@pytest.mark.unit
def test_schedule_empty_state_is_scheduled_specific(client):
    """No scheduled runs → the scheduled-specific empty text, not discovery's."""
    rt = _runtime_with_db()
    hist = MagicMock(return_value={"runs": [], "total": 0})
    p1, p2, p3, p4 = _patch_schedule(rt, hist)
    with p1, p2, p3, p4:
        resp = client.get("/schedule")
    assert resp.status_code == 200
    body = resp.text
    assert "No scheduled runs yet" in body
    # Must NOT show discovery's generic empty wording.
    assert "No discovery runs yet." not in body
