"""FI-1: unit tests for StateDB.feedback_timeline().

Seeds a real StateDB via record_feedback_event() and asserts the grouped
time-series shape. created_at is auto-stamped "now", so all seeded rows land in
the same day/week — these tests assert grouping shape + counts, not dates.
"""
from __future__ import annotations

import pytest

from scripts.core.state_db import StateDB


@pytest.fixture
def db(tmp_path) -> StateDB:
    """A StateDB seeded with 3 'saved' + 2 'dismissed' feedback events."""
    state_db = StateDB(str(tmp_path / "state.db"))
    for i in range(3):
        state_db.record_feedback_event(f"10.1/saved-{i}", "saved", source="test")
    for i in range(2):
        state_db.record_feedback_event(f"10.1/dismissed-{i}", "dismissed", source="test")
    return state_db


@pytest.fixture
def empty_db(tmp_path) -> StateDB:
    """A StateDB with no feedback events."""
    return StateDB(str(tmp_path / "empty.db"))


def test_feedback_timeline_groups_by_period_and_signal(db: StateDB) -> None:
    rows = db.feedback_timeline(granularity="week")
    # rows are {period, signal_type, count}; counts match seeded
    by_sig = {r["signal_type"]: r["count"] for r in rows}
    assert by_sig.get("saved") == 3 and by_sig.get("dismissed") == 2
    assert all({"period", "signal_type", "count"} <= set(r) for r in rows)


def test_feedback_timeline_day_granularity(db: StateDB) -> None:
    rows = db.feedback_timeline(granularity="day")
    assert rows  # groups by day


def test_feedback_timeline_invalid_granularity_falls_back(db: StateDB) -> None:
    # unknown granularity must NOT inject; falls back to week, returns valid rows
    assert db.feedback_timeline(
        granularity="'; DROP TABLE feedback_events;--"
    ) == db.feedback_timeline(granularity="week")


def test_feedback_timeline_empty_db_returns_list(empty_db: StateDB) -> None:
    assert empty_db.feedback_timeline() == []
