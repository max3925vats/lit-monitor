from datetime import date

from lit_monitor.pipelines.discovery import (
    advance_coverage_until,
    read_coverage_until,
    seed_coverage_until,
)
from lit_monitor.search.window import SearchWindow, resolve_window


def test_since_days_mode():
    w = resolve_window(since_days=7, since=None, from_date=None, to_date=None,
                       today=date(2026, 4, 1))
    assert w == SearchWindow(since=date(2026, 3, 25), until=None)


def test_since_date_mode():
    w = resolve_window(since_days=None, since=date(2026, 1, 1),
                       from_date=None, to_date=None, today=date(2026, 4, 1))
    assert w == SearchWindow(since=date(2026, 1, 1), until=None)


def test_range_mode():
    w = resolve_window(since_days=None, since=None,
                       from_date=date(2025, 8, 1), to_date=date(2025, 9, 1),
                       today=date(2026, 4, 1))
    assert w == SearchWindow(since=date(2025, 8, 1), until=date(2025, 9, 1))


def test_no_override_returns_none():
    assert resolve_window(since_days=None, since=None, from_date=None,
                          to_date=None, today=date(2026, 4, 1)) is None


# ---------------------------------------------------------------------------
# Task 2: coverage_until frontier — read / advance / seed
# ---------------------------------------------------------------------------

class _KV:
    def __init__(self, **kv):
        self._kv = dict(kv)

    def get_kv(self, k):
        return self._kv.get(k)

    def set_kv(self, k, v):
        self._kv[k] = v


def test_advance_takes_max_and_ignores_older():
    db = _KV(coverage_until="2026-02-01")
    advance_coverage_until(db, covered_to=date(2025, 9, 1))  # older -> no change
    assert read_coverage_until(db) == date(2026, 2, 1)
    advance_coverage_until(db, covered_to=date(2026, 4, 1))  # newer -> advances
    assert read_coverage_until(db) == date(2026, 4, 1)


def test_seed_from_last_run_date_only_when_unset():
    db = _KV(last_run_date="2026-03-15")
    seed_coverage_until(db)
    assert read_coverage_until(db) == date(2026, 3, 15)
    db.set_kv("last_run_date", "2026-05-01")
    seed_coverage_until(db)  # already set -> no-op
    assert read_coverage_until(db) == date(2026, 3, 15)


def test_read_none_when_unset():
    assert read_coverage_until(_KV()) is None


def test_read_warns_and_resets_on_unparseable(caplog):
    """A corrupt stored value returns None + WARNING (not a crash), so advance()
    then self-heals it — the Task 6 forward-safety path."""
    import logging

    db = _KV(coverage_until="not-a-date")
    with caplog.at_level(logging.WARNING, logger="lit_monitor.pipelines.discovery"):
        assert read_coverage_until(db) is None
    assert "unparseable" in caplog.text
    # self-heal: advancing over the corrupt value writes a clean frontier
    advance_coverage_until(db, covered_to=date(2026, 4, 1))
    assert read_coverage_until(db) == date(2026, 4, 1)
