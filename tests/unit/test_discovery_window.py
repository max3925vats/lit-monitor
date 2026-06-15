from datetime import date
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
