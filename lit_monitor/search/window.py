"""Resolved discovery search window — a leaf module so both the discovery
pipeline and the search layer can import SearchWindow without a cycle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class SearchWindow:
    """A resolved discovery search window. ``until=None`` means 'today'
    (open-ended upper bound)."""
    since: date
    until: date | None


def resolve_window(
    *,
    since_days: int | None,
    since: date | None,
    from_date: date | None,
    to_date: date | None,
    today: date,
) -> SearchWindow | None:
    """Map the mutually-exclusive CLI/web window inputs to a SearchWindow
    override, or None for the default adaptive frontier. The caller is
    responsible for rejecting more than one mode being set (CLI validation)."""
    if since_days is not None:
        return SearchWindow(since=today - timedelta(days=since_days), until=None)
    if since is not None:
        return SearchWindow(since=since, until=None)
    # from_date+to_date: both required. A partial range (only one supplied)
    # falls through to None (a silent no-op) — the caller validates
    # exclusivity/pairing upstream (CLI flags, Task 7), so it never reaches here.
    if from_date is not None and to_date is not None:
        return SearchWindow(since=from_date, until=to_date)
    return None
