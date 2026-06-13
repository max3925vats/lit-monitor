"""Same-day digest disambiguation (ordinal filename suffix).

BUG: two discovery runs on the same date collided on ``Discovery_{date}.md``.
Fix: the day's 1st run keeps the bare name; 2nd, 3rd, ... get a
``_{ordinal}`` suffix. Ordinal = the run's position among same-day
``discovery_runs`` ordered by id.

These are PURE-helper tests:
- ``digest_filename(date, ordinal)`` — filename construction.
- ``discovery_run_same_day_ordinal(state_db, run_id)`` — the COUNT query,
  exercised against a real tmp_path StateDB (never the user's config/state.db).
"""
from __future__ import annotations

import pytest

from lit_monitor.api.queries import discovery_run_same_day_ordinal
from lit_monitor.output.digest_renderer import digest_filename


class TestDigestFilename:
    def test_first_run_keeps_bare_name(self):
        assert digest_filename("2026-06-10", 1) == "Discovery_2026-06-10.md"

    def test_second_run_gets_ordinal_suffix(self):
        assert digest_filename("2026-06-10", 2) == "Discovery_2026-06-10_2.md"

    def test_third_run_gets_ordinal_suffix(self):
        assert digest_filename("2026-06-10", 3) == "Discovery_2026-06-10_3.md"

    def test_ordinal_zero_or_negative_falls_back_to_bare(self):
        # Defensive: ordinal <= 1 → bare name (covers 0 / unexpected inputs).
        assert digest_filename("2026-06-10", 0) == "Discovery_2026-06-10.md"


@pytest.fixture
def seeded_runs(tmp_path):
    """Two discovery runs on the SAME date + one on a DIFFERENT date.

    ``started_at`` defaults to ``datetime('now')`` so all three share today's
    date; we rewrite the third run's date directly so the ordinal counter is
    scoped per-day. Returns the StateDB plus the three run ids in insertion
    order.
    """
    from lit_monitor.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    # Pin all rows to a fixed same-day timestamp so the test is deterministic.
    same_day = "2026-06-10T08:00:00"
    other_day = "2026-06-11T08:00:00"

    rid1 = db.start_discovery_run({"topics": ["a"]})
    rid2 = db.start_discovery_run({"topics": ["b"]})
    rid3 = db.start_discovery_run({"topics": ["c"]})

    with db._connect() as conn:
        conn.execute("UPDATE discovery_runs SET started_at = ? WHERE id = ?", (same_day, rid1))
        conn.execute("UPDATE discovery_runs SET started_at = ? WHERE id = ?", (same_day, rid2))
        conn.execute("UPDATE discovery_runs SET started_at = ? WHERE id = ?", (other_day, rid3))
    return db, rid1, rid2, rid3


class TestSameDayOrdinal:
    def test_first_same_day_run_is_ordinal_1(self, seeded_runs):
        db, rid1, _rid2, _rid3 = seeded_runs
        assert discovery_run_same_day_ordinal(db, rid1) == 1

    def test_second_same_day_run_is_ordinal_2(self, seeded_runs):
        db, _rid1, rid2, _rid3 = seeded_runs
        assert discovery_run_same_day_ordinal(db, rid2) == 2

    def test_other_date_run_resets_to_ordinal_1(self, seeded_runs):
        db, _rid1, _rid2, rid3 = seeded_runs
        assert discovery_run_same_day_ordinal(db, rid3) == 1

    def test_missing_run_returns_1_defensively(self, seeded_runs):
        db, *_ = seeded_runs
        assert discovery_run_same_day_ordinal(db, 99999) == 1
