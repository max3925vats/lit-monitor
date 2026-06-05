"""Unit tests for Bundle K-a per-cluster atrophy feedback weights.

The pure function under test (``compute_per_cluster_feedback_weights``) turns
raw ``feedback_events`` rows + a ``doi -> cluster_id`` mapping into a
``{cluster_id: feedback_weight}`` dict where each weight lives in
``[floor, 1.0]``. Decay happens ONLY on active dismissal; engagement and the
never-touched baseline keep a cluster at full weight.

All tests are pure (no DB): the cluster mapping is injected.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from scripts.clustering.atrophy import (
    compute_per_cluster_feedback_weights,
    find_under_engaged_clusters,
)

# Fixed reference "now" so age computations are deterministic.
NOW = datetime(2026, 6, 5, 12, 0, 0)
FLOOR = 0.1
HALF_LIFE = 90.0


def _event(doi: str, signal_type: str, *, age_days: float = 0.0,
           rating: int | None = None, weight: float = 1.0,
           source: str | None = None) -> dict:
    """Build a feedback_events-shaped row at a given age (days before NOW)."""
    created = NOW - timedelta(days=age_days)
    return {
        "doi": doi,
        "signal_type": signal_type,
        "weight": weight,
        "rating": rating,
        "source": source,
        "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
    }


def test_many_recent_saves_keeps_full_weight() -> None:
    """A cluster with only recent saves (engagement) stays at full weight ~1.0."""
    mapping = {f"10.1/a{i}": 1 for i in range(50)}
    events = [_event(f"10.1/a{i}", "saved", age_days=5.0) for i in range(50)]
    weights = compute_per_cluster_feedback_weights(
        events, mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
    )
    assert weights[1] == pytest.approx(1.0, abs=1e-9)


def test_zero_event_cluster_is_full_weight() -> None:
    """A cluster that has assignments but no feedback events ever → weight 1.0.

    Satisfies the plan's "cluster B with 0 events → weight ≥ floor" (here the
    stronger 1.0: a never-dismissed cluster keeps full weight).
    """
    mapping = {"10.1/only": 7}  # cluster 7 has an assignment but no events
    weights = compute_per_cluster_feedback_weights(
        [], mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
    )
    assert weights[7] == pytest.approx(1.0, abs=1e-9)
    assert weights[7] >= FLOOR


def test_many_recent_dismissals_decays_toward_floor_never_below() -> None:
    """Heavy recent dismissal pushes weight toward the floor but never under it."""
    mapping = {f"10.1/d{i}": 2 for i in range(40)}
    events = [_event(f"10.1/d{i}", "dismissed", age_days=2.0) for i in range(40)]
    weights = compute_per_cluster_feedback_weights(
        events, mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
    )
    assert weights[2] >= FLOOR
    assert weights[2] < 0.5  # clearly decayed, not full


def test_old_dismissals_recover_toward_full() -> None:
    """Dismissals far older than the half-life fade → weight recovers toward 1.0."""
    mapping = {f"10.1/old{i}": 3 for i in range(40)}
    # ~10 half-lives old: decayed mass is negligible.
    events = [_event(f"10.1/old{i}", "dismissed", age_days=900.0)
              for i in range(40)]
    weights = compute_per_cluster_feedback_weights(
        events, mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
    )
    assert weights[3] > 0.9  # almost fully recovered


def test_mixed_is_between_floor_and_full_and_monotone() -> None:
    """Weight is in (floor, 1.0) for mixed feedback and decreases with dismissal
    fraction (more dismissals at fixed saves → lower weight)."""
    mapping = {f"10.1/m{i}": 4 for i in range(60)}

    def weight_for(n_dismiss: int, n_save: int) -> float:
        evs = [_event(f"10.1/m{i}", "dismissed", age_days=3.0)
               for i in range(n_dismiss)]
        evs += [_event(f"10.1/s{i}", "saved", age_days=3.0)
                for i in range(n_save)]
        # ensure save dois are mapped too
        for i in range(n_save):
            mapping[f"10.1/s{i}"] = 4
        return compute_per_cluster_feedback_weights(
            evs, mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
        )[4]

    low = weight_for(5, 20)
    high = weight_for(20, 5)
    assert FLOOR < low < 1.0
    assert FLOOR <= high < 1.0
    # More dismissals (high) → lower weight than fewer dismissals (low).
    assert high < low


def test_doi_not_in_mapping_is_skipped_no_crash() -> None:
    """Events whose DOI is not in the cluster mapping are silently skipped."""
    mapping = {"10.1/known": 9}
    events = [
        _event("10.1/known", "saved", age_days=1.0),
        _event("10.1/unknown", "dismissed", age_days=1.0),  # no mapping
    ]
    weights = compute_per_cluster_feedback_weights(
        events, mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
    )
    # cluster 9 only saw the save → full weight; unknown produced no cluster.
    assert weights == {9: pytest.approx(1.0, abs=1e-9)}


def test_empty_and_nonfinite_inputs_are_finite() -> None:
    """Empty inputs and NaN/garbage events never crash and stay finite."""
    import math

    # Fully empty.
    assert compute_per_cluster_feedback_weights([], {}, now=NOW) == {}

    mapping = {"10.1/x": 1}
    bad = [
        {"doi": "10.1/x", "signal_type": "dismissed",
         "weight": float("nan"), "rating": None, "source": None,
         "created_at": "not-a-date"},
        {"doi": "10.1/x", "signal_type": None, "weight": 1.0,
         "rating": None, "source": None, "created_at": None},
    ]
    weights = compute_per_cluster_feedback_weights(
        bad, mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
    )
    assert 1 in weights
    assert math.isfinite(weights[1])
    assert FLOOR <= weights[1] <= 1.0


def test_rating_classification_negative_decays() -> None:
    """A low rating (≤2) counts as a dismissal and decays the cluster."""
    mapping = {f"10.1/r{i}": 5 for i in range(30)}
    events = [_event(f"10.1/r{i}", "rated", rating=1, age_days=2.0)
              for i in range(30)]
    weights = compute_per_cluster_feedback_weights(
        events, mapping, now=NOW, floor=FLOOR, half_life_days=HALF_LIFE
    )
    assert weights[5] < 0.6
    assert weights[5] >= FLOOR


# --------------------------------------------------------------------------- #
# Bundle K-b: under-engaged cluster detection (pure, injected inputs).
# --------------------------------------------------------------------------- #


def _iso(days_ago: float) -> str:
    """ISO timestamp ``days_ago`` days before NOW (SQLite datetime() shape)."""
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def test_never_surfaced_cluster_is_under_engaged() -> None:
    """A cluster that never surfaced a paper (None) with healthy weight → explore."""
    last = {1: None}
    weights = {1: 1.0}
    assert find_under_engaged_clusters(
        last, weights, now=NOW, quiet_weeks=4, floor=FLOOR
    ) == [1]


def test_old_surface_is_under_engaged_recent_is_not() -> None:
    """Quiet beyond the window → under-engaged; surfaced inside the window → not."""
    # cluster 1 surfaced 10 weeks ago (quiet); cluster 2 surfaced 3 days ago.
    last = {1: _iso(70.0), 2: _iso(3.0)}
    weights = {1: 1.0, 2: 1.0}
    result = find_under_engaged_clusters(
        last, weights, now=NOW, quiet_weeks=4, floor=FLOOR
    )
    assert result == [1]


def test_near_floor_weight_excluded_even_if_quiet() -> None:
    """A quiet cluster the user is actively dismissing (weight near floor) is
    EXCLUDED — exploration must not fight the user's own dismissal signal."""
    last = {1: None, 2: None}
    # cluster 1 at the floor (actively dismissed); cluster 2 healthy.
    weights = {1: FLOOR, 2: 1.0}
    result = find_under_engaged_clusters(
        last, weights, now=NOW, quiet_weeks=4, floor=FLOOR
    )
    assert result == [2]


def test_weight_just_above_threshold_included() -> None:
    """Default exclusion threshold is floor + epsilon (0.05); a weight strictly
    above it is eligible, a weight at/below it is excluded."""
    last = {1: None, 2: None, 3: None}
    weights = {
        1: FLOOR + 0.05 + 1e-6,  # just above threshold → included
        2: FLOOR + 0.05,          # exactly at threshold → excluded (strict >)
        3: FLOOR + 0.04,          # below threshold → excluded
    }
    result = find_under_engaged_clusters(
        last, weights, now=NOW, quiet_weeks=4, floor=FLOOR,
    )
    assert result == [1]


def test_missing_weight_defaults_to_full_and_is_included() -> None:
    """A quiet cluster with no weight entry defaults to 1.0 (never touched) and
    is eligible for exploration."""
    last = {7: None}
    result = find_under_engaged_clusters(
        {**last}, {}, now=NOW, quiet_weeks=4, floor=FLOOR
    )
    assert result == [7]


def test_explicit_min_explore_weight_overrides_default() -> None:
    """An explicit ``min_explore_weight`` overrides the floor+epsilon default."""
    last = {1: None, 2: None}
    weights = {1: 0.5, 2: 0.7}
    # Only weights strictly above 0.6 qualify.
    result = find_under_engaged_clusters(
        last, weights, now=NOW, quiet_weeks=4, floor=FLOOR, min_explore_weight=0.6
    )
    assert result == [2]


def test_unparseable_last_surfaced_counts_as_never() -> None:
    """A malformed last-surfaced timestamp degrades to 'never surfaced' (quiet)
    rather than crashing detection."""
    last = {1: "not-a-date"}
    result = find_under_engaged_clusters(
        last, {1: 1.0}, now=NOW, quiet_weeks=4, floor=FLOOR
    )
    assert result == [1]


def test_empty_inputs_return_empty() -> None:
    """No clusters → no under-engaged clusters (drives the discovery no-op gate)."""
    assert find_under_engaged_clusters({}, {}, now=NOW) == []
