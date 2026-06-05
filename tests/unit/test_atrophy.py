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

from scripts.clustering.atrophy import compute_per_cluster_feedback_weights

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
