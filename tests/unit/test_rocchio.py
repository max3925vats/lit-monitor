"""Unit tests for the Rocchio interest-vector core (Bundle J1).

Covers the pure scalar gates (soft_gate / time_decay_weight), the Rocchio
centroid update (direction + finite guards), and the raw-event orchestrator
(build_interest_inputs: source down-weight, sign classification, embedding-miss
skips, contributor count).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from scripts.learning.rocchio import (
    COLD_START_FLOOR,
    IMPLICIT_SAVE_WEIGHT,
    SOFT_GATE_K,
    build_interest_inputs,
    compute_interest_vector,
    soft_gate,
    time_decay_weight,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- #
# soft_gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [0, 1, 5, 9])
def test_soft_gate_below_floor_is_zero(n):
    assert soft_gate(n) == 0.0


@pytest.mark.parametrize("n", [10, 50, 500])
def test_soft_gate_above_floor_exact(n):
    assert soft_gate(n) == pytest.approx(n / (n + SOFT_GATE_K))


def test_soft_gate_monotone_increasing_above_floor():
    vals = [soft_gate(n) for n in (10, 50, 100, 500, 5000)]
    assert vals == sorted(vals)
    assert all(0.0 < v < 1.0 for v in vals)


def test_soft_gate_floor_boundary():
    # Exactly at the floor it switches on.
    assert soft_gate(COLD_START_FLOOR - 1) == 0.0
    assert soft_gate(COLD_START_FLOOR) > 0.0


# --------------------------------------------------------------------------- #
# time_decay_weight
# --------------------------------------------------------------------------- #

def test_time_decay_at_zero_is_one():
    assert time_decay_weight(0.0) == pytest.approx(1.0)


def test_time_decay_at_half_life_is_half():
    assert time_decay_weight(90.0) == pytest.approx(0.5)


def test_time_decay_at_two_half_lives_is_quarter():
    assert time_decay_weight(180.0) == pytest.approx(0.25)


def test_time_decay_negative_age_clamped():
    # A "future" event must not be up-weighted past 1.0.
    assert time_decay_weight(-30.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# compute_interest_vector
# --------------------------------------------------------------------------- #

def test_interest_moves_toward_positive_away_from_negative():
    dim = 8
    library = np.zeros(dim, dtype=np.float32)
    library[0] = 1.0  # prior points along axis 0

    pos_dir = np.zeros(dim, dtype=np.float32)
    pos_dir[1] = 1.0  # liked papers point along axis 1
    neg_dir = np.zeros(dim, dtype=np.float32)
    neg_dir[2] = 1.0  # dismissed papers point along axis 2

    positive = [(pos_dir, 1.0), (pos_dir, 1.0)]
    negative = [(neg_dir, 1.0)]

    result = compute_interest_vector(library, positive, negative)

    assert np.isfinite(result).all()
    assert result.shape == (dim,)
    # Moved TOWARD the positive direction relative to the prior.
    assert _cos(result, pos_dir) > _cos(library, pos_dir)
    # Moved AWAY from the negative direction relative to the prior.
    assert _cos(result, neg_dir) < _cos(library, neg_dir)


def test_interest_empty_feedback_returns_normalized_library():
    library = np.array([3.0, 0.0, 0.0, 4.0, 0.0], dtype=np.float32)  # norm 5
    result = compute_interest_vector(library, [], [])
    expected = library / np.linalg.norm(library)
    assert np.allclose(result, expected, atol=1e-6)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


def test_interest_nan_in_input_still_finite():
    library = np.array([np.nan, 1.0, 0.0, 0.0], dtype=np.float32)
    pos = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    bad_pos = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    result = compute_interest_vector(library, [(pos, 1.0), (bad_pos, 1.0)], [])
    assert np.isfinite(result).all()


def test_interest_zero_library_with_feedback_is_finite_direction():
    dim = 4
    library = np.zeros(dim, dtype=np.float32)
    pos = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    result = compute_interest_vector(library, [(pos, 1.0)], [])
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# build_interest_inputs
# --------------------------------------------------------------------------- #

def _embed_lookup_factory(mapping):
    def _lookup(doi):
        return mapping.get(doi)
    return _lookup


def test_implicit_save_contributes_half_weight():
    now = datetime(2026, 6, 5)
    created = now.strftime("%Y-%m-%d %H:%M:%S")  # age 0 -> decay 1.0
    vec = np.array([1.0, 0.0], dtype=np.float32)
    mapping = {"10/a": vec, "10/b": vec}

    explicit = {
        "doi": "10/a", "signal_type": "saved", "weight": 1.0,
        "rating": None, "source": "web", "created_at": created,
    }
    implicit = {
        "doi": "10/b", "signal_type": "saved", "weight": 1.0,
        "rating": None, "source": "implicit_zotero_save", "created_at": created,
    }

    pos, neg, n = build_interest_inputs(
        [explicit, implicit], _embed_lookup_factory(mapping), now
    )
    assert n == 2
    assert neg == []
    weights = sorted(w for _, w in pos)
    # implicit weight == explicit weight * IMPLICIT_SAVE_WEIGHT
    assert weights[0] == pytest.approx(weights[1] * IMPLICIT_SAVE_WEIGHT)
    assert weights[1] == pytest.approx(1.0)  # explicit, decay 1.0


def test_signal_classification_pos_neg():
    now = datetime(2026, 6, 5)
    created = now.strftime("%Y-%m-%d %H:%M:%S")
    vec = np.array([1.0, 0.0], dtype=np.float32)
    mapping = {f"10/{i}": vec for i in range(4)}

    events = [
        {"doi": "10/0", "signal_type": "saved", "weight": 1.0, "rating": None,
         "source": "web", "created_at": created},
        {"doi": "10/1", "signal_type": "dismissed", "weight": -0.5, "rating": None,
         "source": "web", "created_at": created},
        {"doi": "10/2", "signal_type": "rated", "weight": 0.5, "rating": 5,
         "source": "web", "created_at": created},
        {"doi": "10/3", "signal_type": "rated", "weight": -0.5, "rating": 1,
         "source": "web", "created_at": created},
    ]
    pos, neg, n = build_interest_inputs(
        events, _embed_lookup_factory(mapping), now
    )
    assert n == 4
    assert len(pos) == 2  # saved + rated 5
    assert len(neg) == 2  # dismissed + rated 1
    # Stored weights are non-negative magnitudes.
    assert all(w > 0 for _, w in pos + neg)


def test_opened_and_rating_three_are_neutral_skipped():
    now = datetime(2026, 6, 5)
    created = now.strftime("%Y-%m-%d %H:%M:%S")
    vec = np.array([1.0, 0.0], dtype=np.float32)
    mapping = {"10/a": vec, "10/b": vec}
    events = [
        {"doi": "10/a", "signal_type": "opened", "weight": 0.3, "rating": None,
         "source": "web", "created_at": created},
        {"doi": "10/b", "signal_type": "rated", "weight": 0.0, "rating": 3,
         "source": "web", "created_at": created},
    ]
    pos, neg, n = build_interest_inputs(
        events, _embed_lookup_factory(mapping), now
    )
    assert (pos, neg, n) == ([], [], 0)


def test_doi_without_embedding_is_skipped():
    now = datetime(2026, 6, 5)
    created = now.strftime("%Y-%m-%d %H:%M:%S")
    mapping = {"10/has": np.array([1.0, 0.0], dtype=np.float32)}  # 10/none absent
    events = [
        {"doi": "10/has", "signal_type": "saved", "weight": 1.0, "rating": None,
         "source": "web", "created_at": created},
        {"doi": "10/none", "signal_type": "saved", "weight": 1.0, "rating": None,
         "source": "web", "created_at": created},
    ]
    pos, neg, n = build_interest_inputs(
        events, _embed_lookup_factory(mapping), now
    )
    assert n == 1
    assert len(pos) == 1


def test_time_decay_folds_into_weight():
    now = datetime(2026, 6, 5)
    old_created = (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    vec = np.array([1.0, 0.0], dtype=np.float32)
    events = [
        {"doi": "10/old", "signal_type": "saved", "weight": 1.0, "rating": None,
         "source": "web", "created_at": old_created},
    ]
    pos, _, n = build_interest_inputs(
        events, _embed_lookup_factory({"10/old": vec}), now
    )
    assert n == 1
    # 90-day-old save: base 1.0 * decay 0.5 == 0.5
    assert pos[0][1] == pytest.approx(0.5, abs=1e-4)
