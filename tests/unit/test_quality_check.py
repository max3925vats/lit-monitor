"""Unit tests for Idea 4 Tier-4 quality-regression scorers.

All tests use a deterministic FAKE embed_fn — no live Ollama. The fake maps
known input strings to fixed orthogonal/parallel/intermediate vectors so the
cosine-derived PASS/WARN/FAIL boundaries are exercised exactly.
"""
from __future__ import annotations

import math

import numpy as np

from lit_monitor.eval.quality_check import (
    FAIL,
    PASS,
    WARN,
    check_null_honesty,
    cosine,
    evaluate_paper,
    score_field,
)

# ---------------------------------------------------------------------------
# Deterministic fake embedder.
#
# Vectors are 2-D so we can dial cosine similarity precisely:
#   - "match_a" / "match_b" are identical → cosine 1.0 → PASS
#   - "orth_a" / "orth_b" are orthogonal → cosine 0.0 → FAIL
#   - "warn_a" / "warn_b" are ~45.6° apart → cosine ~0.70 → WARN
# ---------------------------------------------------------------------------
_WARN_ANGLE = math.acos(0.70)  # angle giving cosine ≈ 0.70 (between the 0.60/0.80 cuts)

_FAKE_VECTORS: dict[str, list[float]] = {
    "match_a": [1.0, 0.0],
    "match_b": [1.0, 0.0],
    "orth_a": [1.0, 0.0],
    "orth_b": [0.0, 1.0],
    "warn_a": [1.0, 0.0],
    "warn_b": [math.cos(_WARN_ANGLE), math.sin(_WARN_ANGLE)],
}


def fake_embed(text: str) -> np.ndarray:
    """Map a known marker string to its fixed vector; unknowns → [1, 0]."""
    return np.array(_FAKE_VECTORS.get(text, [1.0, 0.0]), dtype=np.float64)


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------

def test_cosine_identical_is_one():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_degenerate_vector_returns_zero():
    # Zero-norm vector must not produce NaN.
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_nonfinite_returns_zero():
    # NaN/inf components must be guarded → finite 0.0, never NaN (the exact bug
    # class fixed in ranker.py; pin it here so it can't regress).
    assert cosine([math.nan, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([math.inf, 0.0], [1.0, 0.0]) == 0.0
    assert math.isfinite(cosine([math.nan, math.nan], [math.nan, 0.0]))


# ---------------------------------------------------------------------------
# score_field
# ---------------------------------------------------------------------------

def test_score_field_identical_is_pass():
    score, status = score_field("match_a", "match_b", fake_embed)
    assert status == PASS
    assert score == 1.0


def test_score_field_orthogonal_is_fail():
    score, status = score_field("orth_a", "orth_b", fake_embed)
    assert status == FAIL
    assert score == 0.0


def test_score_field_intermediate_is_warn():
    score, status = score_field("warn_a", "warn_b", fake_embed)
    assert status == WARN
    assert 0.60 <= score < 0.80


def test_score_field_both_empty_is_pass():
    score, status = score_field(None, "", fake_embed)
    assert status == PASS
    assert score == 1.0


def test_score_field_one_empty_is_fail():
    score, status = score_field("match_a", None, fake_embed)
    assert status == FAIL
    assert score == 0.0


# ---------------------------------------------------------------------------
# check_null_honesty
# ---------------------------------------------------------------------------

def test_null_honesty_null_value_passes():
    passed, detail = check_null_honesty(None, "statistical_methods")
    assert passed is True
    assert "statistical_methods" in detail


def test_null_honesty_blank_value_passes():
    passed, _ = check_null_honesty("   ", "data_availability")
    assert passed is True


def test_null_honesty_present_value_fails():
    passed, detail = check_null_honesty("p < 0.05, ANOVA", "statistical_methods")
    assert passed is False
    assert "fabricated" in detail


# ---------------------------------------------------------------------------
# evaluate_paper
# ---------------------------------------------------------------------------

def test_evaluate_paper_mixed_rows():
    # Reference: one text field + one expected_null field.
    reference = {
        "core_finding": "match_b",
        "expected_null": ["statistical_methods"],
    }
    # Actual: text matches; the expected-null field was fabricated.
    actual = {
        "core_finding": "match_a",
        "statistical_methods": "two-way ANOVA",
    }
    rows = evaluate_paper(actual, reference, fake_embed)

    by_field = {r.field: r for r in rows}
    assert set(by_field) == {"core_finding", "statistical_methods"}

    text_row = by_field["core_finding"]
    assert text_row.kind == "text"
    assert text_row.status == PASS
    assert text_row.score == 1.0

    null_row = by_field["statistical_methods"]
    assert null_row.kind == "null"
    assert null_row.status == FAIL
    assert null_row.score is None


def test_evaluate_paper_honest_null_passes():
    reference = {"expected_null": ["software_code"]}
    actual = {"software_code": None}
    rows = evaluate_paper(actual, reference, fake_embed)
    assert len(rows) == 1
    assert rows[0].status == PASS


def test_evaluate_paper_skips_unspecified_text_fields():
    # Reference pins no text fields → no text rows emitted.
    reference = {"expected_null": []}
    actual = {"core_finding": "anything"}
    rows = evaluate_paper(actual, reference, fake_embed)
    assert rows == []
