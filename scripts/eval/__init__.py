"""Idea 4: Tier-4 extraction quality auto-regression.

Pure, testable scoring helpers that compare a paper's live state-DB
extraction against the hand-written expected values in
``docs/internal/reference_papers.yaml``. Used by the Tier-4 section of
``scripts/test_everything.sh`` as an ADVISORY signal — it never changes the
script's exit code.
"""
from __future__ import annotations

from .quality_check import (
    FAIL,
    PASS,
    PASS_THRESHOLD,
    WARN,
    WARN_THRESHOLD,
    FieldResult,
    check_null_honesty,
    cosine,
    evaluate_paper,
    score_field,
)

__all__ = [
    "FAIL",
    "PASS",
    "WARN",
    "PASS_THRESHOLD",
    "WARN_THRESHOLD",
    "FieldResult",
    "check_null_honesty",
    "cosine",
    "evaluate_paper",
    "score_field",
]
