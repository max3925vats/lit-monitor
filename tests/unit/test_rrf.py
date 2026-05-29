"""G8: Reciprocal Rank Fusion tests."""
from __future__ import annotations

import pytest

from scripts.retrieval import reciprocal_rank_fusion as rrf


class TestRRFBasic:
    def test_identical_rankings_preserves_order(self):
        """G8: when both rankings are the same, the order is preserved."""
        result = rrf([["a", "b", "c"], ["a", "b", "c"]])
        dois = [doi for doi, _ in result]
        assert dois == ["a", "b", "c"]
        # Scores monotonically decrease
        scores = [s for _, s in result]
        assert scores[0] > scores[1] > scores[2]

    def test_disjoint_rankings_interleave(self):
        """G8: disjoint rankings interleave by rank position."""
        # Both have same rank-1 contributions: 1/(60+1) = 1/61 ≈ 0.0164
        # rank-2: 1/62, rank-3: 1/63
        result = rrf([["a", "b", "c"], ["x", "y", "z"]])
        dois = [doi for doi, _ in result]
        # Expect interleaving: a (rank 1) vs x (rank 1) — stable tiebreak goes to first seen
        assert dois[0] == "a"  # appears first in input order
        assert dois[1] == "x"
        assert dois[2] == "b"
        assert dois[3] == "y"

    def test_overlap_with_stable_tiebreak(self):
        """G8: overlapping docs get summed; ties broken by first-appearance order."""
        # 'b' appears at rank 2 in both: score = 2/(60+2) = 2/62
        # 'a' appears at rank 1 only in list 1: score = 1/61
        # 'c' appears at rank 1 only in list 2: score = 1/61
        # 'a' first-seen index 0, 'c' first-seen index 1
        # Expected order: b (highest score), a (tied with c, first seen), c
        result = rrf([["a", "b"], ["c", "b"]])
        dois = [doi for doi, _ in result]
        assert dois[0] == "b"
        # Tie between a (1/61) and c (1/61) → stable tiebreak by first appearance
        assert dois[1] == "a"
        assert dois[2] == "c"

    def test_missing_docs_contribute_zero(self):
        """G8: a doc absent from a ranking contributes 0 from that ranking (not 1/k)."""
        # 'a' is in list 1 only: total score = 1/(60+1) = 1/61 ≈ 0.0164
        # If missing contributed 1/k = 1/60, score would be 1/61 + 1/60 ≈ 0.0331
        result = rrf([["a"], ["b"]])
        a_score = next(s for doi, s in result if doi == "a")
        # If correct: a_score is exactly 1/(60+1)
        assert a_score == pytest.approx(1.0 / 61.0)

    def test_k_parameter_affects_ordering(self):
        """G8: changing k changes the score, though usually not the order."""
        result_default = rrf([["a", "b", "c"]])
        result_k10 = rrf([["a", "b", "c"]], k=10)
        # Smaller k → higher absolute scores
        assert result_k10[0][1] > result_default[0][1]
        # Order should still be a > b > c
        assert [doi for doi, _ in result_k10] == ["a", "b", "c"]

    def test_empty_input_returns_empty(self):
        assert rrf([]) == []

    def test_empty_ranking_in_input_is_skipped(self):
        """G8: an empty list among rankings doesn't crash."""
        result = rrf([["a", "b"], [], ["c"]])
        dois = [doi for doi, _ in result]
        # a and c each have one rank-1 contribution; tiebreak by first appearance
        assert dois[0] == "a"  # rank 1 in list 1, first seen
        # c is rank 1 in list 3 (after empty list 2)

    def test_single_ranking_passes_through(self):
        """G8: a single ranking with N items returns N items in order."""
        result = rrf([["a", "b", "c"]])
        dois = [doi for doi, _ in result]
        assert dois == ["a", "b", "c"]


class TestRRFFormulaCorrectness:
    def test_score_formula(self):
        """G8: explicit formula check on a tiny example."""
        # Doc 'a' at rank 1 in two rankings: score = 1/(60+1) + 1/(60+1) = 2/61
        # Doc 'b' at rank 2 in both: score = 2 * 1/62 = 2/62
        result = rrf([["a", "b"], ["a", "b"]])
        a_score = next(s for doi, s in result if doi == "a")
        b_score = next(s for doi, s in result if doi == "b")
        assert a_score == pytest.approx(2.0 / 61.0)
        assert b_score == pytest.approx(2.0 / 62.0)
        # a > b
        assert a_score > b_score
