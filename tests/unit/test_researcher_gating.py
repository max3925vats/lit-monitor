"""Bundle E: tests for should_fire_for_researcher() — graph-neighborhood gating.

Covers:
- Returns True when overlap >= min_overlap
- Returns False when overlap < min_overlap
- Returns True (fail-safe) when graph_db is None
- Returns True (fail-safe) when recent_topic_entities is empty
- Returns True (fail-safe) on graph exception
- Returns True when min_overlap=0 (always fires)
- Handles case where researcher has no papers in graph
"""
from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_db_with_overlap(overlap_count: int) -> MagicMock:
    """Return a graph_db mock whose overlap query returns `overlap_count`."""
    graph_db = MagicMock()
    result = MagicMock()
    result.has_next.side_effect = [True, False]
    result.get_next.return_value = (overlap_count,)
    graph_db._conn.execute.return_value = result
    return graph_db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShouldFireForResearcher:
    def test_returns_true_when_overlap_meets_threshold(self):
        from scripts.search.researcher_tracker import should_fire_for_researcher

        graph_db = _make_graph_db_with_overlap(2)
        assert should_fire_for_researcher(
            "Jane Smith",
            graph_db,
            recent_topic_entities=["chromatography", "filtration"],
            min_overlap=1,
        ) is True

    def test_returns_false_when_overlap_below_threshold(self):
        from scripts.search.researcher_tracker import should_fire_for_researcher

        graph_db = _make_graph_db_with_overlap(0)
        assert should_fire_for_researcher(
            "John Doe",
            graph_db,
            recent_topic_entities=["chromatography"],
            min_overlap=1,
        ) is False

    def test_returns_true_fail_safe_when_graph_none(self):
        from scripts.search.researcher_tracker import should_fire_for_researcher

        assert should_fire_for_researcher(
            "Jane Smith",
            None,
            recent_topic_entities=["chromatography"],
            min_overlap=1,
        ) is True

    def test_returns_true_fail_safe_when_entities_empty(self):
        from scripts.search.researcher_tracker import should_fire_for_researcher

        graph_db = MagicMock()
        assert should_fire_for_researcher(
            "Jane Smith",
            graph_db,
            recent_topic_entities=[],
            min_overlap=1,
        ) is True

    def test_returns_true_fail_open_on_exception(self):
        from scripts.search.researcher_tracker import should_fire_for_researcher

        graph_db = MagicMock()
        graph_db._conn.execute.side_effect = RuntimeError("Graph offline")
        # Must not raise; fail open
        assert should_fire_for_researcher(
            "Jane Smith",
            graph_db,
            recent_topic_entities=["topic"],
            min_overlap=1,
        ) is True

    def test_min_overlap_zero_always_fires(self):
        from scripts.search.researcher_tracker import should_fire_for_researcher

        graph_db = _make_graph_db_with_overlap(0)
        assert should_fire_for_researcher(
            "Jane Smith",
            graph_db,
            recent_topic_entities=["topic"],
            min_overlap=0,
        ) is True

    def test_meets_exact_threshold(self):
        from scripts.search.researcher_tracker import should_fire_for_researcher

        graph_db = _make_graph_db_with_overlap(1)
        assert should_fire_for_researcher(
            "Jane Smith",
            graph_db,
            recent_topic_entities=["topic"],
            min_overlap=1,
        ) is True
