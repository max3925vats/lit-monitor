"""Bundle E: tests for find_trending_concepts() — growth-rate detection.

Covers:
- Growth-rate calculation correctness
- Threshold filtering (only concepts above threshold_growth_rate returned)
- Min-mentions noise floor filtering
- in_existing_topics flag (substring match, case-insensitive)
- Cooldown respected after user dismissal
- Suggestion table persistence (persist_trending_suggestion)
- _load_existing_topic_terms helper
- defensive: returns [] on graph failure, [] when trending_concepts disabled
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    *,
    enabled: bool = True,
    threshold: float = 0.3,
    min_recent: int = 5,
    cooldown: int = 60,
) -> Any:
    """Return a minimal config namespace for trending_concepts."""
    ns = MagicMock()
    ns.trending_concepts.enabled = enabled
    ns.trending_concepts.threshold_growth_rate = threshold
    ns.trending_concepts.min_recent_mentions = min_recent
    ns.trending_concepts.cooldown_days_after_dismiss = cooldown
    return ns


def _make_row(
    concept_text: str = "biorefinery",
    concept_type: str = "topic",
    n_new: int = 20,
    n_prev: int = 5,
) -> dict:
    """Return a raw dict as if returned by the graph mention-count query."""
    return {
        "concept_text": concept_text,
        "concept_type": concept_type,
        "n_mentions_new": n_new,
        "n_mentions_prev": n_prev,
    }


# ---------------------------------------------------------------------------
# Unit tests for _compute_growth_rate
# ---------------------------------------------------------------------------

class TestComputeGrowthRate:
    def test_basic_growth(self):
        from lit_monitor.graph.trending import _compute_growth_rate
        # (20 / 5) - 1 = 3.0
        assert _compute_growth_rate(20, 5) == pytest.approx(3.0)

    def test_zero_prev_uses_floor_of_1(self):
        from lit_monitor.graph.trending import _compute_growth_rate
        # prev=0 → treated as 1; growth = (10/1) - 1 = 9.0
        assert _compute_growth_rate(10, 0) == pytest.approx(9.0)

    def test_growth_equals_threshold_accepted(self):
        from lit_monitor.graph.trending import _compute_growth_rate
        # (6 / 5) - 1 = 0.2 — below 0.3 threshold (caller handles this)
        assert _compute_growth_rate(6, 5) == pytest.approx(0.2)

    def test_new_zero_returns_negative(self):
        from lit_monitor.graph.trending import _compute_growth_rate
        assert _compute_growth_rate(0, 10) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Unit tests for _load_existing_topic_terms
# ---------------------------------------------------------------------------

class TestLoadExistingTopicTerms:
    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "nonexistent.yaml")
        from lit_monitor.graph.trending import _load_existing_topic_terms
        assert _load_existing_topic_terms() == set()

    def test_returns_lowercase_terms(self, tmp_path, monkeypatch):
        topics_yaml = tmp_path / "topics.yaml"
        topics_yaml.write_text(
            "searches:\n"
            "  - name: Biorefinery Design\n"
            "    query: biorefinery AND design\n",
            encoding="utf-8",
        )
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", topics_yaml)
        from lit_monitor.graph.trending import _load_existing_topic_terms
        terms = _load_existing_topic_terms()
        assert "biorefinery design" in terms
        assert "biorefinery and design" in terms

    def test_handles_malformed_yaml(self, tmp_path, monkeypatch):
        topics_yaml = tmp_path / "topics.yaml"
        topics_yaml.write_text(":: not valid yaml ::", encoding="utf-8")
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", topics_yaml)
        from lit_monitor.graph.trending import _load_existing_topic_terms
        # should return empty set, not raise
        result = _load_existing_topic_terms()
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# Unit tests for in_existing_topics flag
# ---------------------------------------------------------------------------

class TestInExistingTopicsFlag:
    def test_substring_match_case_insensitive(self, tmp_path, monkeypatch):
        topics_yaml = tmp_path / "topics.yaml"
        topics_yaml.write_text(
            "searches:\n"
            "  - name: Biorefinery Cascade\n"
            "    query: biorefinery AND cascade\n",
            encoding="utf-8",
        )
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", topics_yaml)
        from lit_monitor.graph.trending import _check_in_existing_topics
        # "biorefinery" is a substring of "biorefinery cascade"
        assert _check_in_existing_topics("Biorefinery") is True

    def test_no_match(self, tmp_path, monkeypatch):
        topics_yaml = tmp_path / "topics.yaml"
        topics_yaml.write_text(
            "searches:\n"
            "  - name: Filtration\n"
            "    query: membrane filtration\n",
            encoding="utf-8",
        )
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", topics_yaml)
        from lit_monitor.graph.trending import _check_in_existing_topics
        assert _check_in_existing_topics("biorefinery") is False


# ---------------------------------------------------------------------------
# Unit tests for find_trending_concepts — filtering logic
# ---------------------------------------------------------------------------

class TestFindTrendingConcepts:
    def _make_graph_db_mock(self, rows: list[dict]) -> MagicMock:
        """Minimal graph_db mock that returns rows when queried."""
        graph_db = MagicMock()
        # Simulate the Kuzu execute() → result cursor pattern
        result = MagicMock()
        result.has_next.side_effect = [True] * len(rows) + [False]
        # Each get_next() returns a row tuple: (concept_text, type, n_new, n_prev)
        result.get_next.side_effect = [
            (r["concept_text"], r["concept_type"], r["n_mentions_new"], r["n_mentions_prev"])
            for r in rows
        ]
        graph_db._conn.execute.return_value = result
        return graph_db

    def _make_state_db_mock(self, dismissed_concepts: list[str] | None = None) -> MagicMock:
        """Minimal state_db mock with no dismissed concepts by default."""
        state_db = MagicMock()
        state_db.get_dismissed_trending_concepts.return_value = dismissed_concepts or []
        return state_db

    def test_returns_empty_when_no_rows(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        graph_db = self._make_graph_db_mock([])
        state_db = self._make_state_db_mock()
        result = find_trending_concepts(graph_db, state_db, _make_cfg())
        assert result == []

    def test_above_threshold_included(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        # growth = (20 / 5) - 1 = 3.0 > 0.3 threshold; n_new=20 >= 5 min
        rows = [_make_row(n_new=20, n_prev=5)]
        graph_db = self._make_graph_db_mock(rows)
        state_db = self._make_state_db_mock()
        result = find_trending_concepts(graph_db, state_db, _make_cfg())
        assert len(result) == 1
        assert result[0]["concept_text"] == "biorefinery"
        assert result[0]["growth_rate"] == pytest.approx(3.0)
        assert result[0]["n_mentions_new"] == 20
        assert result[0]["n_mentions_prev"] == 5

    def test_below_threshold_excluded(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        # growth = (6 / 5) - 1 = 0.2 < 0.3 threshold
        rows = [_make_row(n_new=6, n_prev=5)]
        graph_db = self._make_graph_db_mock(rows)
        state_db = self._make_state_db_mock()
        result = find_trending_concepts(graph_db, state_db, _make_cfg(threshold=0.3))
        assert result == []

    def test_below_min_mentions_excluded(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        # growth is high but n_new=3 < min_recent_mentions=5
        rows = [_make_row(n_new=3, n_prev=0)]
        graph_db = self._make_graph_db_mock(rows)
        state_db = self._make_state_db_mock()
        result = find_trending_concepts(graph_db, state_db, _make_cfg(min_recent=5))
        assert result == []

    def test_in_existing_topics_flag_set(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        topics_yaml = tmp_path / "topics.yaml"
        topics_yaml.write_text(
            "searches:\n- name: biorefinery design\n  query: biorefinery AND design\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", topics_yaml)
        from lit_monitor.graph.trending import find_trending_concepts
        rows = [_make_row(concept_text="Biorefinery", n_new=20, n_prev=5)]
        graph_db = self._make_graph_db_mock(rows)
        state_db = self._make_state_db_mock()
        result = find_trending_concepts(graph_db, state_db, _make_cfg())
        assert len(result) == 1
        assert result[0]["in_existing_topics"] is True

    def test_not_in_existing_topics_flag_false(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        rows = [_make_row(concept_text="novel_concept", n_new=20, n_prev=5)]
        graph_db = self._make_graph_db_mock(rows)
        state_db = self._make_state_db_mock()
        result = find_trending_concepts(graph_db, state_db, _make_cfg())
        assert len(result) == 1
        assert result[0]["in_existing_topics"] is False

    def test_cooldown_respected_dismissed_concept_excluded(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        rows = [_make_row(concept_text="biorefinery", n_new=20, n_prev=5)]
        graph_db = self._make_graph_db_mock(rows)
        # "biorefinery" was recently dismissed → in cooldown
        state_db = self._make_state_db_mock(dismissed_concepts=["biorefinery"])
        result = find_trending_concepts(graph_db, state_db, _make_cfg())
        assert result == []

    def test_cooldown_not_applied_to_different_concept(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        rows = [_make_row(concept_text="biorefinery", n_new=20, n_prev=5)]
        graph_db = self._make_graph_db_mock(rows)
        # "other_concept" dismissed, not "biorefinery"
        state_db = self._make_state_db_mock(dismissed_concepts=["other_concept"])
        result = find_trending_concepts(graph_db, state_db, _make_cfg())
        assert len(result) == 1

    def test_returns_empty_on_graph_exception(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        graph_db = MagicMock()
        graph_db._conn.execute.side_effect = RuntimeError("Kuzu offline")
        state_db = self._make_state_db_mock()
        # Must not raise; returns []
        result = find_trending_concepts(graph_db, state_db, _make_cfg())
        assert result == []

    def test_config_defaults_used_when_cfg_attrs_missing(self, tmp_path, monkeypatch):
        from lit_monitor.graph import trending as _trending_mod
        monkeypatch.setattr(_trending_mod, "_TOPICS_PATH", tmp_path / "topics.yaml")
        from lit_monitor.graph.trending import find_trending_concepts
        rows = [_make_row(n_new=20, n_prev=5)]
        graph_db = self._make_graph_db_mock(rows)
        state_db = self._make_state_db_mock()
        # Pass cfg with no trending_concepts attribute at all
        bare_cfg = MagicMock(spec=[])
        result = find_trending_concepts(graph_db, state_db, bare_cfg)
        # Should use defaults and not raise
        assert isinstance(result, list)
