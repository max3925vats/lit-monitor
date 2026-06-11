"""Bundle E: tests for suggest_expansions() — top-k co-occurring entity lookup.

Covers:
- Returns top-k co-occurring entities for a given topic query string
- Respects top_k limit
- Returns empty list when no entity found for the query
- Returns empty list on graph failure (defensive perimeter)
- Excludes the topic entity itself from results
"""
from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_db_with_entity_and_coentities(
    entity_cid: str | None,
    co_entities: list[tuple[str, int]],  # list of (canonical_id, co_count)
    *,
    top_k: int | None = None,
) -> MagicMock:
    """Mock graph_db that:
    1. Returns `entity_cid` when queried for the topic entity resolution.
    2. Returns `co_entities` (truncated to top_k if provided) for co-occurring entities.

    The real Kuzu query applies LIMIT $k server-side; the mock simulates that by
    truncating co_entities to top_k before building the cursor side_effect.
    """
    graph_db = MagicMock()

    # First execute call: resolve topic query → entity canonical_id
    if entity_cid is None:
        # No entity found
        entity_result = MagicMock()
        entity_result.has_next.return_value = False
        graph_db._conn.execute.return_value = entity_result
    else:
        # Entity found, then co-entities result
        entity_result = MagicMock()
        entity_result.has_next.side_effect = [True, False]
        entity_result.get_next.return_value = (entity_cid,)

        # Simulate LIMIT $k: truncate list to top_k when provided
        limited = co_entities[:top_k] if top_k is not None else co_entities

        co_result = MagicMock()
        co_result.has_next.side_effect = [True] * len(limited) + [False]
        co_result.get_next.side_effect = [(cid, cnt) for cid, cnt in limited]

        graph_db._conn.execute.side_effect = [entity_result, co_result]

    return graph_db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSuggestExpansions:
    def test_returns_top_k_co_entities(self):
        from lit_monitor.graph.query_expansion import suggest_expansions

        co_entities = [
            ("entity_a", 50),
            ("entity_b", 40),
            ("entity_c", 30),
            ("entity_d", 20),
        ]
        # Pass top_k=3 to mock so it simulates Kuzu's LIMIT $k behaviour
        graph_db = _make_graph_db_with_entity_and_coentities("topic_entity", co_entities, top_k=3)
        result = suggest_expansions("myopic distillation", graph_db, top_k=3)
        assert result == ["entity_a", "entity_b", "entity_c"]

    def test_returns_all_if_fewer_than_top_k(self):
        from lit_monitor.graph.query_expansion import suggest_expansions

        co_entities = [("entity_a", 10), ("entity_b", 5)]
        graph_db = _make_graph_db_with_entity_and_coentities("topic_entity", co_entities)
        result = suggest_expansions("myopic distillation", graph_db, top_k=5)
        assert result == ["entity_a", "entity_b"]

    def test_returns_empty_when_no_entity_found(self):
        from lit_monitor.graph.query_expansion import suggest_expansions

        graph_db = _make_graph_db_with_entity_and_coentities(None, [])
        result = suggest_expansions("unknown topic", graph_db, top_k=3)
        assert result == []

    def test_returns_empty_on_graph_exception(self):
        from lit_monitor.graph.query_expansion import suggest_expansions

        graph_db = MagicMock()
        graph_db._conn.execute.side_effect = RuntimeError("Kuzu down")
        # Must not raise; defensive perimeter
        result = suggest_expansions("any topic", graph_db, top_k=3)
        assert result == []

    def test_default_top_k_is_3(self):
        from lit_monitor.graph.query_expansion import suggest_expansions

        co_entities = [("a", 10), ("b", 9), ("c", 8), ("d", 7)]
        # Simulate LIMIT 3 (the default top_k) applied by Kuzu
        graph_db = _make_graph_db_with_entity_and_coentities("ent", co_entities, top_k=3)
        result = suggest_expansions("topic", graph_db)
        assert len(result) == 3

    def test_returns_empty_for_empty_co_entities(self):
        from lit_monitor.graph.query_expansion import suggest_expansions

        graph_db = _make_graph_db_with_entity_and_coentities("topic_entity", [])
        result = suggest_expansions("filtration", graph_db, top_k=3)
        assert result == []
