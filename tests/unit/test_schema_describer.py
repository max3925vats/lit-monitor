"""A1: schema describer tests.

Verifies that describe_schema(graph_db) produces well-formed markdown from a
live KuzuDB instance, that nodes appear before RELs, that all current tables
are represented, that caching works correctly, and that GraphDB.add_paper
invalidates the cache.
"""
from __future__ import annotations

import pytest

from scripts.graph import GraphDB
from scripts.graph.schema_describer import describe_schema, invalidate_schema_cache


@pytest.fixture
def basic_db(tmp_path):
    """Fresh GraphDB with the full schema (Paper, Entity, 10 REL tables)."""
    return GraphDB(persist_dir=str(tmp_path / "a1.kuzu"))


@pytest.mark.unit
class TestDescribeSchemaShape:
    def test_returns_string(self, basic_db):
        result = describe_schema(basic_db)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_paper_node(self, basic_db):
        result = describe_schema(basic_db)
        assert "## Node: Paper" in result

    def test_includes_entity_node(self, basic_db):
        result = describe_schema(basic_db)
        assert "## Node: Entity" in result

    def test_includes_mentions_rel(self, basic_db):
        result = describe_schema(basic_db)
        assert "MENTIONS" in result
        # FROM → TO label
        assert "Paper" in result and "Entity" in result

    def test_includes_typed_predicates(self, basic_db):
        result = describe_schema(basic_db)
        for pred in (
            "CITES",
            "COMPARES_TO",
            "DEPENDS_ON",
            "PROPOSES",
            "LIMITED_BY",
            "INTRODUCES",
            "RAISES_QUESTION",
        ):
            assert pred in result, f"predicate {pred} missing from schema text"

    def test_includes_extends_contradicts_if_present(self, basic_db):
        """R1 added these; describer reports whatever tables show_tables returns."""
        result = describe_schema(basic_db)
        # Fresh DB has all 10 REL tables per current migrations
        assert "EXTENDS" in result
        assert "CONTRADICTS" in result

    def test_nodes_come_before_rels(self, basic_db):
        result = describe_schema(basic_db)
        node_pos = result.find("## Node: Paper")
        rel_pos = result.find("## REL:")
        assert node_pos < rel_pos

    def test_rel_heading_includes_from_to_labels(self, basic_db):
        """A1: REL heading shows FROM → TO table names."""
        result = describe_schema(basic_db)
        # MENTIONS is Paper→Entity; look for the arrow syntax
        assert "→" in result
        # At minimum one REL block contains both node names around an arrow
        assert "Paper → Entity" in result or "Paper →" in result


@pytest.mark.unit
class TestSchemaCache:
    def test_cache_hit_on_second_call(self, basic_db, monkeypatch):
        """A1: second call uses cache (no second show_tables call)."""
        call_count = {"n": 0}
        orig_execute = basic_db._conn.execute

        def counting_execute(query, params=None):
            if "show_tables" in str(query).lower():
                call_count["n"] += 1
            if params is not None:
                return orig_execute(query, params)
            return orig_execute(query)

        monkeypatch.setattr(basic_db._conn, "execute", counting_execute)

        # Clear any prior cache
        invalidate_schema_cache(basic_db)

        result1 = describe_schema(basic_db)
        n_after_first = call_count["n"]

        result2 = describe_schema(basic_db)
        n_after_second = call_count["n"]

        assert result1 == result2
        # Cached → no additional show_tables call
        assert n_after_second == n_after_first

    def test_invalidate_forces_rebuild(self, basic_db):
        """A1: invalidate_schema_cache forces the next call to re-query."""
        first = describe_schema(basic_db)
        invalidate_schema_cache(basic_db)
        # Just confirm second call doesn't crash and returns equivalent shape
        second = describe_schema(basic_db)
        assert isinstance(second, str)
        # Schema is unchanged → result text matches
        assert first == second

    def test_invalidate_called_after_add_paper(self, basic_db):
        """A1: GraphDB.add_paper invalidates the cache so subsequent describes are fresh."""
        import scripts.graph.schema_describer as sd_mod

        describe_schema(basic_db)
        # Cache populated after first call
        assert id(basic_db) in sd_mod._CACHE

        basic_db.add_paper(
            doi="10.0/x",
            entities=[],
            relationships=[],
            paper_metadata={"title": "X", "year": 2024, "journal": "X"},
        )
        # Cache should be invalidated
        assert id(basic_db) not in sd_mod._CACHE
