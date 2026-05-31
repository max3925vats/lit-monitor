"""B2: high-level MCP tool tests.

Covers:
- Validation errors (bad DOI, unknown predicate, unknown entity_type)
- Happy-path shape for all 8 tools via mocked GraphDB
- JSON-serialisability of all return values
- top_k truncation
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.mcp import tools

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_db():
    """Mock GraphDB exposing the methods called by B2 tools."""
    db = MagicMock()

    # resolve_query_entity: return the same string as canonical_id
    db.resolve_query_entity.side_effect = lambda text, type_=None: text

    # find_papers_by_entities: return two (doi, score) tuples
    db.find_papers_by_entities.return_value = [
        ("10.0/a", 2.0),
        ("10.0/b", 1.0),
    ]

    # Paper metadata lookup via _conn.execute (used in find_papers_by_relationship
    # and get_paper_details via get_paper_snapshot)
    mock_cursor = MagicMock()
    mock_cursor.has_next.side_effect = [True, False]  # single row
    mock_cursor.get_next.return_value = ["Test Paper", 2024, "Nature", "PROPOSES"]
    db._conn.execute.return_value = mock_cursor

    return db


def _make_cursor(*rows):
    """Helper: build a mock KuzuDB cursor that emits ``rows`` then stops."""
    cursor = MagicMock()
    has_next_seq = [True] * len(rows) + [False]
    cursor.has_next.side_effect = has_next_seq
    cursor.get_next.side_effect = list(rows)
    return cursor


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_get_paper_details_rejects_bad_doi(self, fake_db):
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with pytest.raises(ValueError, match="DOI"):
                tools.get_paper_details("not-a-doi")

    def test_get_paper_details_rejects_url(self, fake_db):
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with pytest.raises(ValueError, match="DOI"):
                tools.get_paper_details("https://doi.org/10.1234/x")

    def test_find_by_relationship_rejects_unknown_predicate(self, fake_db):
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with pytest.raises(ValueError, match="predicate"):
                tools.find_papers_by_relationship("DOES_NOT_EXIST")

    def test_find_by_relationship_rejects_misspelled_predicate(self, fake_db):
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with pytest.raises(ValueError, match="predicate"):
                tools.find_papers_by_relationship("PROPOSES_TO")

    def test_list_entities_rejects_unknown_type(self, fake_db):
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with pytest.raises(ValueError, match="entity_type"):
                tools.list_entities_by_type("not-a-type")

    def test_list_entities_rejects_capitalised_type(self, fake_db):
        # Vocab is lowercase; "Method" should fail
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with pytest.raises(ValueError, match="entity_type"):
                tools.list_entities_by_type("Method")

    def test_find_papers_by_entity_empty_string_raises(self, fake_db):
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with pytest.raises(ValueError, match="entity"):
                tools.find_papers_by_entity("")


# ---------------------------------------------------------------------------
# Happy-path tests — find_papers_by_entity
# ---------------------------------------------------------------------------


class TestFindPapersByEntity:
    def _make_paper_cursor(self, rows):
        """Cursor that emits one row per MATCH iteration (used in _enrich_papers)."""
        cursor = MagicMock()
        has_seq = [True] * len(rows) + [False]
        cursor.has_next.side_effect = has_seq
        cursor.get_next.side_effect = list(rows)
        return cursor

    def test_returns_list(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_entity("ion exchange")
        assert isinstance(result, list)

    def test_calls_resolve_then_find(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            tools.find_papers_by_entity("ion exchange", top_k=5)
        fake_db.resolve_query_entity.assert_called_once_with("ion exchange", type_=None)
        fake_db.find_papers_by_entities.assert_called_once()

    def test_unresolved_entity_returns_empty(self, fake_db):
        """When resolve_query_entity returns None, tool returns [] without querying."""
        # Override both side_effect AND return_value so None is actually returned.
        fake_db.resolve_query_entity.side_effect = None
        fake_db.resolve_query_entity.return_value = None
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_entity("zzznomatch")
        assert result == []
        fake_db.find_papers_by_entities.assert_not_called()

    def test_paper_shape(self, fake_db):
        """Each item must have at least doi key."""
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_entity("ion exchange")
        for item in result:
            assert "doi" in item

    def test_json_serializable(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_entity("ion exchange")
        json.dumps(result)  # must not raise

    def test_respects_top_k(self, fake_db):
        # find_papers_by_entities returns 2 pairs; top_k=1 should pass k=1 downstream
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            tools.find_papers_by_entity("ion exchange", top_k=1)
        # k=1 was forwarded
        _, kwargs = fake_db.find_papers_by_entities.call_args
        assert kwargs.get("k") == 1 or fake_db.find_papers_by_entities.call_args[0][1] == 1


# ---------------------------------------------------------------------------
# Happy-path tests — find_papers_by_relationship
# ---------------------------------------------------------------------------


class TestFindPapersByRelationship:
    def test_returns_list(self, fake_db):
        cursor = _make_cursor(["10.0/a", "Test Paper", 2024, "Nature"])
        fake_db._conn.execute.return_value = cursor
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_relationship("PROPOSES")
        assert isinstance(result, list)

    def test_paper_shape(self, fake_db):
        cursor = _make_cursor(["10.0/a", "Test Paper", 2024, "Nature"])
        fake_db._conn.execute.return_value = cursor
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_relationship("MENTIONS")
        for item in result:
            assert "doi" in item

    def test_json_serializable(self, fake_db):
        cursor = _make_cursor(["10.0/a", "Test Paper", 2024, "Nature"])
        fake_db._conn.execute.return_value = cursor
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_relationship("CITES")
        json.dumps(result)

    def test_all_valid_predicates_accepted(self, fake_db):
        """Every predicate in VALID_PREDICATES should pass validation."""
        from scripts.graph.relationship_validator import VALID_PREDICATES

        for pred in VALID_PREDICATES:
            cursor = _make_cursor(["10.0/a", "Test Paper", 2024, "Nature"])
            fake_db._conn.execute.return_value = cursor
            with patch.object(tools, "_get_graph_db", return_value=fake_db):
                result = tools.find_papers_by_relationship(pred)
            assert isinstance(result, list), f"predicate {pred!r} should be accepted"


# ---------------------------------------------------------------------------
# Happy-path tests — get_paper_details
# ---------------------------------------------------------------------------


class TestGetPaperDetails:
    def test_returns_required_keys(self, fake_db):
        # get_paper_details delegates to queries.get_paper_snapshot, which
        # has its own Cypher queries; mock _conn.execute to return empty cursors
        empty = _make_cursor()
        fake_db._conn.execute.return_value = empty
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.get_paper_details("10.1234/test")
        assert "metadata" in result
        assert "entities_by_type" in result
        assert "relationships_in" in result
        assert "relationships_out" in result

    def test_json_serializable(self, fake_db):
        empty = _make_cursor()
        fake_db._conn.execute.return_value = empty
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.get_paper_details("10.1234/test")
        json.dumps(result)

    def test_valid_doi_accepted(self, fake_db):
        empty = _make_cursor()
        fake_db._conn.execute.return_value = empty
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.get_paper_details("10.1038/nature12345")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Happy-path tests — get_corpus_stats
# ---------------------------------------------------------------------------


class TestGetCorpusStats:
    def test_returns_dict(self, fake_db):
        # Provide enough cursors for paper_count, entity_count, + 11 predicates
        def make_count_cursor(n):
            c = MagicMock()
            c.has_next.return_value = True
            c.get_next.return_value = [n]
            return c

        fake_db._conn.execute.side_effect = lambda q, *a, **kw: make_count_cursor(0)
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.get_corpus_stats()
        assert isinstance(result, dict)

    def test_required_keys(self, fake_db):
        def make_count_cursor(n):
            c = MagicMock()
            c.has_next.return_value = True
            c.get_next.return_value = [n]
            return c

        fake_db._conn.execute.side_effect = lambda q, *a, **kw: make_count_cursor(42)
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.get_corpus_stats()
        assert "paper_count" in result
        assert "entity_count" in result

    def test_json_serializable(self, fake_db):
        def make_count_cursor(n):
            c = MagicMock()
            c.has_next.return_value = True
            c.get_next.return_value = [n]
            return c

        fake_db._conn.execute.side_effect = lambda q, *a, **kw: make_count_cursor(0)
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.get_corpus_stats()
        json.dumps(result)


# ---------------------------------------------------------------------------
# Happy-path tests — list_entities_by_type
# ---------------------------------------------------------------------------


class TestListEntitiesByType:
    def test_returns_list(self, fake_db):
        cursor = _make_cursor(
            ["ion exchange", "method", "ion exchange", 5]
        )
        fake_db._conn.execute.return_value = cursor
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.list_entities_by_type("method")
        assert isinstance(result, list)

    def test_entity_shape(self, fake_db):
        """Items must include canonical key (mapped from canonical_id)."""
        cursor = _make_cursor(
            ["ion exchange", "method", "ion exchange", 5]
        )
        fake_db._conn.execute.return_value = cursor
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.list_entities_by_type("method")
        for item in result:
            # Spec: MCP-facing key is 'canonical', not 'canonical_id'
            assert "canonical" in item
            assert "type" in item
            assert "mention_count" in item

    def test_json_serializable(self, fake_db):
        cursor = _make_cursor(
            ["ion exchange", "method", "ion exchange", 5]
        )
        fake_db._conn.execute.return_value = cursor
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.list_entities_by_type("method")
        json.dumps(result)

    def test_all_valid_types_accepted(self, fake_db):
        """Every type in _ENTITY_TYPES should pass validation."""
        for etype in tools._ENTITY_TYPES:
            empty = _make_cursor()
            fake_db._conn.execute.return_value = empty
            with patch.object(tools, "_get_graph_db", return_value=fake_db):
                result = tools.list_entities_by_type(etype)
            assert isinstance(result, list), f"type {etype!r} should be accepted"


# ---------------------------------------------------------------------------
# Happy-path tests — get_schema
# ---------------------------------------------------------------------------


class TestGetSchema:
    def test_returns_string(self, fake_db):
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with patch("scripts.mcp.tools._get_schema_text_impl", return_value="# Schema"):
                result = tools.get_schema()
        assert isinstance(result, str)

    def test_delegates_to_schema_impl(self, fake_db):
        """get_schema must call the underlying schema describer."""
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            with patch("scripts.mcp.tools._get_schema_text_impl", return_value="ok") as mock_impl:
                tools.get_schema()
        mock_impl.assert_called_once_with(fake_db)


# ---------------------------------------------------------------------------
# Happy-path tests — find_papers_by_query (graph mode)
# ---------------------------------------------------------------------------


class TestFindPapersByQuery:
    def test_returns_list(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_query("protein folding")
        assert isinstance(result, list)

    def test_no_entity_match_returns_empty(self, fake_db):
        # Clear side_effect so return_value=None takes effect.
        fake_db.resolve_query_entity.side_effect = None
        fake_db.resolve_query_entity.return_value = None
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_query("zzznomatch")
        assert result == []

    def test_paper_shape(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_query("ion exchange")
        for item in result:
            assert "doi" in item

    def test_json_serializable(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_query("ion exchange")
        json.dumps(result)


# ---------------------------------------------------------------------------
# Happy-path tests — find_papers_by_query_hybrid
# ---------------------------------------------------------------------------


class TestFindPapersByQueryHybrid:
    def test_returns_list(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_query_hybrid("chromatography")
        assert isinstance(result, list)

    def test_json_serializable(self, fake_db):
        fake_db._conn.execute.return_value = _make_cursor(
            ["Test Paper", 2024, "Nature"]
        )
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_query_hybrid("chromatography")
        json.dumps(result)

    def test_no_entity_match_returns_empty(self, fake_db):
        # Clear side_effect so return_value=None takes effect.
        fake_db.resolve_query_entity.side_effect = None
        fake_db.resolve_query_entity.return_value = None
        with patch.object(tools, "_get_graph_db", return_value=fake_db):
            result = tools.find_papers_by_query_hybrid("zzznomatch")
        assert result == []
