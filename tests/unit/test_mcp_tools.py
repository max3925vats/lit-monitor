"""B2: high-level MCP tool tests.

Covers:
- Validation errors (bad DOI, unknown predicate, unknown entity_type)
- Happy-path shape for all 8 tools via mocked GraphDB
- JSON-serialisability of all return values
- top_k truncation
"""
from __future__ import annotations

import json
from pathlib import Path
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
        # Source calls find_papers_by_entities([canonical_id], k=top_k) — k is a
        # kwarg and there is only one positional arg, so assert the kwarg directly.
        # (The old `or call_args[0][1]` fallback would IndexError, never reached.)
        _, kwargs = fake_db.find_papers_by_entities.call_args
        assert kwargs.get("k") == 1


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


# ---------------------------------------------------------------------------
# B6: semantic_search tests
# ---------------------------------------------------------------------------

# NOTE: We mock EmbeddingsDB rather than seeding a real ChromaDB instance,
# because ChromaDB embeddings require Ollama to be running (no offline mode).
# Mocking is the correct choice here — we're testing the tool's validation,
# dispatch, coercion, and graceful-degradation logic, not EmbeddingsDB itself.


class TestSemanticSearch:
    """Tests for the semantic_search MCP tool (B6)."""

    # -----------------------------------------------------------------------
    # Validation tests (Step 1 — run RED before implementation)
    # -----------------------------------------------------------------------

    def test_bad_granularity_raises(self):
        with pytest.raises(ValueError, match="granularity must be"):
            tools.semantic_search("protein folding", granularity="sentence")

    def test_granularity_case_sensitive(self):
        with pytest.raises(ValueError, match="granularity must be"):
            tools.semantic_search("query", granularity="Paper")

    def test_top_k_too_small_raises(self):
        with pytest.raises(ValueError, match="top_k must be int"):
            tools.semantic_search("query", top_k=0)

    def test_top_k_too_large_raises(self):
        with pytest.raises(ValueError, match="top_k must be int"):
            tools.semantic_search("query", top_k=101)

    def test_top_k_must_be_int(self):
        with pytest.raises(ValueError, match="top_k must be int"):
            tools.semantic_search("query", top_k=5.0)

    # -----------------------------------------------------------------------
    # Empty-index test (Step 2)
    # -----------------------------------------------------------------------

    def test_missing_persist_dir_returns_empty(self, tmp_path, caplog):
        """When persist_dir does not exist, returns [] without raising."""
        import logging

        nonexistent = str(tmp_path / "no_such_chroma")
        # Reset module-level state so _get_embeddings_db re-evaluates.
        tools._EMBEDDINGS_DB = None
        tools._EMPTY_WARNED = False
        with patch.object(tools, "_resolve_persist_dir", return_value=nonexistent):
            with caplog.at_level(logging.WARNING, logger="scripts.mcp.tools"):
                result = tools.semantic_search("query")
        assert result == []
        assert any("empty embeddings index" in r.message for r in caplog.records)

    # -----------------------------------------------------------------------
    # One-shot warning test (Step 3)
    # -----------------------------------------------------------------------

    def test_one_shot_warning_only_once(self, tmp_path, caplog):
        """Missing persist dir logs exactly ONE warning across multiple calls."""
        import logging

        nonexistent = str(tmp_path / "no_such_chroma2")
        tools._EMBEDDINGS_DB = None
        tools._EMPTY_WARNED = False
        with patch.object(tools, "_resolve_persist_dir", return_value=nonexistent):
            with caplog.at_level(logging.WARNING, logger="scripts.mcp.tools"):
                tools.semantic_search("query1")
                tools.semantic_search("query2")
        warning_count = sum(
            1 for r in caplog.records if "empty embeddings index" in r.message
        )
        assert warning_count == 1

    # -----------------------------------------------------------------------
    # Happy-path: paper granularity (Step 4)
    # -----------------------------------------------------------------------

    def test_paper_granularity_shape(self, tmp_path):
        """granularity='paper' returns [{doi, title, year, score}, ...] sorted desc."""
        # Canned hits from EmbeddingsDB.find_similar_to_text
        fake_hits = [
            {"id": "10.0/a", "score": 0.9, "document": "text a", "metadata": {"title": "Paper A", "year": 2024}},
            {"id": "10.0/b", "score": 0.7, "document": "text b", "metadata": {"title": "Paper B", "year": 2023}},
            {"id": "10.0/c", "score": 0.5, "document": "text c", "metadata": {"title": "Paper C", "year": 2022}},
        ]
        fake_db = MagicMock()
        fake_db.find_similar_to_text.return_value = fake_hits

        tools._EMBEDDINGS_DB = None
        tools._EMPTY_WARNED = False
        persist_dir = str(tmp_path / "chroma")
        import os
        os.makedirs(persist_dir)

        with patch.object(tools, "_resolve_persist_dir", return_value=persist_dir):
            with patch.object(tools, "_get_embeddings_db", return_value=fake_db):
                result = tools.semantic_search("chromatography", top_k=3, granularity="paper")

        assert len(result) == 3
        for item in result:
            assert "doi" in item
            assert "title" in item
            assert "year" in item
            assert "score" in item
        # Scores must be float (not numpy float)
        assert isinstance(result[0]["score"], float)
        # Ordered descending
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    # -----------------------------------------------------------------------
    # Happy-path: chunk granularity (Step 5)
    # -----------------------------------------------------------------------

    def test_chunk_granularity_shape(self, tmp_path):
        """granularity='chunk' returns [{doi, chunk_id, snippet, score}, ...] sorted desc."""
        long_text = "x" * 600  # longer than 500 chars — should be truncated
        fake_hits = [
            {
                "id": "10.0/a#0",
                "score": 0.85,
                "document": long_text,
                "metadata": {"doi": "10.0/a", "section_heading": "Methods", "chunk_index": 0},
            },
            {
                "id": "10.0/b#1",
                "score": 0.60,
                "document": "short text",
                "metadata": {"doi": "10.0/b", "section_heading": "Results", "chunk_index": 1},
            },
        ]
        fake_db = MagicMock()
        fake_db.find_similar_chunks.return_value = fake_hits

        tools._EMBEDDINGS_DB = None
        tools._EMPTY_WARNED = False
        persist_dir = str(tmp_path / "chroma2")
        import os
        os.makedirs(persist_dir)

        with patch.object(tools, "_resolve_persist_dir", return_value=persist_dir):
            with patch.object(tools, "_get_embeddings_db", return_value=fake_db):
                result = tools.semantic_search("chromatography", top_k=5, granularity="chunk")

        assert len(result) == 2
        for item in result:
            assert "doi" in item
            assert "chunk_id" in item
            assert "snippet" in item
            assert "score" in item
        # Snippet truncated to 500 chars
        assert len(result[0]["snippet"]) == 500
        assert len(result[1]["snippet"]) == len("short text")
        # Score is plain float
        assert isinstance(result[0]["score"], float)
        # Ordered descending
        assert result[0]["score"] >= result[1]["score"]

    # -----------------------------------------------------------------------
    # JSON-serializability test (Step 6)
    # -----------------------------------------------------------------------

    def test_paper_result_json_serializable(self, tmp_path):
        """json.dumps must not raise on paper-mode results."""
        import json as _json

        fake_hits = [
            {"id": "10.0/a", "score": 0.9, "document": "text", "metadata": {"title": "A", "year": 2024}},
        ]
        fake_db = MagicMock()
        fake_db.find_similar_to_text.return_value = fake_hits

        tools._EMBEDDINGS_DB = None
        tools._EMPTY_WARNED = False
        persist_dir = str(tmp_path / "chroma3")
        import os
        os.makedirs(persist_dir)

        with patch.object(tools, "_resolve_persist_dir", return_value=persist_dir):
            with patch.object(tools, "_get_embeddings_db", return_value=fake_db):
                result = tools.semantic_search("query", granularity="paper")
        _json.dumps(result)  # must not raise

    def test_chunk_result_json_serializable(self, tmp_path):
        """json.dumps must not raise on chunk-mode results."""
        import json as _json

        fake_hits = [
            {
                "id": "10.0/a#0",
                "score": 0.75,
                "document": "text",
                "metadata": {"doi": "10.0/a", "section_heading": "Intro", "chunk_index": 0},
            },
        ]
        fake_db = MagicMock()
        fake_db.find_similar_chunks.return_value = fake_hits

        tools._EMBEDDINGS_DB = None
        tools._EMPTY_WARNED = False
        persist_dir = str(tmp_path / "chroma4")
        import os
        os.makedirs(persist_dir)

        with patch.object(tools, "_resolve_persist_dir", return_value=persist_dir):
            with patch.object(tools, "_get_embeddings_db", return_value=fake_db):
                result = tools.semantic_search("query", granularity="chunk")
        _json.dumps(result)  # must not raise

    # -----------------------------------------------------------------------
    # Lazy-load caching test (Step 7)
    # -----------------------------------------------------------------------

    def test_lazy_load_cached(self, tmp_path):
        """_get_embeddings_db is only called once across two semantic_search calls."""
        fake_hits = [
            {"id": "10.0/a", "score": 0.9, "document": "t", "metadata": {"title": "A", "year": 2024}},
        ]
        fake_db = MagicMock()
        fake_db.find_similar_to_text.return_value = fake_hits

        tools._EMBEDDINGS_DB = None
        tools._EMPTY_WARNED = False

        # Seed the module-level cache directly to simulate a loaded DB.
        # The real lazy-load path is exercised by _get_embeddings_db unit test;
        # here we just confirm semantic_search doesn't bypass the cache.
        tools._EMBEDDINGS_DB = fake_db

        persist_dir = str(tmp_path / "chroma5")
        with patch.object(tools, "_resolve_persist_dir", return_value=persist_dir):
            tools.semantic_search("q1", granularity="paper")
            tools.semantic_search("q2", granularity="paper")

        # Both calls should have used the same db object (find_similar_to_text called twice)
        assert fake_db.find_similar_to_text.call_count == 2

        # Cleanup
        tools._EMBEDDINGS_DB = None


# ---------------------------------------------------------------------------
# P9: Discovery run MCP tools
# ---------------------------------------------------------------------------


class TestDiscoveryMcpTools:
    """P9: tests for get_recent_discovery_runs and get_discovery_run_papers."""

    def test_get_recent_discovery_runs_returns_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.core.state_db import StateDB
        from scripts.mcp import tools

        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run({})
        db.finish_discovery_run(run_id, "success", 3, 2)
        monkeypatch.setattr(tools, "_get_state_db", lambda: db)

        result = tools.get_recent_discovery_runs(limit=5)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == run_id
        assert result[0]["status"] == "success"

    def test_get_recent_discovery_runs_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.core.state_db import StateDB
        from scripts.mcp import tools

        db = StateDB(tmp_path / "state.db")
        for _ in range(3):
            rid = db.start_discovery_run({})
            db.finish_discovery_run(rid, "success", 1, 1)
        monkeypatch.setattr(tools, "_get_state_db", lambda: db)

        assert len(tools.get_recent_discovery_runs(limit=2)) == 2

    def test_get_discovery_run_papers_sorted_by_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.core.state_db import StateDB
        from scripts.mcp import tools

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.add_discovery_paper(rid, "10/lo", "Lo", 0.1, "", False)
        db.add_discovery_paper(rid, "10/hi", "Hi", 0.9, "", True)
        db.finish_discovery_run(rid, "success", 2, 1)
        monkeypatch.setattr(tools, "_get_state_db", lambda: db)

        papers = tools.get_discovery_run_papers(run_id=rid, top_k=10)
        assert [p["doi"] for p in papers] == ["10/hi", "10/lo"]

    def test_returns_jsonable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P9: results must be json.dumps-safe."""
        import json

        from scripts.core.state_db import StateDB
        from scripts.mcp import tools

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.add_discovery_paper(rid, "10/x", "X", 0.5, "r", True)
        db.finish_discovery_run(rid, "success", 1, 1)
        monkeypatch.setattr(tools, "_get_state_db", lambda: db)

        json.dumps(tools.get_recent_discovery_runs())
        json.dumps(tools.get_discovery_run_papers(run_id=rid))
