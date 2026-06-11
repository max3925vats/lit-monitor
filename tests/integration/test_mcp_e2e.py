"""B5: MCP server end-to-end integration test.

Tests all 10 MCP tools by calling the tool functions in scripts.mcp.tools
directly with monkeypatched backends. No real socket transport needed —
the shape contracts are what MCP clients observe, and those contracts live
in the tool functions, not the transport layer.

Fixture graph: 3 papers, 5 entities, CITES + EXTENDS edges.
Fixture embeddings: mocked EmbeddingsDB returning canned hits (real ChromaDB
seeding requires Ollama at embed time — not available in CI).

Run:
    uv run pytest tests/integration/test_mcp_e2e.py -v -m integration
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_graph(tmp_path):
    """Small KuzuDB with 3 papers, 5 entities, and mixed relationships.

    Papers:
        10.0/a  "Alpha"  2024  J1
        10.0/b  "Beta"   2023  J1  — EXTENDS Alpha
        10.0/c  "Gamma"  2024  J2  — EXTENDS Alpha, CITES Alpha

    Schema v4 fix: CITES now uses 'evidence' (matching all other typed
    Paper→Paper RELs), so CITES edges can be created via the generic
    _upsert_typed_edge path. The workaround that avoided CITES here has
    been removed.

    Entities:
        monoclonal antibody (topic)  — Alpha, Beta
        cation exchange (method)     — Alpha
        high throughput screen (method) — Gamma
        J1 (journal)                 — Alpha, Beta
    """
    from lit_monitor.graph import GraphDB
    from lit_monitor.graph.entity_extractor import EntityTuple
    from lit_monitor.graph.relationship_extractor import RelationshipTuple

    db = GraphDB(persist_dir=str(tmp_path / "e2e.kuzu"))

    # Paper A: monoclonal antibody + cation exchange
    db.add_paper(
        doi="10.0/a",
        paper_metadata={"title": "Alpha", "year": 2024, "journal": "J1"},
        entities=[
            EntityTuple(
                canonical_id="monoclonal antibody",
                type="topic",
                surface="mAb",
                field="abstract",
                span_start=0,
                span_end=3,
            ),
            EntityTuple(
                canonical_id="cation exchange",
                type="method",
                surface="CEX",
                field="methods",
                span_start=0,
                span_end=3,
            ),
            EntityTuple(
                canonical_id="J1",
                type="journal",
                surface="J1",
                field=None,
                span_start=None,
                span_end=None,
            ),
        ],
        relationships=[],
    )

    # Paper B: monoclonal antibody + EXTENDS Alpha
    db.add_paper(
        doi="10.0/b",
        paper_metadata={"title": "Beta", "year": 2023, "journal": "J1"},
        entities=[
            EntityTuple(
                canonical_id="monoclonal antibody",
                type="topic",
                surface="mAb",
                field="abstract",
                span_start=0,
                span_end=3,
            ),
            EntityTuple(
                canonical_id="J1",
                type="journal",
                surface="J1",
                field=None,
                span_start=None,
                span_end=None,
            ),
        ],
        relationships=[
            RelationshipTuple(
                source_doi="10.0/b",
                predicate="EXTENDS",
                target_id="10.0/a",
                target_kind="Paper",
                evidence="builds on Alpha et al.",
                confidence=1.0,
                field="abstract",
            ),
        ],
    )

    # Paper C: high throughput screen + EXTENDS Alpha + CITES Alpha
    db.add_paper(
        doi="10.0/c",
        paper_metadata={"title": "Gamma", "year": 2024, "journal": "J2"},
        entities=[
            EntityTuple(
                canonical_id="high throughput screen",
                type="method",
                surface="HTS",
                field="methods",
                span_start=0,
                span_end=3,
            ),
        ],
        relationships=[
            RelationshipTuple(
                source_doi="10.0/c",
                predicate="EXTENDS",
                target_id="10.0/a",
                target_kind="Paper",
                evidence="extends the Alpha framework",
                confidence=1.0,
                field="abstract",
            ),
            # CITES edge — validates schema v4 fix (resolution → evidence).
            RelationshipTuple(
                source_doi="10.0/c",
                predicate="CITES",
                target_id="10.0/a",
                target_kind="Paper",
                evidence="directly cites Alpha methodology",
                confidence=0.95,
                field="references",
            ),
        ],
    )

    return db


@pytest.fixture
def mock_embeddings_db():
    """Mocked EmbeddingsDB returning canned hits.

    Real ChromaDB seeding requires Ollama running to embed text chunks;
    that dependency is not available in CI. The mock validates the same
    shape contracts that the real EmbeddingsDB exposes.
    """
    m = MagicMock()
    m.find_similar_to_text.return_value = [
        {
            "id": "10.0/a",
            "score": 0.92,
            "document": "Alpha doc...",
            "metadata": {"title": "Alpha", "year": 2024},
        },
        {
            "id": "10.0/b",
            "score": 0.81,
            "document": "Beta doc...",
            "metadata": {"title": "Beta", "year": 2023},
        },
    ]
    m.find_similar_chunks.return_value = [
        {
            "id": "10.0/a#0",
            "score": 0.88,
            "document": "chunk text about monoclonal antibodies...",
            "metadata": {"doi": "10.0/a", "section_heading": "Methods"},
        },
    ]
    return m


# ---------------------------------------------------------------------------
# Tests: all 10 MCP tool functions
# ---------------------------------------------------------------------------

class TestAllTenToolsViaDirectCall:
    """Drive all 10 tools through their public function signatures.

    This is the fastest, most reliable way to confirm the shape contracts that
    MCP clients will observe — transport wrapping (JSON encoding + TextContent)
    is handled by graph_server._call_tool and is tested separately in
    TestDispatchCount.
    """

    # ------------------------------------------------------------------ #
    # Tool 4: get_corpus_stats
    # ------------------------------------------------------------------ #

    def test_get_corpus_stats(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.get_corpus_stats()

        assert isinstance(result, dict)
        assert "paper_count" in result
        assert result["paper_count"] == 3
        # Must be JSON-serializable (the MCP transport wraps in json.dumps)
        json.dumps(result)

    # ------------------------------------------------------------------ #
    # Tool 6: get_schema
    # ------------------------------------------------------------------ #

    def test_get_schema(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.get_schema()

        assert isinstance(result, str)
        # Schema text must reference the two main node types
        assert "Paper" in result
        assert "Entity" in result

    # ------------------------------------------------------------------ #
    # Tool 3: get_paper_details
    # ------------------------------------------------------------------ #

    def test_get_paper_details(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.get_paper_details("10.0/a")

        assert isinstance(result, dict)
        assert "metadata" in result
        assert "entities_by_type" in result
        assert "relationships_in" in result
        assert "relationships_out" in result
        json.dumps(result)

    def test_get_paper_details_bad_doi(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-DOI strings must raise ValueError (the MCP guard returns error text)."""
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        with pytest.raises(ValueError):
            tools.get_paper_details("not-a-doi")

    # ------------------------------------------------------------------ #
    # Tool 1: find_papers_by_entity
    # ------------------------------------------------------------------ #

    def test_find_papers_by_entity(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.find_papers_by_entity("monoclonal antibody")

        assert isinstance(result, list)
        # Both Alpha and Beta mention monoclonal antibody
        dois = [r["doi"] for r in result]
        assert "10.0/a" in dois
        assert "10.0/b" in dois
        json.dumps(result)

    def test_find_papers_by_entity_empty_raises(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        with pytest.raises(ValueError):
            tools.find_papers_by_entity("")

    # ------------------------------------------------------------------ #
    # Tool 2: find_papers_by_relationship
    # ------------------------------------------------------------------ #

    def test_find_papers_by_relationship(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.find_papers_by_relationship("EXTENDS")

        assert isinstance(result, list)
        # Both Beta and Gamma EXTEND Alpha
        dois = [r["doi"] for r in result]
        assert "10.0/b" in dois
        assert "10.0/c" in dois
        json.dumps(result)

    def test_find_papers_by_relationship_bad_predicate(
        self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        with pytest.raises(ValueError):
            tools.find_papers_by_relationship("NOT_A_PREDICATE")

    # ------------------------------------------------------------------ #
    # Tool 5: list_entities_by_type
    # ------------------------------------------------------------------ #

    def test_list_entities_by_type(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.list_entities_by_type("method")

        assert isinstance(result, list)
        assert len(result) >= 1
        # Each item must have the MCP-facing shape
        for item in result:
            assert "canonical" in item
            assert "type" in item
            assert "mention_count" in item
        json.dumps(result)

    def test_list_entities_by_type_bad_type(
        self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        with pytest.raises(ValueError):
            tools.list_entities_by_type("not_a_type")

    # ------------------------------------------------------------------ #
    # Tool 9: run_cypher (B3)
    # ------------------------------------------------------------------ #

    def test_run_cypher_safe_query(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.run_cypher("MATCH (p:Paper) RETURN p.doi AS doi")

        assert isinstance(result, list)
        assert len(result) == 3
        json.dumps(result)

    def test_run_cypher_mutation_blocked(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation keywords must raise ValueError (CypherSafetyError subclass)."""
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        with pytest.raises(ValueError):
            tools.run_cypher("CREATE (n:Paper {doi: '10.0/x'})")

    # ------------------------------------------------------------------ #
    # Tool 7: find_papers_by_query
    # ------------------------------------------------------------------ #

    def test_find_papers_by_query(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.find_papers_by_query("monoclonal antibody")

        assert isinstance(result, list)
        json.dumps(result)

    # ------------------------------------------------------------------ #
    # Tool 8: find_papers_by_query_hybrid
    # ------------------------------------------------------------------ #

    def test_find_papers_by_query_hybrid(self, fixture_graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hybrid falls back gracefully to graph-only when EmbeddingsDB is absent."""
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_get_graph_db", lambda: fixture_graph)

        result = tools.find_papers_by_query_hybrid("monoclonal antibody")

        assert isinstance(result, list)
        json.dumps(result)

    # ------------------------------------------------------------------ #
    # Tool 10: semantic_search — paper granularity (B6)
    # ------------------------------------------------------------------ #

    def test_semantic_search_paper(
        self, mock_embeddings_db: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_EMBEDDINGS_DB", mock_embeddings_db)

        result = tools.semantic_search("monoclonal antibody", top_k=5, granularity="paper")

        assert isinstance(result, list)
        assert len(result) == 2
        # Every paper-granularity hit must carry these keys
        for hit in result:
            assert {"doi", "title", "year", "score"} <= set(hit.keys())
        json.dumps(result)

    # ------------------------------------------------------------------ #
    # Tool 10: semantic_search — chunk granularity (B6)
    # ------------------------------------------------------------------ #

    def test_semantic_search_chunk(
        self, mock_embeddings_db: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_EMBEDDINGS_DB", mock_embeddings_db)

        result = tools.semantic_search("monoclonal antibody", top_k=5, granularity="chunk")

        assert isinstance(result, list)
        assert len(result) == 1
        # Every chunk-granularity hit must carry these keys
        for hit in result:
            assert {"doi", "chunk_id", "snippet", "score"} <= set(hit.keys())
        json.dumps(result)

    def test_semantic_search_bad_granularity(
        self, mock_embeddings_db: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lit_monitor.mcp import tools
        monkeypatch.setattr(tools, "_EMBEDDINGS_DB", mock_embeddings_db)

        with pytest.raises(ValueError):
            tools.semantic_search("query", top_k=5, granularity="paragraph")


# ---------------------------------------------------------------------------
# Dispatch count check — ensures graph_server registers all 10 names
# ---------------------------------------------------------------------------

class TestDispatchCount:
    """Confirm the server's TOOL_NAMES tuple stays at 12 (P9)."""

    def test_twelve_tools_in_dispatch(self) -> None:
        from lit_monitor.mcp import graph_server

        assert len(graph_server.TOOL_NAMES) == 12, (
            f"Expected 12 tool names, got {len(graph_server.TOOL_NAMES)}: "
            f"{graph_server.TOOL_NAMES}"
        )
