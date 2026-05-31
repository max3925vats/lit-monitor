"""H1: shared query layer tests."""
from __future__ import annotations

import json
import sys

import pytest

from scripts.api.queries import (
    _validate_doi,
    get_corpus_stats,
    get_entity_neighborhood,
    get_paper_snapshot,
    get_related_papers,
    get_schema_text,
    list_entities,
)
from scripts.graph import GraphDB
from scripts.graph.entity_extractor import EntityTuple


@pytest.fixture
def populated_graph(tmp_path):
    db = GraphDB(persist_dir=str(tmp_path / "h1.kuzu"))
    entities = [
        EntityTuple(
            canonical_id="ion exchange",
            type="method",
            surface="ion exchange",
            field="methods_summary",
            span_start=0,
            span_end=12,
        ),
        EntityTuple(
            canonical_id="monoclonal antibody",
            type="material",
            surface="monoclonal antibody",
            field="materials_systems",
            span_start=0,
            span_end=20,
        ),
    ]
    db.add_paper(
        doi="10.0/a",
        entities=entities,
        relationships=[],
        paper_metadata={"title": "A", "year": 2024, "journal": "X"},
    )
    db.add_paper(
        doi="10.0/b",
        entities=entities[:1],
        relationships=[],
        paper_metadata={"title": "B", "year": 2024, "journal": "X"},
    )
    return db


class TestValidateDoi:
    def test_valid_doi_passes(self):
        _validate_doi("10.1234/test.123")  # no exception

    def test_bad_doi_raises(self):
        with pytest.raises(ValueError, match="invalid DOI"):
            _validate_doi("not-a-doi")

    def test_empty_doi_raises(self):
        with pytest.raises(ValueError, match="invalid DOI"):
            _validate_doi("")


class TestGetPaperSnapshot:
    def test_returns_required_keys(self, populated_graph):
        result = get_paper_snapshot("10.0/a", populated_graph)
        assert "metadata" in result
        assert "entities_by_type" in result
        assert "relationships_in" in result
        assert "relationships_out" in result

    def test_json_serializable(self, populated_graph):
        result = get_paper_snapshot("10.0/a", populated_graph)
        json.dumps(result)  # must not raise

    def test_unknown_doi_returns_empty_shapes(self, populated_graph):
        result = get_paper_snapshot("10.0/missing", populated_graph)
        # Empty metadata + empty entity / relationship lists, but keys present
        assert result["metadata"] in ({}, None) or not result["metadata"]
        assert result["entities_by_type"] in ({}, [])

    def test_bad_doi_raises_value_error(self, populated_graph):
        with pytest.raises(ValueError):
            get_paper_snapshot("not-a-doi", populated_graph)

    def test_entities_grouped_by_type(self, populated_graph):
        result = get_paper_snapshot("10.0/a", populated_graph)
        ents = result["entities_by_type"]
        # ion exchange (method) + monoclonal antibody (material) — 2 types
        assert isinstance(ents, dict)
        assert len(ents) == 2


class TestGetEntityNeighborhood:
    def test_returns_papers_mentioning(self, populated_graph):
        result = get_entity_neighborhood("ion exchange", populated_graph)
        assert "canonical_id" in result
        assert "papers" in result
        # Two papers mention ion exchange
        assert len(result["papers"]) == 2

    def test_json_serializable(self, populated_graph):
        result = get_entity_neighborhood("ion exchange", populated_graph)
        json.dumps(result)

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            get_entity_neighborhood("", None)


class TestListEntities:
    def test_filters_by_type(self, populated_graph):
        methods = list_entities("method", top_k=10, graph_db=populated_graph)
        assert all(e.get("type") == "method" for e in methods)

    def test_respects_top_k(self, populated_graph):
        result = list_entities("method", top_k=1, graph_db=populated_graph)
        assert len(result) <= 1

    def test_json_serializable(self, populated_graph):
        result = list_entities("method", top_k=10, graph_db=populated_graph)
        json.dumps(result)

    def test_list_entities_counts_distinct_papers_not_edges(self, tmp_path):
        """H1 I1: same canonical_id in two fields of ONE paper = 1 mention, not 2."""
        db = GraphDB(persist_dir=str(tmp_path / "h1_distinct.kuzu"))
        # Same entity appearing in two different fields of the same paper produces
        # two MENTIONS edges. Without DISTINCT, count(p) would return 2.
        entities = [
            EntityTuple(
                canonical_id="chromatography",
                type="method",
                surface="chromatography",
                field="methods_summary",
                span_start=0,
                span_end=14,
            ),
            EntityTuple(
                canonical_id="chromatography",
                type="method",
                surface="chromatography",
                field="discovered_topics",
                span_start=None,
                span_end=None,
            ),
        ]
        db.add_paper(
            doi="10.0/dup",
            entities=entities,
            relationships=[],
            paper_metadata={"title": "Dup", "year": 2024, "journal": "X"},
        )

        result = list_entities("method", top_k=10, graph_db=db)
        chromato = next(e for e in result if e["canonical_id"] == "chromatography")
        # Without DISTINCT: would be 2. With DISTINCT: 1.
        assert chromato["mention_count"] == 1, (
            f"expected 1 (DISTINCT papers), got {chromato['mention_count']}"
        )


class TestGetCorpusStats:
    def test_returns_dict(self, populated_graph):
        result = get_corpus_stats(populated_graph)
        assert isinstance(result, dict)

    def test_json_serializable(self, populated_graph):
        json.dumps(get_corpus_stats(populated_graph))

    def test_contains_paper_count(self, populated_graph):
        result = get_corpus_stats(populated_graph)
        assert "paper_count" in result
        assert result["paper_count"] == 2


class TestGetSchemaText:
    def test_returns_string(self, populated_graph):
        result = get_schema_text(populated_graph)
        assert isinstance(result, str)
        # Either real schema or the fallback message
        assert len(result) > 0

    def test_fallback_when_a1_not_built(self, monkeypatch, populated_graph):
        """H1: when scripts.graph.schema_describer doesn't exist, get_schema_text falls back."""
        # Block the import by setting the module key to None in sys.modules
        monkeypatch.setitem(sys.modules, "scripts.graph.schema_describer", None)
        result = get_schema_text(populated_graph)
        assert isinstance(result, str)
        assert (
            "schema describer" in result.lower()
            or "phase 4a" in result.lower()
            or "a1" in result.lower()
        )


class TestGetRelatedPapers:
    def test_returns_list(self, populated_graph, monkeypatch):
        """get_related_papers wraps retrieve_doi_candidates; monkeypatch to avoid embeddings."""
        from scripts.retrieval import branch as branch_mod

        def fake_retrieve(
            rag_mode: str,
            *,
            seed_doi=None,
            query_text=None,
            entity_ids=None,
            embeddings_db=None,
            graph_db=None,
            k=20,
            **kw,
        ):
            return [("10.0/b", 0.9)]

        monkeypatch.setattr(branch_mod, "retrieve_doi_candidates", fake_retrieve)
        # Re-import inside the test so monkeypatch is active
        from importlib import reload

        import scripts.api.queries as q_mod

        reload(q_mod)
        result = q_mod.get_related_papers("10.0/a", mode="graph", k=10, cfg=None)
        assert isinstance(result, list)
        json.dumps(result)

    def test_invalid_mode_raises(self, populated_graph):
        with pytest.raises(ValueError, match="mode must be"):
            get_related_papers("10.0/a", mode="invalid", k=5, cfg=None)

    def test_bad_doi_raises(self, populated_graph):
        with pytest.raises(ValueError):
            get_related_papers("not-a-doi", mode="graph", k=5, cfg=None)


# ---------------------------------------------------------------------------
# H10: get_papers_by_query — shared free-text retrieval function
# ---------------------------------------------------------------------------

class TestGetPapersByQuery:
    """Unit tests for get_papers_by_query — the H10 single implementation."""

    def test_invalid_mode_raises(self) -> None:
        """Unknown mode raises ValueError before touching any backend."""
        from scripts.api.queries import get_papers_by_query
        with pytest.raises(ValueError, match="unknown mode"):
            get_papers_by_query("antibody", mode="bogus", k=10)

    def test_k_zero_raises(self) -> None:
        """k=0 is below the valid range."""
        from scripts.api.queries import get_papers_by_query
        with pytest.raises(ValueError, match="k must be"):
            get_papers_by_query("antibody", mode="graph", k=0)

    def test_k_negative_raises(self) -> None:
        """Negative k raises ValueError."""
        from scripts.api.queries import get_papers_by_query
        with pytest.raises(ValueError, match="k must be"):
            get_papers_by_query("antibody", mode="graph", k=-5)

    def test_k_above_100_raises(self) -> None:
        """k > 100 raises ValueError."""
        from scripts.api.queries import get_papers_by_query
        with pytest.raises(ValueError, match="k must be"):
            get_papers_by_query("antibody", mode="graph", k=101)

    def test_empty_query_returns_empty_list(self) -> None:
        """Whitespace-only query → [] without contacting any backend."""
        from scripts.api.queries import get_papers_by_query
        assert get_papers_by_query("   ", mode="vector", k=10) == []

    def test_empty_string_returns_empty_list(self) -> None:
        """Empty string query → [] without error."""
        from scripts.api.queries import get_papers_by_query
        assert get_papers_by_query("", mode="graph", k=10) == []

    def test_vector_mode_no_embeddings_db_returns_empty(self) -> None:
        """Vector mode with embeddings_db=None → [] gracefully."""
        from scripts.api.queries import get_papers_by_query
        result = get_papers_by_query("antibody", mode="vector", k=10, embeddings_db=None)
        assert result == []

    def test_vector_mode_returns_doi_title_score(self) -> None:
        """Vector mode: results include doi, title, and score fields."""
        from unittest.mock import MagicMock

        from scripts.api.queries import get_papers_by_query

        mock_edb = MagicMock()
        mock_edb.find_similar_to_text.return_value = [
            {"id": "10.1/x", "score": 0.95, "metadata": {"title": "Paper X"}},
            {"id": "10.1/y", "score": 0.80, "metadata": {"title": "Paper Y"}},
        ]

        results = get_papers_by_query(
            "antibody", mode="vector", k=5, embeddings_db=mock_edb
        )
        assert len(results) == 2
        assert results[0]["doi"] == "10.1/x"
        assert results[0]["title"] == "Paper X"
        assert results[0]["score"] == pytest.approx(0.95)
        assert results[1]["doi"] == "10.1/y"

    def test_graph_mode_returns_score(self) -> None:
        """Graph mode: score field is present in output."""
        from unittest.mock import MagicMock

        from scripts.api.queries import get_papers_by_query

        mock_gdb = MagicMock()
        mock_gdb.resolve_query_entity.return_value = "antibody|method"
        mock_gdb.find_papers_by_entities.return_value = [
            ("10.1/a", 3.0),
            ("10.1/b", 1.0),
        ]
        # Simulate metadata fetch returning nothing so we skip enrich logic
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_conn.execute.return_value = mock_result
        mock_gdb._conn = mock_conn

        results = get_papers_by_query(
            "antibody", mode="graph", k=5, graph_db=mock_gdb
        )
        assert len(results) == 2
        dois = [r["doi"] for r in results]
        assert "10.1/a" in dois
        # Every entry has a score field
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_graph_mode_no_entity_returns_empty(self) -> None:
        """Graph mode: query that resolves to no entity → []."""
        from unittest.mock import MagicMock

        from scripts.api.queries import get_papers_by_query

        mock_gdb = MagicMock()
        mock_gdb.resolve_query_entity.return_value = None  # unresolvable

        results = get_papers_by_query(
            "unresolvable_xyzzy", mode="graph", k=5, graph_db=mock_gdb
        )
        assert results == []

    def test_hybrid_mode_with_no_backends_returns_empty(self) -> None:
        """Hybrid with graph resolving to None and no vector backend → []."""
        from unittest.mock import MagicMock

        from scripts.api.queries import get_papers_by_query

        # Provide graph_db directly so no real kuzu is opened; entity resolves
        # to None so graph leg produces no hits. embeddings_db=None so vector
        # leg is skipped. RRF over two empty lists → [].
        mock_gdb = MagicMock()
        mock_gdb.resolve_query_entity.return_value = None

        result = get_papers_by_query(
            "antibody",
            mode="hybrid",
            k=5,
            graph_db=mock_gdb,
            embeddings_db=None,
        )
        # No vector hits, no graph entity resolved → empty
        assert result == []

    def test_k_1_is_valid(self) -> None:
        """k=1 is the minimum valid value and does not raise."""
        from scripts.api.queries import get_papers_by_query
        # No backends → returns [] without error
        result = get_papers_by_query("x", mode="vector", k=1, embeddings_db=None)
        assert result == []

    def test_k_100_is_valid(self) -> None:
        """k=100 is the maximum valid value and does not raise."""
        from scripts.api.queries import get_papers_by_query
        result = get_papers_by_query("x", mode="vector", k=100, embeddings_db=None)
        assert result == []
