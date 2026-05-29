"""G7: Graph query API tests.

Three read-side query methods on GraphDB:
  - find_similar_papers
  - resolve_query_entity
  - find_papers_by_entities
"""
from __future__ import annotations

import pytest

from scripts.graph import GraphDB
from scripts.graph.entity_extractor import EntityTuple


class _Fixture:
    """Helper: build a small graph (3 papers, 5 entities, 8 MENTIONS) for query tests.

    Shared shape:
      Paper /a → mentions {ion exchange, chromatography, antibody}
      Paper /b → mentions {ion exchange, chromatography}     (2 overlaps with /a)
      Paper /c → mentions {chromatography, ultrafiltration, downstream processing} (1 overlap with /a)
    """

    @staticmethod
    def build(db: GraphDB) -> None:
        entities_a = [
            EntityTuple(
                canonical_id="ion exchange",
                type="method",
                surface="ion exchange",
                field="methods_summary",
                span_start=0,
                span_end=12,
            ),
            EntityTuple(
                canonical_id="chromatography",
                type="method",
                surface="chromatography",
                field="methods_summary",
                span_start=13,
                span_end=27,
            ),
            EntityTuple(
                canonical_id="antibody",
                type="material",
                surface="antibody",
                field="materials_systems",
                span_start=0,
                span_end=8,
            ),
        ]
        entities_b = [
            EntityTuple(
                canonical_id="ion exchange",
                type="method",
                surface="ion exchange",
                field="methods_summary",
                span_start=0,
                span_end=12,
            ),
            EntityTuple(
                canonical_id="chromatography",
                type="method",
                surface="chromatography",
                field="methods_summary",
                span_start=13,
                span_end=27,
            ),
        ]
        entities_c = [
            EntityTuple(
                canonical_id="chromatography",
                type="method",
                surface="chromatography",
                field="methods_summary",
                span_start=0,
                span_end=14,
            ),
            EntityTuple(
                canonical_id="ultrafiltration",
                type="method",
                surface="ultrafiltration",
                field="methods_summary",
                span_start=15,
                span_end=29,
            ),
            EntityTuple(
                canonical_id="downstream processing",
                type="topic",
                surface="downstream processing",
                field="discovered_topics",
                span_start=None,
                span_end=None,
            ),
        ]
        db.add_paper(
            doi="10.0/a",
            entities=entities_a,
            relationships=[],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
        )
        db.add_paper(
            doi="10.0/b",
            entities=entities_b,
            relationships=[],
            paper_metadata={"title": "B", "year": 2024, "journal": "X"},
        )
        db.add_paper(
            doi="10.0/c",
            entities=entities_c,
            relationships=[],
            paper_metadata={"title": "C", "year": 2024, "journal": "Y"},
        )


@pytest.mark.unit
class TestFindSimilarPapers:
    def test_returns_papers_ranked_by_shared_entities(self, tmp_path):
        """G7: find_similar_papers ranks by shared MENTIONS count desc."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        result = db.find_similar_papers("10.0/a", k=10)
        # /b shares 2, /c shares 1. /a never appears (self excluded).
        assert len(result) == 2
        assert result[0][0] == "10.0/b"
        assert result[0][1] == 2.0
        assert result[1][0] == "10.0/c"
        assert result[1][1] == 1.0

    def test_respects_k_limit(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        result = db.find_similar_papers("10.0/a", k=1)
        assert len(result) == 1
        assert result[0][0] == "10.0/b"

    def test_excludes_self(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        result = db.find_similar_papers("10.0/a", k=10)
        assert all(doi != "10.0/a" for doi, _ in result)

    def test_unknown_doi_returns_empty(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        result = db.find_similar_papers("10.0/missing", k=10)
        assert result == []


@pytest.mark.unit
class TestResolveQueryEntity:
    def test_exact_canonical_resolves(self, tmp_path):
        """G7: a known canonical entity resolves to itself."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        assert db.resolve_query_entity("ion exchange", type_="method") == "ion exchange"

    def test_fuzzy_typo_resolves_at_query_threshold(self, tmp_path):
        """G7: a typo at the 85 threshold resolves (would fail at ingest 90)."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        # "ion exchnage" (typo) should match "ion exchange" at min_ratio=85
        result = db.resolve_query_entity("ion exchnage", type_="method")
        assert result == "ion exchange"

    def test_unknown_entity_returns_none(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        result = db.resolve_query_entity("nonexistent thing", type_="method")
        assert result is None

    def test_type_scope_respected(self, tmp_path):
        """G7: looking up a method entity with type_='topic' shouldn't fuzz-match it."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        # "ion exchange" is a method, not a topic — query with type_='topic' should miss
        result = db.resolve_query_entity("ion exchange", type_="topic")
        assert result is None


@pytest.mark.unit
class TestFindPapersByEntities:
    def test_returns_papers_ranked_by_overlap(self, tmp_path):
        """G7: papers with most MENTIONS into the given entity set rank highest."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        # /a has 3 MENTIONS into {ion exchange, chromatography, antibody} = 3
        # /b has 2 (ion exchange, chromatography)
        # /c has 1 (chromatography)
        result = db.find_papers_by_entities(
            ["ion exchange", "chromatography", "antibody"], k=10
        )
        assert len(result) == 3
        assert result[0][0] == "10.0/a"
        assert result[0][1] == 3.0
        assert result[1][0] == "10.0/b"
        assert result[1][1] == 2.0
        assert result[2][0] == "10.0/c"
        assert result[2][1] == 1.0

    def test_respects_k_limit(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        result = db.find_papers_by_entities(["ion exchange", "chromatography"], k=2)
        assert len(result) == 2

    def test_empty_entity_list_returns_empty(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        result = db.find_papers_by_entities([], k=10)
        assert result == []


@pytest.mark.unit
class TestFindSimilarPapersDistinct:
    """G7: count(DISTINCT e) — same canonical_id in different fields counted once."""

    def test_same_entity_in_two_fields_counts_as_one_shared(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # Paper /a mentions 'chromatography' in TWO fields → two MENTIONS edges
        entities_a = [
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
        # Paper /b mentions 'chromatography' in ONE field
        entities_b = [
            EntityTuple(
                canonical_id="chromatography",
                type="method",
                surface="chromatography",
                field="methods_summary",
                span_start=0,
                span_end=14,
            ),
        ]
        db.add_paper(
            doi="10.0/a",
            entities=entities_a,
            relationships=[],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
        )
        db.add_paper(
            doi="10.0/b",
            entities=entities_b,
            relationships=[],
            paper_metadata={"title": "B", "year": 2024, "journal": "X"},
        )

        result = db.find_similar_papers("10.0/a", k=10)
        # Without DISTINCT: would be 2 (two MENTIONS paths through chromatography).
        # With DISTINCT: 1 (one shared entity, regardless of field count).
        assert len(result) == 1
        assert result[0][0] == "10.0/b"
        assert result[0][1] == 1.0, f"expected 1.0 with DISTINCT, got {result[0][1]}"


@pytest.mark.unit
class TestFindPapersByEntitiesDistinct:
    """G7: count(DISTINCT e) — same canonical_id in different fields counted once."""

    def test_same_entity_in_two_fields_counts_once(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # Paper /a mentions chromatography in TWO fields → two MENTIONS edges
        entities_a = [
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
            doi="10.0/a",
            entities=entities_a,
            relationships=[],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
        )

        result = db.find_papers_by_entities(["chromatography"], k=10)
        # Without DISTINCT: 2 (two MENTIONS edges).
        # With DISTINCT: 1.
        assert len(result) == 1
        assert result[0][0] == "10.0/a"
        assert result[0][1] == 1.0


@pytest.mark.unit
class TestInvalidateQueryCache:
    def test_invalidate_clears_cached_normalizer(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _Fixture.build(db)
        # Trigger cache build
        result1 = db.resolve_query_entity("ion exchange", type_="method")
        assert result1 == "ion exchange"
        assert db._query_normalizer is not None
        # Invalidate
        db.invalidate_query_cache()
        assert db._query_normalizer is None
        # Next call rebuilds
        result2 = db.resolve_query_entity("ion exchange", type_="method")
        assert result2 == "ion exchange"
        assert db._query_normalizer is not None
