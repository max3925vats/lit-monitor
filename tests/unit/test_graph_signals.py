"""Bundle D: graph-in-ranker tests.

Covers:
- get_graph_signals_for_candidate(): shared entities, citations, authors,
  extends, compares_to, empty library, unknown candidate, case-insensitive
  author matching, semicolon-separated author parsing.
- ranker.py: graph signal kwargs, weighted sum adds to score, score_breakdown
  shape, zero weights preserve v0.8/A/B/C behavior.
- Config: ranking.weights graph defaults are 0.0.
- Normalization: counts are capped to [0, 1] range before weighting.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(canonical_id: str, type_: str = "topic") -> Any:
    """Create a minimal EntityTuple for use in add_paper()."""
    from scripts.graph.entity_extractor import EntityTuple

    return EntityTuple(
        canonical_id=canonical_id,
        type=type_,
        surface=canonical_id,
        field="abstract",
        span_start=0,
        span_end=len(canonical_id),
    )


def _make_cites_rel(source_doi: str, target_doi: str) -> Any:
    """Create a CITES RelationshipTuple."""
    from scripts.graph.relationship_extractor import RelationshipTuple

    return RelationshipTuple(
        source_doi=source_doi,
        predicate="CITES",
        target_id=target_doi,
        target_kind="Paper",
        evidence="cites",
        confidence=1.0,
        field="references",
    )


def _make_extends_rel(source_doi: str, target_doi: str) -> Any:
    """Create an EXTENDS RelationshipTuple."""
    from scripts.graph.relationship_extractor import RelationshipTuple

    return RelationshipTuple(
        source_doi=source_doi,
        predicate="EXTENDS",
        target_id=target_doi,
        target_kind="Paper",
        evidence="extends",
        confidence=1.0,
        field="abstract",
    )


def _make_compares_rel(source_doi: str, target_doi: str) -> Any:
    """Create a COMPARES_TO RelationshipTuple."""
    from scripts.graph.relationship_extractor import RelationshipTuple

    return RelationshipTuple(
        source_doi=source_doi,
        predicate="COMPARES_TO",
        target_id=target_doi,
        target_kind="Paper",
        evidence="compared",
        confidence=1.0,
        field="abstract",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_graph(tmp_path):
    """Small KuzuDB with 3 library papers + 2 candidate scenarios.

    Library:
      lib-a: entities=[mAb], authors="Smith; Jones"
      lib-b: entities=[CEX], authors="Lee"
      lib-c: entities=[mAb], authors="Smith"  (also CITES lib-a)

    Candidates:
      cand-shared-entities: entities=[mAb, CEX], CITES lib-b, EXTENDS lib-a
                            authors="Garcia"
      cand-shared-authors:  entities=[], authors="Smith"
    """
    from scripts.graph import GraphDB

    db = GraphDB(persist_dir=str(tmp_path / "d.kuzu"))

    # --- Library papers ---
    db.add_paper(
        doi="10.0/lib-a",
        paper_metadata={"title": "A", "year": 2024, "journal": "J", "authors": "Smith; Jones"},
        entities=[_make_entity("mAb")],
        relationships=[],
    )
    db.add_paper(
        doi="10.0/lib-b",
        paper_metadata={"title": "B", "year": 2024, "journal": "J", "authors": "Lee"},
        entities=[_make_entity("CEX", type_="method")],
        relationships=[],
    )
    db.add_paper(
        doi="10.0/lib-c",
        paper_metadata={"title": "C", "year": 2024, "journal": "J", "authors": "Smith"},
        entities=[_make_entity("mAb")],
        relationships=[_make_cites_rel("10.0/lib-c", "10.0/lib-a")],
    )

    # --- Candidates ---
    db.add_paper(
        doi="10.0/cand-shared-entities",
        paper_metadata={"title": "Candidate", "year": 2024, "journal": "J", "authors": "Garcia"},
        entities=[_make_entity("mAb"), _make_entity("CEX", type_="method")],
        relationships=[
            _make_cites_rel("10.0/cand-shared-entities", "10.0/lib-b"),
            _make_extends_rel("10.0/cand-shared-entities", "10.0/lib-a"),
        ],
    )
    db.add_paper(
        doi="10.0/cand-shared-authors",
        paper_metadata={"title": "Author Candidate", "year": 2024, "journal": "J", "authors": "Smith"},
        entities=[],
        relationships=[],
    )

    return db


@pytest.fixture
def library_dois():
    return ["10.0/lib-a", "10.0/lib-b", "10.0/lib-c"]


# ---------------------------------------------------------------------------
# TestGraphSignalsExtraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGraphSignalsExtraction:
    """get_graph_signals_for_candidate() signal extraction tests."""

    def test_shared_entities_counted(self, fixture_graph, library_dois):
        """Candidate mentions mAb (in lib-a, lib-c) and CEX (in lib-b) — 2 shared."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate(
            "10.0/cand-shared-entities", fixture_graph, library_dois
        )
        assert sig["n_shared_entities"] == 2
        assert set(sig["shared_entity_canonical_ids"]) == {"mAb", "CEX"}

    def test_cites_in_library(self, fixture_graph, library_dois):
        """Candidate CITES lib-b (in library) → count 1."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate(
            "10.0/cand-shared-entities", fixture_graph, library_dois
        )
        assert sig["n_cites_in_library"] == 1

    def test_cited_by_library_counts_incoming_cites(self, fixture_graph):
        """lib-c CITES lib-a → lib-a sees n_cited_by_library=1 when lib-c is library."""
        from scripts.api.queries import get_graph_signals_for_candidate

        # lib-a is the candidate; lib-c (which cites lib-a) is in the "library"
        sig = get_graph_signals_for_candidate(
            "10.0/lib-a", fixture_graph, ["10.0/lib-c"]
        )
        assert sig["n_cited_by_library"] == 1

    def test_extends_in_library(self, fixture_graph, library_dois):
        """Candidate EXTENDS lib-a (in library) → count 1."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate(
            "10.0/cand-shared-entities", fixture_graph, library_dois
        )
        assert sig["n_extends_in_library"] == 1

    def test_compares_to_library_zero_when_none(self, fixture_graph, library_dois):
        """No COMPARES_TO edge → n_compares_to_library == 0."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate(
            "10.0/cand-shared-entities", fixture_graph, library_dois
        )
        assert sig["n_compares_to_library"] == 0

    def test_compares_to_library_counted(self, fixture_graph):
        """When COMPARES_TO edge is present it is counted."""
        from scripts.graph import GraphDB
        from scripts.api.queries import get_graph_signals_for_candidate

        # Re-use the fixture_graph connection directly to add a COMPARES_TO edge
        fixture_graph.add_paper(
            doi="10.0/cand-compares",
            paper_metadata={"title": "Cmp", "year": 2024, "journal": "J"},
            entities=[],
            relationships=[_make_compares_rel("10.0/cand-compares", "10.0/lib-b")],
        )
        sig = get_graph_signals_for_candidate(
            "10.0/cand-compares", fixture_graph, ["10.0/lib-b"]
        )
        assert sig["n_compares_to_library"] == 1

    def test_shared_authors_detected(self, fixture_graph, library_dois):
        """Smith appears in lib-a and lib-c; cand-shared-authors is by Smith."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate(
            "10.0/cand-shared-authors", fixture_graph, library_dois
        )
        assert sig["n_shared_authors"] == 1
        assert "Smith" in sig["shared_authors_sample"]

    def test_shared_authors_sample_capped_at_three(self, tmp_path):
        """shared_authors_sample contains at most 3 names."""
        from scripts.graph import GraphDB
        from scripts.api.queries import get_graph_signals_for_candidate

        db = GraphDB(persist_dir=str(tmp_path / "authors3.kuzu"))
        db.add_paper(
            doi="10.0/lib-multi",
            paper_metadata={"authors": "Alpha; Beta; Gamma; Delta", "title": "L", "year": 2024, "journal": "J"},
            entities=[],
            relationships=[],
        )
        db.add_paper(
            doi="10.0/cand-multi",
            paper_metadata={"authors": "Alpha; Beta; Gamma; Delta", "title": "C", "year": 2024, "journal": "J"},
            entities=[],
            relationships=[],
        )
        sig = get_graph_signals_for_candidate("10.0/cand-multi", db, ["10.0/lib-multi"])
        assert sig["n_shared_authors"] == 4
        # Sample must be capped at 3 entries for UI annotation
        assert len(sig["shared_authors_sample"]) <= 3

    def test_no_overlap_returns_zeros(self, fixture_graph, library_dois):
        """Unknown candidate (not in graph) returns all-zero dict."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate("10.0/unknown", fixture_graph, library_dois)
        assert sig["n_shared_entities"] == 0
        assert sig["n_cites_in_library"] == 0
        assert sig["n_cited_by_library"] == 0
        assert sig["n_extends_in_library"] == 0
        assert sig["n_shared_authors"] == 0
        assert sig["shared_entity_canonical_ids"] == []
        assert sig["shared_authors_sample"] == []

    def test_empty_library_dois_returns_zeros(self, fixture_graph):
        """Empty library → no overlap possible, all zeros."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate("10.0/cand-shared-entities", fixture_graph, [])
        assert sig["n_shared_entities"] == 0

    def test_none_graph_db_returns_zeros(self):
        """graph_db=None returns the default zero dict without raising."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate("10.0/anything", None, ["10.0/lib"])
        assert sig["n_shared_entities"] == 0
        assert sig["n_shared_authors"] == 0

    def test_result_always_contains_all_keys(self, fixture_graph, library_dois):
        """Returned dict always has all 8 required keys, even for unknown candidate."""
        from scripts.api.queries import get_graph_signals_for_candidate

        sig = get_graph_signals_for_candidate("10.0/unknown", fixture_graph, library_dois)
        required = {
            "n_shared_entities", "shared_entity_canonical_ids",
            "n_cites_in_library", "n_cited_by_library",
            "n_extends_in_library", "n_compares_to_library",
            "n_shared_authors", "shared_authors_sample",
        }
        assert required.issubset(sig.keys())

    def test_case_insensitive_author_match(self, tmp_path):
        """Author overlap is case-insensitive: 'SMITH' in library matches 'Smith' in candidate."""
        from scripts.graph import GraphDB
        from scripts.api.queries import get_graph_signals_for_candidate

        db = GraphDB(persist_dir=str(tmp_path / "case.kuzu"))
        db.add_paper(
            doi="10.0/lib-upper",
            paper_metadata={"authors": "SMITH", "title": "L", "year": 2024, "journal": "J"},
            entities=[],
            relationships=[],
        )
        db.add_paper(
            doi="10.0/cand-lower",
            paper_metadata={"authors": "Smith", "title": "C", "year": 2024, "journal": "J"},
            entities=[],
            relationships=[],
        )
        sig = get_graph_signals_for_candidate("10.0/cand-lower", db, ["10.0/lib-upper"])
        assert sig["n_shared_authors"] == 1


# ---------------------------------------------------------------------------
# TestRankerIntegration
# ---------------------------------------------------------------------------


def _make_mock_db(score: float = 0.5) -> MagicMock:
    db = MagicMock()
    db.find_similar_to_text.return_value = [
        {"score": score, "id": "x", "document": "", "metadata": {}}
    ]
    return db


def _make_mock_llm(doi: str = "10.1/a") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = json.dumps({doi: "relevant"})
    return llm


def _make_candidate(doi: str, score: float = 0.5) -> dict[str, Any]:
    return {
        "doi": doi,
        "title": f"Paper {doi}",
        "abstract": "Abstract.",
        "_embedding": np.array([score, 0.0, 0.0], dtype=np.float32),
    }


@pytest.mark.unit
class TestRankerIntegration:
    """rank_papers() with graph_signals and graph_weights kwargs."""

    def test_default_weights_zero_no_behavior_change(self):
        """With all graph weights at 0, scores are byte-for-byte identical (regression)."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.7), _make_candidate("10.1/b", 0.3)]
        db = MagicMock()
        db.find_similar_to_text.side_effect = [
            [{"score": 0.7, "id": "x"}],
            [{"score": 0.3, "id": "y"}],
        ]
        llm = MagicMock()
        llm.complete.return_value = '{"10.1/a": "r", "10.1/b": "r"}'

        # Baseline — no graph kwargs
        ranked_base = rank_papers(candidates[:], db, llm, top_k=2)
        scores_base = {p["doi"]: p["similarity_score"] for p in ranked_base}

        db.find_similar_to_text.side_effect = [
            [{"score": 0.7, "id": "x"}],
            [{"score": 0.3, "id": "y"}],
        ]
        llm.complete.return_value = '{"10.1/a": "r", "10.1/b": "r"}'
        ranked_zero = rank_papers(
            candidates[:], db, llm, top_k=2,
            graph_signals={"10.1/a": {"n_shared_entities": 5}},
            graph_weights={"entity_overlap": 0.0, "citation": 0.0, "shared_authors": 0.0},
        )
        scores_zero = {p["doi"]: p["similarity_score"] for p in ranked_zero}

        assert scores_base == scores_zero

    def test_entity_overlap_weight_adds_to_score(self):
        """Nonzero graph_entity_overlap weight adds normalized signal to score."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")

        base = rank_papers(candidates[:], db, llm, top_k=1)
        base_score = base[0]["similarity_score"]

        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        signals = {"10.1/a": {"n_shared_entities": 5}}
        weights = {"entity_overlap": 0.2, "citation": 0.0, "shared_authors": 0.0}
        with_graph = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        assert with_graph[0]["similarity_score"] > base_score

    def test_citation_weight_adds_to_score(self):
        """Nonzero citation weight (cites_in_library) adds to score."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        base = rank_papers(candidates[:], db, llm, top_k=1)
        base_score = base[0]["similarity_score"]

        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        signals = {"10.1/a": {"n_cites_in_library": 3}}
        weights = {"entity_overlap": 0.0, "citation": 0.15, "shared_authors": 0.0}
        with_graph = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        assert with_graph[0]["similarity_score"] > base_score

    def test_shared_authors_weight_adds_to_score(self):
        """Nonzero shared_authors weight adds to score."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        base = rank_papers(candidates[:], db, llm, top_k=1)
        base_score = base[0]["similarity_score"]

        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        signals = {"10.1/a": {"n_shared_authors": 3}}
        weights = {"entity_overlap": 0.0, "citation": 0.0, "shared_authors": 0.1}
        with_graph = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        assert with_graph[0]["similarity_score"] > base_score

    def test_score_breakdown_contains_graph_keys(self):
        """Bundle B integration: score_breakdown always has graph_* keys."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")

        ranked = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals={"10.1/a": {"n_shared_entities": 2}},
            graph_weights={"entity_overlap": 0.2, "citation": 0.0, "shared_authors": 0.0},
        )
        bd = ranked[0]["score_breakdown"]
        assert "graph_entity_overlap" in bd
        assert "graph_citation" in bd
        assert "graph_shared_authors" in bd

    def test_score_breakdown_has_graph_keys_even_without_graph_signals(self):
        """score_breakdown always has the graph_* keys (default 0.0) even when no signals."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        ranked = rank_papers(candidates[:], db, llm, top_k=1)
        bd = ranked[0]["score_breakdown"]
        assert bd.get("graph_entity_overlap") == 0.0
        assert bd.get("graph_citation") == 0.0
        assert bd.get("graph_shared_authors") == 0.0

    def test_shared_authors_sample_in_breakdown(self):
        """graph_shared_authors_sample is surfaced on output paper for Bundle B UI."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        signals = {
            "10.1/a": {
                "n_shared_authors": 2,
                "shared_authors_sample": ["Smith", "Jones"],
            }
        }
        weights = {"entity_overlap": 0.0, "citation": 0.0, "shared_authors": 0.1}
        ranked = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        assert ranked[0].get("graph_shared_authors_sample") == ["Smith", "Jones"]

    def test_weighted_sum_not_rrf(self):
        """Verify the combination is additive (weighted sum), not RRF.

        If weights are w=0.2 and normalized signal=1.0, the score must be
        base + 0.2, not RRF-fused.
        """
        from scripts.llm.ranker import rank_papers

        base_score = 0.6
        candidates = [_make_candidate("10.1/a", base_score)]
        db = _make_mock_db(base_score)
        llm = _make_mock_llm("10.1/a")
        # 5 shared entities → normalized = min(5/5, 1.0) = 1.0
        signals = {"10.1/a": {"n_shared_entities": 5}}
        w = 0.2
        weights = {"entity_overlap": w, "citation": 0.0, "shared_authors": 0.0}
        ranked = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        expected = base_score + w * 1.0
        assert abs(ranked[0]["similarity_score"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# TestConfigDefaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigDefaults:
    """Config defaults for Bundle D graph weights must be 0.0."""

    def test_ranking_weights_graph_defaults_zero(self, tmp_path):
        """ranking.weights.graph_* all default to 0.0 when not set in YAML."""
        from scripts.core.config import Config

        # Write a minimal extraction.yaml with no graph weight keys.
        # Config takes extraction_yaml as a Path kwarg (not positional).
        cfg_path = tmp_path / "extraction.yaml"
        cfg_path.write_text("# minimal config\n")

        cfg = Config(extraction_yaml=cfg_path)
        assert cfg.ranking.weights.graph_entity_overlap == 0.0
        assert cfg.ranking.weights.graph_citation == 0.0
        assert cfg.ranking.weights.graph_shared_authors == 0.0

    def test_ranking_weights_graph_reads_from_yaml(self, tmp_path):
        """graph weights from YAML are loaded into config correctly."""
        from scripts.core.config import Config

        cfg_path = tmp_path / "extraction.yaml"
        cfg_path.write_text(
            "ranking:\n"
            "  weights:\n"
            "    graph_entity_overlap: 0.15\n"
            "    graph_citation: 0.10\n"
            "    graph_shared_authors: 0.05\n"
        )
        cfg = Config(extraction_yaml=cfg_path)
        assert cfg.ranking.weights.graph_entity_overlap == 0.15
        assert cfg.ranking.weights.graph_citation == 0.10
        assert cfg.ranking.weights.graph_shared_authors == 0.05


# ---------------------------------------------------------------------------
# TestNormalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalization:
    """Normalization of raw counts to [0, 1] range."""

    def test_zero_entities_normalized_to_zero(self):
        """0 shared entities → normalized value 0.0."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        signals = {"10.1/a": {"n_shared_entities": 0}}
        weights = {"entity_overlap": 0.5, "citation": 0.0, "shared_authors": 0.0}
        ranked = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        # entity_overlap contribution = 0 * 0.5 = 0.0
        assert ranked[0]["score_breakdown"]["graph_entity_overlap"] == 0.0

    def test_five_entities_normalized_to_one(self):
        """5+ shared entities → normalized value 1.0 (cap at 1.0)."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        signals = {"10.1/a": {"n_shared_entities": 5}}
        w = 0.3
        weights = {"entity_overlap": w, "citation": 0.0, "shared_authors": 0.0}
        ranked = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        assert abs(ranked[0]["score_breakdown"]["graph_entity_overlap"] - w) < 1e-6

    def test_beyond_cap_entities_normalized_to_one(self):
        """10 shared entities still normalizes to 1.0 (min(10/5, 1.0) = 1.0)."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        signals = {"10.1/a": {"n_shared_entities": 10}}
        w = 0.3
        weights = {"entity_overlap": w, "citation": 0.0, "shared_authors": 0.0}
        ranked = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        # Should be same as 5 entities (capped at 1.0)
        assert abs(ranked[0]["score_breakdown"]["graph_entity_overlap"] - w) < 1e-6

    def test_citation_normalized_combines_all_citation_signals(self):
        """citation signal combines cites_in + cited_by + extends + compares_to."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_mock_db(0.5)
        llm = _make_mock_llm("10.1/a")
        # Total citation score = 1 + 1 + 1 + 0 = 3 → normalized = min(3/3, 1.0) = 1.0
        signals = {
            "10.1/a": {
                "n_cites_in_library": 1,
                "n_cited_by_library": 1,
                "n_extends_in_library": 1,
                "n_compares_to_library": 0,
            }
        }
        w = 0.2
        weights = {"entity_overlap": 0.0, "citation": w, "shared_authors": 0.0}
        ranked = rank_papers(
            candidates[:], db, llm, top_k=1,
            graph_signals=signals, graph_weights=weights,
        )
        assert abs(ranked[0]["score_breakdown"]["graph_citation"] - w) < 1e-6
