"""G4 tests: schema-source relationship extractor + RelationshipValidator.

TDD order: write tests first (RED), then implement to make them GREEN.

Field map verified against config/extraction_schema.yaml:
  assumptions        → DEPENDS_ON
  future_work        → PROPOSES
  limitations        → LIMITED_BY
  novelty_statement  → INTRODUCES
  open_questions     → RAISES_QUESTION
  comparison_to_prior → COMPARES_TO (Paper→Paper via citation DOI resolution)
"""
from __future__ import annotations

from unittest.mock import MagicMock

from scripts.graph.relationship_extractor import (
    RelationshipTuple,
    _excerpt,
    extract_relationships,
)
from scripts.graph.relationship_validator import (
    VALID_PREDICATES,
    RelationshipValidator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tuple(**overrides) -> RelationshipTuple:
    """Build a valid RelationshipTuple with sane defaults."""
    defaults = dict(
        source_doi="10.0/a",
        predicate="PROPOSES",
        target_id="some method",
        target_kind="Entity",
        evidence="ev",
        confidence=1.0,
        field="future_work",
    )
    defaults.update(overrides)
    return RelationshipTuple(**defaults)


# ---------------------------------------------------------------------------
# RelationshipValidator
# ---------------------------------------------------------------------------

class TestRelationshipValidator:
    def test_valid_tuple_passes(self):
        assert RelationshipValidator.is_valid(_make_tuple())

    def test_empty_source_doi_rejected(self):
        assert not RelationshipValidator.is_valid(_make_tuple(source_doi=""))

    def test_unknown_predicate_rejected(self):
        assert not RelationshipValidator.is_valid(_make_tuple(predicate="MADE_UP"))

    def test_empty_target_id_rejected(self):
        assert not RelationshipValidator.is_valid(_make_tuple(target_id=""))

    def test_bad_target_kind_rejected(self):
        assert not RelationshipValidator.is_valid(_make_tuple(target_kind="Node"))

    def test_paper_target_kind_passes(self):
        assert RelationshipValidator.is_valid(
            _make_tuple(target_kind="Paper", predicate="COMPARES_TO")
        )

    def test_all_phase1_predicates_in_valid_set(self):
        for p in {
            "MENTIONS",
            "CITES",
            "COMPARES_TO",
            "DEPENDS_ON",
            "PROPOSES",
            "LIMITED_BY",
            "INTRODUCES",
            "RAISES_QUESTION",
        }:
            assert p in VALID_PREDICATES, f"{p!r} missing from VALID_PREDICATES"


# ---------------------------------------------------------------------------
# _excerpt helper
# ---------------------------------------------------------------------------

class TestExcerpt:
    def test_short_text_unchanged(self):
        assert _excerpt("hello world") == "hello world"

    def test_long_text_truncated(self):
        long = "word " * 100  # 500 chars
        result = _excerpt(long)
        assert len(result) <= 201  # 200 + ellipsis char
        assert result.endswith("…")

    def test_truncation_at_word_boundary(self):
        # Should not break mid-word
        long = "hello world " * 20  # 240 chars
        result = _excerpt(long)
        assert not result[:-1].endswith(" ")  # trimmed trailing space before ellipsis


# ---------------------------------------------------------------------------
# Entity-target predicates
# ---------------------------------------------------------------------------

class TestExtractRelationshipsEntityTargets:
    def test_proposes_from_future_work_list(self):
        """PROPOSES: list field → one edge per resolved entity."""
        extraction = {
            "future_work": ["multi-objective optimization", "ML for chromatography"]
        }
        entity_lookup = {
            ("multi-objective optimization", "method"): "multi-objective optimization",
            ("ml for chromatography", "method"): "ml for chromatography",
        }
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        proposes = [t for t in result if t.predicate == "PROPOSES"]
        assert len(proposes) == 2
        for t in proposes:
            assert t.source_doi == "10.0/a"
            assert t.target_kind == "Entity"
            assert t.confidence == 1.0
            assert t.field == "future_work"

    def test_limited_by_from_limitations(self):
        """LIMITED_BY: list field → one edge per resolved entity."""
        extraction = {"limitations": ["scale-up uncertainty"]}
        entity_lookup = {("scale-up uncertainty", "topic"): "scale up uncertainty"}
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        limited = [t for t in result if t.predicate == "LIMITED_BY"]
        assert len(limited) == 1
        assert limited[0].field == "limitations"
        assert limited[0].target_id == "scale up uncertainty"

    def test_depends_on_from_assumptions_list(self):
        """DEPENDS_ON: list field → one edge per resolved entity."""
        extraction = {"assumptions": ["constant temperature", "ideal mixing"]}
        entity_lookup = {
            ("constant temperature", "topic"): "constant temperature",
            ("ideal mixing", "topic"): "ideal mixing",
        }
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        depends = [t for t in result if t.predicate == "DEPENDS_ON"]
        assert len(depends) == 2

    def test_introduces_from_novelty_statement_string(self):
        """INTRODUCES: string field → single edge."""
        extraction = {"novelty_statement": "novel column chromatography approach"}
        entity_lookup = {
            ("novel column chromatography approach", "topic"): "novel column chromatography approach"
        }
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        intros = [t for t in result if t.predicate == "INTRODUCES"]
        assert len(intros) == 1
        assert intros[0].field == "novelty_statement"

    def test_raises_question_from_open_questions(self):
        """RAISES_QUESTION: list field → one edge per resolved entity."""
        extraction = {"open_questions": ["what is the role of buffer pH?"]}
        entity_lookup = {
            ("what is the role of buffer ph?", "topic"): "what is the role of buffer ph"
        }
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        raises = [t for t in result if t.predicate == "RAISES_QUESTION"]
        assert len(raises) == 1

    def test_missing_field_produces_no_edge(self):
        """Empty extraction_json → no tuples."""
        result = extract_relationships(
            extraction_json={},
            source_doi="10.0/a",
            entity_lookup={},
            state_db=None,
        )
        assert result == []

    def test_null_field_value_skipped(self):
        """Null field values (from LLM null-honesty) produce no edges."""
        result = extract_relationships(
            extraction_json={"future_work": None, "limitations": None},
            source_doi="10.0/a",
            entity_lookup={},
            state_db=None,
        )
        assert result == []

    def test_unknown_target_entity_skipped(self):
        """Surface not in entity_lookup → no edge emitted."""
        extraction = {"future_work": ["something not in lookup"]}
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup={},
            state_db=None,
        )
        assert result == []

    def test_confidence_is_always_1_0(self):
        """All schema-source edges have confidence=1.0."""
        extraction = {"assumptions": ["steady state"]}
        entity_lookup = {("steady state", "topic"): "steady state"}
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        assert all(t.confidence == 1.0 for t in result)

    def test_string_field_treated_as_single_item(self):
        """A string value for a normally-list field is treated as one item."""
        # limitations can sometimes come back as a string from the LLM
        extraction = {"limitations": "only lab scale tested"}
        entity_lookup = {("only lab scale tested", "topic"): "only lab scale tested"}
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        limited = [t for t in result if t.predicate == "LIMITED_BY"]
        assert len(limited) == 1

    def test_multiple_predicates_in_one_extraction(self):
        """Multiple fields in one extraction_json → edges for all resolved entities."""
        extraction = {
            "assumptions": ["ideal flow"],
            "future_work": ["scale-up study"],
        }
        entity_lookup = {
            ("ideal flow", "topic"): "ideal flow",
            ("scale-up study", "method"): "scale-up study",
        }
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        predicates = {t.predicate for t in result}
        assert "DEPENDS_ON" in predicates
        assert "PROPOSES" in predicates

    def test_evidence_trimmed_to_200_chars(self):
        """Evidence field is capped at 200 chars."""
        long_surface = "a " * 120  # 240 chars
        extraction = {"future_work": [long_surface]}
        entity_lookup = {(long_surface.lower().strip(), "method"): "long entity"}
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup=entity_lookup,
            state_db=None,
        )
        for t in result:
            assert len(t.evidence) <= 201  # 200 + ellipsis char


# ---------------------------------------------------------------------------
# COMPARES_TO — Paper→Paper
# ---------------------------------------------------------------------------

class TestExtractRelationshipsComparesTo:
    def test_compares_to_resolves_via_state_db(self):
        """COMPARES_TO emits Paper→Paper when state_db returns a citation edge."""
        extraction = {"comparison_to_prior": "compared to Smith 2020 we achieve higher yield"}
        mock_state_db = MagicMock()
        mock_state_db.get_citation_edges.return_value = [
            {"target_doi": "10.0/smith2020", "ref_id": "ref1", "resolution": "author_year_fuzzy"},
        ]
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup={},
            state_db=mock_state_db,
        )
        compares = [t for t in result if t.predicate == "COMPARES_TO"]
        assert len(compares) >= 1
        assert compares[0].target_kind == "Paper"
        assert compares[0].target_id == "10.0/smith2020"
        assert compares[0].field == "comparison_to_prior"
        assert compares[0].confidence == 1.0
        mock_state_db.get_citation_edges.assert_called_once_with("10.0/a")

    def test_compares_to_skipped_when_no_state_db(self):
        """COMPARES_TO not emitted when state_db=None."""
        extraction = {"comparison_to_prior": "compared to Smith 2020..."}
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup={},
            state_db=None,
        )
        assert all(t.predicate != "COMPARES_TO" for t in result)

    def test_compares_to_skipped_when_no_resolved_dois(self):
        """COMPARES_TO not emitted when state_db returns no resolved citation edges."""
        extraction = {"comparison_to_prior": "compared to Smith 2020..."}
        mock_state_db = MagicMock()
        mock_state_db.get_citation_edges.return_value = []
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup={},
            state_db=mock_state_db,
        )
        assert all(t.predicate != "COMPARES_TO" for t in result)

    def test_compares_to_skips_null_target_doi(self):
        """COMPARES_TO skips citation_edges with target_doi=None (unresolved)."""
        extraction = {"comparison_to_prior": "compared to Jones 2019..."}
        mock_state_db = MagicMock()
        mock_state_db.get_citation_edges.return_value = [
            {"target_doi": None, "ref_id": "[12]", "resolution": "unresolved"},
        ]
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup={},
            state_db=mock_state_db,
        )
        assert all(t.predicate != "COMPARES_TO" for t in result)

    def test_compares_to_multiple_resolved_targets(self):
        """COMPARES_TO emits one edge per resolved citation target."""
        extraction = {"comparison_to_prior": "outperforms Smith 2020 and Jones 2019"}
        mock_state_db = MagicMock()
        mock_state_db.get_citation_edges.return_value = [
            {"target_doi": "10.0/smith2020", "ref_id": "ref1", "resolution": "author_year_fuzzy"},
            {"target_doi": "10.0/jones2019", "ref_id": "ref2", "resolution": "author_year_fuzzy"},
        ]
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup={},
            state_db=mock_state_db,
        )
        compares = [t for t in result if t.predicate == "COMPARES_TO"]
        assert len(compares) == 2
        target_dois = {t.target_id for t in compares}
        assert "10.0/smith2020" in target_dois
        assert "10.0/jones2019" in target_dois

    def test_compares_to_state_db_error_handled_gracefully(self):
        """COMPARES_TO handles state_db exceptions without crashing."""
        extraction = {"comparison_to_prior": "compared to Smith 2020..."}
        mock_state_db = MagicMock()
        mock_state_db.get_citation_edges.side_effect = RuntimeError("db error")
        # Should not raise; simply produces no COMPARES_TO edges
        result = extract_relationships(
            extraction_json=extraction,
            source_doi="10.0/a",
            entity_lookup={},
            state_db=mock_state_db,
        )
        assert all(t.predicate != "COMPARES_TO" for t in result)
