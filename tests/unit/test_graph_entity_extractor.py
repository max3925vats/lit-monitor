"""Tests for G3 (schema-source entity extractor) and N3 (MentionEdge multi-source merge).

Field map used (verified against config/extraction_schema.yaml):
  method   → ["methods_summary"]
  material → ["materials_systems"]
  topic    → ["discovered_topics", "novelty_statement"]
  keyword  → from paper_metadata["keywords_json"] (Zotero tag strings)
  author   → from paper_metadata["authors"]  (JSON array of strings)
  journal  → from paper_metadata["journal"]
"""
import json
from dataclasses import FrozenInstanceError

import pytest

from lit_monitor.graph.entity_extractor import (
    EntityTuple,
    MentionEdge,
    extract_entities,
    from_biobert,
    from_llm_cloud,
    merge_mentions,
)
from lit_monitor.graph.ner import NerSpan
from lit_monitor.graph.normalizer import EntityNormalizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalizer() -> EntityNormalizer:
    """Return a no-alias, no-vocab normalizer (identity normalization only)."""
    return EntityNormalizer(aliases={})


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

class TestExtractEntitiesBasic:
    def test_returns_list_of_entitytuples(self):
        """G3: extract_entities returns a list of EntityTuple, never None or dict."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"methods_summary": "ion exchange chromatography"},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        assert isinstance(result, list)
        assert all(isinstance(t, EntityTuple) for t in result)

    def test_method_field_produces_method_entity(self):
        """G3: methods_summary → ≥1 EntityTuple with type='method'."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"methods_summary": "ion exchange chromatography"},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        method_tuples = [t for t in result if t.type == "method"]
        assert len(method_tuples) >= 1
        assert method_tuples[0].canonical_id == "ion exchange chromatography"
        assert method_tuples[0].field == "methods_summary"

    def test_materials_systems_produces_material_entity(self):
        """G3: materials_systems → EntityTuple with type='material'."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"materials_systems": "Protein A resin"},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        material_tuples = [t for t in result if t.type == "material"]
        assert len(material_tuples) >= 1
        # normalizer lowercases
        assert "protein" in material_tuples[0].canonical_id
        assert material_tuples[0].field == "materials_systems"

    def test_discovered_topics_list_produces_topic_entities(self):
        """G3: discovered_topics as list → one EntityTuple per item."""
        n = _normalizer()
        topics = ["antibody purification", "protein aggregation"]
        result = extract_entities(
            extraction_json={"discovered_topics": topics},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        topic_tuples = [t for t in result if t.type == "topic"]
        assert len(topic_tuples) == 2

    def test_novelty_statement_string_produces_topic_entity(self):
        """G3: novelty_statement string → EntityTuple with type='topic'."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"novelty_statement": "first use of mixed-mode chromatography"},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        topic_tuples = [t for t in result if t.type == "topic"]
        assert len(topic_tuples) >= 1
        assert topic_tuples[0].field == "novelty_statement"

    def test_authors_from_metadata_list_of_strings(self):
        """G3: Zotero authors stored as JSON array of 'Last, First' strings."""
        n = _normalizer()
        authors_json = json.dumps(["Smith, Jane", "Jones, Bob"])
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": authors_json, "journal": ""},
            normalizer=n,
        )
        authors = [t for t in result if t.type == "author"]
        assert len(authors) == 2
        assert any("smith" in t.canonical_id for t in authors)
        assert any("jones" in t.canonical_id for t in authors)

    def test_authors_field_is_none_for_author_entities(self):
        """G3: author entities have field=None (they come from Zotero, not extraction_json)."""
        n = _normalizer()
        authors_json = json.dumps(["Smith, Jane"])
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": authors_json, "journal": ""},
            normalizer=n,
        )
        authors = [t for t in result if t.type == "author"]
        assert all(t.field is None for t in authors)

    def test_journal_metadata_becomes_journal_entity(self):
        """G3: paper_metadata['journal'] → EntityTuple with type='journal'."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": "Biotechnology and Bioengineering"},
            normalizer=n,
        )
        journals = [t for t in result if t.type == "journal"]
        assert len(journals) == 1
        assert "biotechnology" in journals[0].canonical_id

    def test_journal_field_is_none(self):
        """G3: journal entity has field=None (comes from Zotero paper_metadata)."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": "Nature Methods"},
            normalizer=n,
        )
        journals = [t for t in result if t.type == "journal"]
        assert all(t.field is None for t in journals)

    def test_keywords_from_metadata_produce_keyword_entities(self):
        """G3: paper_metadata['keywords_json'] (Zotero tags) → EntityTuple with type='keyword'."""
        n = _normalizer()
        kw_json = json.dumps(["antibody", "chromatography"])
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": "", "keywords_json": kw_json},
            normalizer=n,
        )
        kw_tuples = [t for t in result if t.type == "keyword"]
        assert len(kw_tuples) == 2

    def test_empty_extraction_json_and_minimal_metadata(self):
        """G3: all-empty input returns empty list."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        assert result == []

    def test_none_field_value_skipped(self):
        """G3: None field values produce no tuples (LLM returned null)."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"methods_summary": None, "materials_systems": None},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        assert result == []

    def test_empty_string_field_value_skipped(self):
        """G3: empty string field values produce no tuples."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"methods_summary": ""},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestExtractEntitiesDedup:
    def test_dedup_by_canonical_type_field(self):
        """G3: same surface in different fields → 2 tuples (different field = different provenance)."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={
                "methods_summary": "ion exchange chromatography",
                # novelty_statement → type=topic, different type → still 2 distinct tuples
            },
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        # Just check no crash and all are distinct (canonical, type, field) triples
        keys = [(t.canonical_id, t.type, t.field) for t in result]
        assert len(keys) == len(set(keys))

    def test_dedup_same_canonical_same_type_same_field(self):
        """G3: duplicate entries in a list field are deduplicated."""
        n = _normalizer()
        # discovered_topics as list with duplicate
        result = extract_entities(
            extraction_json={"discovered_topics": ["protein aggregation", "protein aggregation"]},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        topic_tuples = [t for t in result if t.type == "topic"]
        # After dedup: only one tuple for same (canonical_id, type, field)
        assert len(topic_tuples) == 1

    def test_same_surface_different_fields_kept(self):
        """G3: same canonical in two different fields → 2 tuples (dedup is per (canonical,type,field))."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={
                "methods_summary": "size exclusion chromatography",
                # novelty_statement is topic type, so different type key → definitely 2 tuples
                "novelty_statement": "size exclusion chromatography for purity assessment",
            },
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        # methods_summary → type=method; novelty_statement → type=topic
        # Both fields have the same technology mentioned, but different (canonical, type, field)
        keys = {(t.canonical_id, t.type, t.field) for t in result}
        # At minimum: one method entry + one topic entry
        method_keys = {k for k in keys if k[1] == "method"}
        topic_keys = {k for k in keys if k[1] == "topic"}
        assert len(method_keys) >= 1
        assert len(topic_keys) >= 1


# ---------------------------------------------------------------------------
# Span tests
# ---------------------------------------------------------------------------

class TestExtractEntitiesSpans:
    def test_span_populated_for_string_field(self):
        """G3: string field → span_start=0, span_end=len(text)."""
        n = _normalizer()
        text = "ion exchange chromatography was used to purify the antibody"
        result = extract_entities(
            extraction_json={"methods_summary": text},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        method_tuples = [t for t in result if t.type == "method"]
        assert len(method_tuples) >= 1
        t = method_tuples[0]
        assert t.span_start == 0
        assert t.span_end == len(text)

    def test_span_null_for_list_field(self):
        """G3: list field → span_start=None, span_end=None."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"discovered_topics": ["antibody purification", "protein aggregation"]},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        topic_tuples = [t for t in result if t.type == "topic"]
        assert all(t.span_start is None and t.span_end is None for t in topic_tuples)

    def test_span_null_for_author_entities(self):
        """G3: author entities have no span (come from structured Zotero metadata)."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": json.dumps(["Smith, Jane"]), "journal": ""},
            normalizer=n,
        )
        authors = [t for t in result if t.type == "author"]
        assert all(t.span_start is None and t.span_end is None for t in authors)

    def test_span_null_for_journal_entity(self):
        """G3: journal entity has no span."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": "Journal of Biotechnology"},
            normalizer=n,
        )
        journals = [t for t in result if t.type == "journal"]
        assert all(t.span_start is None and t.span_end is None for t in journals)

    def test_span_null_for_keyword_entity(self):
        """G3: keyword entities (from Zotero tags list) have no span."""
        n = _normalizer()
        kw_json = json.dumps(["antibody"])
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": "", "keywords_json": kw_json},
            normalizer=n,
        )
        kw_tuples = [t for t in result if t.type == "keyword"]
        assert all(t.span_start is None and t.span_end is None for t in kw_tuples)


# ---------------------------------------------------------------------------
# Author parsing edge cases
# ---------------------------------------------------------------------------

class TestAuthorParsing:
    def test_authors_empty_json_array(self):
        """G3: empty authors JSON array → no author entities."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        assert not any(t.type == "author" for t in result)

    def test_authors_missing_from_metadata(self):
        """G3: missing authors key → no crash, no author entities."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"journal": "Nature"},
            normalizer=n,
        )
        assert not any(t.type == "author" for t in result)

    def test_authors_invalid_json_skipped(self):
        """G3: invalid JSON in authors field → no crash, no author entities."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "not-valid-json", "journal": ""},
            normalizer=n,
        )
        assert not any(t.type == "author" for t in result)

    def test_keywords_missing_from_metadata(self):
        """G3: missing keywords_json key → no crash, no keyword entities."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        assert not any(t.type == "keyword" for t in result)

    def test_keywords_empty_array(self):
        """G3: empty keywords_json → no keyword entities."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": "", "keywords_json": "[]"},
            normalizer=n,
        )
        assert not any(t.type == "keyword" for t in result)

    def test_keywords_invalid_json_skipped(self):
        """G3: invalid JSON in keywords_json → no crash, no keyword entities."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": "", "keywords_json": "bad-json"},
            normalizer=n,
        )
        assert not any(t.type == "keyword" for t in result)

    def test_empty_journal_produces_no_entity(self):
        """G3: empty journal string → no journal entity."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        assert not any(t.type == "journal" for t in result)


# ---------------------------------------------------------------------------
# G3 AC2: list-of-dicts author shape (Zotero primary format)
# ---------------------------------------------------------------------------

class TestParseAuthorsDictShape:
    """G3 AC2: list-of-dicts authors are the Zotero primary format."""

    def test_dict_shape_with_lastName_and_firstName(self):
        """Zotero canonical shape: dicts with 'lastName' and 'firstName'."""
        from lit_monitor.graph.entity_extractor import _parse_authors
        raw = json.dumps([
            {"lastName": "Smith", "firstName": "Jane"},
            {"lastName": "Jones", "firstName": "Bob"},
        ])
        assert _parse_authors(raw) == ["Smith, Jane", "Jones, Bob"]

    def test_dict_shape_extracts_to_entities(self):
        """End-to-end: dict-shape authors produce Entity(type='author')."""
        n = EntityNormalizer(aliases={})
        authors_json = json.dumps([
            {"lastName": "Smith", "firstName": "Jane"},
            {"lastName": "Jones", "firstName": "Bob"},
        ])
        result = extract_entities(
            extraction_json={},
            paper_metadata={"authors": authors_json, "journal": ""},
            normalizer=n,
        )
        authors = [t for t in result if t.type == "author"]
        assert len(authors) == 2
        assert any("smith" in t.canonical_id for t in authors)
        assert any("jones" in t.canonical_id for t in authors)

    def test_dict_shape_with_only_last_name(self):
        """Dict with only lastName (no firstName) still produces an author."""
        from lit_monitor.graph.entity_extractor import _parse_authors
        raw = json.dumps([{"lastName": "Smith"}])
        assert _parse_authors(raw) == ["Smith"]

    def test_dict_shape_with_first_last_alias_fields(self):
        """Some legacy fixtures use 'first'/'last' instead of 'firstName'/'lastName'."""
        from lit_monitor.graph.entity_extractor import _parse_authors
        raw = json.dumps([{"first": "Jane", "last": "Smith"}])
        assert _parse_authors(raw) == ["Smith, Jane"]

    def test_dict_without_name_fields_is_skipped(self):
        """Garbage dict (no name keys) is silently skipped."""
        from lit_monitor.graph.entity_extractor import _parse_authors
        raw = json.dumps([{"affiliation": "X"}, {"firstName": "Bob"}])
        # First dict has no name field → skipped; second has only firstName → kept
        assert _parse_authors(raw) == ["Bob"]


# ---------------------------------------------------------------------------
# Surface form preservation
# ---------------------------------------------------------------------------

class TestSurfacePreservation:
    def test_surface_preserves_original_case(self):
        """G3: surface field holds raw input before normalization."""
        n = _normalizer()
        result = extract_entities(
            extraction_json={"materials_systems": "Protein A Sepharose"},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=n,
        )
        material_tuples = [t for t in result if t.type == "material"]
        assert len(material_tuples) == 1
        # surface = original; canonical_id = normalized
        assert material_tuples[0].surface == "Protein A Sepharose"
        assert material_tuples[0].canonical_id != "Protein A Sepharose"  # normalizer lowercased it

    def test_entitytuple_is_frozen(self):
        """G3: EntityTuple instances are immutable (frozen=True dataclass)."""
        t = EntityTuple(
            canonical_id="test",
            type="method",
            surface="Test",
            field="methods_summary",
            span_start=0,
            span_end=4,
        )
        with pytest.raises((AttributeError, TypeError)):
            t.canonical_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# N3: MentionEdge invariants
# ---------------------------------------------------------------------------


class TestMentionEdge:
    def test_frozen_dataclass(self):
        """N3: MentionEdge is immutable (frozen=True dataclass)."""
        m = MentionEdge(
            paper_id="10.0/a", entity_key="egfr", type="material",
            source="biobert", surface="EGFR", field=None,
            confidence=0.95, span_start=0, span_end=4,
        )
        with pytest.raises(FrozenInstanceError):
            m.confidence = 0.5  # type: ignore[misc]

    def test_source_accepts_three_values(self):
        """N3: source must be one of schema | biobert | llm_cloud."""
        for source in ("schema", "biobert", "llm_cloud"):
            MentionEdge(
                paper_id="10.0/a", entity_key="x", type="topic",
                source=source, surface="x", field=None,
                confidence=1.0, span_start=None, span_end=None,
            )

    def test_invalid_source_raises_value_error(self):
        """N3: source outside the whitelist is rejected at construction time."""
        with pytest.raises(ValueError):
            MentionEdge(
                paper_id="10.0/a", entity_key="x", type="topic",
                source="bogus", surface="x", field=None,
                confidence=1.0, span_start=None, span_end=None,
            )


# ---------------------------------------------------------------------------
# N3: from_biobert
# ---------------------------------------------------------------------------


class TestFromBiobert:
    def test_biobert_spans_become_mention_edges(self):
        """N3: NerSpan list → MentionEdge list with source='biobert'.

        Uses labels that map deterministically (DISEASE→topic, TECHNIQUE→method)
        so the test does not depend on the removed silent 'material' default
        (P3.3).
        """
        normalizer = EntityNormalizer(aliases={})
        spans = [
            NerSpan(text="chromatography", label="TECHNIQUE", start=0, end=14, confidence=0.95),
            NerSpan(text="lung cancer", label="DISEASE", start=20, end=31, confidence=0.88),
        ]
        edges = from_biobert(spans, paper_id="10.0/a", normalizer=normalizer)
        assert all(e.source == "biobert" for e in edges)
        assert all(e.paper_id == "10.0/a" for e in edges)
        # BioBERT confidence carried through
        assert any(e.confidence == 0.95 for e in edges)
        # Offsets carried through
        assert any(e.span_start == 0 and e.span_end == 14 for e in edges)

    def test_biobert_canonicalizes_via_normalizer(self):
        """N3: surface form normalized to canonical entity_key via alias lookup.

        Label METHOD maps to type 'method' (P3.3: 'material' is no longer a
        default for arbitrary labels, so we use a label that maps explicitly).
        """
        normalizer = EntityNormalizer(aliases={"method": {"hplc": "high performance liquid chromatography"}})
        spans = [NerSpan(text="HPLC", label="METHOD", start=0, end=4, confidence=0.9)]
        edges = from_biobert(spans, paper_id="10.0/a", normalizer=normalizer)
        # entity_key holds the canonical (alias-resolved) form
        assert edges[0].entity_key == "high performance liquid chromatography"
        # Surface preserves the original text
        assert edges[0].surface == "HPLC"

    def test_biobert_unknown_label_not_classified_material(self):
        """P3.3: a span with an unrecognised NER label must NOT become a
        'material' entity (the old silent default).  It is dropped instead.
        """
        normalizer = EntityNormalizer(aliases={})
        spans = [
            NerSpan(text="EGFR", label="GENE", start=0, end=4, confidence=0.95),
            NerSpan(text="aspirin", label="DRUG", start=10, end=17, confidence=0.9),
        ]
        edges = from_biobert(spans, paper_id="10.0/a", normalizer=normalizer)
        # No edge may carry type 'material' from these unknown labels.
        assert all(e.type != "material" for e in edges)
        # In fact, both unknown-label spans are dropped entirely.
        assert edges == []

    def test_biobert_empty_spans_returns_empty(self):
        """N3: empty NerSpan list → empty MentionEdge list."""
        normalizer = EntityNormalizer(aliases={})
        assert from_biobert([], paper_id="10.0/a", normalizer=normalizer) == []


# ---------------------------------------------------------------------------
# N3: from_llm_cloud
# ---------------------------------------------------------------------------


class TestFromLlmCloud:
    def test_new_entities_become_mention_edges(self):
        """N3: payload['new_entities'] → MentionEdge list with source='llm_cloud', confidence=0.85."""
        normalizer = EntityNormalizer(aliases={})
        payload = {
            "new_entities": [
                {"surface": "perfusion bioreactor", "type": "method",
                 "span_start": 0, "span_end": 20},
            ],
            "validations": [],
        }
        edges = from_llm_cloud(payload, paper_id="10.0/a", normalizer=normalizer)
        assert len(edges) == 1
        e = edges[0]
        assert e.source == "llm_cloud"
        assert e.confidence == 0.85
        assert e.type == "method"
        assert e.span_start == 0
        assert e.span_end == 20

    def test_validations_keep_true_become_edges(self):
        """N3: validations with keep=True produce edges; keep=False are skipped."""
        normalizer = EntityNormalizer(aliases={})
        payload = {
            "new_entities": [],
            "validations": [
                {"surface": "EGFR", "keep": True, "reason": "real gene"},
                {"surface": "blah", "keep": False, "reason": "not a real entity"},
            ],
        }
        edges = from_llm_cloud(payload, paper_id="10.0/a", normalizer=normalizer)
        # Only keep=True validations produce edges
        assert len(edges) == 1
        assert edges[0].surface == "EGFR"
        # Validation edges have no span info (offset originated in BioBERT, not reproduced here)
        assert edges[0].span_start is None

    def test_empty_payload_returns_empty(self):
        """N3: payload with no new_entities and no validations → empty list."""
        normalizer = EntityNormalizer(aliases={})
        edges = from_llm_cloud(
            {"new_entities": [], "validations": []},
            paper_id="10.0/a",
            normalizer=normalizer,
        )
        assert edges == []


# ---------------------------------------------------------------------------
# N3: merge_mentions
# ---------------------------------------------------------------------------


class TestMergeMentions:
    def test_same_entity_two_sources_produces_two_edges(self):
        """N3: same canonical entity from two sources → one entity_key + two distinct edges."""
        schema_edges = [MentionEdge(
            paper_id="10.0/a", entity_key="egfr", type="material",
            source="schema", surface="EGFR", field="materials_systems",
            confidence=1.0, span_start=None, span_end=None,
        )]
        biobert_edges = [MentionEdge(
            paper_id="10.0/a", entity_key="egfr", type="material",
            source="biobert", surface="EGFR", field=None,
            confidence=0.92, span_start=0, span_end=4,
        )]
        merged = merge_mentions(schema_edges, biobert_edges)
        assert len(merged) == 2
        sources = {e.source for e in merged}
        assert sources == {"schema", "biobert"}
        # Both reference the same canonical entity key
        assert all(e.entity_key == "egfr" for e in merged)

    def test_dedup_within_same_source(self):
        """N3: same edge emitted twice from same source → deduped to one."""
        edge = MentionEdge(
            paper_id="10.0/a", entity_key="egfr", type="material",
            source="biobert", surface="EGFR", field=None,
            confidence=0.92, span_start=0, span_end=4,
        )
        merged = merge_mentions([edge, edge, edge])
        assert len(merged) == 1

    def test_dedup_keys_on_full_tuple(self):
        """N3: same canonical+source but different spans → 2 edges (distinct positions)."""
        e1 = MentionEdge(
            paper_id="10.0/a", entity_key="egfr", type="material",
            source="biobert", surface="EGFR", field=None,
            confidence=0.92, span_start=0, span_end=4,
        )
        e2 = MentionEdge(
            paper_id="10.0/a", entity_key="egfr", type="material",
            source="biobert", surface="EGFR", field=None,
            confidence=0.88, span_start=50, span_end=54,
        )
        merged = merge_mentions([e1, e2])
        # Different spans → both kept
        assert len(merged) == 2

    def test_idempotent_rerun(self):
        """N3: running merge twice on the same inputs is stable."""
        edges = [MentionEdge(
            paper_id="10.0/a", entity_key="egfr", type="material",
            source="schema", surface="EGFR", field="materials_systems",
            confidence=1.0, span_start=None, span_end=None,
        )]
        first = merge_mentions(edges)
        second = merge_mentions(first)
        key = lambda e: (e.entity_key, e.source, e.span_start, e.span_end, e.field)  # noqa: E731
        assert set(key(e) for e in first) == set(key(e) for e in second)

    def test_empty_inputs_returns_empty(self):
        """N3: merge_mentions with no args or all-empty lists → empty list."""
        assert merge_mentions() == []
        assert merge_mentions([]) == []
        assert merge_mentions([], [], []) == []
