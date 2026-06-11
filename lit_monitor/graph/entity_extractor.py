"""G3: schema-source entity extractor for the lit-monitor knowledge graph.

Maps already-extracted JSON fields (no new LLM calls) and Zotero paper_metadata
to EntityTuple rows ready for G6 to write as MENTIONS edges with source='schema'.

Six entity types in Phase 1: topic, method, material, author, journal, keyword.

Field map (verified against config/extraction_schema.yaml as of v0.4.0):
  method   → extraction_json["methods_summary"]          (string field)
  material → extraction_json["materials_systems"]        (string field)
  topic    → extraction_json["discovered_topics"]        (list or string field)
             extraction_json["novelty_statement"]        (string field)
  keyword  → paper_metadata["keywords_json"]             (JSON array of Zotero tags)
  author   → paper_metadata["authors"]                   (JSON array of dicts or strings)
  journal  → paper_metadata["journal"]                   (plain string)

Deviation from the original brief:
  - "methodology" does not exist in the schema → removed.
  - "materials"   does not exist in the schema → replaced with "materials_systems".
  - "topics"      does not exist in the schema → replaced with "discovered_topics".
  - "keywords_json" is a Zotero tag column in state.db (paper_metadata), NOT an
    extraction_json field; keyword extraction therefore reads paper_metadata.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from lit_monitor.graph.normalizer import EntityNormalizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _SCHEMA_ENTITY_SOURCES — extraction_json field map
# ---------------------------------------------------------------------------
# Maps entity type → list of extraction_json field names to scan.
# Authors, journal, and keywords come from paper_metadata (Zotero), not here.
_SCHEMA_ENTITY_SOURCES: dict[str, list[str]] = {
    "method":   ["methods_summary"],
    "material": ["materials_systems"],
    "topic":    ["discovered_topics", "novelty_statement"],
}


# ---------------------------------------------------------------------------
# EntityTuple
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityTuple:
    """One entity occurrence in a paper, ready for a MENTIONS edge in G6.

    Attributes
    ----------
    canonical_id:
        Normalized canonical form from EntityNormalizer (lowercase, ASCII-folded,
        singularized, alias-resolved).
    type:
        One of: topic / method / material / author / journal / keyword.
    surface:
        Raw surface form before normalization (preserves original casing).
    field:
        Source field name in extraction_json (e.g. ``"methods_summary"``),
        or ``None`` for Zotero-origin entities (author, journal, keyword).
    span_start:
        Character offset of the surface form's start position within the field's
        string value (via ``str.find``).  ``None`` when the field is a list or the
        entity comes from structured metadata (author / journal / keyword).
    span_end:
        Character offset one past the surface form's end.  ``None`` in same cases
        as ``span_start``.
    """

    canonical_id: str
    type: str
    surface: str
    field: str | None
    span_start: int | None
    span_end: int | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emit_from_text_field(
    text: str,
    field: str,
    type_: str,
    normalizer: EntityNormalizer,
) -> list[EntityTuple]:
    """Produce a single EntityTuple from a non-empty string field.

    Phase 1 treats the entire field value as one entity surface (most
    extraction_json string fields contain a short noun phrase or paragraph
    that is the entity itself, not embedded entity mentions).  Span is
    always (0, len(text)).

    Phase 2 BioBERT NER will replace this with substring-level extraction.
    """
    text = text.strip()
    if not text:
        return []
    canonical, _via = normalizer.normalize(text, type_=type_)
    return [
        EntityTuple(
            canonical_id=canonical,
            type=type_,
            surface=text,
            field=field,
            span_start=0,
            span_end=len(text),
        )
    ]


def _emit_from_list_field(
    items: list[Any],
    field: str,
    type_: str,
    normalizer: EntityNormalizer,
) -> list[EntityTuple]:
    """Produce one EntityTuple per non-empty item in a list field.

    Spans are None because list items have no character position in a parent
    string — there is no single field text to search within.
    """
    out: list[EntityTuple] = []
    for item in items:
        if not item:
            continue
        surface = str(item).strip()
        if not surface:
            continue
        canonical, _via = normalizer.normalize(surface, type_=type_)
        out.append(EntityTuple(
            canonical_id=canonical,
            type=type_,
            surface=surface,
            field=field,
            span_start=None,
            span_end=None,
        ))
    return out


def _parse_json_string_array(raw: Any) -> list[str]:
    """Parse a JSON array of strings (e.g. Zotero keywords_json).

    Skips non-string items with a debug log so unexpected shapes are observable.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw:
        if isinstance(x, str):
            stripped = x.strip()
            if stripped:
                out.append(stripped)
        elif x:
            logger.debug("_parse_json_string_array: skipping non-str %r", type(x).__name__)
    return out


def _parse_authors(raw: Any) -> list[str]:
    """Parse Zotero authors field — supports both list-of-strings and list-of-dicts.

    Zotero stores authors as either:
      - list of dicts: [{"lastName": "Smith", "firstName": "Jane"}, ...]
      - list of strings: ["Smith, Jane", "Jones, Bob"]

    Returns canonical surface forms ("Last, First" preferred, falls back to
    whichever name part is present).
    """
    # Reuse the JSON-decode + type-check from the existing helper, but bail
    # before the str() coercion so dicts survive the parse.
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    for item in raw:
        if not item:
            continue
        if isinstance(item, dict):
            # Zotero shape: {"firstName": "Jane", "lastName": "Smith"}
            # Also tolerate the "first"/"last" alias seen in some test fixtures.
            last = (item.get("lastName") or item.get("last") or "").strip()
            first = (item.get("firstName") or item.get("first") or "").strip()
            if last and first:
                name = f"{last}, {first}"
            elif last:
                name = last
            elif first:
                name = first
            else:
                logger.debug("_parse_authors: dict without name fields: %r", item)
                continue
            out.append(name)
        elif isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
        else:
            logger.debug("_parse_authors: skipping non-str/dict item: %r", type(item).__name__)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(
    extraction_json: dict[str, Any],
    paper_metadata: dict[str, Any],
    normalizer: EntityNormalizer,
) -> list[EntityTuple]:
    """Extract all schema-source entities from one paper.

    No LLM calls are made.  All input data must already be available.

    Parameters
    ----------
    extraction_json:
        The JSON blob stored in state.db ``papers.extraction_json``, already
        parsed into a dict.  May be empty if the paper has not been extracted yet.
    paper_metadata:
        Zotero-origin fields from the same state.db row:
        ``authors`` (JSON array of dicts or strings), ``journal`` (plain string),
        and optionally ``keywords_json`` (JSON array of Zotero tag strings).
    normalizer:
        EntityNormalizer instance from G2.  Deduplication/fuzzy-collapse scope
        is the normalizer's current vocab; the caller is responsible for calling
        ``normalizer.add_to_vocab()`` after each entity is committed to KuzuDB.

    Returns
    -------
    list[EntityTuple]
        Deduplicated by ``(canonical_id, type, field)``; same entity appearing
        in two different fields produces two tuples (different provenance).
        List order follows the declaration order in ``_SCHEMA_ENTITY_SOURCES``
        then Zotero fields (keywords → authors → journal).
    """
    raw: list[EntityTuple] = []

    # 1. extraction_json fields: method, material, topic
    # Phase 1: whole-field span (0, len(text)). Phase 2's BioBERT NER will
    # replace this with substring-level extraction across the field text.
    for type_, field_names in _SCHEMA_ENTITY_SOURCES.items():
        for field in field_names:
            value = extraction_json.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                raw.extend(_emit_from_text_field(value, field, type_, normalizer))
            elif isinstance(value, list):
                raw.extend(_emit_from_list_field(value, field, type_, normalizer))
            else:
                logger.debug(
                    "extract_entities: skipping unsupported type for field %r (%s)",
                    field, type(value).__name__,
                )

    # 2. Keywords from paper_metadata (Zotero tags — not in extraction_json)
    kw_raw = paper_metadata.get("keywords_json")
    for kw in _parse_json_string_array(kw_raw):
        canonical, _via = normalizer.normalize(kw, type_="keyword")
        raw.append(EntityTuple(
            canonical_id=canonical,
            type="keyword",
            surface=kw,
            field=None,     # Zotero origin → no extraction_json field
            span_start=None,
            span_end=None,
        ))

    # 3. Authors from paper_metadata (Zotero — stored as JSON array of dicts or strings)
    for author in _parse_authors(paper_metadata.get("authors")):
        canonical, _via = normalizer.normalize(author, type_="author")
        raw.append(EntityTuple(
            canonical_id=canonical,
            type="author",
            surface=author,
            field=None,
            span_start=None,
            span_end=None,
        ))

    # 4. Journal from paper_metadata (Zotero — plain string)
    journal_val = paper_metadata.get("journal")
    if journal_val:
        surface = str(journal_val).strip()
        if surface:
            canonical, _via = normalizer.normalize(surface, type_="journal")
            raw.append(EntityTuple(
                canonical_id=canonical,
                type="journal",
                surface=surface,
                field=None,
                span_start=None,
                span_end=None,
            ))

    # 5. Deduplicate by (canonical_id, type, field) — preserve first-seen order
    seen: set[tuple[str, str, str | None]] = set()
    dedup: list[EntityTuple] = []
    for t in raw:
        key = (t.canonical_id, t.type, t.field)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)

    return dedup


# ---------------------------------------------------------------------------
# N3: MentionEdge + multi-source merge helpers
# ---------------------------------------------------------------------------
# Fuses Phase 1 (schema), N1 (BioBERT), and N2 (cloud-Ollama long-tail) into
# a single set of MentionEdges with per-source provenance.
#
# Same entity from two sources → ONE Entity node + TWO MENTIONS edges.
# Dedup key: (paper_id, entity_key, source, span_start, span_end, field).

_VALID_SOURCES: frozenset[str] = frozenset({"schema", "biobert", "llm_cloud"})

# Default confidence per source when no span-level score is available.
_LLM_CLOUD_CONFIDENCE: float = 0.85
_SCHEMA_CONFIDENCE: float = 1.0


@dataclass(frozen=True)
class MentionEdge:
    """One MENTIONS edge occurrence with per-source provenance.

    Attributes
    ----------
    paper_id:
        Source paper DOI or other stable identifier.
    entity_key:
        Canonical entity id after normalization (lowercase, alias-resolved).
    type:
        One of the 6 closed entity types: topic / method / material /
        author / journal / keyword.
    source:
        Extraction source — must be one of ``"schema"``, ``"biobert"``,
        or ``"llm_cloud"``.
    surface:
        Original surface form before normalization (original casing).
    field:
        extraction_json field name (e.g. ``"methods_summary"``) or ``None``
        when the entity originates from structured metadata or a NER span
        without a field association.
    confidence:
        Per-source confidence: BioBERT span probability, 0.85 for LLM
        cloud, 1.0 for schema.
    span_start:
        Character offset of the span start in the source field text.
        ``None`` when offsets are not available (schema / validation edges).
    span_end:
        Character offset one past the span end.  ``None`` in same cases.
    """

    paper_id: str
    entity_key: str
    type: str
    source: str
    surface: str
    field: str | None
    confidence: float
    span_start: int | None
    span_end: int | None

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"MentionEdge.source must be one of {sorted(_VALID_SOURCES)!r}, "
                f"got {self.source!r}"
            )


def _ner_label_to_type(label: str) -> str | None:
    """N3: map a BioBERT NER label to one of the closed entity types.

    Conservative heuristic for Phase 2.  Phase 3 can refine with per-corpus
    learning (N6 propose-aliases).

    P3.3: a label we do not recognise returns ``None`` (caller drops the span)
    rather than silently bucketing it as ``"material"``.  The old default
    conflated distinct biomedical entity classes (genes, proteins, drugs) into
    ``material``, polluting that type.  Dropping is the safe, schema-stable
    choice: it introduces no new ``type`` value into the closed vocabulary
    (topic / method / material / author / journal / keyword) and loses only
    spans we cannot confidently type.  See the escalation note in the P3 report
    about whether to add an explicit ``"other"`` catch-all type instead.

    Args:
        label: Raw NER label from the BioBERT pipeline (e.g. ``"GENE"``,
               ``"DISEASE"``).

    Returns:
        One of ``"topic"`` / ``"method"``, or ``None`` for an unrecognised
        label (the caller skips spans that map to ``None``).
    """
    label_upper = (label or "").upper()
    if "DISEASE" in label_upper or "CONDITION" in label_upper:
        return "topic"
    if "METHOD" in label_upper or "TECHNIQUE" in label_upper:
        return "method"
    # P3.3: unrecognised label — do NOT default to "material" (that conflated
    # genes/proteins/drugs into the material bucket).  Return None so the
    # caller drops the span instead of misclassifying it.
    return None


def from_biobert(
    spans: list[Any],
    paper_id: str,
    normalizer: EntityNormalizer,
) -> list[MentionEdge]:
    """N3: convert BioBERT NerSpans into MentionEdges with source='biobert'.

    Args:
        spans: List of ``NerSpan`` objects from ``BiobertNER.extract()``.
        paper_id: Identifier of the source paper (DOI or state.db key).
        normalizer: ``EntityNormalizer`` instance from G2 for canonical lookup.

    Returns:
        List of ``MentionEdge`` objects, one per span. Empty if spans is empty
        or all spans normalize to empty strings.
    """
    out: list[MentionEdge] = []
    for span in spans:
        type_ = _ner_label_to_type(span.label)
        # P3.3: drop spans whose NER label we cannot map to a known type
        # instead of silently bucketing them as "material".
        if type_ is None:
            continue
        canonical, _via = normalizer.normalize(span.text, type_=type_)
        if not canonical:
            continue
        out.append(MentionEdge(
            paper_id=paper_id,
            entity_key=canonical,
            type=type_,
            source="biobert",
            surface=span.text,
            field=None,
            confidence=float(span.confidence),
            span_start=int(span.start),
            span_end=int(span.end),
        ))
    return out


def from_llm_cloud(
    payload: dict[str, Any],
    paper_id: str,
    normalizer: EntityNormalizer,
) -> list[MentionEdge]:
    """N3: convert N2's long-tail+validation payload into MentionEdges.

    Payload shape (from N2 cloud-Ollama extractor)::

        {
          "new_entities": [
              {"surface": "...", "type": "...", "span_start": int, "span_end": int},
              ...
          ],
          "validations": [
              {"surface": "...", "keep": bool, "reason": "..."},
              ...
          ],
        }

    - ``new_entities`` → MentionEdge with span offsets when present; source='llm_cloud'.
    - ``validations[keep=True]`` → MentionEdge, no span offsets (offsets lived in BioBERT).
    - ``validations[keep=False]`` → silently skipped.

    Args:
        payload: Dict conforming to the N2 output schema.
        paper_id: Identifier of the source paper.
        normalizer: ``EntityNormalizer`` instance from G2.

    Returns:
        List of ``MentionEdge`` objects with source='llm_cloud'.
    """
    out: list[MentionEdge] = []

    for ent in payload.get("new_entities") or []:
        if not isinstance(ent, dict):
            continue
        surface = ent.get("surface")
        type_ = ent.get("type")
        if not isinstance(surface, str) or not surface:
            continue
        if not isinstance(type_, str) or not type_:
            continue
        canonical, _via = normalizer.normalize(surface, type_=type_)
        if not canonical:
            continue
        # Carry span offsets only when both are integers.
        span_start = ent.get("span_start")
        span_end = ent.get("span_end")
        if not isinstance(span_start, int) or not isinstance(span_end, int):
            span_start = None
            span_end = None
        out.append(MentionEdge(
            paper_id=paper_id,
            entity_key=canonical,
            type=type_,
            source="llm_cloud",
            surface=surface,
            field=None,
            confidence=_LLM_CLOUD_CONFIDENCE,
            span_start=span_start,
            span_end=span_end,
        ))

    for val in payload.get("validations") or []:
        if not isinstance(val, dict):
            continue
        if not val.get("keep"):
            continue
        surface = val.get("surface")
        if not isinstance(surface, str) or not surface:
            continue
        # Validation edges default to type='material' (BioBERT genetic NER default).
        # No offsets: the original span lived in BioBERT and isn't reproduced here.
        canonical, _via = normalizer.normalize(surface, type_="material")
        if not canonical:
            continue
        out.append(MentionEdge(
            paper_id=paper_id,
            entity_key=canonical,
            type="material",
            source="llm_cloud",
            surface=surface,
            field=None,
            confidence=_LLM_CLOUD_CONFIDENCE,
            span_start=None,
            span_end=None,
        ))

    return out


def merge_mentions(*edge_lists: list[MentionEdge]) -> list[MentionEdge]:
    """N3: merge and deduplicate MentionEdge lists from multiple sources.

    Dedup key: ``(paper_id, entity_key, source, span_start, span_end, field)``.

    Same entity from two different sources → two edges are kept (one per source).
    Same edge emitted twice from the same source → collapsed to one.

    Args:
        *edge_lists: Any number of ``list[MentionEdge]`` positional arguments.

    Returns:
        Flat deduplicated list in encounter order.
    """
    seen: set[tuple] = set()
    out: list[MentionEdge] = []
    for edges in edge_lists:
        for e in edges:
            key = (e.paper_id, e.entity_key, e.source, e.span_start, e.span_end, e.field)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    return out
