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

from scripts.graph.normalizer import EntityNormalizer

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
