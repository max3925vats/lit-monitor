"""G4: schema-source typed-relationship extractor.

Maps 6 extraction_json fields to typed predicate edges. No LLM calls.
Reuses citation_edges from state.db for COMPARES_TO DOI resolution.

Field map (verified against config/extraction_schema.yaml):
  assumptions        → DEPENDS_ON   (list or string)
  future_work        → PROPOSES     (list or string)
  limitations        → LIMITED_BY   (list or string)
  novelty_statement  → INTRODUCES   (string)
  open_questions     → RAISES_QUESTION (list or string)
  comparison_to_prior → COMPARES_TO  (string; Paper→Paper via citation DOI lookup)

No deviations from the brief — all 6 field names match the schema exactly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from scripts.graph.relationship_validator import RelationshipValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field → predicate map — Entity-target predicates only
# ---------------------------------------------------------------------------
# All field names verified against config/extraction_schema.yaml (2026-05-29).
_FIELD_TO_ENTITY_PREDICATE: dict[str, str] = {
    "assumptions": "DEPENDS_ON",
    "future_work": "PROPOSES",
    "limitations": "LIMITED_BY",
    "novelty_statement": "INTRODUCES",
    "open_questions": "RAISES_QUESTION",
}

# COMPARES_TO is Paper→Paper — handled separately via DOI resolution.
_COMPARES_TO_FIELD = "comparison_to_prior"

# Maximum characters to store as the evidence snippet on each edge.
_EVIDENCE_MAX_CHARS = 200

# Type priority order for entity lookup — try these before giving up.
_ENTITY_TYPE_HINTS = ("method", "material", "topic", "keyword")


# ---------------------------------------------------------------------------
# RelationshipTuple
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationshipTuple:
    """One typed edge: source_doi → target_id with predicate + evidence.

    Attributes:
        source_doi:  DOI of the source paper.
        predicate:   One of VALID_PREDICATES.
        target_id:   canonical_id (Entity) or DOI (Paper).
        target_kind: "Entity" or "Paper".
        evidence:    Field-text excerpt, capped at _EVIDENCE_MAX_CHARS.
        confidence:  Always 1.0 for schema-source edges.
        field:       The extraction_json key this edge was extracted from.
    """

    source_doi: str
    predicate: str
    target_id: str
    target_kind: str
    evidence: str
    confidence: float
    field: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _excerpt(text: str, max_chars: int = _EVIDENCE_MAX_CHARS) -> str:
    """Trim text to max_chars at a word boundary, appending '…' if truncated.

    Args:
        text:      Raw evidence string.
        max_chars: Character budget before truncation (default 200).

    Returns:
        Possibly-truncated string, stripped of surrounding whitespace.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Prefer breaking at a word boundary to avoid a mid-word cut.
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "…"


def _resolve_target_entity(
    surface: str,
    entity_lookup: dict[tuple[str, str], str],
) -> str | None:
    """Return canonical_id for a surface form, trying multiple entity types.

    entity_lookup: {(surface_lower, type_hint): canonical_id}
    Tries each type in _ENTITY_TYPE_HINTS; returns the first match or None.
    """
    key_base = surface.lower().strip()
    for type_hint in _ENTITY_TYPE_HINTS:
        canonical = entity_lookup.get((key_base, type_hint))
        if canonical:
            return canonical
    return None


def _resolve_comparison_targets(
    text: str,
    source_doi: str,
    state_db: Any,
) -> list[str]:
    """Look up all resolved citation-edge target_dois for a source paper.

    Phase 1 strategy: return every resolved (target_doi != None) citation edge
    for this paper from state.db. The comparison_to_prior text is intentionally
    not parsed for author/year in Phase 1 — citation edge resolution already
    anchors the DOIs to real papers.  Phase 3's LLM extractor will do proper
    fuzzy author-year matching against the comparison text.

    Args:
        text:       comparison_to_prior field value (used only for guard checks).
        source_doi: DOI of the source paper.
        state_db:   StateDB instance.

    Returns:
        List of resolved target DOIs (may be empty).
    """
    if not text:
        return []
    try:
        cite_edges = state_db.get_citation_edges(source_doi)
    except Exception as exc:  # pragma: no cover — tested via mock
        logger.warning(
            "COMPARES_TO: state_db.get_citation_edges failed for %s: %s",
            source_doi,
            exc,
        )
        return []

    resolved: list[str] = []
    for edge in cite_edges:
        target = edge.get("target_doi") if isinstance(edge, dict) else None
        if target:
            resolved.append(target)
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_relationships(
    extraction_json: dict[str, Any],
    source_doi: str,
    entity_lookup: dict[tuple[str, str], str],
    state_db: Any | None = None,
) -> list[RelationshipTuple]:
    """Extract all schema-source typed relationships for one paper.

    Covers all 6 Phase 1 predicates:
      - 5 Entity-target predicates from _FIELD_TO_ENTITY_PREDICATE
      - 1 Paper-target predicate (COMPARES_TO) via citation DOI resolution

    Args:
        extraction_json: Flat dict from the LLM extraction pipeline.
        source_doi:      DOI of the paper being processed.
        entity_lookup:   {(surface_lower, type): canonical_id} built by G6
                         from G3's just-extracted EntityTuple rows.
        state_db:        Optional StateDB instance. Required only for
                         COMPARES_TO; ignored (no edges emitted) if None.

    Returns:
        List of validated RelationshipTuples (malformed tuples filtered out).
    """
    tuples: list[RelationshipTuple] = []

    # ── Entity-target predicates ──────────────────────────────────────────────
    for field, predicate in _FIELD_TO_ENTITY_PREDICATE.items():
        value = extraction_json.get(field)
        if value is None:
            continue

        # Normalise to a list of surface strings regardless of field shape.
        if isinstance(value, str):
            items: list[str] = [value.strip()] if value.strip() else []
        elif isinstance(value, list):
            items = [str(x).strip() for x in value if x is not None and str(x).strip()]
        else:
            continue

        for surface in items:
            target_id = _resolve_target_entity(surface, entity_lookup)
            if not target_id:
                # Surface not in lookup — no edge; validator would drop it anyway.
                logger.debug(
                    "extract_relationships: entity %r not in lookup for %s",
                    surface,
                    predicate,
                )
                continue

            tuples.append(
                RelationshipTuple(
                    source_doi=source_doi,
                    predicate=predicate,
                    target_id=target_id,
                    target_kind="Entity",
                    evidence=_excerpt(surface),
                    confidence=1.0,
                    field=field,
                )
            )

    # ── COMPARES_TO — Paper → Paper ───────────────────────────────────────────
    compare_text = extraction_json.get(_COMPARES_TO_FIELD)
    if compare_text and state_db is not None:
        for target_doi in _resolve_comparison_targets(
            compare_text, source_doi, state_db
        ):
            tuples.append(
                RelationshipTuple(
                    source_doi=source_doi,
                    predicate="COMPARES_TO",
                    target_id=target_doi,
                    target_kind="Paper",
                    evidence=_excerpt(compare_text),
                    confidence=1.0,
                    field=_COMPARES_TO_FIELD,
                )
            )

    # ── Final validation pass ─────────────────────────────────────────────────
    return [t for t in tuples if RelationshipValidator.is_valid(t)]
