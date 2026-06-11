"""G4: closed-vocabulary validator for typed-relationship edges.

Single source of truth for the Phase 1 predicate vocabulary.
RelationshipValidator.is_valid() is the gate before any tuple reaches the graph.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Phase 1 closed predicate vocabulary (8 predicates), extended in Phase 3 (R2)
# with EXTENDS + CONTRADICTS — the two LLM-only predicates. Total: 10 names.
# EXTENDS and CONTRADICTS are sourced exclusively from the LLM relationship
# extractor (R2); the other 8 can come from either G4 schema extraction or
# LLM augmentation.
VALID_PREDICATES: frozenset[str] = frozenset(
    {
        "MENTIONS",
        "CITES",
        "COMPARES_TO",
        "DEPENDS_ON",
        "PROPOSES",
        "LIMITED_BY",
        "INTRODUCES",
        "RAISES_QUESTION",
        "EXTENDS",       # R2 — LLM-extracted Paper→Paper edge
        "CONTRADICTS",   # R2 — LLM-extracted Paper→Paper edge
    }
)

if TYPE_CHECKING:
    from lit_monitor.graph.relationship_extractor import RelationshipTuple


class RelationshipValidator:
    """Drops malformed RelationshipTuples before they hit the graph.

    Validation rules:
    - predicate must be in VALID_PREDICATES
    - source_doi must be non-empty
    - target_id must be non-empty
    - target_kind must be 'Entity' or 'Paper'
    """

    @staticmethod
    def is_valid(tup: RelationshipTuple) -> bool:
        """Return True iff the tuple passes all validation rules."""
        if not tup.source_doi:
            logger.info(
                "Validator drop: empty source_doi (predicate=%s)", tup.predicate
            )
            return False
        if tup.predicate not in VALID_PREDICATES:
            logger.info("Validator drop: unknown predicate %r", tup.predicate)
            return False
        if not tup.target_id:
            logger.info(
                "Validator drop: empty target_id (predicate=%s)", tup.predicate
            )
            return False
        if tup.target_kind not in ("Entity", "Paper"):
            logger.info("Validator drop: bad target_kind %r", tup.target_kind)
            return False
        return True
