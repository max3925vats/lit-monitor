"""
Markdown text processing helpers for the lit-monitor extraction pipeline.

Phase M (2026-05-16): the pipeline reads pre-converted Markdown files from
Zotero attachments.  This module provides text-level helpers that operate on
that Markdown content.

Public API:
    strip_end_matter(text)        — remove end-matter before simple-phase extraction
    strip_references_section(text) — remove references/bibliography only
"""
from __future__ import annotations

import logging
import re as _re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heading patterns
# ---------------------------------------------------------------------------

# Headings that mark the start of a references/bibliography section only.
_REFERENCE_HEADING = _re.compile(
    r"^(?:#{1,3}\s*)?"
    r"(?:references|bibliography|works\s+cited|literature\s+cited"
    r"|reference\s+list|cited\s+works|citations)"
    r"\s*:?\s*$",
    _re.IGNORECASE | _re.MULTILINE,
)

# Headings for all end-matter: references plus acknowledgements, funding, etc.
_END_MATTER_HEADING = _re.compile(
    r"^(?:#{1,3}\s*)?"
    r"(?:references|bibliography|works\s+cited|literature\s+cited"
    r"|reference\s+list|cited\s+works|citations"
    r"|acknowledg(?:e?ment|e?ments)"
    r"|funding(?:\s+information)?"
    r"|conflict(?:s)?\s+of\s+interest(?:s)?"
    r"|competing\s+interests?"
    r"|author\s+contributions?"
    r"|data\s+availability(?:\s+statement)?"
    r"|code\s+availability(?:\s+statement)?"
    r"|supplementary\s+(?:information|material|data)"
    r"|supporting\s+information"
    r")"
    r"\s*:?\s*$",
    _re.IGNORECASE | _re.MULTILINE,
)


def strip_end_matter(text: str) -> str:
    """Remove end-matter sections from extracted paper text.

    Strips: references, bibliography, acknowledgements, funding,
    conflict-of-interest, author contributions, data availability,
    code availability, and supplementary information headings.

    Applies to the final 30% of the document only to avoid false positives
    on inline phrases. Used exclusively before the simple extraction phase;
    the complex phase keeps the full text (bibliography needed for key_citations).

    Returns the original text unchanged if no end-matter heading is found.
    """
    if not text:
        return text
    search_start = int(len(text) * 0.70)
    match = _END_MATTER_HEADING.search(text, pos=search_start)
    if match is None:
        return text
    return text[: match.start()].rstrip()


def strip_references_section(text: str) -> str:
    """Remove the references/bibliography section from extracted paper text.

    Searches only the final 30% of the document to avoid false positives on
    inline phrases like 'see references above' or 'as cited in the references'.
    Returns the original text unchanged if no reference-section heading is found.

    Handles both plain-text headings (REFERENCES, Bibliography) and markdown
    headings (## References, # Works Cited).
    """
    if not text:
        return text
    search_start = int(len(text) * 0.70)
    match = _REFERENCE_HEADING.search(text, pos=search_start)
    if match is None:
        return text
    return text[: match.start()].rstrip()
