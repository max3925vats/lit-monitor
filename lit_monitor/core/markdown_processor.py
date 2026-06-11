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

# Fraction of the document (counted from the start) that is *skipped* when
# searching for end-matter / references headings.  Searching only the final
# 1 - _END_MATTER_SEARCH_WINDOW fraction avoids false positives on inline
# phrases like "see references above" that appear in the body of the paper.
# 0.30 means "search the last 30% of the document" (search_start = 70% mark).
_END_MATTER_SEARCH_WINDOW = 0.30

# ---------------------------------------------------------------------------
# Heading patterns
# ---------------------------------------------------------------------------

# Headings that mark the start of a references/bibliography section only.
#
# Whitespace classes are restricted to horizontal whitespace ``[ \t]`` (never
# the newline-matching ``\s``) and the trailing run is written as a single
# ``[ \t]*(?::[ \t]*)?`` so no two quantifiers can match the same characters.
# Both choices keep matching provably linear-time (CodeQL py/polynomial-redos):
# under ``\s`` + MULTILINE a crafted line of trailing whitespace could force
# superlinear backtracking against the ``$`` anchor.
_REFERENCE_HEADING = _re.compile(
    r"^(?:#{1,3}[ \t]*)?"
    r"(?:references|bibliography|works[ \t]+cited|literature[ \t]+cited"
    r"|reference[ \t]+list|cited[ \t]+works|citations)"
    r"[ \t]*(?::[ \t]*)?$",
    _re.IGNORECASE | _re.MULTILINE,
)

# Headings for all end-matter: references plus acknowledgements, funding, etc.
#
# Quantifiers here are deliberately unambiguous to keep matching linear-time
# (CodeQL py/polynomial-redos).  ``acknowledge?ments?`` replaces the earlier
# ``acknowledg(?:e?ment|e?ments)`` whose two branches overlapped (both could
# match "acknowledgement"), forcing backtracking; ``conflicts?`` / ``interests?``
# replace ``conflict(?:s)?`` / ``interest(?:s)?`` for the same reason. All
# whitespace is horizontal-only ``[ \t]`` and the trailing run is a single
# non-overlapping ``[ \t]*(?::[ \t]*)?`` (see _REFERENCE_HEADING).
_END_MATTER_HEADING = _re.compile(
    r"^(?:#{1,3}[ \t]*)?"
    r"(?:references|bibliography|works[ \t]+cited|literature[ \t]+cited"
    r"|reference[ \t]+list|cited[ \t]+works|citations"
    r"|acknowledge?ments?"
    r"|funding(?:[ \t]+information)?"
    r"|conflicts?[ \t]+of[ \t]+interests?"
    r"|competing[ \t]+interests?"
    r"|author[ \t]+contributions?"
    r"|data[ \t]+availability(?:[ \t]+statement)?"
    r"|code[ \t]+availability(?:[ \t]+statement)?"
    r"|supplementary[ \t]+(?:information|material|data)"
    r"|supporting[ \t]+information"
    r")"
    r"[ \t]*(?::[ \t]*)?$",
    _re.IGNORECASE | _re.MULTILINE,
)


def strip_end_matter(text: str) -> str:
    """Remove end-matter sections from extracted paper text.

    Strips: references, bibliography, acknowledgements, funding,
    conflict-of-interest, author contributions, data availability,
    code availability, and supplementary information headings.

    Applies to the final ``_END_MATTER_SEARCH_WINDOW`` fraction of the
    document only (currently the last 30%) to avoid false positives on inline
    phrases. Used exclusively before the simple extraction phase; the complex
    phase keeps the full text (bibliography needed for key_citations).

    Returns the original text unchanged if no end-matter heading is found.
    """
    if not text:
        return text
    search_start = int(len(text) * (1.0 - _END_MATTER_SEARCH_WINDOW))
    match = _END_MATTER_HEADING.search(text, pos=search_start)
    if match is None:
        return text
    return text[: match.start()].rstrip()


def strip_references_section(text: str) -> str:
    """Remove the references/bibliography section from extracted paper text.

    Searches only the final ``_END_MATTER_SEARCH_WINDOW`` fraction of the
    document (currently the last 30%) to avoid false positives on inline
    phrases like 'see references above' or 'as cited in the references'.
    Returns the original text unchanged if no reference-section heading is found.

    Handles both plain-text headings (REFERENCES, Bibliography) and markdown
    headings (## References, # Works Cited).
    """
    if not text:
        return text
    search_start = int(len(text) * (1.0 - _END_MATTER_SEARCH_WINDOW))
    match = _REFERENCE_HEADING.search(text, pos=search_start)
    if match is None:
        return text
    return text[: match.start()].rstrip()
