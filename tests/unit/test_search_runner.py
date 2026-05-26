"""
Unit tests for scripts/search/search_runner.py helpers.

Covers Audit_31 H2: _format_query() must always run the canonical formatting
path, even when the input already contains '[' or ']' characters. Previously
the function short-circuited on any '[' and returned the topic unchanged,
which let unvalidated YAML values bypass the bracket-wrapping logic.
"""
from __future__ import annotations


def test_format_query_round_trips_bracketed_input():
    """Audit_31 H2 — pre-bracketed inputs MUST be re-canonicalised.

    Use whitespace/casing variation so the (old) early-return path would
    produce a literally different string from the canonical path. The
    assertion pins the canonical output; if anyone restores the early-return
    short-circuit, this test fails because the early-return would preserve
    the input verbatim instead of normalising whitespace + bracket layout.
    """
    from scripts.search.search_runner import _format_query
    # Input with extra whitespace inside brackets — early-return would
    # preserve the spaces verbatim; canonical path strips and re-wraps.
    messy = _format_query("[ Topic ]   AND   [ Other ]")
    canonical = _format_query("Topic AND Other")
    assert messy == canonical, f"messy={messy!r}, canonical={canonical!r}"
    # Pin the exact output too, to catch any future tweak to the formatter.
    assert "[" in canonical and "]" in canonical, f"expected brackets in {canonical!r}"


def test_format_query_canonical_form():
    """Pin the exact canonical output so silent formatter drift is caught."""
    from scripts.search.search_runner import _format_query
    assert _format_query("Topic AND Other") == "[Topic] AND [Other]"
