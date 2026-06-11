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
    from lit_monitor.search.search_runner import _format_query
    # Input with extra whitespace inside brackets — early-return would
    # preserve the spaces verbatim; canonical path strips and re-wraps.
    messy = _format_query("[ Topic ]   AND   [ Other ]")
    canonical = _format_query("Topic AND Other")
    assert messy == canonical, f"messy={messy!r}, canonical={canonical!r}"
    # Pin the exact output too, to catch any future tweak to the formatter.
    assert "[" in canonical and "]" in canonical, f"expected brackets in {canonical!r}"


def test_format_query_canonical_form():
    """Pin the exact canonical output so silent formatter drift is caught."""
    from lit_monitor.search.search_runner import _format_query
    assert _format_query("Topic AND Other") == "[Topic] AND [Other]"


# ---------------------------------------------------------------------------
# AR-4: canonical DOI normalization at the search-dedup boundary
# ---------------------------------------------------------------------------

class _FakeStateDB:
    """Minimal stand-in exposing only known_dois() for filter_known_dois."""

    def __init__(self, known):
        self._known = set(known)

    def known_dois(self):
        return self._known


def test_filter_known_dois_url_wrap_collision():
    """AR-4 STASH-VERIFY target: a URL-wrapped candidate DOI must be recognised
    as already-known when the bare canonical form is in state.db.

    Before AR-4 the candidate was only ``.strip().lower()``-ed, so
    ``https://doi.org/10.1/a`` did NOT match the stored bare ``10.1/a`` and the
    paper was wrongly re-ingested as new.
    """
    from lit_monitor.search.search_runner import filter_known_dois
    state_db = _FakeStateDB(known={"10.1/a"})
    papers = [{"doi": "https://doi.org/10.1/A", "title": "wrapped dup"}]
    out = filter_known_dois(papers, state_db)
    assert out == [], "URL-wrapped DOI should dedup against the bare known DOI"


def test_filter_known_dois_keeps_genuinely_new():
    """Sanity: a non-matching DOI is still kept (no over-aggressive dedup)."""
    from lit_monitor.search.search_runner import filter_known_dois
    state_db = _FakeStateDB(known={"10.1/a"})
    papers = [{"doi": "10.2/b", "title": "new"}]
    out = filter_known_dois(papers, state_db)
    assert len(out) == 1


def test_convert_findpapers_results_normalizes_stored_doi():
    """AR-4: the stored DOI on the converted record is canonicalized (URL
    unwrapped + lowercased), consistent with how the rest of the system stores
    bare lowercased DOIs."""
    from types import SimpleNamespace

    from lit_monitor.search.search_runner import _convert_findpapers_results

    fake_paper = SimpleNamespace(
        doi="https://doi.org/10.1234/ABC",
        title="t", abstract="a", authors=["x"],
        publication=None, publication_date=None,
        keywords=None, databases=None,
    )
    out = _convert_findpapers_results([fake_paper], tracked_author=False)
    assert out[0]["doi"] == "10.1234/abc"
