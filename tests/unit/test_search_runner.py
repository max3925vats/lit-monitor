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
    """Minimal stand-in for filter_known_dois.

    The old AR-4 tests treated the 'known' set as "everything already seen"
    (the old known_dois() semantic). Under the lifecycle relaxation (Task 3),
    filter_known_dois now calls library_dois() + surfaced_dois() instead of
    known_dois(). We map the old 'known' set to library_dois() here so the
    AR-4 tests continue to exercise their intended behavior: a DOI that is in
    the library is suppressed.
    """

    def __init__(self, known):
        self._known = set(known)

    def library_dois(self):
        return self._known

    def surfaced_dois(self):
        return set()

    def known_dois(self):
        # kept for reference but must NOT be called by filter_known_dois
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


# ---------------------------------------------------------------------------
# Task 3 / lifecycle Part B: filter_known_dois must use library/surfaced only
# ---------------------------------------------------------------------------

class _FakeStateDB_lifecycle:
    """Stand-in that raises if known_dois() is called — the new filter must NOT use it."""

    def __init__(self, library, surfaced):
        self._library = set(library)
        self._surfaced = set(surfaced)

    def library_dois(self):
        return set(self._library)

    def surfaced_dois(self):
        return set(self._surfaced)

    # known_dois must NOT be consulted by the new filter:
    def known_dois(self):
        raise AssertionError("filter_known_dois must use library/surfaced, not known_dois")


def test_filter_known_dois_suppresses_library_and_shown_keeps_unshown():
    from lit_monitor.search.search_runner import filter_known_dois

    papers = [
        {"doi": "10.1/lib"},                    # canonical candidate vs non-canonical stored -> drop
        {"doi": "10.1/shown"},                  # canonical candidate vs non-canonical stored -> drop
        {"doi": "10.1/unshown"},                # retrieved-only -> keep
        {"doi": ""},                            # no doi -> keep
    ]
    # STORED sides are deliberately non-canonical (URL-wrapped / upper-cased) so the
    # match only succeeds if filter_known_dois normalizes the library/surfaced sets,
    # not just the candidate side.
    db = _FakeStateDB_lifecycle(
        library={"HTTPS://DOI.ORG/10.1/LIB"},
        surfaced={"10.1/SHOWN"},
    )
    out = filter_known_dois(papers, db)
    dois = [p["doi"] for p in out]
    assert "10.1/unshown" in dois and "" in dois
    assert "10.1/lib" not in dois and "10.1/shown" not in dois


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


# ---------------------------------------------------------------------------
# Tasks 3+4+5: SearchWindow threading (since + until)
# ---------------------------------------------------------------------------

def _make_empty_search_result():
    from unittest.mock import MagicMock
    r = MagicMock()
    r.papers = []
    return r


def test_run_searches_passes_since_and_until_to_findpapers():
    """run_searches with a bounded SearchWindow passes both since AND until to
    _findpapers.search.  Monkeypatches _findpapers.search to capture kwargs."""
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch

    from lit_monitor.search.search_runner import run_searches
    from lit_monitor.search.window import SearchWindow

    window = SearchWindow(since=date(2025, 1, 1), until=date(2025, 2, 1))
    captured_kwargs: list[dict] = []

    def _fake_search(**kwargs):
        captured_kwargs.append(kwargs)

    config = SimpleNamespace(topics=["filtration"])
    with patch("lit_monitor.search.search_runner._S2_AVAILABLE", False):
        with patch("lit_monitor.search.search_runner._findpapers") as mock_fp:
            mock_fp.search.side_effect = _fake_search
            with patch(
                "lit_monitor.search.search_runner._fp_load",
                return_value=_make_empty_search_result(),
            ):
                with patch(
                    "lit_monitor.search.search_runner._load_api_secrets",
                    return_value={},
                ):
                    run_searches(config, window=window)

    assert len(captured_kwargs) == 1, "findpapers.search must be called once"
    assert captured_kwargs[0]["since"] == date(2025, 1, 1)
    assert captured_kwargs[0]["until"] == date(2025, 2, 1)


def test_run_searches_open_ended_window_passes_none_until():
    """An open-ended window (until=None) passes until=None to _findpapers.search."""
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch

    from lit_monitor.search.search_runner import run_searches
    from lit_monitor.search.window import SearchWindow

    window = SearchWindow(since=date(2025, 1, 1), until=None)
    captured_kwargs: list[dict] = []

    def _fake_search(**kwargs):
        captured_kwargs.append(kwargs)

    config = SimpleNamespace(topics=["filtration"])
    with patch("lit_monitor.search.search_runner._S2_AVAILABLE", False):
        with patch("lit_monitor.search.search_runner._findpapers") as mock_fp:
            mock_fp.search.side_effect = _fake_search
            with patch(
                "lit_monitor.search.search_runner._fp_load",
                return_value=_make_empty_search_result(),
            ):
                with patch(
                    "lit_monitor.search.search_runner._load_api_secrets",
                    return_value={},
                ):
                    run_searches(config, window=window)

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["since"] == date(2025, 1, 1)
    assert captured_kwargs[0]["until"] is None
