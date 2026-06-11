"""
Unit tests for E1 — scripts/search/citation_graph.py and StateDB citation_edges methods.
All S2 network calls are mocked; zero live requests.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.core.state_db import StateDB
from scripts.search.citation_graph import (
    _doi_from_ref,
    _extract_first_author,
    _extract_year,
    _fetch_s2_references,
    _ref_search_string,
    _resolve_ref_id,
    build_citation_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_s2_ref(
    *,
    paper_id: str = "abc123",
    doi: str | None = None,
    title: str = "Test paper title",
    year: int = 2020,
    first_author: str = "Smith, John",
) -> SimpleNamespace:
    """Build a minimal mock S2 reference object."""
    ext_ids = {"DOI": doi} if doi else {}
    author = SimpleNamespace(name=first_author)
    return SimpleNamespace(
        paperId=paper_id,
        externalIds=ext_ids,
        title=title,
        year=year,
        authors=[author],
    )


def _make_state_db() -> StateDB:
    tmp = tempfile.mkdtemp()
    return StateDB(Path(tmp) / "test.db")


# ---------------------------------------------------------------------------
# StateDB: citation_edges CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_upsert_and_get_citation_edges():
    db = _make_state_db()
    db.upsert_citation_edge(
        source_doi="10.1/src",
        ref_id="[1]",
        target_doi="10.1/tgt",
        target_s2_id="s2abc",
        context="foundational method",
        resolution="numeric_index",
    )
    edges = db.get_citation_edges("10.1/src")
    assert len(edges) == 1
    assert edges[0]["ref_id"] == "[1]"
    assert edges[0]["target_doi"] == "10.1/tgt"
    assert edges[0]["resolution"] == "numeric_index"


@pytest.mark.unit
def test_upsert_overwrites_resolved_with_resolved():
    """A better-resolved edge can overwrite a previous one."""
    db = _make_state_db()
    db.upsert_citation_edge("10.1/src", "[1]", "10.1/old", None, "ctx", "numeric_index")
    db.upsert_citation_edge("10.1/src", "[1]", "10.1/new", "s2id", "ctx2", "author_year_fuzzy")
    edges = db.get_citation_edges("10.1/src")
    assert len(edges) == 1
    assert edges[0]["target_doi"] == "10.1/new"
    assert edges[0]["resolution"] == "author_year_fuzzy"


@pytest.mark.unit
def test_upsert_coalesce_preserves_resolved_edge():
    """An unresolved re-run must NOT overwrite a previously-resolved edge."""
    db = _make_state_db()
    db.upsert_citation_edge("10.1/src", "[1]", "10.1/tgt", "s2id", "ctx", "numeric_index")
    db.upsert_citation_edge("10.1/src", "[1]", None, None, "ctx2", "unresolved")
    edges = db.get_citation_edges("10.1/src")
    assert edges[0]["target_doi"] == "10.1/tgt"
    assert edges[0]["resolution"] == "numeric_index"


@pytest.mark.unit
def test_get_citation_edges_empty():
    db = _make_state_db()
    assert db.get_citation_edges("10.1/unknown") == []


@pytest.mark.unit
def test_get_papers_citing_doi():
    db = _make_state_db()
    db.upsert_citation_edge("10.1/A", "[1]", "10.1/tgt", None, "", "numeric_index")
    db.upsert_citation_edge("10.1/B", "[2]", "10.1/tgt", None, "", "numeric_index")
    db.upsert_citation_edge("10.1/C", "[3]", "10.1/other", None, "", "numeric_index")
    citing = db.get_papers_citing_doi("10.1/tgt")
    assert set(citing) == {"10.1/A", "10.1/B"}


@pytest.mark.unit
def test_get_papers_citing_doi_unresolved_excluded():
    """Unresolved edges (target_doi=None) must not appear in get_papers_citing_doi."""
    db = _make_state_db()
    db.upsert_citation_edge("10.1/A", "[1]", None, None, "", "unresolved")
    assert db.get_papers_citing_doi("") == []


# ---------------------------------------------------------------------------
# _ref_search_string
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ref_search_string_basic():
    ref = _make_s2_ref(first_author="Pearson, Glen", year=2011, title="Membrane Filtration")
    s = _ref_search_string(ref)
    assert "Pearson" in s
    assert "2011" in s
    assert "Membrane Filtration" in s


@pytest.mark.unit
def test_ref_search_string_empty_authors():
    ref = SimpleNamespace(title="Some Title", year=2020, authors=[])
    s = _ref_search_string(ref)
    assert "2020" in s
    assert "Some Title" in s


# ---------------------------------------------------------------------------
# _doi_from_ref
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_doi_from_ref_present():
    ref = _make_s2_ref(doi="10.1016/ABCD")
    assert _doi_from_ref(ref) == "10.1016/abcd"


@pytest.mark.unit
def test_doi_from_ref_missing():
    ref = SimpleNamespace(externalIds={})
    assert _doi_from_ref(ref) is None


# ---------------------------------------------------------------------------
# _resolve_ref_id: numeric
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resolve_numeric_first_ref():
    refs = [
        _make_s2_ref(paper_id="p1", doi="10.1/one", title="First"),
        _make_s2_ref(paper_id="p2", doi="10.1/two", title="Second"),
    ]
    edge = _resolve_ref_id("[1]", "used as baseline", refs)
    assert edge.resolution == "numeric_index"
    assert edge.target_doi == "10.1/one"
    assert edge.target_s2_id == "p1"


@pytest.mark.unit
def test_resolve_numeric_second_ref():
    refs = [
        _make_s2_ref(paper_id="p1", doi="10.1/one"),
        _make_s2_ref(paper_id="p2", doi="10.1/two"),
    ]
    edge = _resolve_ref_id("[2]", "ctx", refs)
    assert edge.resolution == "numeric_index"
    assert edge.target_doi == "10.1/two"


@pytest.mark.unit
def test_resolve_numeric_out_of_range():
    """Numeric index beyond reference list → falls through to unresolved."""
    refs = [_make_s2_ref(paper_id="p1")]
    edge = _resolve_ref_id("[99]", "ctx", refs)
    assert edge.resolution == "unresolved"


@pytest.mark.unit
def test_resolve_numeric_empty_refs():
    edge = _resolve_ref_id("[1]", "ctx", [])
    assert edge.resolution == "unresolved"


# ---------------------------------------------------------------------------
# _resolve_ref_id: author-year fuzzy
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resolve_author_year_fuzzy_match():
    refs = [
        _make_s2_ref(
            paper_id="p_author2",
            doi="10.1/author2_2011",
            first_author="Pearson, Glen",
            year=2011,
            title="Process Filtration Studies",
        ),
    ]
    edge = _resolve_ref_id("Pearson et al. (2011)", "baseline model", refs)
    assert edge.resolution == "author_year_fuzzy"
    assert edge.target_doi == "10.1/author2_2011"


@pytest.mark.unit
def test_resolve_author_year_no_match():
    refs = [
        _make_s2_ref(
            paper_id="px",
            doi="10.1/unrelated",
            first_author="Jones, Mary",
            year=2015,
            title="Chromatography Review",
        ),
    ]
    edge = _resolve_ref_id("Smith 1999", "ctx", refs)
    assert edge.resolution == "unresolved"


@pytest.mark.unit
def test_resolve_author_year_wrong_year_not_matched():
    """Same author but different year must NOT match (year hard filter)."""
    refs = [
        _make_s2_ref(
            paper_id="px",
            doi="10.1/smith2015",
            first_author="Smith, John",
            year=2015,
            title="Membrane Filtration",
        ),
    ]
    edge = _resolve_ref_id("Smith 1999", "ctx", refs)
    assert edge.resolution == "unresolved"


# ---------------------------------------------------------------------------
# _extract_year / _extract_first_author
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_year_standard():
    assert _extract_year("Pearson et al. (2011)") == 2011


@pytest.mark.unit
def test_extract_year_bare():
    assert _extract_year("Smith 1999") == 1999


@pytest.mark.unit
def test_extract_year_none():
    assert _extract_year("[20]") is None


@pytest.mark.unit
def test_extract_first_author_standard():
    assert _extract_first_author("Pearson et al. (2011)") == "Pearson"


@pytest.mark.unit
def test_extract_first_author_with_paren():
    assert _extract_first_author("(Smith, 2020)") == "Smith"


# ---------------------------------------------------------------------------
# _fetch_s2_references: rate-limit backoff
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_s2_references_rate_limit_retry_then_success():
    """Rate-limit on first attempt; success on second → returns references."""
    call_count = 0

    class _FakePaper:
        references = [_make_s2_ref(paper_id="p1")]

    class _FakeS2:
        def get_paper(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("429 Too many requests")
            return _FakePaper()

    with patch("scripts.search.citation_graph._SemanticScholar", return_value=_FakeS2()), \
         patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch("time.sleep"):
        refs = _fetch_s2_references("10.1/test", max_retries=2)

    assert len(refs) == 1
    assert call_count == 2


@pytest.mark.unit
def test_fetch_s2_references_connection_refused_treated_as_rate_limit():
    """P6.6: the S2 client maps HTTP 429 to ConnectionRefusedError.

    Detection must key off that exception type, not just the message
    substring, so a 429 with a message that lacks "429"/"rate"/"too many"
    still triggers a retry.
    """
    call_count = 0

    class _FakePaper:
        references = [_make_s2_ref(paper_id="p1")]

    class _FakeS2:
        def get_paper(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # No "429"/"rate"/"too many" substring on purpose.
                raise ConnectionRefusedError("Too busy right now.")
            return _FakePaper()

    with patch("scripts.search.citation_graph._SemanticScholar", return_value=_FakeS2()), \
         patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch("time.sleep"):
        refs = _fetch_s2_references("10.1/test", max_retries=2)

    assert len(refs) == 1
    assert call_count == 2  # retried after the ConnectionRefusedError


@pytest.mark.unit
def test_fetch_s2_references_passes_timeout_to_constructor():
    """H3: SemanticScholar must be instantiated with timeout=30 to bound network calls."""
    mock_sch = MagicMock()
    mock_sch.get_paper.return_value = None
    with patch(
        "scripts.search.citation_graph._SemanticScholar", return_value=mock_sch
    ) as mock_cls, \
         patch("scripts.search.citation_graph._S2_AVAILABLE", True):
        _fetch_s2_references("10.1/test", api_key="my-s2-key")
    mock_cls.assert_called_once_with(api_key="my-s2-key", timeout=30)


@pytest.mark.unit
def test_fetch_s2_references_exhausted_retries():
    """Exceeds max_retries → returns empty list."""

    class _FakeS2:
        def get_paper(self, *a, **kw):
            raise RuntimeError("429 rate limit exceeded")

    with patch("scripts.search.citation_graph._SemanticScholar", return_value=_FakeS2()), \
         patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch("time.sleep"):
        refs = _fetch_s2_references("10.1/test", max_retries=2)

    assert refs == []


@pytest.mark.unit
def test_fetch_s2_references_default_retry_count_and_backoff_sequence():
    """AR-5 Fix 2 (Finding 6): with the DEFAULT retry policy, a persistently
    rate-limited DOI makes exactly 1 initial + 3 backoff retries = 4 total HTTP
    attempts, with backoff sleeps [1.0, 2.0, 4.0] seconds.

    STASH-VERIFY: the off-by-one default (``max_retries=4`` over
    ``range(max_retries + 1)``) produces 5 attempts and sleeps
    [1.0, 2.0, 4.0, 8.0] → both assertions fail RED.
    """
    call_count = 0

    class _FakeS2:
        def get_paper(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("429 rate limit exceeded")

    sleeps: list[float] = []

    with patch("scripts.search.citation_graph._SemanticScholar", return_value=_FakeS2()), \
         patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
        # No max_retries override → exercise the default policy.
        refs = _fetch_s2_references("10.1/test")

    assert refs == []
    assert call_count == 4  # 1 initial + 3 retries
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.unit
def test_fetch_s2_references_non_rate_limit_error():
    """Non-rate-limit errors are not retried."""
    call_count = 0

    class _FakeS2:
        def get_paper(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network failure")

    with patch("scripts.search.citation_graph._SemanticScholar", return_value=_FakeS2()), \
         patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch("time.sleep"):
        refs = _fetch_s2_references("10.1/test", max_retries=4)

    assert refs == []
    assert call_count == 1  # no retry for non-rate-limit errors


# ---------------------------------------------------------------------------
# build_citation_graph
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_citation_graph_no_extraction():
    db = _make_state_db()
    result = build_citation_graph("10.1/missing", db)
    assert result.n_resolved == 0
    assert result.edges == []


@pytest.mark.unit
def test_build_citation_graph_no_key_citations():
    db = _make_state_db()
    db.upsert_paper({
        "doi": "10.1/nodoi",
        "title": "Test",
        "extraction_json": json.dumps({"core_finding": "something"}),
    })
    result = build_citation_graph("10.1/nodoi", db)
    assert result.n_resolved == 0
    assert result.edges == []


@pytest.mark.unit
def test_build_citation_graph_numeric_resolved():
    """Numeric ref_id '[1]' → resolved via S2 reference index 0."""
    db = _make_state_db()
    kc = [{"ref_id": "[1]", "context": "foundational method"}]
    db.upsert_paper({
        "doi": "10.1/src",
        "title": "Source Paper",
        "extraction_json": json.dumps({"key_citations": kc}),
    })

    mock_ref = _make_s2_ref(paper_id="tgt_s2", doi="10.1/tgt", title="Target")

    with patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch(
             "scripts.search.citation_graph._fetch_s2_references",
             return_value=[mock_ref],
         ):
        result = build_citation_graph("10.1/src", db)

    assert result.n_resolved == 1
    assert result.n_unresolved == 0
    assert result.edges[0].resolution == "numeric_index"
    assert result.edges[0].target_doi == "10.1/tgt"

    # Verify persisted to DB
    edges = db.get_citation_edges("10.1/src")
    assert len(edges) == 1
    assert edges[0]["target_doi"] == "10.1/tgt"


@pytest.mark.unit
def test_build_citation_graph_unresolved_stored():
    """Unresolved ref_ids are stored with target_doi=None."""
    db = _make_state_db()
    kc = [{"ref_id": "Doe 1999", "context": "background"}]
    db.upsert_paper({
        "doi": "10.1/src",
        "title": "Source",
        "extraction_json": json.dumps({"key_citations": kc}),
    })

    with patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch("scripts.search.citation_graph._fetch_s2_references", return_value=[]):
        result = build_citation_graph("10.1/src", db)

    assert result.n_unresolved == 1
    assert result.n_resolved == 0
    edges = db.get_citation_edges("10.1/src")
    assert edges[0]["resolution"] == "unresolved"
    assert edges[0]["target_doi"] is None


@pytest.mark.unit
def test_build_citation_graph_mixed_resolved_unresolved():
    """Mix of numeric (resolved) and author-year (unresolved) in one paper."""
    db = _make_state_db()
    kc = [
        {"ref_id": "[1]", "context": "method"},
        {"ref_id": "Unknown 1800", "context": "background"},
    ]
    db.upsert_paper({
        "doi": "10.1/src",
        "title": "Source",
        "extraction_json": json.dumps({"key_citations": kc}),
    })

    mock_ref = _make_s2_ref(paper_id="p1", doi="10.1/ref1")

    with patch("scripts.search.citation_graph._S2_AVAILABLE", True), \
         patch(
             "scripts.search.citation_graph._fetch_s2_references",
             return_value=[mock_ref],
         ):
        result = build_citation_graph("10.1/src", db)

    assert result.n_resolved == 1
    assert result.n_unresolved == 1
    assert result.s2_references_count == 1
