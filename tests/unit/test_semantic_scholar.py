"""
Unit tests for scripts.search.semantic_scholar (D3).

All tests mock ``scripts.search.semantic_scholar._SemanticScholar`` so no live
network calls are made.  The Semantic Scholar response structure mirrors the real
API objects: Paper objects with .paperId, .citationCount, etc.
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lit_monitor.search.semantic_scholar import (
    _strip_findpapers_format,
    enrich_paper,
    search_semantic_scholar,
)

# ---------------------------------------------------------------------------
# Helpers to build minimal mock Paper objects
# ---------------------------------------------------------------------------

def _mock_paper(
    *,
    paper_id: str = "abc123",
    citation_count: int = 42,
    influential: int = 5,
    oa_pdf_url: str | None = "https://example.com/paper.pdf",
) -> MagicMock:
    """Build a minimal mock Paper for enrich_paper() tests."""
    paper = MagicMock()
    paper.paperId = paper_id
    paper.citationCount = citation_count
    paper.influentialCitationCount = influential
    paper.openAccessPdf = {"url": oa_pdf_url} if oa_pdf_url else None
    return paper


def _mock_search_result(
    *,
    doi: str = "10.1016/j.foo.2024.001",
    title: str = "Test Paper",
    year: int = 2024,
    authors: list[str] | None = None,
    journal: str = "Journal of Tests",
    abstract: str = "An abstract.",
    fos: list[str] | None = None,
) -> MagicMock:
    """Build a minimal mock Paper for search_semantic_scholar() tests."""
    p = MagicMock()
    p.externalIds = {"DOI": doi}
    p.title = title
    p.year = year
    p.abstract = abstract
    p.fieldsOfStudy = fos or ["Biology"]
    # Venue
    venue = MagicMock()
    venue.name = journal
    p.publicationVenue = venue
    # Authors
    _authors = authors or ["Smith, J.", "Doe, A."]
    p.authors = [SimpleNamespace(name=a) for a in _authors]
    return p


# ---------------------------------------------------------------------------
# enrich_paper
# ---------------------------------------------------------------------------

class TestEnrichPaper:
    def test_returns_all_s2_fields_on_success(self):
        mock_sch = MagicMock()
        mock_sch.get_paper.return_value = _mock_paper()
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            result = enrich_paper("10.1016/j.foo.2024.001")
        assert result["s2_paper_id"] == "abc123"
        assert result["s2_citation_count"] == 42
        assert result["s2_influential_citations"] == 5
        assert result["s2_open_access_pdf"] == "https://example.com/paper.pdf"

    def test_prefixes_doi_with_DOI_colon(self):
        """get_paper must be called with 'DOI:<doi>' not the raw DOI string."""
        mock_sch = MagicMock()
        mock_sch.get_paper.return_value = _mock_paper()
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            enrich_paper("10.1016/j.foo.2024.001")
        call_args = mock_sch.get_paper.call_args
        assert call_args.args[0] == "DOI:10.1016/j.foo.2024.001"

    def test_returns_empty_on_api_error(self):
        mock_sch = MagicMock()
        mock_sch.get_paper.side_effect = RuntimeError("connection refused")
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            result = enrich_paper("10.1016/j.foo.2024.001")
        assert result == {}

    def test_returns_empty_for_empty_doi(self):
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar") as mock_cls:
            assert enrich_paper("") == {}
            assert enrich_paper("   ") == {}
        mock_cls.assert_not_called()

    def test_returns_empty_when_paper_not_found(self):
        """get_paper returning None → empty dict (not an error)."""
        mock_sch = MagicMock()
        mock_sch.get_paper.return_value = None
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            result = enrich_paper("10.1016/j.foo.2024.001")
        assert result == {}

    def test_omits_open_access_pdf_when_absent(self):
        mock_sch = MagicMock()
        mock_sch.get_paper.return_value = _mock_paper(oa_pdf_url=None)
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            result = enrich_paper("10.1016/j.foo.2024.001")
        assert "s2_open_access_pdf" not in result

    def test_passes_api_key_to_constructor(self):
        mock_sch = MagicMock()
        mock_sch.get_paper.return_value = _mock_paper()
        with patch(
            "lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch
        ) as mock_cls:
            enrich_paper("10.1016/j.foo.2024.001", api_key="my-s2-key")
        mock_cls.assert_called_once_with(api_key="my-s2-key", timeout=30)

    def test_citation_counts_coerced_to_int(self):
        """Ensure counts are stored as int, not as raw API type."""
        mock_sch = MagicMock()
        paper = _mock_paper(citation_count=77, influential=3)
        mock_sch.get_paper.return_value = paper
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            result = enrich_paper("10.1/x")
        assert isinstance(result["s2_citation_count"], int)
        assert isinstance(result["s2_influential_citations"], int)


# ---------------------------------------------------------------------------
# search_semantic_scholar
# ---------------------------------------------------------------------------

class TestSearchSemanticScholar:
    def test_returns_papers_in_pipeline_format(self):
        from lit_monitor.search.window import SearchWindow
        mock_sch = MagicMock()
        mock_sch.search_paper.return_value = [_mock_search_result()]
        window = SearchWindow(since=date.today() - timedelta(days=14), until=None)
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            results = search_semantic_scholar("filtration", window=window)
        assert len(results) == 1
        p = results[0]
        assert p["doi"] == "10.1016/j.foo.2024.001"
        assert p["title"] == "Test Paper"
        assert p["year"] == 2024
        assert p["source_databases"] == ["semantic_scholar"]
        assert p["tracked_author"] is False

    def test_doi_is_lowercased(self):
        mock_sch = MagicMock()
        result_paper = _mock_search_result()
        result_paper.externalIds = {"DOI": "10.1016/J.FOO.2024.001"}
        mock_sch.search_paper.return_value = [result_paper]
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            results = search_semantic_scholar("query")
        assert results[0]["doi"] == "10.1016/j.foo.2024.001"

    def test_returns_empty_on_api_error(self):
        mock_sch = MagicMock()
        mock_sch.search_paper.side_effect = RuntimeError("timeout")
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            results = search_semantic_scholar("filtration")
        assert results == []

    def test_returns_empty_for_empty_query(self):
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar") as mock_cls:
            assert search_semantic_scholar("") == []
            assert search_semantic_scholar("   ") == []
        mock_cls.assert_not_called()

    def test_passes_publication_date_filter(self):
        """publication_date_or_year should be '<since>:' for an open-ended window."""
        from lit_monitor.search.window import SearchWindow
        mock_sch = MagicMock()
        mock_sch.search_paper.return_value = []
        expected_date = date.today() - timedelta(days=30)
        window = SearchWindow(since=expected_date, until=None)
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            search_semantic_scholar("query", window=window)
        call_kwargs = mock_sch.search_paper.call_args.kwargs
        assert call_kwargs["publication_date_or_year"] == f"{expected_date}:"

    def test_strips_findpapers_brackets_before_querying(self):
        """[term] AND [other] should become 'term other' for S2."""
        mock_sch = MagicMock()
        mock_sch.search_paper.return_value = []
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            search_semantic_scholar("[filtration] AND [model protein]")
        call_args = mock_sch.search_paper.call_args
        query_sent = call_args.args[0]
        assert "[" not in query_sent
        assert "AND" not in query_sent
        assert "filtration" in query_sent
        assert "model protein" in query_sent

    def test_handles_paper_with_no_doi(self):
        """Papers without an external DOI should have doi='' in result."""
        mock_sch = MagicMock()
        nodoi = _mock_search_result()
        nodoi.externalIds = {}
        mock_sch.search_paper.return_value = [nodoi]
        with patch("lit_monitor.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
            results = search_semantic_scholar("query")
        assert results[0]["doi"] == ""


# ---------------------------------------------------------------------------
# _strip_findpapers_format
# ---------------------------------------------------------------------------

class TestStripFindpapersFormat:
    def test_strips_brackets(self):
        assert _strip_findpapers_format("[filtration]") == "filtration"

    def test_strips_brackets_and_boolean_operators(self):
        out = _strip_findpapers_format("[filtration] AND [model protein]")
        assert "[" not in out and "AND" not in out
        assert "filtration" in out
        assert "model protein" in out

    def test_strips_or_operator(self):
        out = _strip_findpapers_format("[TopicX] OR [filtration]")
        assert "OR" not in out

    def test_strips_and_not_operator(self):
        out = _strip_findpapers_format("[protein] AND NOT [DNA]")
        assert "AND NOT" not in out

    def test_plain_query_unchanged(self):
        assert _strip_findpapers_format("filtration") == "filtration"

    def test_collapses_whitespace(self):
        out = _strip_findpapers_format("[a] AND [b] AND [c]")
        # Should not have leading/trailing whitespace or double spaces
        assert out == out.strip()
        assert "  " not in out


# ---------------------------------------------------------------------------
# Tasks 3+4+5: SearchWindow date-filter tests
# ---------------------------------------------------------------------------

class TestSearchSemanticScholarWindow:
    """Verify that SearchWindow produces the correct publication_date_or_year string."""

    def test_bounded_window_builds_since_colon_until_filter(self):
        """A bounded window (since + until) must produce '<since>:<until>'."""
        from datetime import date
        from unittest.mock import MagicMock, patch

        from lit_monitor.search.semantic_scholar import search_semantic_scholar
        from lit_monitor.search.window import SearchWindow

        window = SearchWindow(since=date(2025, 1, 1), until=date(2025, 2, 1))
        mock_sch = MagicMock()
        mock_sch.search_paper.return_value = []
        with patch(
            "lit_monitor.search.semantic_scholar._SemanticScholar",
            return_value=mock_sch,
        ):
            search_semantic_scholar("filtration", window=window)

        call_kwargs = mock_sch.search_paper.call_args.kwargs
        assert call_kwargs["publication_date_or_year"] == "2025-01-01:2025-02-01"

    def test_open_ended_window_builds_since_colon_filter(self):
        """An open-ended window (until=None) must produce '<since>:' (no end date)."""
        from datetime import date
        from unittest.mock import MagicMock, patch

        from lit_monitor.search.semantic_scholar import search_semantic_scholar
        from lit_monitor.search.window import SearchWindow

        window = SearchWindow(since=date(2025, 1, 1), until=None)
        mock_sch = MagicMock()
        mock_sch.search_paper.return_value = []
        with patch(
            "lit_monitor.search.semantic_scholar._SemanticScholar",
            return_value=mock_sch,
        ):
            search_semantic_scholar("filtration", window=window)

        call_kwargs = mock_sch.search_paper.call_args.kwargs
        # until=None → open-ended → "<since>:"
        assert call_kwargs["publication_date_or_year"] == "2025-01-01:"
