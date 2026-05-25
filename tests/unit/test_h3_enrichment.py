"""
Unit tests for H3 — pre-extraction S2 enrichment + publicationTypes + CrossRef tracking.

Covers:
  - enrich_paper() returns s2_publication_types as a list of strings
  - enrich_paper() handles missing / None publicationTypes gracefully
  - BuildSummary has crossref_resolved field (empty by default)
  - s2_publication_types key is present in enrich_paper() result for reviews
  - enrich_paper() still returns existing s2_* fields alongside publicationTypes
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# enrich_paper() — publicationTypes field (H3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_h3_enrich_paper_returns_s2_publication_types():
    """enrich_paper() includes s2_publication_types list in return dict when available."""
    from scripts.search.semantic_scholar import enrich_paper

    mock_paper = MagicMock()
    mock_paper.paperId = "abc123"
    mock_paper.citationCount = 42
    mock_paper.influentialCitationCount = 5
    mock_paper.openAccessPdf = None
    mock_paper.publicationTypes = ["JournalArticle"]

    mock_sch = MagicMock()
    mock_sch.get_paper.return_value = mock_paper

    with patch("scripts.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
        result = enrich_paper("10.1234/test")

    assert "s2_publication_types" in result
    assert result["s2_publication_types"] == ["JournalArticle"]


@pytest.mark.unit
def test_h3_enrich_paper_returns_review_publication_type():
    """enrich_paper() returns ['Review'] for review articles — used by H4 detect_review()."""
    from scripts.search.semantic_scholar import enrich_paper

    mock_paper = MagicMock()
    mock_paper.paperId = "rev456"
    mock_paper.citationCount = 100
    mock_paper.influentialCitationCount = 20
    mock_paper.openAccessPdf = None
    mock_paper.publicationTypes = ["Review"]

    mock_sch = MagicMock()
    mock_sch.get_paper.return_value = mock_paper

    with patch("scripts.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
        result = enrich_paper("10.9999/review")

    assert result["s2_publication_types"] == ["Review"]


@pytest.mark.unit
def test_h3_enrich_paper_handles_none_publication_types():
    """enrich_paper() omits s2_publication_types key when publicationTypes is None."""
    from scripts.search.semantic_scholar import enrich_paper

    mock_paper = MagicMock()
    mock_paper.paperId = "xyz"
    mock_paper.citationCount = 10
    mock_paper.influentialCitationCount = 1
    mock_paper.openAccessPdf = None
    mock_paper.publicationTypes = None

    mock_sch = MagicMock()
    mock_sch.get_paper.return_value = mock_paper

    with patch("scripts.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
        result = enrich_paper("10.1111/none")

    # Key must be absent when the field is None — don't write null to DB
    assert "s2_publication_types" not in result


@pytest.mark.unit
def test_h3_enrich_paper_handles_empty_publication_types():
    """enrich_paper() returns an empty list when publicationTypes is [] (not omitted)."""
    from scripts.search.semantic_scholar import enrich_paper

    mock_paper = MagicMock()
    mock_paper.paperId = "empty"
    mock_paper.citationCount = 5
    mock_paper.influentialCitationCount = 0
    mock_paper.openAccessPdf = None
    mock_paper.publicationTypes = []

    mock_sch = MagicMock()
    mock_sch.get_paper.return_value = mock_paper

    with patch("scripts.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
        result = enrich_paper("10.2222/empty")

    assert "s2_publication_types" in result
    assert result["s2_publication_types"] == []


@pytest.mark.unit
def test_h3_enrich_paper_includes_existing_s2_fields_alongside_pub_types():
    """publicationTypes is returned alongside the existing s2_* fields, not instead of them."""
    from scripts.search.semantic_scholar import enrich_paper

    mock_paper = MagicMock()
    mock_paper.paperId = "full"
    mock_paper.citationCount = 30
    mock_paper.influentialCitationCount = 7
    mock_paper.openAccessPdf = {"url": "https://example.com/paper.pdf"}
    mock_paper.publicationTypes = ["JournalArticle"]

    mock_sch = MagicMock()
    mock_sch.get_paper.return_value = mock_paper

    with patch("scripts.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
        result = enrich_paper("10.3333/full")

    assert result["s2_paper_id"] == "full"
    assert result["s2_citation_count"] == 30
    assert result["s2_influential_citations"] == 7
    assert result["s2_open_access_pdf"] == "https://example.com/paper.pdf"
    assert result["s2_publication_types"] == ["JournalArticle"]


@pytest.mark.unit
def test_h3_enrich_paper_coerces_pub_types_to_strings():
    """s2_publication_types entries are coerced to str — handles non-string API values."""
    from scripts.search.semantic_scholar import enrich_paper

    mock_paper = MagicMock()
    mock_paper.paperId = "coerce"
    mock_paper.citationCount = 0
    mock_paper.influentialCitationCount = 0
    mock_paper.openAccessPdf = None
    # Simulate non-string objects in the list (shouldn't happen but be defensive)
    mock_paper.publicationTypes = [MagicMock(__str__=lambda self: "Review")]

    mock_sch = MagicMock()
    mock_sch.get_paper.return_value = mock_paper

    with patch("scripts.search.semantic_scholar._SemanticScholar", return_value=mock_sch):
        result = enrich_paper("10.4444/coerce")

    assert isinstance(result["s2_publication_types"], list)
    assert all(isinstance(t, str) for t in result["s2_publication_types"])


# ---------------------------------------------------------------------------
# BuildSummary — crossref_resolved field (H3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_h3_build_summary_crossref_resolved_default_empty():
    """BuildSummary.crossref_resolved is an empty list on construction."""
    from scripts.pipelines.brain_build import BuildSummary

    summary = BuildSummary()
    assert hasattr(summary, "crossref_resolved")
    assert summary.crossref_resolved == []


@pytest.mark.unit
def test_h3_build_summary_crossref_resolved_is_list():
    """BuildSummary.crossref_resolved is a mutable list (not a tuple or None)."""
    from scripts.pipelines.brain_build import BuildSummary

    summary = BuildSummary()
    summary.crossref_resolved.append("10.1234/test-doi")
    assert summary.crossref_resolved == ["10.1234/test-doi"]


@pytest.mark.unit
def test_h3_build_summary_crossref_resolved_independent_across_instances():
    """Two BuildSummary instances have independent crossref_resolved lists."""
    from scripts.pipelines.brain_build import BuildSummary

    s1 = BuildSummary()
    s2 = BuildSummary()
    s1.crossref_resolved.append("10.1/a")
    assert s2.crossref_resolved == [], "Dataclass mutable default must not be shared"
