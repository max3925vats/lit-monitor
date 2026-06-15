"""
Phase 8 unit tests — Search layer.
All findpapers Engine calls are mocked. No network access required.
"""
from __future__ import annotations

import datetime
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lit_monitor.search.window import SearchWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_config(topics=None, researchers=None, wos_key=None, scopus_key=None):
    return SimpleNamespace(
        topics=["filtration AND ProteinA"] if topics is None else topics,
        researchers=[] if researchers is None else researchers,
        wos_api_key=wos_key,
        scopus_api_key=scopus_key,
        email="test@example.com",
    )
def _make_mock_paper(
    doi="10.1234/test.2024",
    title="Test Paper on TopicX",
    authors=None,
    pub_date="2024-01-15",
    journal="J Membrane Sci",
    abstract="Summary here.",
    keywords=None,
    databases=None,
):
    """Build a mock findpapers Paper object with the required attributes."""
    paper = MagicMock()
    paper.doi = doi
    paper.title = title
    paper.authors = authors or ["Smith, John", "Jones, Alice"]
    # Must be a real date object: source reads paper.publication_date.year
    paper.publication_date = datetime.date.fromisoformat(pub_date) if pub_date else None
    paper.year = None
    # Must be an object with .title: source reads getattr(paper.publication, "title", "")
    paper.publication = SimpleNamespace(title=journal)
    paper.abstract = abstract
    paper.keywords = keywords or {"filtration", "ProteinA"}
    paper.databases = databases or ["openalex"]
    return paper
def _make_mock_result(papers):
    result = MagicMock()
    result.papers = papers
    return result
# ---------------------------------------------------------------------------
# search_runner tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_search_runner_converts_findpapers_output(tmp_path):
    """run_searches returns correctly structured paper dicts."""
    from lit_monitor.search.search_runner import run_searches
    mock_result = _make_mock_result([_make_mock_paper()])
    window = SearchWindow(since=date.today() - timedelta(days=7), until=None)
    with patch("lit_monitor.search.search_runner._S2_AVAILABLE", False):
        with patch("lit_monitor.search.search_runner._findpapers"):
            with patch("lit_monitor.search.search_runner._fp_load", return_value=mock_result):
                results = run_searches(_make_config(), window=window)
    assert len(results) == 1
    paper = results[0]
    assert paper["doi"] == "10.1234/test.2024"
    assert paper["title"] == "Test Paper on TopicX"
    assert isinstance(paper["authors"], list)
    assert paper["year"] == 2024
    assert paper["journal"] == "J Membrane Sci"
    assert paper["tracked_author"] is False
@pytest.mark.unit
def test_search_runner_deduplicates_same_doi():
    """Same DOI from multiple topics appears only once."""
    from lit_monitor.search.search_runner import run_searches
    mock_result = _make_mock_result([_make_mock_paper(doi="10.1/same")])
    config = _make_config(topics=["query1 AND ProteinA", "query2 AND UF"])
    window = SearchWindow(since=date.today() - timedelta(days=7), until=None)
    with patch("lit_monitor.search.search_runner._S2_AVAILABLE", False):
        with patch("lit_monitor.search.search_runner._findpapers"):
            with patch("lit_monitor.search.search_runner._fp_load", return_value=mock_result):
                results = run_searches(config, window=window)
    assert len(results) == 1
@pytest.mark.unit
def test_known_dois_filtered(tmp_path):
    """filter_known_dois removes papers already in state DB."""
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.search.search_runner import filter_known_dois
    db = StateDB(str(tmp_path / "state.db"))
    db.upsert_paper({"doi": "10.1/known", "title": "Known Paper", "source_type": "paper"})
    papers = [
        {"doi": "10.1/known", "title": "Known"},
        {"doi": "10.1/new", "title": "New"},
        {"doi": "", "title": "No DOI"},
    ]
    result = filter_known_dois(papers, db)
    dois = [p["doi"] for p in result]
    assert "10.1/known" not in dois
    assert "10.1/new" in dois
    assert "" in dois  # no-DOI papers always kept
@pytest.mark.unit
def test_missing_api_key_logs_warning_not_error(caplog, tmp_path):
    """Missing Scopus key produces a warning, not an exception."""
    import logging

    from lit_monitor.search.search_runner import run_searches
    mock_result = _make_mock_result([])
    config = _make_config(wos_key=None, scopus_key=None)
    window = SearchWindow(since=date.today() - timedelta(days=7), until=None)
    with caplog.at_level(logging.WARNING):
        with patch("lit_monitor.search.search_runner._S2_AVAILABLE", False):
            with patch("lit_monitor.search.search_runner._findpapers"):
                with patch("lit_monitor.search.search_runner._fp_load", return_value=mock_result):
                    run_searches(config, window=window)
    assert any("Scopus" in rec.message for rec in caplog.records)
@pytest.mark.unit
def test_search_runner_topic_search_failure_continues():
    """If one topic search fails, others still run."""
    from lit_monitor.search.search_runner import run_searches
    second_result = _make_mock_result([_make_mock_paper(doi="10.1/second")])
    config = _make_config(topics=["query_one", "query_two"])
    window = SearchWindow(since=date.today() - timedelta(days=7), until=None)
    with patch("lit_monitor.search.search_runner._S2_AVAILABLE", False):
        with patch("lit_monitor.search.search_runner._findpapers") as mock_fp:
            # First call to _findpapers.search raises; second succeeds (no-op)
            mock_fp.search.side_effect = [RuntimeError("Network error"), None]
            with patch("lit_monitor.search.search_runner._fp_load", return_value=second_result):
                results = run_searches(config, window=window)
    assert any(p["doi"] == "10.1/second" for p in results)
@pytest.mark.unit
def test_no_topics_returns_empty():
    """run_searches with empty topics returns [] without hitting findpapers."""
    from lit_monitor.search.search_runner import run_searches
    config = _make_config(topics=[])
    with patch("lit_monitor.search.search_runner._findpapers") as mock_fp:
        results = run_searches(config)
    mock_fp.search.assert_not_called()
    assert results == []
# ---------------------------------------------------------------------------
# researcher_tracker tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_researcher_tracker_sets_tracked_flag():
    """run_researcher_searches returns papers with tracked_author=True."""
    from lit_monitor.search.researcher_tracker import run_researcher_searches
    researcher = {"name": "Author2 G"}
    config = _make_config(researchers=[researcher])
    mock_result = _make_mock_result([_make_mock_paper(doi="10.1/author2_2024")])
    window = SearchWindow(since=date.today() - timedelta(days=7), until=None)
    with patch("lit_monitor.search.researcher_tracker._findpapers"):
        with patch("lit_monitor.search.researcher_tracker._fp_load", return_value=mock_result):
            results = run_researcher_searches(config, window=window)
    assert all(p["tracked_author"] is True for p in results)
@pytest.mark.unit
def test_researcher_tracker_deduplicates():
    """Same DOI from two author searches appears only once."""
    from lit_monitor.search.researcher_tracker import run_researcher_searches
    config = _make_config(researchers=[{"name": "Author2 G"}, {"name": "Author1 A"}])
    mock_result = _make_mock_result([_make_mock_paper(doi="10.1/shared")])
    window = SearchWindow(since=date.today() - timedelta(days=7), until=None)
    with patch("lit_monitor.search.researcher_tracker._findpapers"):
        with patch("lit_monitor.search.researcher_tracker._fp_load", return_value=mock_result):
            results = run_researcher_searches(config, window=window)
    assert sum(1 for p in results if p["doi"] == "10.1/shared") == 1
@pytest.mark.unit
def test_researcher_tracker_no_researchers_returns_empty():
    """No researchers configured → empty list without calling findpapers."""
    from lit_monitor.search.researcher_tracker import run_researcher_searches
    config = _make_config(researchers=[])
    with patch("lit_monitor.search.researcher_tracker._findpapers") as mock_fp:
        results = run_researcher_searches(config)
    mock_fp.search.assert_not_called()
    assert results == []
@pytest.mark.unit
def test_openalex_abstract_reconstruction():
    """_reconstruct_abstract correctly reconstructs from inverted index."""
    from lit_monitor.search.researcher_tracker import _reconstruct_abstract
    inverted = {"The": [0], "quick": [1], "brown": [2], "fox": [3]}
    result = _reconstruct_abstract(inverted)
    assert result == "The quick brown fox"
@pytest.mark.unit
def test_openalex_abstract_none_returns_empty():
    from lit_monitor.search.researcher_tracker import _reconstruct_abstract
    assert _reconstruct_abstract(None) == ""
