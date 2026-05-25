"""
Unit tests for scripts.core.doi_resolver (D2).

All tests mock ``scripts.core.doi_resolver.Crossref`` so no live network
calls are made.  The CrossRef response structure mirrors the real API:
  result["message"]["items"] — list of dicts with "DOI" and "score" keys.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.core.doi_resolver import resolve_doi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_crossref_response(doi: str, score: float) -> dict:
    """Build a minimal CrossRef API response dict."""
    return {
        "message": {
            "items": [
                {"DOI": doi, "score": score},
            ]
        }
    }


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

def test_doi_resolver_returns_doi_on_high_score():
    """High-score CrossRef hit → DOI returned."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = _make_crossref_response(
        "10.1016/j.foo.2023.001", score=95.0
    )
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        result = resolve_doi(
            title="A Novel Filtration Study",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
        )
    assert result == "10.1016/j.foo.2023.001"


def test_doi_resolver_returns_none_on_low_score():
    """Score below threshold → None (false-positive guard)."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = _make_crossref_response(
        "10.1016/j.foo.2023.001", score=30.0
    )
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        result = resolve_doi(
            title="A Novel Filtration Study",
            authors=["Smith, John"],
            year=2023,
        )
    assert result is None


def test_doi_resolver_returns_none_on_empty_title():
    """Empty / whitespace-only title → None immediately, no API call."""
    with patch("scripts.core.doi_resolver.Crossref") as mock_cls:
        assert resolve_doi("", []) is None
        assert resolve_doi("   ", []) is None
    mock_cls.assert_not_called()


def test_doi_resolver_returns_none_on_api_error():
    """Network / API exception → None (logged as warning, not raised)."""
    mock_cr = MagicMock()
    mock_cr.works.side_effect = RuntimeError("connection refused")
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        result = resolve_doi("Some Title", ["Author, One"])
    assert result is None


def test_doi_resolver_returns_none_on_empty_items():
    """CrossRef returns an empty items list → None."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = {"message": {"items": []}}
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        result = resolve_doi("Some Title", [])
    assert result is None


def test_doi_resolver_returns_none_when_top_item_has_no_doi():
    """Top result has score above threshold but empty DOI field → None."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = {"message": {"items": [{"DOI": "", "score": 90.0}]}}
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        result = resolve_doi("Some Title With No DOI", ["Author, X"])
    assert result is None


# ---------------------------------------------------------------------------
# Author extraction
# ---------------------------------------------------------------------------

def test_doi_resolver_uses_up_to_three_author_last_names():
    """Only the first 3 authors' last names are sent as query.author."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = _make_crossref_response("10.1/x", score=80.0)
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        resolve_doi(
            "Title",
            ["Alpha, A", "Beta, B", "Gamma, C", "Delta, D"],  # 4 authors
        )
    call_kwargs = mock_cr.works.call_args.kwargs
    # Only first 3 last names joined
    assert call_kwargs["query_author"] == "Alpha, Beta, Gamma"


def test_doi_resolver_works_with_empty_authors_list():
    """No authors → query_author is None, API call still made."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = _make_crossref_response("10.1/y", score=85.0)
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        result = resolve_doi("Some Title", [])
    assert result == "10.1/y"
    call_kwargs = mock_cr.works.call_args.kwargs
    assert call_kwargs["query_author"] is None


# ---------------------------------------------------------------------------
# mailto / polite pool
# ---------------------------------------------------------------------------

def test_doi_resolver_passes_mailto_to_crossref():
    """mailto kwarg is forwarded to Crossref constructor."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = _make_crossref_response("10.1/z", score=75.0)
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr) as mock_cls:
        resolve_doi("Title", [], mailto="test@example.com")
    mock_cls.assert_called_once_with(mailto="test@example.com")


def test_doi_resolver_creates_crossref_without_mailto_when_none():
    """When mailto is None, Crossref() is called without the kwarg."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = _make_crossref_response("10.1/z", score=75.0)
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr) as mock_cls:
        resolve_doi("Title", [], mailto=None)
    mock_cls.assert_called_once_with()


# ---------------------------------------------------------------------------
# min_score threshold
# ---------------------------------------------------------------------------

def test_doi_resolver_accepts_custom_min_score():
    """Lower min_score accepts results that the default 70.0 would reject."""
    mock_cr = MagicMock()
    mock_cr.works.return_value = _make_crossref_response("10.1/low", score=50.0)
    with patch("scripts.core.doi_resolver.Crossref", return_value=mock_cr):
        result = resolve_doi("Title", [], min_score=40.0)
    assert result == "10.1/low"
