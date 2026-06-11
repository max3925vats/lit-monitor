"""
Unit tests for H4 — detect_review() in scripts/core/item_router.py.

Covers all three detection paths:
  - S2 publicationTypes == "Review" (priority 1)
  - Title/abstract keyword match (priority 2)
  - Journal name substring match (priority 3)
  - Conservative false fallback for ambiguous/non-review papers
"""
from __future__ import annotations

import pytest

from lit_monitor.core.item_router import detect_review

# ---------------------------------------------------------------------------
# Priority 1 — S2 publicationTypes
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_s2_review_type_detected():
    """S2 publicationTypes=['Review'] → True (highest priority)."""
    assert detect_review(["Review"], "", "", "") is True


@pytest.mark.unit
def test_s2_review_type_case_insensitive():
    """S2 publicationTypes matching is case-insensitive."""
    assert detect_review(["review"], "", "", "") is True


@pytest.mark.unit
def test_s2_journal_article_not_detected():
    """S2 publicationTypes=['JournalArticle'] alone → False."""
    assert detect_review(["JournalArticle"], "", "", "") is False


@pytest.mark.unit
def test_s2_mixed_types_with_review_detected():
    """S2 publicationTypes containing 'Review' alongside others → True."""
    assert detect_review(["JournalArticle", "Review"], "", "", "") is True


@pytest.mark.unit
def test_s2_none_falls_through_to_keyword():
    """s2_types=None falls through to keyword detection."""
    assert detect_review(None, "A systematic review of TopicX", "", "") is True


@pytest.mark.unit
def test_s2_empty_list_falls_through_to_keyword():
    """s2_types=[] falls through to keyword detection."""
    assert detect_review([], "A systematic review of TopicX", "", "") is True


# ---------------------------------------------------------------------------
# Priority 2 — title/abstract keyword match
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_keyword_systematic_review_in_title():
    """'systematic review' in title → True."""
    assert detect_review(None, "A Systematic Review of System Fouling", "", "") is True


@pytest.mark.unit
def test_keyword_meta_analysis_in_title():
    """'meta-analysis' in title → True."""
    assert detect_review(None, "Meta-Analysis of Filtration Outcomes", "", "") is True


@pytest.mark.unit
def test_keyword_scoping_review_in_abstract():
    """'scoping review' in abstract → True."""
    abstract = "We conducted a scoping review of published TopicX studies."
    assert detect_review(None, "Membrane Study", abstract, "") is True


@pytest.mark.unit
def test_keyword_narrative_review_in_abstract():
    """'narrative review' in abstract → True."""
    abstract = "This narrative review synthesises evidence from 45 studies."
    assert detect_review(None, "Overview", abstract, "") is True


@pytest.mark.unit
def test_keyword_literature_review_in_title():
    """'literature review' in title → True."""
    assert detect_review(None, "A Literature Review on MethodB", "", "") is True


@pytest.mark.unit
def test_keyword_matching_is_case_insensitive():
    """Keyword matching is case-insensitive."""
    assert detect_review(None, "SYSTEMATIC REVIEW OF TANGENTIAL FLOW", "", "") is True


@pytest.mark.unit
def test_keyword_not_matched_for_primary_research():
    """Primary research paper without review keywords → keyword path returns False."""
    title = "Protein aggregation during filtration"
    abstract = "We measured flux decline under various conditions."
    assert detect_review(None, title, abstract, "") is False


# ---------------------------------------------------------------------------
# Priority 3 — journal name substring
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_journal_reviews_substring_detected():
    """Journal name containing 'reviews' → True."""
    assert detect_review(None, "", "", "Biotechnology Reviews") is True


@pytest.mark.unit
def test_journal_annual_review_detected():
    """Journal name containing 'annual review' → True."""
    assert detect_review(None, "", "", "Annual Review of Chemical Engineering") is True


@pytest.mark.unit
def test_journal_matching_is_case_insensitive():
    """Journal name matching is case-insensitive."""
    assert detect_review(None, "", "", "ANNUAL REVIEW OF BIOCHEMISTRY") is True


@pytest.mark.unit
def test_journal_normal_name_not_detected():
    """Non-review journal name → journal path returns False."""
    assert detect_review(None, "", "", "Journal of Mock Domain") is False


# ---------------------------------------------------------------------------
# Conservative False — all paths negative
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_returns_false_when_all_signals_absent():
    """Returns False when no review signals are present anywhere."""
    result = detect_review(
        s2_types=["JournalArticle"],
        title="Protein aggregation in UF modules",
        abstract="We measured pressure drop across membranes.",
        journal="Journal of Mock Domain",
    )
    assert result is False


@pytest.mark.unit
def test_returns_false_for_empty_inputs():
    """Returns False when all inputs are empty strings and s2_types is None."""
    assert detect_review(None, "", "", "") is False


@pytest.mark.unit
def test_s2_priority_overrides_non_review_journal():
    """S2 Review classification wins even when journal name is ambiguous."""
    result = detect_review(
        s2_types=["Review"],
        title="Overview of chromatography",
        abstract="",
        journal="Journal of Chromatography A",  # not a review journal
    )
    assert result is True
