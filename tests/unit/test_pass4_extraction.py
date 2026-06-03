"""
Unit tests for M3 complex-phase extraction.

Replaces the old G2 / Pass-4 tests.  Covers:
  - extract_complex() sends full text to LLM without end-matter stripping
  - extract_complex() never chunks — exactly one LLM call even for long text
  - extract_complex() logs WARNING when text exceeds the context budget
  - PAPER_COMPLEX_FIELDS contains key_citations and comparison_to_prior
  - re_extract() with phases=["complex"] routes to extract_paper(phases=("complex",))
  - COALESCE: complex re-run preserves existing simple-phase fields
  - re_extract() default (no phases arg) runs both phases
  - re_extract_all_failed_phase("complex") targets _complex_error papers
  - re_extract_all_failed_phase("simple") targets _simple_error papers
  - re_extract_all_failed_phase skips healthy papers and raises on bad phase name
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm.llm_client import MockLLMClient

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "This paper presents a novel TopicX approach.\n\n"
    "We compare against Smith et al. (2020) [1] as the benchmark.\n\n"
    "## References\n\n"
    "[1] Smith, J. et al. (2020). System Fouling. J. Mock Sci., 500, 100–110.\n"
)


def _make_state_db_with(record: dict) -> MagicMock:
    db = MagicMock()
    db.get_paper.return_value = record
    return db


# ---------------------------------------------------------------------------
# extract_complex() — full-text pass-through (no end-matter stripping)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_complex_sends_full_text_including_references():
    """extract_complex() must NOT strip end-matter — full text is sent to LLM.

    Key difference from extract_simple: references are needed for
    key_citations substantiveness judgment.
    """
    from scripts.llm.extractor import extract_complex

    seen_user_prompts: list[str] = []

    class CapturingMock(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 12288) -> str:
            seen_user_prompts.append(user)
            return super().complete(system, user, max_tokens)

    llm = CapturingMock(mock_response_key="paper_pass3")
    extract_complex(SAMPLE_TEXT, llm)

    assert seen_user_prompts, "LLM was never called"
    user_prompt = seen_user_prompts[0]
    # The bibliography line must still be in the prompt.
    assert "Smith, J. et al. (2020)" in user_prompt, (
        "References must NOT be stripped for the complex phase"
    )
    assert "## References" in user_prompt


@pytest.mark.unit
def test_extract_complex_does_not_call_strip_end_matter():
    """extract_complex() must not call strip_end_matter (that's for extract_simple)."""
    from scripts.llm.extractor import extract_complex

    llm = MockLLMClient(mock_response_key="paper_pass3")
    with patch("scripts.llm.extractor.strip_end_matter") as mock_strip:
        extract_complex(SAMPLE_TEXT, llm)

    mock_strip.assert_not_called()


# ---------------------------------------------------------------------------
# extract_complex() — no chunking, always one LLM call
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_complex_single_llm_call_for_short_text():
    """Short text → exactly one LLM call, n_chunks == 1."""
    from scripts.llm.extractor import extract_complex

    llm = MockLLMClient(mock_response_key="paper_pass3")
    pr = extract_complex(SAMPLE_TEXT, llm)

    assert llm.call_count == 1
    assert pr.n_chunks == 1


@pytest.mark.unit
def test_extract_complex_single_llm_call_for_long_text():
    """extract_complex() makes exactly one LLM call regardless of text length.

    Even text that would trigger A6 chunking in extract_simple must not
    cause multiple calls here — whole-paper context is required for
    citation substantiveness judgment.
    """
    from scripts.llm.extractor import extract_complex

    # Force a very small num_ctx so any chunking would be triggered.
    class SmallCtxMock(MockLLMClient):
        num_ctx = 512

    llm = SmallCtxMock(mock_response_key="paper_pass3")
    long_text = "Important scientific content. " * 5_000  # ~150k chars
    pr = extract_complex(long_text, llm)

    assert llm.call_count == 1, (
        f"extract_complex() must never chunk — expected 1 call, got {llm.call_count}"
    )
    assert pr.n_chunks == 1


@pytest.mark.unit
def test_extract_complex_warns_when_text_exceeds_budget(caplog):
    """extract_complex() logs WARNING when fulltext exceeds the estimated input budget."""
    from scripts.llm.extractor import extract_complex

    class TinyCtxMock(MockLLMClient):
        num_ctx = 512  # tiny → budget ≈ 4000 chars (clamped to _MIN_CHUNK_CHARS)

    llm = TinyCtxMock(mock_response_key="paper_pass3")
    long_text = "A" * 100_000  # well beyond any realistic budget

    with caplog.at_level(logging.WARNING, logger="scripts.llm.extractor"):
        extract_complex(long_text, llm)

    assert any("complex phase" in r.message.lower() for r in caplog.records), (
        "A WARNING about text length should be logged for the complex phase"
    )


# ---------------------------------------------------------------------------
# PAPER_COMPLEX_FIELDS — schema check
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_paper_complex_fields_contains_citation_fields():
    """key_citations and comparison_to_prior must be in PAPER_COMPLEX_FIELDS (M3)."""
    from scripts.llm.extractor import PAPER_COMPLEX_FIELDS
    assert "key_citations" in PAPER_COMPLEX_FIELDS
    assert "comparison_to_prior" in PAPER_COMPLEX_FIELDS


@pytest.mark.unit
def test_paper_complex_fields_contains_discovered_topics():
    """discovered_topics must be in PAPER_COMPLEX_FIELDS (M4 field, complex phase)."""
    from scripts.llm.extractor import PAPER_COMPLEX_FIELDS
    assert "discovered_topics" in PAPER_COMPLEX_FIELDS


@pytest.mark.unit
def test_paper_complex_fields_does_not_overlap_simple_fields():
    """Simple and complex field sets must be disjoint."""
    from scripts.llm.extractor import PAPER_COMPLEX_FIELDS, PAPER_SIMPLE_FIELDS
    overlap = set(PAPER_SIMPLE_FIELDS) & set(PAPER_COMPLEX_FIELDS)
    assert not overlap, f"Fields appear in both phases: {overlap}"


# ---------------------------------------------------------------------------
# re_extract() — complex phase routing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_re_extract_phases_complex_calls_extract_paper_with_complex():
    """re_extract(phases=["complex"]) must call extract_paper with phases=("complex",)."""
    from scripts.obsidian_tools.re_extract import re_extract

    record = {
        "doi": "10.1/test",
        "source_type": "paper",
        "extraction_json": json.dumps({"core_finding": "Existing.", "core_finding_confidence": "explicit"}),
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }
    state_db = _make_state_db_with(record)
    llm = MockLLMClient()
    config = SimpleNamespace()

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value=SAMPLE_TEXT), \
         patch("scripts.obsidian_tools.re_extract.extract_paper") as mock_ep:
        mock_ep.return_value = {
            "novelty_statement": "Novel.",
            "novelty_statement_confidence": "explicit",
            "_overall_confidence": 0.9,
            "_max_n_chunks": 1,
        }
        re_extract("10.1/test", config, state_db, llm, phases=["complex"], rerender=False)

    mock_ep.assert_called_once()
    call_kwargs = mock_ep.call_args.kwargs
    assert call_kwargs.get("phases") == ("complex",), (
        f"extract_paper must be called with phases=('complex',), got {call_kwargs.get('phases')}"
    )


@pytest.mark.unit
def test_re_extract_complex_passes_existing_extraction_to_extract_paper():
    """COALESCE guarantee: re_extract passes existing extraction to extract_paper.

    The merge itself happens inside extract_paper (via existing_extraction=).
    This test verifies that re_extract passes the loaded existing extraction
    so that extract_paper can merge rather than start from scratch.
    """
    from scripts.obsidian_tools.re_extract import re_extract

    existing = {
        "core_finding": "Simple phase result.",
        "core_finding_confidence": "explicit",
        "_overall_confidence": 0.7,
    }
    record = {
        "doi": "10.1/coalesce",
        "source_type": "paper",
        "extraction_json": json.dumps(existing),
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }
    state_db = _make_state_db_with(record)

    # Simulate what real extract_paper does: start from existing, add complex fields.
    def merge_mock(fulltext, llm, *, content_type, ocr_heavy, phases, existing_extraction):
        merged = dict(existing_extraction) if existing_extraction else {}
        merged.update({
            "novelty_statement": "Novel approach.",
            "novelty_statement_confidence": "explicit",
            "_overall_confidence": 0.9,
            "_max_n_chunks": 1,
        })
        return merged

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value=SAMPLE_TEXT), \
         patch("scripts.obsidian_tools.re_extract.extract_paper", side_effect=merge_mock) as mock_ep:
        re_extract("10.1/coalesce", SimpleNamespace(), state_db, MockLLMClient(),
                   phases=["complex"], rerender=False)

    # Verify existing_extraction was passed through
    call_kwargs = mock_ep.call_args.kwargs
    assert call_kwargs.get("existing_extraction") == existing, (
        "re_extract must pass the loaded existing extraction to extract_paper "
        "so that extract_paper can merge (COALESCE) rather than start empty"
    )

    # Verify update_extraction_json received the merged result (both old + new fields).
    saved_data = state_db.update_extraction_json.call_args.args[1]
    assert saved_data.get("core_finding") == "Simple phase result.", (
        "Merged result must include the simple-phase core_finding from existing"
    )
    assert saved_data.get("novelty_statement") == "Novel approach.", (
        "Merged result must include the new complex-phase novelty_statement"
    )


@pytest.mark.unit
def test_re_extract_default_runs_both_phases():
    """re_extract() with no phases arg must call extract_paper with both phases."""
    from scripts.obsidian_tools.re_extract import re_extract

    record = {
        "doi": "10.1/default",
        "source_type": "paper",
        "extraction_json": "{}",
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }
    state_db = _make_state_db_with(record)

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value=SAMPLE_TEXT), \
         patch("scripts.obsidian_tools.re_extract.extract_paper") as mock_ep:
        mock_ep.return_value = {"_overall_confidence": 0.5, "_max_n_chunks": 1}
        re_extract("10.1/default", SimpleNamespace(), state_db, MockLLMClient(), rerender=False)

    mock_ep.assert_called_once()
    call_kwargs = mock_ep.call_args.kwargs
    phases = call_kwargs.get("phases")
    assert "simple" in phases and "complex" in phases, (
        f"Default re_extract must run both phases; got phases={phases}"
    )


@pytest.mark.unit
def test_re_extract_raises_for_unknown_source_type():
    """re_extract() raises ValueError for unrecognised source_type."""
    from scripts.obsidian_tools.re_extract import re_extract

    record = {
        "doi": "10.1/bad",
        "source_type": "dataset",  # not 'paper' or 'review'
        "extraction_json": "{}",
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }
    state_db = _make_state_db_with(record)

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value=SAMPLE_TEXT), \
         pytest.raises(ValueError, match="source_type"):
        re_extract("10.1/bad", SimpleNamespace(), state_db, MockLLMClient(), rerender=False)


# ---------------------------------------------------------------------------
# re_extract_all_failed_phase() — bulk complex/simple re-extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_re_extract_all_failed_phase_targets_complex_error_papers():
    """re_extract_all_failed_phase("complex") re-runs papers with _complex_error."""
    from scripts.obsidian_tools.re_extract import re_extract_all_failed_phase

    errored = {
        "doi": "10.1/complex_fail",
        "source_type": "paper",
        "extraction_json": json.dumps({
            "core_finding": "Simple ran fine.",
            "_complex_error": "LLM timeout during complex phase.",
        }),
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }
    state_db = MagicMock()
    state_db.get_all_by_source_type.return_value = [errored]
    state_db.get_paper.return_value = errored

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value=SAMPLE_TEXT), \
         patch("scripts.obsidian_tools.re_extract.extract_paper") as mock_ep:
        mock_ep.return_value = {
            "novelty_statement": "OK now.",
            "_overall_confidence": 0.8,
            "_max_n_chunks": 1,
        }
        stats = re_extract_all_failed_phase(
            phase="complex",
            config=SimpleNamespace(),
            state_db=state_db,
            llm=MockLLMClient(),
        )

    assert stats["re_extracted"] == 1
    assert stats["failed"] == 0
    mock_ep.assert_called_once()


@pytest.mark.unit
def test_re_extract_all_failed_phase_targets_simple_error_papers():
    """re_extract_all_failed_phase("simple") re-runs papers with _simple_error."""
    from scripts.obsidian_tools.re_extract import re_extract_all_failed_phase

    errored = {
        "doi": "10.1/simple_fail",
        "source_type": "paper",
        "extraction_json": json.dumps({
            "_simple_error": "Chunking failed unexpectedly.",
        }),
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }
    state_db = MagicMock()
    state_db.get_all_by_source_type.return_value = [errored]
    state_db.get_paper.return_value = errored

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value=SAMPLE_TEXT), \
         patch("scripts.obsidian_tools.re_extract.extract_paper") as mock_ep:
        mock_ep.return_value = {
            "core_finding": "Got it now.",
            "_overall_confidence": 0.9,
            "_max_n_chunks": 1,
        }
        stats = re_extract_all_failed_phase(
            phase="simple",
            config=SimpleNamespace(),
            state_db=state_db,
            llm=MockLLMClient(),
        )

    assert stats["re_extracted"] == 1
    assert stats["failed"] == 0


@pytest.mark.unit
def test_re_extract_all_failed_phase_skips_healthy_papers():
    """Papers without the phase-specific error key are skipped."""
    from scripts.obsidian_tools.re_extract import re_extract_all_failed_phase

    healthy = {
        "doi": "10.1/healthy",
        "source_type": "paper",
        "extraction_json": json.dumps({
            "core_finding": "All good.",
            "novelty_statement": "Novel.",
        }),
    }
    state_db = MagicMock()
    state_db.get_all_by_source_type.return_value = [healthy]

    with patch("scripts.obsidian_tools.re_extract.extract_paper") as mock_ep:
        stats = re_extract_all_failed_phase(
            phase="complex",
            config=SimpleNamespace(),
            state_db=state_db,
            llm=MockLLMClient(),
        )

    assert stats["re_extracted"] == 0
    assert stats["failed"] == 0
    mock_ep.assert_not_called()


@pytest.mark.unit
def test_re_extract_all_failed_phase_raises_on_invalid_phase():
    """re_extract_all_failed_phase() raises ValueError for unrecognised phase names."""
    from scripts.obsidian_tools.re_extract import re_extract_all_failed_phase

    state_db = MagicMock()
    state_db.get_all_by_source_type.return_value = []

    with pytest.raises(ValueError, match="phase must be"):
        re_extract_all_failed_phase(
            phase="pass1",  # old pass-based name — must be rejected
            config=SimpleNamespace(),
            state_db=state_db,
            llm=MockLLMClient(),
        )


@pytest.mark.unit
def test_re_extract_all_failed_phase_counts_individual_failures():
    """Single-item failures increment the 'failed' counter and do not abort the batch."""
    from scripts.obsidian_tools.re_extract import re_extract_all_failed_phase

    records = [
        {
            "doi": f"10.1/fail{i}",
            "source_type": "paper",
            "extraction_json": json.dumps({"_complex_error": "timeout"}),
            "note_path": "",
            "ocr_heavy": 0,
            "zotero_key": "",
        }
        for i in range(3)
    ]
    state_db = MagicMock()
    state_db.get_all_by_source_type.return_value = records
    # get_paper returns the record for each DOI
    state_db.get_paper.side_effect = records.__iter__().__next__

    with patch("scripts.obsidian_tools.re_extract._load_fulltext",
               side_effect=RuntimeError("disk error")):
        stats = re_extract_all_failed_phase(
            phase="complex",
            config=SimpleNamespace(),
            state_db=state_db,
            llm=MockLLMClient(),
        )

    assert stats["re_extracted"] == 0
    assert stats["failed"] == 3


@pytest.mark.unit
def test_re_extract_all_failed_phase_raises_under_strict():
    """P4.4: in --strict a per-paper re-extract failure escalates to a raise
    instead of being silently counted in stats['failed']."""
    import scripts.core.strict_mode as _sm
    from scripts.obsidian_tools.re_extract import re_extract_all_failed_phase

    record = {
        "doi": "10.1/fail",
        "source_type": "paper",
        "extraction_json": json.dumps({"_complex_error": "timeout"}),
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }
    state_db = MagicMock()
    state_db.get_all_by_source_type.return_value = [record]
    state_db.get_paper.return_value = record

    _sm.set_strict(True)
    with patch("scripts.obsidian_tools.re_extract._load_fulltext",
               side_effect=RuntimeError("disk error")):
        with pytest.raises(Exception, match="Phase re-extract failed"):
            re_extract_all_failed_phase(
                phase="complex",
                config=SimpleNamespace(),
                state_db=state_db,
                llm=MockLLMClient(),
            )
