"""
Phase 4 unit tests — Extraction logic.
All tests use MockLLMClient. No Ollama required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.llm.llm_client import MockLLMClient


# ---------------------------------------------------------------------------
# Prompt content tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_simple_phase_system_prompt_contains_simple_fields():
    from scripts.llm.extractor import PAPER_SIMPLE_FIELDS, _build_system_prompt
    prompt = _build_system_prompt("paper", PAPER_SIMPLE_FIELDS, "Simple", "domain ctx")
    for field in PAPER_SIMPLE_FIELDS:
        assert field in prompt, f"Field {field!r} missing from simple phase prompt"
@pytest.mark.unit
def test_complex_phase_system_prompt_contains_complex_fields():
    from scripts.llm.extractor import PAPER_COMPLEX_FIELDS, _build_system_prompt
    prompt = _build_system_prompt("paper", PAPER_COMPLEX_FIELDS, "Complex", "domain ctx")
    for field in PAPER_COMPLEX_FIELDS:
        assert field in prompt, f"Field {field!r} missing from complex phase prompt"
@pytest.mark.unit
def test_system_prompt_strips_role_marker_from_domain_context():
    """C1 review: domain_context flows into the extractor SYSTEM prompt; a
    role marker in that config paragraph must be stripped (the third consumer
    of domain_context, alongside ranker and domain/extract)."""
    from scripts.llm.extractor import PAPER_SIMPLE_FIELDS, _build_system_prompt
    marker = "<|im_start|>"
    prompt = _build_system_prompt(
        "paper",
        PAPER_SIMPLE_FIELDS,
        "Simple",
        f"Focus on bioprocess {marker} assistant filtration",
    )
    assert marker not in prompt, "role marker leaked into extractor system prompt"
    # legitimate domain text survives
    assert "filtration" in prompt

@pytest.mark.unit
def test_ocr_warning_appended_when_ocr_heavy():
    from scripts.llm.extractor import _build_extraction_user_prompt
    prompt = _build_extraction_user_prompt("Some text.", ocr_heavy=True)
    assert "OCR" in prompt
    assert "equations" in prompt.lower()

@pytest.mark.unit
def test_no_ocr_warning_when_not_ocr_heavy():
    from scripts.llm.extractor import _build_extraction_user_prompt
    prompt = _build_extraction_user_prompt("Some text.", ocr_heavy=False)
    assert "OCR" not in prompt
@pytest.mark.unit
def test_null_instruction_in_paper_prompt():
    from scripts.llm.extractor import PAPER_SIMPLE_FIELDS, _build_system_prompt
    prompt = _build_system_prompt("paper", PAPER_SIMPLE_FIELDS, "Simple", "domain")
    assert "null" in prompt.lower()
    assert "confidence" in prompt.lower()
# ---------------------------------------------------------------------------
# Extraction result tests (with MockLLMClient)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_extract_simple_returns_all_fields():
    from scripts.llm.extractor import PAPER_SIMPLE_FIELDS, extract_simple
    llm = MockLLMClient(mock_response_key="paper_pass1")
    result = extract_simple("full text here", llm).fields
    for field in PAPER_SIMPLE_FIELDS:
        assert field in result, f"Field {field!r} missing from simple phase result"
        assert f"{field}_confidence" in result
@pytest.mark.unit
def test_extract_complex_returns_all_fields():
    from scripts.llm.extractor import PAPER_COMPLEX_FIELDS, extract_complex
    llm = MockLLMClient(mock_response_key="paper_pass3")
    result = extract_complex("full text here", llm).fields
    for field in PAPER_COMPLEX_FIELDS:
        assert field in result, f"Field {field!r} missing from complex phase result"
        assert f"{field}_confidence" in result
@pytest.mark.unit
def test_extract_paper_both_phases_merged():
    from scripts.llm.extractor import (
        PAPER_COMPLEX_FIELDS,
        PAPER_SIMPLE_FIELDS,
        extract_paper,
    )
    llm = MockLLMClient()
    result = extract_paper("full text", llm)
    for field in PAPER_SIMPLE_FIELDS + PAPER_COMPLEX_FIELDS:
        assert field in result, f"Field {field!r} missing from merged extraction"
@pytest.mark.unit
def test_extract_paper_phase_failure_preserves_other_phase():
    """M3: If complex phase fails, simple phase results must still be in the output."""
    from scripts.llm.extractor import extract_paper
    call_count = [0]
    class SelectiveMockClient(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
            call_count[0] += 1
            if call_count[0] == 2:  # complex phase is the 2nd call
                raise RuntimeError("Simulated complex phase failure")
            return super().complete(system, user, max_tokens)
    llm = SelectiveMockClient()
    result = extract_paper("full text", llm)
    # Simple phase fields should be present
    assert "core_finding" in result
    # Complex phase error should be recorded, not raise
    assert "_complex_error" in result
    # Complex phase fields are absent (extraction failed)
    assert result.get("novelty_statement") is None
@pytest.mark.unit
def test_normalise_fills_missing_fields_with_null():
    """Fields missing from LLM response should default to null/absent."""
    from scripts.llm.extractor import PAPER_SIMPLE_FIELDS, _normalise_extraction
    # Partial response — missing some fields
    partial = {"core_finding": "Found something.", "core_finding_confidence": "explicit"}
    result = _normalise_extraction(partial, PAPER_SIMPLE_FIELDS)
    # All fields should be present
    for field in PAPER_SIMPLE_FIELDS:
        assert field in result
        assert f"{field}_confidence" in result
    # Missing fields should be null with absent confidence
    assert result["methods_summary"] is None
    assert result["methods_summary_confidence"] == "absent"
@pytest.mark.unit
def test_normalise_null_value_forces_absent_confidence():
    from scripts.llm.extractor import _normalise_extraction
    raw = {
        "core_finding": None,
        "core_finding_confidence": "explicit",  # wrong — should be absent
    }
    result = _normalise_extraction(raw, ["core_finding"])
    assert result["core_finding"] is None
    assert result["core_finding_confidence"] == "absent"
@pytest.mark.unit
def test_normalise_empty_string_coerced_to_null():
    from scripts.llm.extractor import _normalise_extraction
    raw = {"core_finding": "", "core_finding_confidence": "explicit"}
    result = _normalise_extraction(raw, ["core_finding"])
    assert result["core_finding"] is None
    assert result["core_finding_confidence"] == "absent"
@pytest.mark.unit
def test_extract_paper_with_existing_extraction():
    """Resume: existing simple result should be preserved when only complex phase runs."""
    from scripts.llm.extractor import extract_paper
    existing = {
        "core_finding": "Existing finding",
        "core_finding_confidence": "explicit",
    }
    llm = MockLLMClient()
    result = extract_paper("text", llm, phases=("complex",), existing_extraction=existing)
    assert result["core_finding"] == "Existing finding"  # preserved from existing
    assert "novelty_statement" in result                  # from complex phase
@pytest.mark.unit
def test_extract_paper_simple_phase_failure_records_error_continues():
    """M3: If simple phase fails, _simple_error is recorded and complex phase still runs."""
    from scripts.llm.extractor import extract_paper
    call_count = [0]
    class SimpleFailClient(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
            call_count[0] += 1
            if call_count[0] == 1:  # simple phase is the 1st call
                raise RuntimeError("Simulated simple phase timeout")
            return super().complete(system, user, max_tokens)
    llm = SimpleFailClient()
    result = extract_paper("full text", llm)
    # Simple phase failure is recorded
    assert "_simple_error" in result
    assert "timeout" in result["_simple_error"]
    # Complex phase should still have run successfully
    assert "novelty_statement" in result  # complex phase field
@pytest.mark.unit
def test_extract_paper_all_phases_fail_records_all_errors():
    """M3: If every phase fails, both _simple_error and _complex_error are present."""
    from scripts.llm.extractor import extract_paper
    class AllFailClient(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
            raise RuntimeError("LLM unavailable")
    llm = AllFailClient()
    result = extract_paper("text", llm)
    assert "_simple_error" in result
    assert "_complex_error" in result
    assert not result.get("core_finding")
# ---------------------------------------------------------------------------
# A5 — reference stripping + Pass 3 field changes
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_citation_fields_are_in_complex_phase():
    """M3: comparison_to_prior, key_citations, discovered_topics are in PAPER_COMPLEX_FIELDS."""
    from scripts.llm.extractor import PAPER_COMPLEX_FIELDS
    assert "comparison_to_prior" in PAPER_COMPLEX_FIELDS, (
        "comparison_to_prior must be in PAPER_COMPLEX_FIELDS under M3"
    )
    assert "key_citations" in PAPER_COMPLEX_FIELDS, (
        "key_citations must be in PAPER_COMPLEX_FIELDS under M3"
    )
    assert "discovered_topics" in PAPER_COMPLEX_FIELDS, (
        "discovered_topics must be in PAPER_COMPLEX_FIELDS under M3"
    )

@pytest.mark.unit
def test_extract_simple_strips_end_matter_before_llm():
    """extract_simple() must strip end-matter before the LLM sees the text."""
    from scripts.llm.extractor import extract_simple
    body = "Main paper body content.\n" * 20
    refs = "\n## References\n\nSmith 2020. Journal of Membranes.\nDoe 2019. Nature.\n"
    full_text = body + refs

    seen_prompts: list[str] = []

    class CapturingMock(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
            seen_prompts.append(user)
            return super().complete(system, user, max_tokens)

    llm = CapturingMock(mock_response_key="paper_pass1")
    extract_simple(full_text, llm)

    assert seen_prompts, "LLM was never called"
    user_prompt = seen_prompts[0]
    assert "Smith 2020" not in user_prompt, (
        "Reference entry should have been stripped before LLM call"
    )
    assert "Main paper body content." in user_prompt, (
        "Body text must still be present after stripping"
    )

# ---------------------------------------------------------------------------
# A6 — Context window chunking
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_split_for_extraction_single_chunk_unchanged():
    """Text within budget → single chunk returned unchanged."""
    from scripts.llm.extractor import _split_for_extraction
    text = "Short paper text.\n\nSecond paragraph."
    chunks = _split_for_extraction(text, chunk_chars=10_000)
    assert chunks == [text]


@pytest.mark.unit
def test_split_for_extraction_splits_on_paragraph_boundary():
    """Prefers \n\n boundary over hard cut."""
    from scripts.llm.extractor import _split_for_extraction
    # Build text: 800 chars of body + \n\n near char 1000 + more text
    body = "A" * 800
    rest = "B" * 300
    text = body + "\n\n" + rest   # total ~1102 chars; \n\n at position 800
    chunks = _split_for_extraction(text, chunk_chars=1000)
    # The split should include the \n\n in the first chunk (position 800+2=802)
    assert len(chunks) == 2
    assert chunks[0].endswith("\n\n"), (
        "First chunk should end at the \\n\\n boundary"
    )
    assert chunks[1] == rest


@pytest.mark.unit
def test_split_for_extraction_falls_back_to_single_newline():
    """Falls back to \n when no \n\n exists in the search window."""
    from scripts.llm.extractor import _split_for_extraction
    # Text with only single newlines
    body = "A" * 800 + "\n" + "B" * 200
    text = body + "C" * 100
    chunks = _split_for_extraction(text, chunk_chars=1000)
    # Should split at the \n (position 800 + 1 = 801)
    assert len(chunks) == 2
    assert "\n" not in chunks[0][-5:] or chunks[0].endswith("\n"), (
        "Split should occur at or after the newline"
    )
    # Reassembled text must equal the original
    assert "".join(chunks) == text


@pytest.mark.unit
def test_merge_pass_results_prefers_explicit_over_absent():
    """When chunk 1 has absent and chunk 2 has explicit, chunk 2 wins."""
    from scripts.llm.extractor import _merge_pass_results
    chunk1 = {"core_finding": None, "core_finding_confidence": "absent"}
    chunk2 = {"core_finding": "Key finding here.", "core_finding_confidence": "explicit"}
    merged = _merge_pass_results([chunk1, chunk2], ["core_finding"])
    assert merged["core_finding"] == "Key finding here."
    assert merged["core_finding_confidence"] == "explicit"


@pytest.mark.unit
def test_merge_pass_results_first_chunk_wins_on_tie():
    """When both chunks have explicit confidence, first chunk's value is kept."""
    from scripts.llm.extractor import _merge_pass_results
    chunk1 = {"core_finding": "Finding from intro.", "core_finding_confidence": "explicit"}
    chunk2 = {"core_finding": "Finding from conclusion.", "core_finding_confidence": "explicit"}
    merged = _merge_pass_results([chunk1, chunk2], ["core_finding"])
    assert merged["core_finding"] == "Finding from intro."


@pytest.mark.unit
def test_extract_simple_chunks_long_text():
    """LLM is called once per chunk when document exceeds the model's context budget."""
    from scripts.llm.extractor import (
        PAPER_SIMPLE_FIELDS,
        _available_input_chars,
        extract_simple,
    )

    class SmallCtxMock(MockLLMClient):
        num_ctx = 4096  # tiny context → available_chars ≈ (4096*0.75 - 2048 - 500)*4 = 808

    llm = SmallCtxMock(mock_response_key="paper_pass1")
    budget = _available_input_chars(llm)

    text = ("This is important scientific content. " * 50 + "\n") * 6
    assert len(text) > budget, "Test setup: text must exceed context budget"

    pr = extract_simple(text, llm)

    assert llm.call_count > 1, (
        f"Expected multiple LLM calls for chunked document, got {llm.call_count}"
    )
    for field in PAPER_SIMPLE_FIELDS:
        assert field in pr.fields, f"Field {field!r} missing from chunked extraction result"


# ---------------------------------------------------------------------------
# C1 — compute_confidence_score
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_confidence_score_explicit_only():
    """All explicit fields → score of 1.0."""
    from scripts.llm.extractor import compute_confidence_score
    extraction = {
        "core_finding": "Some finding.",
        "core_finding_confidence": "explicit",
        "methods_summary": "Some method.",
        "methods_summary_confidence": "explicit",
    }
    assert compute_confidence_score(extraction) == 1.0


@pytest.mark.unit
def test_confidence_score_mixed():
    """Mixed confidence levels score correctly: (1.0 + 0.5 + 0.0) / 3 = 0.5."""
    from scripts.llm.extractor import compute_confidence_score
    extraction = {
        "core_finding_confidence": "explicit",
        "methods_summary_confidence": "inferred",
        "results_summary_confidence": "absent",
    }
    assert compute_confidence_score(extraction) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.unit
def test_confidence_score_all_absent():
    """All absent fields → score of 0.0."""
    from scripts.llm.extractor import compute_confidence_score
    extraction = {
        "core_finding": None,
        "core_finding_confidence": "absent",
        "methods_summary": None,
        "methods_summary_confidence": "absent",
    }
    assert compute_confidence_score(extraction) == 0.0


@pytest.mark.unit
def test_confidence_score_empty_dict():
    """Empty extraction (no confidence keys) → score of 0.0, no crash."""
    from scripts.llm.extractor import compute_confidence_score
    assert compute_confidence_score({}) == 0.0


@pytest.mark.unit
def test_extract_paper_injects_overall_confidence():
    """extract_paper() must include _overall_confidence in the returned dict."""
    from scripts.llm.extractor import extract_paper
    llm = MockLLMClient(mock_response_key="paper_pass1")
    result = extract_paper("Some paper text.", llm, phases=("simple",))
    assert "_overall_confidence" in result
    assert isinstance(result["_overall_confidence"], float)
    assert 0.0 <= result["_overall_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# F15 — per-pass dict[int, LLMClient] support in extract_paper
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_extract_paper_returns_both_phase_fields():
    """extract_paper() with default phases returns fields from both phases."""
    from scripts.llm.extractor import extract_paper
    llm = MockLLMClient()
    result = extract_paper("Some paper text.", llm)
    assert "core_finding" in result        # simple phase
    assert "novelty_statement" in result   # complex phase
    assert "_overall_confidence" in result


@pytest.mark.unit
def test_extract_paper_simple_only_returns_simple_fields():
    """extract_paper(phases=('simple',)) returns simple fields and skips complex."""
    from scripts.llm.extractor import extract_paper
    llm = MockLLMClient()
    result = extract_paper("Some paper text.", llm, phases=("simple",))
    assert "core_finding" in result
    # No complex-phase error — complex simply wasn't run
    assert "_complex_error" not in result


# ---------------------------------------------------------------------------
# E3 — extract_fields() — targeted field extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_fields_returns_only_requested_fields():
    """extract_fields() result contains only the requested fields + their confidence."""
    from scripts.llm.extractor import extract_fields
    # Mock returns paper_pass1 dict which includes core_finding etc.
    # The result should only contain the single requested field + confidence.
    llm = MockLLMClient(mock_response_key="paper_pass1")
    result = extract_fields("Paper text.", ["core_finding"], llm)
    assert "core_finding" in result
    assert "core_finding_confidence" in result
    # Fields from other passes must not appear
    assert "background_motivation" not in result
    assert "novelty_statement" not in result


@pytest.mark.unit
def test_extract_fields_system_prompt_contains_only_requested_fields():
    """System prompt sent to LLM lists only the requested fields — no extras."""
    from scripts.llm.extractor import extract_fields

    seen_prompts: list[str] = []

    class CapturingMock(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
            seen_prompts.append(system)
            return super().complete(system, user, max_tokens)

    llm = CapturingMock(mock_response_key="paper_pass1")
    extract_fields("Some paper text.", ["actionable_insights"], llm)

    assert seen_prompts, "LLM was never called"
    system_prompt = seen_prompts[0]
    # The field list uses "  - fieldname" format — check targeted field is listed.
    assert "  - actionable_insights" in system_prompt
    # Other pass fields must NOT appear in the field-list section.
    # (Note: core_finding appears in the hardcoded prompt examples, so check the
    # list-item format specifically rather than raw substring presence.)
    assert "  - background_motivation" not in system_prompt
    assert "  - core_finding" not in system_prompt


@pytest.mark.unit
def test_extract_fields_raises_on_unknown_field():
    """ValueError raised immediately for any field not in ALL_PAPER_FIELDS."""
    from scripts.llm.extractor import extract_fields
    llm = MockLLMClient()
    with pytest.raises(ValueError, match="Unknown field"):
        extract_fields("Paper text.", ["nonexistent_field"], llm)


@pytest.mark.unit
def test_extract_fields_raises_on_empty_fields():
    """ValueError raised when fields list is empty."""
    from scripts.llm.extractor import extract_fields
    llm = MockLLMClient()
    with pytest.raises(ValueError, match="non-empty"):
        extract_fields("Paper text.", [], llm)


@pytest.mark.unit
def test_extract_fields_strips_references_before_llm():
    """References section is stripped before the LLM sees the text."""
    from scripts.llm.extractor import extract_fields
    body = "Main text.\n" * 20
    refs = "\n## References\n\nSmith 2020. J. Mem.\nDoe 2019. Nature.\n"
    full_text = body + refs

    seen_user_prompts: list[str] = []

    class CapturingMock(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
            seen_user_prompts.append(user)
            return super().complete(system, user, max_tokens)

    llm = CapturingMock(mock_response_key="paper_pass1")
    extract_fields(full_text, ["core_finding"], llm)

    assert seen_user_prompts, "LLM was never called"
    user_prompt = seen_user_prompts[0]
    assert "Smith 2020" not in user_prompt, "Reference entry should be stripped"
    assert "Main text." in user_prompt, "Body text must still be present"


@pytest.mark.unit
def test_extract_fields_chunks_long_text():
    """Long text is split into multiple LLM calls and results merged."""
    from scripts.llm.extractor import extract_fields

    call_count = [0]

    class CountingMock(MockLLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
            call_count[0] += 1
            return super().complete(system, user, max_tokens)

    # Build text that exceeds a tiny context window (4000 char minimum chunk).
    # Patch num_ctx to force chunking at a very small budget.
    long_text = "Word. " * 5_000  # ~30 000 chars — well above any realistic chunk
    llm = CountingMock(mock_response_key="paper_pass1")
    llm.num_ctx = 512  # artificially tiny → forces at least 2 chunks

    result = extract_fields(long_text, ["core_finding"], llm)
    assert call_count[0] >= 2, "Expected multiple LLM calls for a long document"
    assert "core_finding" in result


@pytest.mark.unit
def test_extract_fields_citation_fields_now_valid():
    """M3: key_citations and comparison_to_prior are in ALL_PAPER_FIELDS (complex phase)."""
    from scripts.llm.extractor import ALL_PAPER_FIELDS, extract_fields
    assert "key_citations" in ALL_PAPER_FIELDS
    assert "comparison_to_prior" in ALL_PAPER_FIELDS
    # Should NOT raise — these are now valid schema fields
    llm = MockLLMClient()
    result = extract_fields("text", ["key_citations"], llm)
    assert "key_citations" in result


@pytest.mark.unit
def test_extract_fields_sets_overall_confidence():
    """N8c: extract_fields() must set _overall_confidence (was missing, causing 0.0 in notes)."""
    from scripts.llm.extractor import extract_fields
    llm = MockLLMClient(mock_response_key="paper_pass1")
    result = extract_fields("Paper text.", ["core_finding"], llm)
    assert "_overall_confidence" in result, "_overall_confidence absent from extract_fields result"
    assert isinstance(result["_overall_confidence"], float)
    assert 0.0 <= result["_overall_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# E3 — re_extract() routing: targeted vs full-pass
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_re_extract_uses_extract_fields_when_fields_given_without_passes():
    """re_extract() calls extract_fields() — not extract_paper() — when fields
    is given without an explicit passes argument."""
    from unittest.mock import patch

    from scripts.obsidian_tools.re_extract import re_extract

    state_db = MagicMock()
    state_db.get_paper.return_value = {
        "doi": "10.1/test",
        "source_type": "paper",
        "extraction_json": "{}",
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }

    llm = MockLLMClient(mock_response_key="paper_pass1")
    config = MagicMock()

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value="Body text."):
        with patch("scripts.obsidian_tools.re_extract.extract_fields") as mock_ef:
            mock_ef.return_value = {
                "actionable_insights": "Insight text.",
                "actionable_insights_confidence": "explicit",
            }
            re_extract(
                "10.1/test", config, state_db, llm,
                fields=["actionable_insights"],
                rerender=False,
            )

    mock_ef.assert_called_once()
    # The second positional arg to extract_fields is the fields list.
    assert mock_ef.call_args.args[1] == ["actionable_insights"]


@pytest.mark.unit
def test_re_extract_uses_extract_paper_when_phase_and_field_both_given():
    """When both --phase and --field are specified, extract_paper() is called
    (not extract_fields()), and the result is filtered to the requested fields."""
    from unittest.mock import patch

    from scripts.obsidian_tools.re_extract import re_extract

    state_db = MagicMock()
    state_db.get_paper.return_value = {
        "doi": "10.1/test",
        "source_type": "paper",
        "extraction_json": "{}",
        "note_path": "",
        "ocr_heavy": 0,
        "zotero_key": "",
    }

    llm = MockLLMClient(mock_response_key="paper_pass1")
    config = MagicMock()

    with patch("scripts.obsidian_tools.re_extract._load_fulltext", return_value="Body text."):
        with patch("scripts.obsidian_tools.re_extract.extract_fields") as mock_ef:
            with patch("scripts.obsidian_tools.re_extract.extract_paper") as mock_ep:
                mock_ep.return_value = {
                    "core_finding": "Finding.",
                    "core_finding_confidence": "explicit",
                    "_overall_confidence": 1.0,
                }
                re_extract(
                    "10.1/test", config, state_db, llm,
                    phases=["simple"],
                    fields=["core_finding"],
                    rerender=False,
                )

    # extract_paper (phase-based) must be called; extract_fields must NOT be called.
    mock_ep.assert_called_once()
    mock_ef.assert_not_called()


# ---------------------------------------------------------------------------
# I3 — PassResult structure and _max_n_chunks propagation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_simple_returns_passresult():
    """extract_simple() returns a PassResult, not a plain dict."""
    from scripts.llm.extractor import PassResult, extract_simple
    llm = MockLLMClient(mock_response_key="paper_pass1")
    pr = extract_simple("some text", llm)
    assert isinstance(pr, PassResult)
    assert hasattr(pr, "fields")
    assert hasattr(pr, "n_chunks")


@pytest.mark.unit
def test_extract_simple_n_chunks_1_for_short_text():
    """Short text that fits in one call → n_chunks == 1."""
    from scripts.llm.extractor import extract_simple
    llm = MockLLMClient(mock_response_key="paper_pass1")
    pr = extract_simple("Short paper text.", llm)
    assert pr.n_chunks == 1


@pytest.mark.unit
def test_extract_simple_n_chunks_gt_1_for_long_text():
    """Chunked text → n_chunks reflects the actual chunk count."""
    from scripts.llm.extractor import _available_input_chars, extract_simple

    class TinyCtxMock(MockLLMClient):
        num_ctx = 512  # forces chunking

    llm = TinyCtxMock(mock_response_key="paper_pass1")
    budget = _available_input_chars(llm)
    text = ("Scientific content. " * 60 + "\n") * 5
    assert len(text) > budget

    pr = extract_simple(text, llm)
    assert pr.n_chunks > 1
    assert pr.n_chunks == llm.call_count


@pytest.mark.unit
def test_extract_paper_includes_max_n_chunks():
    """extract_paper() propagates _max_n_chunks from all phase results."""
    from scripts.llm.extractor import extract_paper
    llm = MockLLMClient()
    result = extract_paper("some text", llm)
    assert "_max_n_chunks" in result
    assert isinstance(result["_max_n_chunks"], int)
    assert result["_max_n_chunks"] >= 1


@pytest.mark.unit
def test_extract_paper_max_n_chunks_reflects_worst_phase():
    """_max_n_chunks reflects the chunk count from the simple phase."""
    from scripts.llm.extractor import _available_input_chars, extract_paper

    class TinyCtxMock(MockLLMClient):
        num_ctx = 512

    llm = TinyCtxMock()
    budget = _available_input_chars(llm)
    text = ("Long scientific content. " * 60 + "\n") * 5
    assert len(text) > budget

    result = extract_paper(text, llm, phases=("simple",))
    assert result["_max_n_chunks"] > 1


# ---------------------------------------------------------------------------
# V-7 — extract_fields() schema-aware field validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_fields_paper_unchanged():
    """
    extract_fields() with all ALL_PAPER_FIELDS and content_type='paper' must
    return a dict containing every field — regression guard against V-7 changes
    accidentally removing paper fields from the valid set.
    """
    from scripts.llm.extractor import ALL_PAPER_FIELDS, extract_fields

    llm = MockLLMClient()
    result = extract_fields("Full paper text.", ALL_PAPER_FIELDS, llm, content_type="paper")
    for field in ALL_PAPER_FIELDS:
        assert field in result, f"Paper field {field!r} missing from extract_fields output"


@pytest.mark.unit
def test_extract_fields_review_schema():
    """M3: review schema has 23 fields (paper minus study_type, plus comparison_to_prior/
    key_citations/discovered_topics in the complex phase).  All must be extractable."""
    from scripts.llm.extraction_schema import all_fields_for_schema

    review_fields = all_fields_for_schema("review")
    assert len(review_fields) == 23, (
        f"Expected 23 review fields after M3, got {len(review_fields)}: {review_fields}"
    )
    assert "study_type" not in review_fields
    # M3 complex fields present in review schema
    assert "key_citations" in review_fields
    assert "comparison_to_prior" in review_fields
    assert "discovered_topics" in review_fields

    llm = MockLLMClient()
    from scripts.llm.extractor import extract_fields
    result = extract_fields("Review text.", review_fields, llm, content_type="review")
    for field in review_fields:
        assert field in result


@pytest.mark.unit
def test_extract_fields_unknown_schema_field_raises():
    """
    extract_fields() with a field not in the schema for content_type must raise
    ValueError — guards against typos or schema drift.
    """
    from scripts.llm.extractor import extract_fields

    llm = MockLLMClient()
    with pytest.raises(ValueError, match="Unknown field"):
        extract_fields("text.", ["nonexistent_field_xyz"], llm, content_type="paper")
