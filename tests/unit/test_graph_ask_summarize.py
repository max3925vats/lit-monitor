"""A4 (Phase 4a): Cypher result → prose summarization tests.

Single LLM call. Defensive perimeter — returns None on failure, never raises.
Empty-result short-circuit returns a deterministic message WITHOUT calling
the LLM.

Test coverage:
  - happy path: returns prose from LLM.
  - single-call invariant: exactly one client.complete() per question.
  - empty-result short-circuit: '_(no results)_' input → deterministic
    message, zero LLM calls.
  - error handling: LLM exception / empty response → None, never raises.
  - prompt round-trip: YAML carries required placeholders, frames the LLM
    as a literature-corpus analyst (not a Cypher tutor), and forbids
    inventing facts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from scripts.graph.ask import summarize_results


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestSummarizeResultsHappyPath:
    def test_returns_prose_from_llm(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "Four papers from 2024 are in the corpus."
        )
        result = summarize_results(
            "show me papers",
            "MATCH (p) RETURN p",
            "| doi |\n|---|\n| 10.0/a |",
            client=mock_client,
        )
        assert result == "Four papers from 2024 are in the corpus."

    def test_single_llm_call(self) -> None:
        """A4 invariant: ONE Ollama call per summarization."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "answer"
        summarize_results(
            "q", "MATCH (p) RETURN p", "| x |\n|---|\n| 1 |", client=mock_client
        )
        assert mock_client.complete.call_count == 1

    def test_passes_question_into_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "answer"
        unique_q = "UNIQUE_QUESTION_MARKER_77"
        summarize_results(
            unique_q,
            "MATCH (p) RETURN p",
            "| x |\n|---|\n| 1 |",
            client=mock_client,
        )
        _, kwargs = mock_client.complete.call_args
        full_call = (kwargs.get("system", "") or "") + (kwargs.get("user", "") or "")
        assert unique_q in full_call

    def test_passes_rows_into_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "answer"
        unique_rows = "UNIQUE_ROWS_MARKER_123"
        summarize_results(
            "q",
            "MATCH (p) RETURN p",
            unique_rows,
            client=mock_client,
        )
        _, kwargs = mock_client.complete.call_args
        full_call = (kwargs.get("system", "") or "") + (kwargs.get("user", "") or "")
        assert unique_rows in full_call

    def test_strips_whitespace_from_response(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "  \n  Answer with padding.  \n  "
        result = summarize_results(
            "q", "MATCH (p) RETURN p", "| x |\n|---|\n| 1 |", client=mock_client
        )
        assert result == "Answer with padding."


# ---------------------------------------------------------------------------
# Empty-result short-circuit — MUST NOT call the LLM
# ---------------------------------------------------------------------------
class TestEmptyResultShortCircuit:
    def test_no_llm_call_when_rows_empty(self) -> None:
        """A4: '_(no results)_' input means we skip the LLM call entirely."""
        mock_client = MagicMock()
        result = summarize_results(
            "show me papers from 1900",
            "MATCH (p:Paper) WHERE p.year = 1900 RETURN p",
            "_(no results)_",
            client=mock_client,
        )
        assert mock_client.complete.call_count == 0
        assert result is not None
        assert "No matching results" in result

    def test_short_circuit_message_includes_question(self) -> None:
        result = summarize_results(
            "find rare unicorn methods",
            "MATCH (p)-[:MENTIONS]->(e) RETURN p",
            "_(no results)_",
            client=MagicMock(),
        )
        assert result is not None
        assert "find rare unicorn methods" in result

    def test_short_circuit_when_rows_have_whitespace(self) -> None:
        """Defensive: '_(no results)_' with surrounding whitespace still
        triggers the short-circuit. A3 returns the bare token but a future
        caller might pad it."""
        mock_client = MagicMock()
        result = summarize_results(
            "q",
            "MATCH (p) RETURN p",
            "   _(no results)_  \n",
            client=mock_client,
        )
        assert mock_client.complete.call_count == 0
        assert result is not None
        assert "No matching results" in result


# ---------------------------------------------------------------------------
# Error handling — defensive perimeter
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_llm_exception_returns_none(self, caplog) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("network")
        with caplog.at_level(logging.INFO):
            result = summarize_results(
                "q",
                "MATCH (p) RETURN p",
                "| x |\n|---|\n| 1 |",
                client=mock_client,
            )
        assert result is None

    def test_empty_response_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "   "
        result = summarize_results(
            "q", "MATCH (p) RETURN p", "| x |\n|---|\n| 1 |", client=mock_client
        )
        assert result is None

    def test_none_response_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = None
        result = summarize_results(
            "q", "MATCH (p) RETURN p", "| x |\n|---|\n| 1 |", client=mock_client
        )
        assert result is None

    def test_never_raises_on_llm_error(self) -> None:
        """A4 defensive perimeter: arbitrary LLM exceptions never propagate."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = ValueError("bad")
        # Must not raise.
        result = summarize_results(
            "q", "MATCH (p) RETURN p", "| x |\n|---|\n| 1 |", client=mock_client
        )
        assert result is None

    def test_empty_question_returns_none(self) -> None:
        mock_client = MagicMock()
        result = summarize_results(
            "", "MATCH (p) RETURN p", "| x |\n|---|\n| 1 |", client=mock_client
        )
        assert result is None
        # And the LLM is not called for an empty question.
        assert mock_client.complete.call_count == 0


# ---------------------------------------------------------------------------
# Prompt YAML round-trip
# ---------------------------------------------------------------------------
class TestPromptRoundTrip:
    def test_yaml_exists(self) -> None:
        assert Path("config/prompts/ask_summarize.example.yaml").exists()

    def test_yaml_has_required_placeholders(self) -> None:
        raw = Path("config/prompts/ask_summarize.example.yaml").read_text()
        for placeholder in ("{question}", "{cypher}", "{rows}"):
            assert placeholder in raw, f"missing placeholder {placeholder}"

    def test_yaml_frames_as_analyst_not_cypher_tutor(self) -> None:
        raw = Path("config/prompts/ask_summarize.example.yaml").read_text().lower()
        # Must frame the role as analyst / answering, NOT teaching Cypher.
        assert "analyst" in raw or "answer the question" in raw
        # Must explicitly forbid explaining the query.
        assert (
            "do not explain" in raw
            or "not explain the cypher" in raw
            or "not the mechanism" in raw
        )

    def test_yaml_forbids_invention(self) -> None:
        raw = Path("config/prompts/ask_summarize.example.yaml").read_text().lower()
        # Must forbid inventing facts not in the rows.
        assert (
            "never invent" in raw
            or "do not invent" in raw
            or "only facts" in raw
        )

    def test_prompt_loads_via_registry(self) -> None:
        """Round-trip through the registry — confirms required-placeholder
        validation passes and Pydantic model accepts the YAML."""
        from scripts.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("ask_summarize")
        assert prompt.system
        assert prompt.user_template
        # Verify render_user works with all three placeholders.
        rendered = prompt.render_user(
            question="test question",
            cypher="MATCH (p) RETURN p",
            rows="| x |\n|---|\n| 1 |",
        )
        assert "test question" in rendered
        assert "MATCH (p) RETURN p" in rendered
        assert "| x |" in rendered
