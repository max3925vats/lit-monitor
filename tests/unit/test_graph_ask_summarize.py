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

    def test_yaml_has_cluster_context_placeholder(self) -> None:
        """Bundle I: {cluster_context} must be present in the YAML user_template."""
        raw = Path("config/prompts/ask_summarize.example.yaml").read_text()
        assert "{cluster_context}" in raw

    def test_yaml_required_placeholders_include_cluster_context(self) -> None:
        raw = Path("config/prompts/ask_summarize.example.yaml").read_text()
        assert "cluster_context" in raw

    def test_prompt_loads_via_registry(self) -> None:
        """Round-trip through the registry — confirms required-placeholder
        validation passes and Pydantic model accepts the YAML."""
        from scripts.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("ask_summarize")
        assert prompt.system
        assert prompt.user_template
        # Verify render_user works with all four placeholders (Bundle I adds
        # cluster_context).
        rendered = prompt.render_user(
            question="test question",
            cypher="MATCH (p) RETURN p",
            rows="| x |\n|---|\n| 1 |",
            cluster_context="",
        )
        assert "test question" in rendered
        assert "MATCH (p) RETURN p" in rendered
        assert "| x |" in rendered


# ---------------------------------------------------------------------------
# Bundle I: cluster-context injection
# ---------------------------------------------------------------------------
class TestClusterContextInjection:
    """summarize_results injects cluster_context into the prompt when provided."""

    def test_cluster_context_appears_in_prompt(self) -> None:
        """When cluster_context is passed, the framing blurb reaches the LLM."""
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "Based on your Antibody Purification theme: four papers found."
        )
        result = summarize_results(
            "find purification papers",
            "MATCH (p) RETURN p",
            "| doi |\n|---|\n| 10.0/a |",
            client=mock_client,
            cluster_context={
                "display_name": "Antibody Purification",
                "n_papers": 42,
                "similarity": 0.78,
            },
        )
        # Verify the LLM was called and the prompt carried the cluster blurb.
        assert mock_client.complete.call_count == 1
        _, kwargs = mock_client.complete.call_args
        full_call = (kwargs.get("system", "") or "") + (kwargs.get("user", "") or "")
        assert "Antibody Purification" in full_call
        assert "42" in full_call
        assert result is not None

    def test_no_cluster_context_sends_empty_blurb(self) -> None:
        """cluster_context=None → {cluster_context} placeholder is empty string."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "answer"
        summarize_results(
            "q",
            "MATCH (p) RETURN p",
            "| x |\n|---|\n| 1 |",
            client=mock_client,
            cluster_context=None,
        )
        assert mock_client.complete.call_count == 1

    def test_cluster_context_does_not_affect_empty_result_short_circuit(
        self,
    ) -> None:
        """Empty rows → deterministic message even with cluster_context set."""
        mock_client = MagicMock()
        result = summarize_results(
            "find papers from 1900",
            "MATCH (p) RETURN p",
            "_(no results)_",
            client=mock_client,
            cluster_context={
                "display_name": "Downstream Processing",
                "n_papers": 7,
                "similarity": 0.55,
            },
        )
        # LLM must NOT be called for an empty result set, even with cluster ctx.
        assert mock_client.complete.call_count == 0
        assert result is not None
        assert "No matching results" in result

    def test_cluster_context_blurb_includes_theme_name(self) -> None:
        """The injected blurb references both the theme name and paper count."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "answer"
        unique_theme = "UNIQUE_THEME_XYZ_789"
        summarize_results(
            "q",
            "MATCH (p) RETURN p",
            "| x |\n|---|\n| 1 |",
            client=mock_client,
            cluster_context={
                "display_name": unique_theme,
                "n_papers": 99,
                "similarity": 0.61,
            },
        )
        _, kwargs = mock_client.complete.call_args
        full_call = (kwargs.get("system", "") or "") + (kwargs.get("user", "") or "")
        assert unique_theme in full_call
        assert "99" in full_call


# ---------------------------------------------------------------------------
# Bundle I: _identify_relevant_cluster unit tests
# ---------------------------------------------------------------------------
class TestIdentifyRelevantCluster:
    """_identify_relevant_cluster behaves correctly across edge cases."""

    def _make_centroid_blob(self, vec: list[float]) -> bytes:
        """Serialise a float list to the BLOB format used by Bundle C."""
        import numpy as np

        return np.array(vec, dtype=np.float32).tobytes()

    def test_returns_none_on_empty_cluster_list(self) -> None:
        from scripts.graph.ask import _identify_relevant_cluster

        mock_state_db = MagicMock()
        mock_state_db.list_active_clusters.return_value = []
        mock_emb_db = MagicMock()

        result = _identify_relevant_cluster("question", mock_state_db, mock_emb_db)
        assert result is None
        # embed_text must NOT be called — no clusters to compare against.
        mock_emb_db.embed_text.assert_not_called()

    def test_returns_none_when_similarity_below_threshold(self) -> None:
        import numpy as np

        from scripts.graph.ask import _identify_relevant_cluster

        mock_state_db = MagicMock()
        # Centroid and question orthogonal → cosine similarity ≈ 0.
        centroid = [1.0, 0.0, 0.0, 0.0]
        mock_state_db.list_active_clusters.return_value = [
            {
                "id": 1,
                "display_name": "Theme A",
                "n_papers": 10,
                "centroid_blob": self._make_centroid_blob(centroid),
            }
        ]
        mock_emb_db = MagicMock()
        mock_emb_db.embed_text.return_value = np.array(
            [0.0, 1.0, 0.0, 0.0], dtype=np.float32
        )

        result = _identify_relevant_cluster(
            "question", mock_state_db, mock_emb_db, threshold=0.3
        )
        assert result is None

    def test_returns_cluster_dict_when_similarity_above_threshold(self) -> None:
        import numpy as np

        from scripts.graph.ask import _identify_relevant_cluster

        mock_state_db = MagicMock()
        centroid = [1.0, 0.0, 0.0, 0.0]
        mock_state_db.list_active_clusters.return_value = [
            {
                "id": 1,
                "display_name": "Protein A Capture",
                "n_papers": 25,
                "centroid_blob": self._make_centroid_blob(centroid),
            }
        ]
        mock_emb_db = MagicMock()
        # Parallel to centroid → cosine similarity = 1.0
        mock_emb_db.embed_text.return_value = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        result = _identify_relevant_cluster(
            "protein A chromatography methods",
            mock_state_db,
            mock_emb_db,
            threshold=0.3,
        )
        assert result is not None
        assert result["display_name"] == "Protein A Capture"
        assert result["n_papers"] == 25
        assert result["similarity"] == 1.0

    def test_picks_highest_similarity_cluster(self) -> None:
        import numpy as np

        from scripts.graph.ask import _identify_relevant_cluster

        mock_state_db = MagicMock()
        mock_state_db.list_active_clusters.return_value = [
            {
                "id": 1,
                "display_name": "Low Match",
                "n_papers": 5,
                "centroid_blob": self._make_centroid_blob([0.0, 1.0, 0.0, 0.0]),
            },
            {
                "id": 2,
                "display_name": "High Match",
                "n_papers": 15,
                "centroid_blob": self._make_centroid_blob([1.0, 0.0, 0.0, 0.0]),
            },
        ]
        mock_emb_db = MagicMock()
        mock_emb_db.embed_text.return_value = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        result = _identify_relevant_cluster(
            "q", mock_state_db, mock_emb_db, threshold=0.3
        )
        assert result is not None
        assert result["display_name"] == "High Match"

    def test_graceful_degradation_on_state_db_exception(self) -> None:
        """list_active_clusters raises → returns None, does not propagate."""
        from scripts.graph.ask import _identify_relevant_cluster

        mock_state_db = MagicMock()
        mock_state_db.list_active_clusters.side_effect = RuntimeError("DB error")
        mock_emb_db = MagicMock()

        result = _identify_relevant_cluster("q", mock_state_db, mock_emb_db)
        assert result is None

    def test_graceful_degradation_on_embed_exception(self) -> None:
        """embed_text raises → returns None, does not propagate."""
        from scripts.graph.ask import _identify_relevant_cluster

        mock_state_db = MagicMock()
        mock_state_db.list_active_clusters.return_value = [
            {
                "id": 1,
                "display_name": "Theme",
                "n_papers": 10,
                "centroid_blob": self._make_centroid_blob([1.0, 0.0]),
            }
        ]
        mock_emb_db = MagicMock()
        mock_emb_db.embed_text.side_effect = ValueError("empty text")

        result = _identify_relevant_cluster("q", mock_state_db, mock_emb_db)
        assert result is None

    def test_skips_cluster_with_none_centroid_blob(self) -> None:
        """Clusters with None centroid_blob are skipped without error."""
        import numpy as np

        from scripts.graph.ask import _identify_relevant_cluster

        mock_state_db = MagicMock()
        mock_state_db.list_active_clusters.return_value = [
            {
                "id": 1,
                "display_name": "No Centroid",
                "n_papers": 5,
                "centroid_blob": None,  # not yet computed
            }
        ]
        mock_emb_db = MagicMock()
        mock_emb_db.embed_text.return_value = np.array(
            [1.0, 0.0], dtype=np.float32
        )

        result = _identify_relevant_cluster("q", mock_state_db, mock_emb_db)
        assert result is None
