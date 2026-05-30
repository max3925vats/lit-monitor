"""
N7 (Phase 2): Unit tests for obsidian re-extract --rag-mode graph.

Covers:
  - vector mode is unchanged (no graph_context in prompt, extract_paper called)
  - graph mode injects a context block (LLM sees "Top N most-related papers")
  - graph mode falls back to vector when graph DB is unavailable (WARNING logged)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> SimpleNamespace:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Literature" / "Papers").mkdir(parents=True)
    return SimpleNamespace(
        obsidian=SimpleNamespace(
            vault_path=str(vault),
            papers_folder="Literature/Papers",
        ),
    )


def _make_state_db(tmp_path: Path):
    from scripts.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")


def _seed_paper(state_db, doi: str, tmp_path: Path) -> None:
    """Insert a minimal paper record so re_extract can find it."""
    note_path = str(
        tmp_path / "vault" / "Literature" / "Papers" / "Smith2021_Test.md"
    )
    Path(note_path).write_text("# Dummy note\nWe studied filtration.", encoding="utf-8")
    state_db.upsert_paper({
        "doi": doi,
        "title": "A Study on Filtration",
        "abstract": "We studied filtration with Protein A resin.",
        "source_type": "paper",
        "extraction_json": json.dumps({"core_finding": "old finding"}),
        "note_path": note_path,
        "status": "extraction_complete",
    })


# Minimal extraction dict returned by mock extract_paper — needs the required
# fields so compute_confidence_score doesn't blow up.
_MOCK_EXTRACTION = {
    "core_finding": "Updated finding.",
    "core_finding_confidence": "explicit",
}


# ---------------------------------------------------------------------------
# TestVectorModeUnchanged
# ---------------------------------------------------------------------------

class TestVectorModeUnchanged:
    """N7 AC4: --rag-mode vector must be identical to the pre-N7 path."""

    @pytest.mark.unit
    def test_vector_mode_calls_extract_paper(self, tmp_path):
        """vector mode routes through extract_paper, not the graph path."""
        from scripts.obsidian_tools.re_extract import re_extract

        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)
        doi = "10.1/vec-test"
        _seed_paper(state_db, doi, tmp_path)

        with (
            patch(
                "scripts.obsidian_tools.re_extract.extract_paper",
                return_value=_MOCK_EXTRACTION,
            ) as mock_extract,
            patch("scripts.obsidian_tools.re_extract.rerender_note"),
        ):
            result = re_extract(
                doi=doi,
                config=config,
                state_db=state_db,
                llm=MagicMock(),
                phases=["simple"],
                rag_mode="vector",
            )

        # extract_paper must have been called (standard path)
        mock_extract.assert_called_once()
        assert result["core_finding"] == "Updated finding."

    @pytest.mark.unit
    def test_vector_mode_does_not_open_graph_db(self, tmp_path):
        """vector mode must NOT touch GraphDB at all."""
        from scripts.obsidian_tools.re_extract import re_extract

        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)
        doi = "10.1/vec-nograph"
        _seed_paper(state_db, doi, tmp_path)

        with (
            patch(
                "scripts.obsidian_tools.re_extract.extract_paper",
                return_value=_MOCK_EXTRACTION,
            ),
            patch("scripts.obsidian_tools.re_extract.rerender_note"),
            patch(
                "scripts.obsidian_tools.re_extract._open_graph_db_safe"
            ) as mock_open,
        ):
            re_extract(
                doi=doi,
                config=config,
                state_db=state_db,
                llm=MagicMock(),
                phases=["simple"],
                rag_mode="vector",
            )

        # _open_graph_db_safe must NOT be called in vector mode
        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# TestGraphModeInjectsContext
# ---------------------------------------------------------------------------

class TestGraphModeInjectsContext:
    """N7 AC2+AC3: graph mode injects related papers + entities into the prompt."""

    def _make_mock_graph_db(self) -> MagicMock:
        """Build a realistic graph_db mock with canned find_similar_papers output."""
        graph_db = MagicMock()

        # find_similar_papers returns [(doi, score), ...]
        graph_db.find_similar_papers.return_value = [
            ("10.1/related-a", 5.0),
            ("10.1/related-b", 3.0),
        ]

        # _conn.execute is called for title lookups and MENTIONS query.
        # We need it to behave differently per call; use side_effect.
        title_result_a = MagicMock()
        title_result_a.has_next.side_effect = [True, False]
        title_result_a.get_next.return_value = ["Related Paper A"]

        title_result_b = MagicMock()
        title_result_b.has_next.side_effect = [True, False]
        title_result_b.get_next.return_value = ["Related Paper B"]

        # MENTIONS query result — two entities
        mentions_result = MagicMock()
        mentions_result.has_next.side_effect = [True, True, False]
        mentions_result.get_next.side_effect = [
            ["Protein A resin", "material"],
            ["cation exchange", "method"],
        ]

        graph_db._conn.execute.side_effect = [
            title_result_a,
            title_result_b,
            mentions_result,
        ]
        return graph_db

    @pytest.mark.unit
    def test_graph_mode_renders_context_in_llm_prompt(self, tmp_path):
        """graph mode calls llm.complete with a user message containing graph context."""
        from scripts.obsidian_tools.re_extract import re_extract

        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)
        doi = "10.1/graph-test"
        _seed_paper(state_db, doi, tmp_path)

        mock_graph_db = self._make_mock_graph_db()
        llm = MagicMock()
        llm.complete.return_value = json.dumps({"core_finding": "graph-enriched finding"})

        with (
            patch(
                "scripts.obsidian_tools.re_extract._open_graph_db_safe",
                return_value=mock_graph_db,
            ),
            patch("scripts.obsidian_tools.re_extract.rerender_note"),
        ):
            result = re_extract(
                doi=doi,
                config=config,
                state_db=state_db,
                llm=llm,
                rag_mode="graph",
            )

        # llm.complete must have been called exactly once
        llm.complete.assert_called_once()
        call_kwargs = llm.complete.call_args

        # Check the user message contains the graph context block
        user_msg: str = call_kwargs.kwargs.get("user") or call_kwargs.args[1]
        assert "most-related papers" in user_msg, (
            "Expected 'most-related papers' header in graph context"
        )
        assert "10.1/related-a" in user_msg
        assert "Related Paper A" in user_msg
        assert "Protein A resin" in user_msg

        # The result should reflect what the LLM returned
        assert result["core_finding"] == "graph-enriched finding"

    @pytest.mark.unit
    def test_graph_mode_persists_merged_extraction(self, tmp_path):
        """graph mode writes merged extraction back to state_db."""
        from scripts.obsidian_tools.re_extract import re_extract

        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)
        doi = "10.1/graph-persist"
        _seed_paper(state_db, doi, tmp_path)

        mock_graph_db = self._make_mock_graph_db()
        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            {"core_finding": "new finding", "core_finding_confidence": "explicit"}
        )

        with (
            patch(
                "scripts.obsidian_tools.re_extract._open_graph_db_safe",
                return_value=mock_graph_db,
            ),
            patch("scripts.obsidian_tools.re_extract.rerender_note"),
        ):
            re_extract(
                doi=doi,
                config=config,
                state_db=state_db,
                llm=llm,
                rag_mode="graph",
            )

        updated = state_db.get_extraction_json(doi)
        assert updated["core_finding"] == "new finding"


# ---------------------------------------------------------------------------
# TestGraphModeFallback
# ---------------------------------------------------------------------------

class TestGraphModeFallback:
    """N7 AC5: graph mode falls back to vector when GraphDB is unavailable."""

    @pytest.mark.unit
    def test_falls_back_when_graph_db_none(self, tmp_path, caplog):
        """Missing [graph] extra → fall back to vector mode with WARNING."""
        from scripts.obsidian_tools.re_extract import re_extract

        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)
        doi = "10.1/fallback-none"
        _seed_paper(state_db, doi, tmp_path)

        with (
            patch(
                "scripts.obsidian_tools.re_extract._open_graph_db_safe",
                return_value=None,
            ),
            patch(
                "scripts.obsidian_tools.re_extract.extract_paper",
                return_value=_MOCK_EXTRACTION,
            ) as mock_extract,
            patch("scripts.obsidian_tools.re_extract.rerender_note"),
            caplog.at_level(logging.WARNING, logger="scripts.obsidian_tools.re_extract"),
        ):
            result = re_extract(
                doi=doi,
                config=config,
                state_db=state_db,
                llm=MagicMock(),
                phases=["simple"],
                rag_mode="graph",
            )

        # WARNING must have been logged
        assert any(
            "falling back to vector" in record.message.lower()
            or "rag-mode graph" in record.message.lower()
            or "graph db unavailable" in record.message.lower()
            for record in caplog.records
        ), f"Expected fallback WARNING; got: {[r.message for r in caplog.records]}"

        # Vector path (extract_paper) must have been used instead
        mock_extract.assert_called_once()
        assert result["core_finding"] == "Updated finding."

    @pytest.mark.unit
    def test_open_graph_db_safe_returns_none_on_kuzu_error(self, tmp_path):
        """_open_graph_db_safe() returns None (never raises) when Kuzu errors.

        This is the unit test for _open_graph_db_safe's own error handling;
        re_extract treats None as a fallback signal (covered by the None test above).
        """
        from scripts.obsidian_tools.re_extract import _open_graph_db_safe

        with patch(
            "scripts.graph.import_citations.safe_graph_db",
            side_effect=RuntimeError("kuzu database locked"),
        ):
            result = _open_graph_db_safe()

        # Must return None, not raise
        assert result is None
