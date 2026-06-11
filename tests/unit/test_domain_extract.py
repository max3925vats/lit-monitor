"""Bundle G (v0.9): tests for the domain focus LLM extractor + state_db helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from lit_monitor.core.path_utils import resolve_path


# ---------------------------------------------------------------------------
# Prompt YAML round-trip + registry registration
# ---------------------------------------------------------------------------
class TestPromptYaml:
    def test_yaml_file_present_and_loads(self) -> None:
        """The example prompt ships with the package and parses as YAML."""
        path = resolve_path(Path("config/prompts/domain_extraction.example.yaml"))
        assert path.exists(), "domain_extraction.example.yaml should ship with repo"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "system" in data
        assert "user_template" in data
        assert "{domain_context}" in data["user_template"]

    def test_registered_in_prompt_registry(self) -> None:
        """The prompt is registered with the required placeholder set."""
        from lit_monitor.llm.prompt_registry import _REQUIRED_PLACEHOLDERS

        assert "domain_extraction" in _REQUIRED_PLACEHOLDERS
        assert "domain_context" in _REQUIRED_PLACEHOLDERS["domain_extraction"]

    def test_load_prompt_renders(self) -> None:
        """load_prompt resolves the example file and validates the schema."""
        from lit_monitor.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("domain_extraction")
        assert prompt.system
        assert "{domain_context}" in prompt.user_template
        rendered = prompt.render_user(domain_context="I work on X.")
        assert "I work on X." in rendered


# ---------------------------------------------------------------------------
# analyze_domain — defensive perimeter
# ---------------------------------------------------------------------------
class TestAnalyzeDomain:
    def test_happy_path_parses_json(self) -> None:
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "topics": ["bioprocessing"],
                "methods": ["chromatography"],
                "materials": ["antibodies"],
                "adjacent_fields": [],
                "exclusions": [],
            }
        )
        result = analyze_domain("I work on bioprocessing.", llm=mock_llm)
        assert result is not None
        assert result["topics"] == ["bioprocessing"]
        assert result["methods"] == ["chromatography"]
        assert result["materials"] == ["antibodies"]
        assert result["adjacent_fields"] == []
        assert result["exclusions"] == []

    def test_empty_input_returns_none(self) -> None:
        from lit_monitor.domain.extract import analyze_domain

        assert analyze_domain("", llm=MagicMock()) is None
        assert analyze_domain("   \n  ", llm=MagicMock()) is None

    def test_none_input_returns_none(self) -> None:
        from lit_monitor.domain.extract import analyze_domain

        # type: ignore[arg-type] — defensive: callers shouldn't but we tolerate.
        assert analyze_domain(None, llm=MagicMock()) is None  # type: ignore[arg-type]

    def test_malformed_json_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "not actually JSON {broken"
        with caplog.at_level(logging.INFO, logger="lit_monitor.domain.extract"):
            result = analyze_domain("real input", llm=mock_llm)
        assert result is None
        assert any("JSON parse failed" in rec.message for rec in caplog.records)

    def test_llm_exception_returns_none(self) -> None:
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("network")
        result = analyze_domain("input", llm=mock_llm)
        assert result is None

    def test_empty_llm_response_returns_none(self) -> None:
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "   "
        assert analyze_domain("input", llm=mock_llm) is None

    def test_missing_required_keys_returns_none(self) -> None:
        """JSON missing required schema keys → fail soft, return None."""
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({"topics": ["x"]})
        assert analyze_domain("input", llm=mock_llm) is None

    def test_non_object_json_returns_none(self) -> None:
        """JSON array (not object) → None."""
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "[1, 2, 3]"
        assert analyze_domain("input", llm=mock_llm) is None

    def test_fenced_output_stripped(self) -> None:
        """LLM may wrap output in ```json ...``` fences — should still parse."""
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '```json\n'
            '{"topics": ["x"], "methods": [], "materials": [], '
            '"adjacent_fields": [], "exclusions": []}\n'
            '```'
        )
        result = analyze_domain("input", llm=mock_llm)
        assert result is not None
        assert result["topics"] == ["x"]

    def test_thinking_block_stripped(self) -> None:
        """Ollama thinking-mode <think>...</think> block is stripped."""
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            "<think>let me think about this</think>"
            '{"topics": ["x"], "methods": [], "materials": [], '
            '"adjacent_fields": [], "exclusions": []}'
        )
        result = analyze_domain("input", llm=mock_llm)
        assert result is not None
        assert result["topics"] == ["x"]

    def test_caps_items_at_eight(self) -> None:
        """Defense-in-depth: cap each list at 8 even if the LLM ignores the prompt."""
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        twelve = [f"t{i}" for i in range(12)]
        mock_llm.complete.return_value = json.dumps(
            {
                "topics": twelve,
                "methods": [],
                "materials": [],
                "adjacent_fields": [],
                "exclusions": [],
            }
        )
        result = analyze_domain("input", llm=mock_llm)
        assert result is not None
        assert len(result["topics"]) == 8

    def test_dedup_case_insensitive(self) -> None:
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "topics": ["Bioprocessing", "bioprocessing", "BIOPROCESSING", "other"],
                "methods": [],
                "materials": [],
                "adjacent_fields": [],
                "exclusions": [],
            }
        )
        result = analyze_domain("input", llm=mock_llm)
        assert result is not None
        # First-seen casing preserved; duplicates dropped
        assert result["topics"] == ["Bioprocessing", "other"]

    def test_non_list_field_coerced_to_empty(self) -> None:
        from lit_monitor.domain.extract import analyze_domain

        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "topics": "not a list",  # malformed
                "methods": [],
                "materials": [],
                "adjacent_fields": [],
                "exclusions": [],
            }
        )
        result = analyze_domain("input", llm=mock_llm)
        assert result is not None
        assert result["topics"] == []


# ---------------------------------------------------------------------------
# StateDB: domain_focus_extracted table + helpers
# ---------------------------------------------------------------------------
class TestDomainFocusTable:
    def test_table_exists(self, tmp_path: Path) -> None:
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='domain_focus_extracted'"
            ).fetchall()
        assert rows

    def test_save_extraction_counts(self, tmp_path: Path) -> None:
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        db.save_domain_extraction(
            {
                "topics": ["A", "B"],
                "methods": ["C"],
                "materials": [],
                "adjacent_fields": [],
                "exclusions": ["D"],
            }
        )
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT field_type, value FROM domain_focus_extracted "
                "ORDER BY field_type, value"
            ).fetchall()
        assert len(rows) == 4
        types = {r[0] for r in rows}
        assert types == {"topic", "method", "exclusion"}

    def test_save_replaces_previous(self, tmp_path: Path) -> None:
        """save_domain_extraction has REPLACE semantics — wipes old rows."""
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        empty: dict[str, list[str]] = {
            "topics": [], "methods": [], "materials": [],
            "adjacent_fields": [], "exclusions": [],
        }
        db.save_domain_extraction({**empty, "topics": ["A"]})
        db.save_domain_extraction({**empty, "topics": ["B"]})
        with db._connect() as conn:
            rows = conn.execute("SELECT value FROM domain_focus_extracted").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "B"

    def test_list_extracted_grouped_by_plural(self, tmp_path: Path) -> None:
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        db.save_domain_extraction(
            {
                "topics": ["T1", "T2"],
                "methods": ["M1"],
                "materials": [],
                "adjacent_fields": [],
                "exclusions": [],
            }
        )
        result = db.list_domain_extraction()
        # Stable shape — all five plural keys always present
        assert set(result.keys()) == {
            "topics", "methods", "materials", "adjacent_fields", "exclusions",
        }
        assert len(result["topics"]) == 2
        assert len(result["methods"]) == 1
        assert result["materials"] == []

    def test_list_row_shape(self, tmp_path: Path) -> None:
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        db.save_domain_extraction(
            {
                "topics": ["X"],
                "methods": [], "materials": [],
                "adjacent_fields": [], "exclusions": [],
            }
        )
        result = db.list_domain_extraction()
        row = result["topics"][0]
        assert set(row.keys()) >= {"id", "value", "user_confirmed", "last_analyzed_at"}
        assert row["value"] == "X"
        assert row["user_confirmed"] is False

    def test_clear_returns_rowcount(self, tmp_path: Path) -> None:
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        db.save_domain_extraction(
            {
                "topics": ["a", "b", "c"],
                "methods": [], "materials": [],
                "adjacent_fields": [], "exclusions": [],
            }
        )
        n = db.clear_domain_extraction()
        assert n == 3
        # Idempotent on empty
        assert db.clear_domain_extraction() == 0

    def test_set_confirmed_toggle(self, tmp_path: Path) -> None:
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        db.save_domain_extraction(
            {
                "topics": ["X"],
                "methods": [], "materials": [],
                "adjacent_fields": [], "exclusions": [],
            }
        )
        row_id = db.list_domain_extraction()["topics"][0]["id"]
        db.set_domain_extraction_confirmed(row_id, True)
        assert db.list_domain_extraction()["topics"][0]["user_confirmed"] is True
        db.set_domain_extraction_confirmed(row_id, False)
        assert db.list_domain_extraction()["topics"][0]["user_confirmed"] is False

    def test_save_skips_empty_strings(self, tmp_path: Path) -> None:
        """Whitespace-only / empty items shouldn't pollute the table."""
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        db.save_domain_extraction(
            {
                "topics": ["A", "", "   ", "B"],
                "methods": [], "materials": [],
                "adjacent_fields": [], "exclusions": [],
            }
        )
        assert len(db.list_domain_extraction()["topics"]) == 2
