"""A5: lit-monitor ask CLI command tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scripts.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_graph_db() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Helpers — shared patch targets
# ---------------------------------------------------------------------------
_SAFE_GRAPH_DB = "scripts.graph.safe_graph_db"
_GET_CONFIG = "scripts.core.config.get_config"
_DESCRIBE_SCHEMA = "scripts.graph.schema_describer.describe_schema"
_GENERATE_CYPHER = "scripts.graph.ask.generate_cypher"
_EXECUTE_CYPHER = "scripts.graph.ask.execute_cypher"
_SUMMARIZE = "scripts.graph.ask.summarize_results"


class TestHappyPath:
    def test_basic_ask_prints_prose_and_table(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: happy path emits prose summary followed by a markdown table."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="## Node: Paper\n- doi: STRING\n"),
            patch(_GENERATE_CYPHER, return_value="MATCH (p:Paper) RETURN p.doi LIMIT 100"),
            patch(_EXECUTE_CYPHER, return_value=[{"doi": "10.0/a"}]),
            patch(_SUMMARIZE, return_value="One paper is in the corpus."),
        ):
            result = runner.invoke(main, ["ask", "show", "me", "papers"])

        assert result.exit_code == 0, result.output
        assert "One paper is in the corpus." in result.output
        assert "10.0/a" in result.output  # rendered in table cell

    def test_multi_word_question_without_quotes(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: nargs=-1 allows multi-word questions without shell quotes."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value="MATCH (p) RETURN p") as gen_mock,
            patch(_EXECUTE_CYPHER, return_value=[{"doi": "x"}]),
            patch(_SUMMARIZE, return_value="summary"),
        ):
            result = runner.invoke(main, ["ask", "how", "many", "papers", "exist"])

        assert result.exit_code == 0, result.output
        # The question should be joined with spaces before reaching generate_cypher
        call_question = gen_mock.call_args[0][0]
        assert "how many papers exist" == call_question


class TestCypherOnly:
    def test_cypher_only_skips_execution_and_summary(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: --cypher-only prints the Cypher and returns without calling execute/summarize."""
        execute_called: dict[str, int] = {"n": 0}
        summarize_called: dict[str, int] = {"n": 0}

        def fake_execute(db: MagicMock, c: str, **kw: object) -> list[dict]:
            execute_called["n"] += 1
            return [{"doi": "x"}]

        def fake_summarize(q: str, c: str, r: str, **kw: object) -> str:
            summarize_called["n"] += 1
            return "summary"

        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value="MATCH (p) RETURN p"),
            patch(_EXECUTE_CYPHER, fake_execute),
            patch(_SUMMARIZE, fake_summarize),
        ):
            result = runner.invoke(main, ["ask", "--cypher-only", "show", "papers"])

        assert result.exit_code == 0, result.output
        assert "MATCH (p) RETURN p" in result.output
        assert execute_called["n"] == 0
        assert summarize_called["n"] == 0


class TestVerbose:
    def test_verbose_prints_stage_labels(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: --verbose emits stage labels for each pipeline step."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value="MATCH (p) RETURN p"),
            patch(_EXECUTE_CYPHER, return_value=[{"x": 1}]),
            patch(_SUMMARIZE, return_value="answer"),
        ):
            result = runner.invoke(main, ["ask", "--verbose", "show", "papers"])

        assert result.exit_code == 0, result.output
        out = result.output.lower()
        # Should show at least one stage label mentioning cypher generation
        assert "cypher" in out
        # Should indicate execution stage
        assert "execut" in out
        # Should indicate summarizing stage
        assert "summari" in out

    def test_verbose_prints_generated_cypher(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: --verbose echoes the Cypher string so users can inspect it."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value="MATCH (p:Paper) RETURN p.title"),
            patch(_EXECUTE_CYPHER, return_value=[{"title": "Foo"}]),
            patch(_SUMMARIZE, return_value="One paper found."),
        ):
            result = runner.invoke(main, ["ask", "--verbose", "list", "papers"])

        assert "MATCH (p:Paper) RETURN p.title" in result.output


class TestFailureHandling:
    def test_safe_graph_db_none_exits_1(self, runner: CliRunner) -> None:
        """A5: safe_graph_db None → helpful error about graph build + exit 1."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=None),
            patch(_GET_CONFIG, return_value=MagicMock()),
        ):
            result = runner.invoke(main, ["ask", "show", "papers"])

        assert result.exit_code != 0
        # Must mention graph backend + how to fix
        out = result.output.lower()
        assert "graph" in out
        assert "build" in out or "unavailable" in out

    def test_generate_cypher_none_exits_1(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: generate_cypher None → translation error + rephrase hint + exit 1."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value=None),
        ):
            result = runner.invoke(main, ["ask", "do", "something"])

        assert result.exit_code != 0
        out = result.output.lower()
        # Should hint at rephrasing
        assert "translate" in out or "rephras" in out

    def test_execute_cypher_none_exits_1(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: execute_cypher None → show the failed Cypher so user can debug + exit 1."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value="BAD CYPHER"),
            patch(_EXECUTE_CYPHER, return_value=None),
        ):
            result = runner.invoke(main, ["ask", "go"])

        assert result.exit_code != 0
        # Must show what Cypher was tried so user can debug
        assert "BAD CYPHER" in result.output

    def test_summarize_none_still_prints_table_exit_0(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: summarize_results None → show raw table + note; exit 0 (we have data)."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value="MATCH p RETURN p"),
            patch(_EXECUTE_CYPHER, return_value=[{"doi": "10.0/a"}]),
            patch(_SUMMARIZE, return_value=None),
        ):
            result = runner.invoke(main, ["ask", "go"])

        assert result.exit_code == 0, result.output
        # Table cell must be present
        assert "10.0/a" in result.output
        # Must mention that summarization failed
        out = result.output.lower()
        assert "summariz" in out and "fail" in out

    def test_describe_schema_empty_exits_1(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: empty schema string from describe_schema → error + exit 1."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value=""),
        ):
            result = runner.invoke(main, ["ask", "find", "papers"])

        assert result.exit_code != 0


class TestEmptyResults:
    def test_empty_results_renders_short_circuit_message(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: empty rows from execute_cypher → A4 short-circuits; exit 0."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(
                _GENERATE_CYPHER,
                return_value="MATCH p WHERE p.year = 1900 RETURN p",
            ),
            patch(_EXECUTE_CYPHER, return_value=[]),
            # Don't mock summarize_results — let it short-circuit naturally
        ):
            result = runner.invoke(
                main, ["ask", "find", "papers", "from", "1900"]
            )

        assert result.exit_code == 0, result.output
        assert "no matching results" in result.output.lower()


class TestMaxRows:
    def test_max_rows_passed_to_render_rows(
        self, runner: CliRunner, mock_graph_db: MagicMock
    ) -> None:
        """A5: --max-rows N is forwarded to render_rows as max_rows=N."""
        with (
            patch(_SAFE_GRAPH_DB, return_value=mock_graph_db),
            patch(_GET_CONFIG, return_value=MagicMock()),
            patch(_DESCRIBE_SCHEMA, return_value="schema"),
            patch(_GENERATE_CYPHER, return_value="MATCH (p) RETURN p"),
            patch(_EXECUTE_CYPHER, return_value=[{"doi": "x"}]),
            patch(_SUMMARIZE, return_value="summary"),
            patch("scripts.graph.ask.render_rows", return_value="| doi |\n|---|\n| x |") as render_mock,
        ):
            result = runner.invoke(main, ["ask", "--max-rows", "5", "show", "papers"])

        assert result.exit_code == 0, result.output
        # render_rows must have been called with max_rows=5
        render_mock.assert_called_once()
        _, kwargs = render_mock.call_args
        assert kwargs.get("max_rows") == 5
