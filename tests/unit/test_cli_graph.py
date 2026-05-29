"""G10: CLI tests for graph backfill + rebuild."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from scripts.cli import main


class TestGraphBackfillCLI:
    def test_help_shows_subcommand(self):
        """G10: graph --help lists backfill and rebuild subcommands."""
        runner = CliRunner()
        result = runner.invoke(main, ["graph", "--help"])
        assert result.exit_code == 0
        assert "backfill" in result.output
        assert "rebuild" in result.output

    def test_backfill_requires_a_flag(self):
        """G10: missing all/doi/since -> UsageError."""
        runner = CliRunner()
        result = runner.invoke(main, ["graph", "backfill"])
        assert result.exit_code != 0
        assert "Must specify" in result.output

    def test_backfill_without_graph_extra_errors(self):
        """G10: safe_graph_db() returning None gives a helpful error."""
        runner = CliRunner()
        with patch("scripts.graph.safe_graph_db", return_value=None):
            result = runner.invoke(main, ["graph", "backfill", "--all"])
        assert result.exit_code != 0
        assert "uv sync --extra graph" in result.output

    def test_backfill_all_succeeds_with_mock_graph(self, tmp_path):
        """G10: backfill --all calls backfill_papers and echoes count."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_papers", return_value=3),
            # get_config is imported inside the command function body so patch the source.
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--all"])
        assert result.exit_code == 0, result.output
        assert "3" in result.output

    def test_rebuild_requires_a_flag(self):
        """G10: rebuild with no flag -> UsageError."""
        runner = CliRunner()
        # Pass --yes to skip the confirmation prompt
        result = runner.invoke(main, ["graph", "rebuild", "--yes"])
        assert result.exit_code != 0
        assert "Must specify" in result.output

    def test_rebuild_all_and_aliases_only_are_mutually_exclusive(self):
        """G10: --all and --aliases-only together -> UsageError."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "graph", "rebuild", "--all", "--aliases-only", "--yes"
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_rebuild_without_graph_extra_errors(self):
        """G10: rebuild without graph extra gives helpful error."""
        runner = CliRunner()
        with patch("scripts.graph.safe_graph_db", return_value=None):
            result = runner.invoke(main, ["graph", "rebuild", "--all", "--yes"])
        assert result.exit_code != 0
        assert "uv sync --extra graph" in result.output
