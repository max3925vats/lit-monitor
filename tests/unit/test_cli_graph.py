"""G10/G13: CLI tests for graph backfill + rebuild + status subcommand."""
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


class TestGraphStatusCLI:
    """G13: `lit-monitor graph status` command."""

    def test_status_exits_zero_when_graph_extra_missing(self):
        """G13: ImportError during [graph] import -> friendly message, exit 0."""
        runner = CliRunner()
        # Simulate kuzu not installed by making the scripts.graph module raise
        # ImportError when imported inside the command body.
        import builtins
        _real_import = builtins.__import__

        def _mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "scripts.graph":
                raise ImportError("[graph] not installed")
            return _real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            result = runner.invoke(main, ["graph", "status"])
        # Must not crash; informative message expected
        assert result.exit_code == 0

    def test_status_exits_zero_when_no_graph_db(self):
        """G13: safe_graph_db() returns None -> friendly message, exit 0."""
        runner = CliRunner()
        with patch("scripts.graph.safe_graph_db", return_value=None):
            result = runner.invoke(main, ["graph", "status"])
        assert result.exit_code == 0
        assert "backfill" in result.output.lower() or "graph" in result.output.lower()

    def test_status_prints_paper_count(self):
        """G13: when graph DB is available, output includes Paper and Entity rows."""
        runner = CliRunner()
        mock_conn = MagicMock()

        # Make res.has_next() return True once then False, giving count 42 for Paper
        # and count 7 for Entity; all rels return 0.
        def _execute_side_effect(sql: str) -> MagicMock:
            res = MagicMock()
            res.has_next.return_value = True
            if "Paper" in sql:
                res.get_next.return_value = [42]
            elif "Entity" in sql:
                res.get_next.return_value = [7]
            else:
                res.get_next.return_value = [0]
            return res

        mock_conn.execute.side_effect = _execute_side_effect
        mock_graph_db = MagicMock()
        mock_graph_db._conn = mock_conn

        with patch("scripts.graph.safe_graph_db", return_value=mock_graph_db):
            result = runner.invoke(main, ["graph", "status"])

        assert result.exit_code == 0
        # The table must contain node types
        assert "Paper" in result.output
        assert "Entity" in result.output
        # And the numeric count
        assert "42" in result.output
