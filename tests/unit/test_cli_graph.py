"""G10/G13/N5: CLI tests for graph backfill + rebuild + status subcommand."""
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


class TestGraphBackfillNerCLI:
    """N5: CLI --ner / --ner-with-llm flags on `lit-monitor graph backfill`."""

    def test_backfill_ner_flag_triggers_backfill_ner(self, tmp_path):
        """N5: --ner invokes backfill_ner and echoes the summary."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 5, "edges_added": 12, "failures": 0}
        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_ner", return_value=summary),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--ner"])
        assert result.exit_code == 0, result.output
        assert "5" in result.output
        assert "12" in result.output

    def test_backfill_ner_with_llm_flag(self, tmp_path):
        """N5: --ner-with-llm sets with_llm=True in backfill_ner call."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 1, "edges_added": 3, "failures": 0}
        captured: dict[str, object] = {}

        def fake_backfill_ner(state_db, graph_db, *, with_llm, **kwargs):
            captured["with_llm"] = with_llm
            return summary

        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_ner", side_effect=fake_backfill_ner),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--ner-with-llm"])
        assert result.exit_code == 0, result.output
        assert captured["with_llm"] is True

    def test_backfill_ner_partial_failure_exits_one(self, tmp_path):
        """N5: partial failures (failures > 0) cause exit code 1."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 3, "edges_added": 7, "failures": 2}
        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_ner", return_value=summary),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--ner"])
        assert result.exit_code == 1

    def test_backfill_ner_help_mentions_nlp_extra(self):
        """N5: --ner help text references the [nlp] extra."""
        runner = CliRunner()
        result = runner.invoke(main, ["graph", "backfill", "--help"])
        assert result.exit_code == 0
        assert "nlp" in result.output

    def test_backfill_ner_no_other_flags_required(self, tmp_path):
        """N5: --ner alone is sufficient (no --all / --doi / --since needed)."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 0, "edges_added": 0, "failures": 0}
        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_ner", return_value=summary),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--ner"])
        assert result.exit_code == 0, result.output

    def test_backfill_limit_flag_accepted(self, tmp_path):
        """N5: --limit N is accepted alongside --ner."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        captured: dict[str, object] = {}

        def fake_backfill_ner(state_db, graph_db, *, limit, **kwargs):
            captured["limit"] = limit
            return {"papers_processed": 0, "edges_added": 0, "failures": 0}

        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_ner", side_effect=fake_backfill_ner),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--ner", "--limit", "10"])
        assert result.exit_code == 0, result.output
        assert captured["limit"] == 10


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

    def test_graph_status_by_source_shows_table(self):
        """N8: --by-source prints a second table with schema/biobert/llm_cloud rows."""
        runner = CliRunner()
        mock_conn = MagicMock()

        def _execute_side_effect(sql: str) -> MagicMock:
            res = MagicMock()
            if "m.source" in sql:
                # Simulate 3 source rows returned in sequence
                _rows = [
                    ["biobert", 7, 5],
                    ["llm_cloud", 3, 2],
                    ["schema", 12, 9],
                ]
                _iter = iter(_rows)

                def _has_next() -> bool:
                    # peek without consuming
                    try:
                        res._peek = next(_iter)
                        return True
                    except StopIteration:
                        return False

                def _get_next() -> list:
                    return res._peek  # type: ignore[attr-defined]

                res.has_next.side_effect = _has_next
                res.get_next.side_effect = _get_next
            else:
                res.has_next.return_value = True
                if "Paper" in sql:
                    res.get_next.return_value = [10]
                elif "Entity" in sql:
                    res.get_next.return_value = [3]
                else:
                    res.get_next.return_value = [0]
            return res

        mock_conn.execute.side_effect = _execute_side_effect
        mock_graph_db = MagicMock()
        mock_graph_db._conn = mock_conn

        with patch("scripts.graph.safe_graph_db", return_value=mock_graph_db):
            result = runner.invoke(main, ["graph", "status", "--by-source"])

        assert result.exit_code == 0, result.output
        # The by-source table header and at least one source name must appear
        assert "MENTIONS by source" in result.output or "schema" in result.output or "biobert" in result.output


# ---------------------------------------------------------------------------
# R5: CLI --relationships / --relationships-with-llm flags
# ---------------------------------------------------------------------------

class TestCliGraphBackfillRelationshipsFlag:
    """R5: --relationships / --relationships-with-llm on `lit-monitor graph backfill`."""

    def test_relationships_flag_invokes_backfill_relationships(self, tmp_path):
        """R5: --relationships calls backfill_relationships (not backfill_papers/backfill_ner)."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 4, "edges_added": 9, "failures": 0}
        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_relationships", return_value=summary),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--relationships"])
        assert result.exit_code == 0, result.output
        assert "4" in result.output
        assert "9" in result.output

    def test_relationships_with_llm_implies_relationships(self, tmp_path):
        """R5: --relationships-with-llm alone is sufficient; implies --relationships."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 2, "edges_added": 5, "failures": 0}
        captured: dict[str, object] = {}

        def fake_backfill_rel(state_db, graph_db, *, with_llm, **kwargs):
            captured["with_llm"] = with_llm
            return summary

        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_relationships", side_effect=fake_backfill_rel),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--relationships-with-llm"])
        assert result.exit_code == 0, result.output
        assert captured["with_llm"] is True

    def test_relationships_with_llm_false_without_flag(self, tmp_path):
        """R5: --relationships alone sets with_llm=False."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        captured: dict[str, object] = {}

        def fake_backfill_rel(state_db, graph_db, *, with_llm, **kwargs):
            captured["with_llm"] = with_llm
            return {"papers_processed": 0, "edges_added": 0, "failures": 0}

        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_relationships", side_effect=fake_backfill_rel),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--relationships"])
        assert result.exit_code == 0, result.output
        assert captured["with_llm"] is False

    def test_relationships_partial_failure_exits_one(self, tmp_path):
        """R5: failures > 0 causes exit code 1."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 3, "edges_added": 6, "failures": 1}
        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_relationships", return_value=summary),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--relationships"])
        assert result.exit_code == 1

    def test_relationships_no_other_flags_required(self, tmp_path):
        """R5: --relationships alone is sufficient (no --all / --doi / --since needed)."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        summary = {"papers_processed": 0, "edges_added": 0, "failures": 0}
        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_relationships", return_value=summary),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--relationships"])
        assert result.exit_code == 0, result.output

    def test_relationships_help_text(self):
        """R5: --relationships appears in backfill --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["graph", "backfill", "--help"])
        assert result.exit_code == 0
        assert "relationships" in result.output

    def test_relationships_limit_accepted(self, tmp_path):
        """R5: --limit N is accepted alongside --relationships."""
        runner = CliRunner()
        mock_graph_db = MagicMock()
        captured: dict[str, object] = {}

        def fake_backfill_rel(state_db, graph_db, *, limit, **kwargs):
            captured["limit"] = limit
            return {"papers_processed": 0, "edges_added": 0, "failures": 0}

        with (
            patch("scripts.graph.safe_graph_db", return_value=mock_graph_db),
            patch("scripts.graph.backfill.backfill_relationships", side_effect=fake_backfill_rel),
            patch("scripts.core.config.get_config") as mock_cfg,
        ):
            mock_cfg.return_value.state_db.path = str(tmp_path / "state.db")
            result = runner.invoke(main, ["graph", "backfill", "--relationships", "--limit", "5"])
        assert result.exit_code == 0, result.output
        assert captured["limit"] == 5
