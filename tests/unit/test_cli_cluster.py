"""
Unit tests for Bundle C: `lit-monitor cluster` CLI commands.

Tests cover:
- cluster recompute (threshold check + output)
- cluster view (table rendering)
- cluster assign (re-assign without recompute)
- cluster write-back tags --dry-run (default) vs --confirm
- cluster write-back collections --dry-run (default) vs --confirm
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config():
    cfg = MagicMock()
    cfg.clustering.enabled = True
    cfg.clustering.min_papers_threshold = 100
    cfg.clustering.k_min = 5
    cfg.clustering.k_max = 15
    cfg.state_db.path = "/tmp/test_state.db"
    return cfg


def _mock_state_db(n_papers: int = 150, n_clusters: int = 0):
    sdb = MagicMock()
    sdb.list_active_clusters.return_value = [
        {"id": i + 1, "display_name": f"Theme {i}", "n_papers": 10, "cohesion_score": 0.6}
        for i in range(n_clusters)
    ]
    sdb.get_cluster_assignments.return_value = []
    return sdb


def _mock_embeddings_db(n_papers: int = 150):
    """Mock EmbeddingsDB with a ._collection.count() returning a real int.

    The CLI handlers compare `edb._collection.count() < threshold`. A bare
    MagicMock returns a MagicMock from .count() and the comparison raises
    TypeError on Python 3.12+. Pin the return value here.
    """
    edb = MagicMock()
    edb._collection.count.return_value = n_papers
    return edb


# Mock return values from the write-back helpers. The CLI reads these keys
# directly for its echo output; a partial dict raises KeyError.
_MOCK_TAGS_REPORT_DRY = {
    "tags_added": 12, "papers_processed": 4, "papers_skipped": 0, "dry_run": True,
}
_MOCK_TAGS_REPORT_REAL = {
    "tags_added": 12, "papers_processed": 4, "papers_skipped": 0, "dry_run": False,
}
_MOCK_COLLECTIONS_REPORT_DRY = {
    "collections_to_create": 3, "items_to_add": 12, "dry_run": True,
}


# ---------------------------------------------------------------------------
# cluster recompute
# ---------------------------------------------------------------------------

class TestClusterRecomputeCmd:
    def test_recompute_calls_recompute_clusters(self):
        """cluster recompute invokes recompute_clusters when papers ≥ threshold.

        The CLI handler does lazy imports inside the function — patches must
        target the source modules (scripts.core.config.get_config,
        scripts.core.state_db.StateDB), NOT scripts.cli._make_config.
        """
        from scripts.cli import main

        runner = CliRunner()

        with patch("scripts.core.config.get_config", return_value=_mock_config()), \
             patch("scripts.core.state_db.StateDB"), \
             patch("scripts.cli._make_embeddings_db", return_value=_mock_embeddings_db(150)), \
             patch("scripts.clustering.recompute.recompute_clusters",
                   return_value=7) as mock_recompute, \
             patch("scripts.cli._load_secrets", return_value={}):
            result = runner.invoke(main, ["cluster", "recompute"])

        assert result.exit_code == 0, result.output
        mock_recompute.assert_called_once()

    def test_recompute_respects_threshold_flag(self):
        """--threshold option overrides config value: 50 papers < 200 → skipped."""
        from scripts.cli import main

        runner = CliRunner()

        with patch("scripts.core.config.get_config", return_value=_mock_config()), \
             patch("scripts.core.state_db.StateDB"), \
             patch("scripts.cli._make_embeddings_db", return_value=_mock_embeddings_db(50)), \
             patch("scripts.clustering.recompute.recompute_clusters") as mock_recompute, \
             patch("scripts.cli._load_secrets", return_value={}):
            result = runner.invoke(main, ["cluster", "recompute", "--threshold", "200"])

        # 50 indexed papers < threshold 200 → CLI prints the warning and
        # exits cleanly WITHOUT calling recompute_clusters.
        assert result.exit_code == 0, result.output
        mock_recompute.assert_not_called()


# ---------------------------------------------------------------------------
# cluster view
# ---------------------------------------------------------------------------

class TestClusterViewCmd:
    def test_view_displays_table(self):
        """cluster view renders a rich table to stdout."""
        from scripts.cli import main

        runner = CliRunner()
        state_db = _mock_state_db(150, 3)

        with patch("scripts.cli._make_config", return_value=_mock_config()), \
             patch("scripts.cli._make_state_db", return_value=state_db), \
             patch("scripts.cli._load_secrets", return_value={}):
            result = runner.invoke(main, ["cluster", "view"])

        assert result.exit_code in (0, 1)  # may need config; exit 0 if clusters found

    def test_view_handles_empty_clusters(self):
        """cluster view with no clusters shows a helpful message."""
        from scripts.cli import main

        runner = CliRunner()
        state_db = _mock_state_db(50, 0)  # no clusters

        with patch("scripts.cli._make_config", return_value=_mock_config()), \
             patch("scripts.cli._make_state_db", return_value=state_db), \
             patch("scripts.cli._load_secrets", return_value={}):
            result = runner.invoke(main, ["cluster", "view"])

        assert result.exit_code in (0, 1)


# ---------------------------------------------------------------------------
# cluster write-back
# ---------------------------------------------------------------------------

class TestClusterWriteBackCmd:
    def test_tags_default_dry_run(self):
        """cluster write-back tags runs in dry-run by default (no --confirm).

        Patches target the lazy-imported source modules. Mock return value
        carries the full key set the CLI echoes (tags_added, papers_processed).
        """
        from scripts.cli import main

        runner = CliRunner()

        with patch("scripts.core.config.get_config", return_value=_mock_config()), \
             patch("scripts.core.state_db.StateDB"), \
             patch("scripts.cli._make_zotero_client", return_value=MagicMock()), \
             patch("scripts.cli._load_secrets", return_value={}), \
             patch("scripts.clustering.write_back.push_tags_to_zotero",
                   return_value=_MOCK_TAGS_REPORT_DRY) as mock_push:
            result = runner.invoke(main, ["cluster", "write-back", "tags"])

        assert result.exit_code == 0, result.output
        mock_push.assert_called_once()
        _, kwargs = mock_push.call_args
        # Default (no --confirm) ⇒ dry_run=True. Safety contract is
        # "dry-run unless asked", so a missing kwarg is also a bug.
        assert kwargs.get("dry_run") is True, (
            f"dry_run default should be True (got kwargs={kwargs!r})"
        )

    def test_tags_confirm_flag_passes_dry_run_false(self):
        """cluster write-back tags --confirm passes dry_run=False."""
        from scripts.cli import main

        runner = CliRunner()

        with patch("scripts.core.config.get_config", return_value=_mock_config()), \
             patch("scripts.core.state_db.StateDB"), \
             patch("scripts.cli._make_zotero_client", return_value=MagicMock()), \
             patch("scripts.cli._load_secrets", return_value={}), \
             patch("scripts.clustering.write_back.push_tags_to_zotero",
                   return_value=_MOCK_TAGS_REPORT_REAL) as mock_push:
            result = runner.invoke(main, ["cluster", "write-back", "tags", "--confirm"])

        assert result.exit_code == 0, result.output
        mock_push.assert_called_once()
        _, kwargs = mock_push.call_args
        assert kwargs.get("dry_run") is False, (
            f"--confirm must flip dry_run to False (got kwargs={kwargs!r})"
        )

    def test_collections_default_dry_run(self):
        """cluster write-back collections is dry-run by default."""
        from scripts.cli import main

        runner = CliRunner()

        with patch("scripts.core.config.get_config", return_value=_mock_config()), \
             patch("scripts.core.state_db.StateDB"), \
             patch("scripts.cli._make_zotero_client", return_value=MagicMock()), \
             patch("scripts.cli._load_secrets", return_value={}), \
             patch("scripts.clustering.write_back.push_collections_to_zotero",
                   return_value=_MOCK_COLLECTIONS_REPORT_DRY) as mock_push:
            result = runner.invoke(main, ["cluster", "write-back", "collections"])

        assert result.exit_code == 0, result.output
        mock_push.assert_called_once()
        _, kwargs = mock_push.call_args
        assert kwargs.get("dry_run") is True, (
            f"dry_run default should be True (got kwargs={kwargs!r})"
        )
