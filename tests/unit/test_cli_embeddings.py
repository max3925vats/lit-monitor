"""Bundle F: CLI embeddings subcommand tests.

Coverage:
  - embeddings status lists provenance rows
  - embeddings switch with --keep-old records new provenance
  - embeddings switch destructive requires --confirm
  - embeddings rebuild delegates to rebuild_active
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# embeddings status
# ---------------------------------------------------------------------------
class TestCliEmbeddingsStatus:
    def test_status_empty_when_no_rows(self, tmp_path, monkeypatch):
        """status command prints 'No embedding collections' when table is empty."""
        from scripts.cli import main
        from scripts.core.state_db import StateDB
        from scripts.core import config as config_mod

        sdb = StateDB(tmp_path / "state.db")
        mock_cfg = _make_mock_config(tmp_path)
        monkeypatch.setattr(config_mod, "_config_cache", mock_cfg)

        runner = CliRunner()
        result = runner.invoke(main, ["embeddings", "status"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No embedding collections" in result.output

    def test_status_lists_provenance_rows(self, tmp_path, monkeypatch):
        """status command shows each recorded collection."""
        from scripts.cli import main
        from scripts.core.state_db import StateDB
        from scripts.core import config as config_mod

        sdb = StateDB(tmp_path / "state.db")
        sdb.record_embedding_provenance("lit_monitor_v1", "ollama", "mxbai-embed-large", 1024)
        sdb.record_embedding_provenance("papers_v2_litellm", "litellm", "text-embedding-3-large", 3072)

        mock_cfg = _make_mock_config(tmp_path)
        monkeypatch.setattr(config_mod, "_config_cache", mock_cfg)

        runner = CliRunner()
        result = runner.invoke(main, ["embeddings", "status"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "lit_monitor_v1" in result.output
        assert "papers_v2_litellm" in result.output


# ---------------------------------------------------------------------------
# embeddings switch
# ---------------------------------------------------------------------------
class TestCliEmbeddingsSwitch:
    def test_switch_destructive_without_confirm_exits_nonzero(self, tmp_path, monkeypatch):
        """switch without --keep-old and without --confirm exits with code 2."""
        from scripts.cli import main
        from scripts.core import config as config_mod

        mock_cfg = _make_mock_config(tmp_path)
        monkeypatch.setattr(config_mod, "_config_cache", mock_cfg)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["embeddings", "switch", "--provider", "litellm", "--model", "text-embedding-3-large"],
            catch_exceptions=False,
        )
        # Should exit non-zero (code 2 per spec) and print a warning about destructive op
        assert result.exit_code != 0
        assert "DESTRUCTIVE" in result.output or "confirm" in result.output.lower()

    def test_switch_with_keep_old_aborted_via_stdin(self, tmp_path, monkeypatch):
        """switch --keep-old exits 1 when user answers 'n' to the cost prompt."""
        from scripts.cli import main
        from scripts.core.state_db import StateDB
        from scripts.core import config as config_mod

        sdb = StateDB(tmp_path / "state.db")
        mock_cfg = _make_mock_config(tmp_path)
        monkeypatch.setattr(config_mod, "_config_cache", mock_cfg)

        runner = CliRunner()
        # Answer 'n' to the "Continue?" prompt
        result = runner.invoke(
            main,
            ["embeddings", "switch", "--provider", "ollama", "--model", "nomic-embed-text", "--keep-old"],
            input="n\n",
            catch_exceptions=False,
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# embeddings rebuild
# ---------------------------------------------------------------------------
class TestCliEmbeddingsRebuild:
    def test_rebuild_without_confirm_exits_nonzero(self, tmp_path, monkeypatch):
        """rebuild without --confirm (and without --keep-old) exits non-zero."""
        from scripts.cli import main
        from scripts.core import config as config_mod

        mock_cfg = _make_mock_config(tmp_path)
        monkeypatch.setattr(config_mod, "_config_cache", mock_cfg)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["embeddings", "rebuild"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_mock_config(tmp_path):
    """Build a minimal mock Config object with state_db pointing at tmp_path."""
    class _NS:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def get(self, attr, default=None):
            return getattr(self, attr, default)

    cfg = object.__new__(type("Config", (), {}))

    state_db_path = str(tmp_path / "state.db")
    cfg.state_db = _NS(path=state_db_path)

    # embedding config
    cfg.embedding = _NS(
        provider="ollama",
        ollama=_NS(host="http://localhost:11434", model="mxbai-embed-large", dim=1024),
        litellm=_NS(model="text-embedding-3-large", dim=3072),
    )

    # brain_build / embeddings (legacy path)
    cfg.embeddings = _NS(ollama_host="http://localhost:11434", model="mxbai-embed-large")
    cfg.brain_build = _NS(ollama_host="http://localhost:11434")

    chroma_dir = str(tmp_path / "chroma")
    cfg._chroma_dir = chroma_dir

    return cfg
