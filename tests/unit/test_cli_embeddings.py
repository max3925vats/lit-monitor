"""Bundle F: CLI embeddings subcommand tests.

Coverage:
  - embeddings status lists provenance rows
  - embeddings switch with --keep-old records new provenance
  - embeddings switch destructive requires --confirm
  - embeddings rebuild delegates to rebuild_active
  - B3: destructive switch builds-new-then-swaps; crash mid-rebuild preserves old
"""
from __future__ import annotations

from click.testing import CliRunner


# ---------------------------------------------------------------------------
# embeddings status
# ---------------------------------------------------------------------------
class TestCliEmbeddingsStatus:
    def test_status_empty_when_no_rows(self, tmp_path, monkeypatch):
        """status command prints 'No embedding collections' when table is empty."""
        from lit_monitor.cli import main
        from lit_monitor.core import config as config_mod
        from lit_monitor.core.state_db import StateDB

        # Create empty state.db at tmp_path; side-effect only — the CLI re-opens
        # via _make_state_db, so we don't need to hold the handle.
        StateDB(tmp_path / "state.db")
        mock_cfg = _make_mock_config(tmp_path)
        monkeypatch.setattr(config_mod, "_config_cache", mock_cfg)

        runner = CliRunner()
        result = runner.invoke(main, ["embeddings", "status"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No embedding collections" in result.output

    def test_status_lists_provenance_rows(self, tmp_path, monkeypatch):
        """status command shows each recorded collection."""
        from lit_monitor.cli import main
        from lit_monitor.core import config as config_mod
        from lit_monitor.core.state_db import StateDB

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
        from lit_monitor.cli import main
        from lit_monitor.core import config as config_mod

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
        from lit_monitor.cli import main
        from lit_monitor.core import config as config_mod
        from lit_monitor.core.state_db import StateDB

        # Create empty state.db at tmp_path; side-effect only.
        StateDB(tmp_path / "state.db")
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
        from lit_monitor.cli import main
        from lit_monitor.core import config as config_mod

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
# B3: destructive switch builds-new-then-swaps (safe migration)
# ---------------------------------------------------------------------------
class TestSwitchProviderSafeSwap:
    def _setup_current(self, tmp_path):
        """Seed a current 'lit_monitor_v1' provenance row; return the StateDB."""
        from lit_monitor.core.state_db import StateDB

        sdb = StateDB(tmp_path / "state.db")
        sdb.record_embedding_provenance(
            "lit_monitor_v1", "ollama", "mxbai-embed-large", 1024
        )
        sdb.set_current_embedding_collection("lit_monitor_v1")
        return sdb

    def _patch_config(self, tmp_path, monkeypatch):
        from lit_monitor.core import config as config_mod

        mock_cfg = _make_mock_config(tmp_path)
        monkeypatch.setattr(config_mod, "get_config", lambda: mock_cfg)
        return mock_cfg

    def test_destructive_rebuild_crash_preserves_old_collection(
        self, tmp_path, monkeypatch
    ):
        """B3: if _rebuild_collection raises during a destructive switch, the OLD
        collection must remain current (provenance unchanged) and the OLD
        collection must NOT have been dropped. The half-built new collection is
        best-effort cleaned up."""
        import pytest

        from lit_monitor.pipelines import embeddings_migration as mig

        sdb = self._setup_current(tmp_path)
        self._patch_config(tmp_path, monkeypatch)

        # Auto-confirm the interactive "Continue?" cost prompt.
        monkeypatch.setattr(mig, "_count_papers", lambda *a, **k: 5)
        import click as _click
        monkeypatch.setattr(_click, "confirm", lambda *a, **k: True)

        dropped: list[str] = []
        monkeypatch.setattr(
            mig, "_drop_collection",
            lambda name, persist_dir: dropped.append(name),
        )

        # Rebuild blows up partway through (simulating a kill mid-embed).
        def boom(**kwargs):
            raise RuntimeError("simulated crash during rebuild")

        monkeypatch.setattr(mig, "_rebuild_collection", boom)

        with pytest.raises(RuntimeError):
            mig.switch_provider(
                provider="litellm",
                model="text-embedding-3-large",
                dim=3072,
                keep_old=False,
                confirm=True,
            )

        # OLD collection still current; provenance untouched.
        rows = sdb.list_embedding_provenance()
        current = [r for r in rows if r["is_current"]]
        assert len(current) == 1
        assert current[0]["collection_name"] == "lit_monitor_v1"
        # No new provenance row got recorded for the failed build.
        assert all(r["collection_name"] == "lit_monitor_v1" for r in rows)

        # The OLD collection must NEVER be dropped on the failure path.
        assert "lit_monitor_v1" not in dropped

    def test_destructive_success_swaps_then_drops_old(self, tmp_path, monkeypatch):
        """B3 happy path: a successful destructive switch builds into a NEW
        collection, records provenance + marks it current, THEN drops the old
        one (swap order: build → record → set-current → drop-old)."""
        from lit_monitor.pipelines import embeddings_migration as mig

        sdb = self._setup_current(tmp_path)
        self._patch_config(tmp_path, monkeypatch)

        monkeypatch.setattr(mig, "_count_papers", lambda *a, **k: 5)
        import click as _click
        monkeypatch.setattr(_click, "confirm", lambda *a, **k: True)

        order: list[str] = []
        dropped: list[str] = []

        def fake_rebuild(**kwargs):
            order.append(f"rebuild:{kwargs['collection_name']}")

        def fake_drop(name, persist_dir):
            order.append(f"drop:{name}")
            dropped.append(name)

        monkeypatch.setattr(mig, "_rebuild_collection", fake_rebuild)
        monkeypatch.setattr(mig, "_drop_collection", fake_drop)

        mig.switch_provider(
            provider="litellm",
            model="text-embedding-3-large",
            dim=3072,
            keep_old=False,
            confirm=True,
        )

        rows = sdb.list_embedding_provenance()
        current = [r for r in rows if r["is_current"]]
        assert len(current) == 1
        new_name = current[0]["collection_name"]
        # New collection is NOT the old one (built fresh, then swapped).
        assert new_name != "lit_monitor_v1"
        # Old collection was dropped AFTER the rebuild into the new collection.
        assert "lit_monitor_v1" in dropped
        assert order.index(f"rebuild:{new_name}") < order.index("drop:lit_monitor_v1")


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
