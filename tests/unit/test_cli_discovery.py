"""P6: lit-monitor discovery view CLI tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    from scripts.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    run_id = db.start_discovery_run({"topics": ["x"]})
    db.add_discovery_paper(
        run_id, doi="10.0/a", title="Alpha", score=0.9, rationale="r1", ingested=True
    )
    db.add_discovery_paper(
        run_id, doi="10.0/b", title="Beta", score=0.7, rationale="r2", ingested=False
    )
    db.finish_discovery_run(run_id, "success", 2, 1)

    # Make get_config return a cfg whose state_db.path points here
    fake_cfg = MagicMock()
    fake_cfg.state_db.path = str(tmp_path / "state.db")
    # The CLI resolves config via ``from scripts.core.config import get_config``
    # at call time, so patching the source symbol is what actually controls it.
    # (A former second patch on "scripts.cli.get_config" was dead: that name is
    # never bound on the cli module, and raising=False made it a silent no-op.)
    monkeypatch.setattr("scripts.core.config.get_config", lambda: fake_cfg)
    return db, run_id


class TestDiscoveryView:
    def test_no_runs_friendly_message(self, runner, tmp_path, monkeypatch):
        fake_cfg = MagicMock()
        fake_cfg.state_db.path = str(tmp_path / "empty.db")
        # Patch the source symbol only; "scripts.cli.get_config" is never bound.
        monkeypatch.setattr("scripts.core.config.get_config", lambda: fake_cfg)
        from scripts.cli import main

        result = runner.invoke(main, ["discovery", "view"])
        assert result.exit_code == 0
        assert "no discovery runs" in result.output.lower()

    def test_view_with_data(self, runner, seeded_db):
        from scripts.cli import main

        result = runner.invoke(main, ["discovery", "view"])
        assert result.exit_code == 0, result.output
        assert "Alpha" in result.output
        assert "10.0/a" in result.output

    def test_top_k_limits_rows(self, runner, seeded_db):
        from scripts.cli import main

        result = runner.invoke(main, ["discovery", "view", "--top-k", "1"])
        assert result.exit_code == 0
        # Highest-scoring paper present; second one absent
        assert "Alpha" in result.output
        assert "Beta" not in result.output

    def test_specific_run_id(self, runner, seeded_db):
        from scripts.cli import main

        db, run_id = seeded_db
        result = runner.invoke(main, ["discovery", "view", "--run-id", str(run_id)])
        assert result.exit_code == 0
        assert "Alpha" in result.output

    def test_unknown_run_id_exits_nonzero(self, runner, seeded_db):
        from scripts.cli import main

        result = runner.invoke(main, ["discovery", "view", "--run-id", "99999"])
        assert result.exit_code != 0
