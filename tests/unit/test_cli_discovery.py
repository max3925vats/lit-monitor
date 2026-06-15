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
    from lit_monitor.core.state_db import StateDB

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
    # The CLI resolves config via ``from lit_monitor.core.config import get_config``
    # at call time, so patching the source symbol is what actually controls it.
    # (A former second patch on "lit_monitor.cli.get_config" was dead: that name is
    # never bound on the cli module, and raising=False made it a silent no-op.)
    monkeypatch.setattr("lit_monitor.core.config.get_config", lambda: fake_cfg)
    return db, run_id


class TestDiscoveryView:
    def test_no_runs_friendly_message(self, runner, tmp_path, monkeypatch):
        fake_cfg = MagicMock()
        fake_cfg.state_db.path = str(tmp_path / "empty.db")
        # Patch the source symbol only; "lit_monitor.cli.get_config" is never bound.
        monkeypatch.setattr("lit_monitor.core.config.get_config", lambda: fake_cfg)
        from lit_monitor.cli import main

        result = runner.invoke(main, ["discovery", "view"])
        assert result.exit_code == 0
        assert "no discovery runs" in result.output.lower()

    def test_view_with_data(self, runner, seeded_db):
        from lit_monitor.cli import main

        result = runner.invoke(main, ["discovery", "view"])
        assert result.exit_code == 0, result.output
        assert "Alpha" in result.output
        assert "10.0/a" in result.output

    def test_top_k_limits_rows(self, runner, seeded_db):
        from lit_monitor.cli import main

        result = runner.invoke(main, ["discovery", "view", "--top-k", "1"])
        assert result.exit_code == 0
        # Highest-scoring paper present; second one absent
        assert "Alpha" in result.output
        assert "Beta" not in result.output

    def test_specific_run_id(self, runner, seeded_db):
        from lit_monitor.cli import main

        db, run_id = seeded_db
        result = runner.invoke(main, ["discovery", "view", "--run-id", str(run_id)])
        assert result.exit_code == 0
        assert "Alpha" in result.output

    def test_unknown_run_id_exits_nonzero(self, runner, seeded_db):
        from lit_monitor.cli import main

        result = runner.invoke(main, ["discovery", "view", "--run-id", "99999"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Task 5: backfill-ingested unit tests
# ---------------------------------------------------------------------------


def test_backfill_ingested_flips_recommended_and_library_papers(tmp_path):
    """Papers in the library (status != 'discovered') that appear in a digest
    should have their discovery_paper_results.ingested flipped to 1; papers not
    yet in the library should remain 0."""
    from lit_monitor.api.queries import get_discovery_run_papers
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.pipelines.discovery_backfill import backfill_ingested

    db = StateDB(db_path=tmp_path / "s.db")
    run_id = db.start_discovery_run({}, trigger="manual")
    db.add_discovery_paper(
        run_id=run_id,
        doi="10.1/in-lib",
        title="t",
        score=0.5,
        rationale="",
        ingested=False,
    )
    db.add_discovery_paper(
        run_id=run_id,
        doi="10.1/not-in-lib",
        title="t",
        score=0.5,
        rationale="",
        ingested=False,
    )
    db.upsert_paper({"doi": "10.1/in-lib", "status": "extraction_complete"})

    flipped = backfill_ingested(db)
    assert flipped == 1

    papers = {p["doi"]: p for p in get_discovery_run_papers(db, run_id, top_k=100)}
    assert papers["10.1/in-lib"]["ingested"] == 1
    assert papers["10.1/not-in-lib"]["ingested"] == 0


def test_backfill_ingested_is_idempotent(tmp_path):
    """Running backfill_ingested twice should not double-count: second call
    still returns 1 (the row matched) but the timestamp is unchanged
    (COALESCE keeps the first ingested_at)."""
    from lit_monitor.api.queries import get_discovery_run_papers
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.pipelines.discovery_backfill import backfill_ingested

    db = StateDB(db_path=tmp_path / "s.db")
    run_id = db.start_discovery_run({}, trigger="manual")
    db.add_discovery_paper(
        run_id=run_id,
        doi="10.2/paper",
        title="t",
        score=0.8,
        rationale="",
        ingested=False,
    )
    db.upsert_paper({"doi": "10.2/paper", "status": "extraction_complete"})

    first = backfill_ingested(db)
    second = backfill_ingested(db)
    assert first == 1
    assert second == 1  # still matches 1 DOI

    papers = {p["doi"]: p for p in get_discovery_run_papers(db, run_id, top_k=10)}
    assert papers["10.2/paper"]["ingested"] == 1


def test_backfill_ingested_cli_command(tmp_path, monkeypatch):
    """The CLI subcommand should print a summary line and exit 0."""
    from unittest.mock import MagicMock

    from click.testing import CliRunner

    from lit_monitor.core.state_db import StateDB

    db = StateDB(db_path=tmp_path / "s.db")
    run_id = db.start_discovery_run({}, trigger="manual")
    db.add_discovery_paper(
        run_id=run_id,
        doi="10.3/x",
        title="t",
        score=0.6,
        rationale="",
        ingested=False,
    )
    db.upsert_paper({"doi": "10.3/x", "status": "extraction_complete"})

    fake_cfg = MagicMock()
    fake_cfg.state_db.path = str(tmp_path / "s.db")
    monkeypatch.setattr("lit_monitor.core.config.get_config", lambda: fake_cfg)

    from lit_monitor.cli import main

    result = CliRunner().invoke(main, ["discovery", "backfill-ingested"])
    assert result.exit_code == 0, result.output
    assert "1" in result.output
