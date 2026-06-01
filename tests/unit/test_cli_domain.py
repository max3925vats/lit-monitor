"""Bundle G (v0.9): `lit-monitor domain` CLI tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_cfg_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a tmp state.db + a fake_cfg patched into both get_config sites."""
    from scripts.core.state_db import StateDB

    db_path = tmp_path / "state.db"
    db = StateDB(db_path)

    fake_cfg = MagicMock()
    fake_cfg.state_db.path = str(db_path)
    monkeypatch.setattr("scripts.core.config.get_config", lambda: fake_cfg)
    return db, db_path, tmp_path


def _seed_domain_yaml(cwd: Path, text: str = "I work on bioprocessing.") -> None:
    cfg_dir = cwd / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "domain_context.yaml").write_text(
        f"domain_focus: |\n  {text}\n",
        encoding="utf-8",
    )


class TestDomainView:
    def test_empty_table_friendly_message(
        self, runner: CliRunner, fake_cfg_and_db
    ) -> None:
        from scripts.cli import main

        result = runner.invoke(main, ["domain", "view"])
        assert result.exit_code == 0, result.output
        assert "No domain extraction" in result.output

    def test_view_populated(self, runner: CliRunner, fake_cfg_and_db) -> None:
        from scripts.cli import main

        db, _, _ = fake_cfg_and_db
        db.save_domain_extraction(
            {
                "topics": ["alpha"],
                "methods": ["beta"],
                "materials": [],
                "adjacent_fields": [],
                "exclusions": [],
            }
        )
        result = runner.invoke(main, ["domain", "view"])
        assert result.exit_code == 0, result.output
        assert "alpha" in result.output
        assert "beta" in result.output


class TestDomainAnalyze:
    def test_missing_yaml_exits_nonzero(
        self,
        runner: CliRunner,
        fake_cfg_and_db,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts.cli import main

        # Use isolated FS so config/domain_context.yaml is genuinely absent
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["domain", "analyze"])
        assert result.exit_code != 0
        assert "domain_context.yaml" in result.output

    def test_analyze_writes_to_db(
        self,
        runner: CliRunner,
        fake_cfg_and_db,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts.cli import main

        db, _, _ = fake_cfg_and_db
        # Stub analyze_domain at its call site in scripts.cli
        fake_extraction = {
            "topics": ["bioprocessing"],
            "methods": ["chromatography"],
            "materials": [],
            "adjacent_fields": [],
            "exclusions": [],
        }
        monkeypatch.setattr(
            "scripts.domain.extract.analyze_domain",
            lambda text: fake_extraction,
        )

        with runner.isolated_filesystem():
            _seed_domain_yaml(Path.cwd())
            result = runner.invoke(main, ["domain", "analyze"])

        assert result.exit_code == 0, result.output
        assert "bioprocessing" in result.output
        # Verify persistence
        rows = db.list_domain_extraction()
        assert len(rows["topics"]) == 1
        assert rows["topics"][0]["value"] == "bioprocessing"

    def test_analyze_failure_exits_nonzero(
        self,
        runner: CliRunner,
        fake_cfg_and_db,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts.cli import main

        monkeypatch.setattr(
            "scripts.domain.extract.analyze_domain",
            lambda text: None,
        )
        with runner.isolated_filesystem():
            _seed_domain_yaml(Path.cwd())
            result = runner.invoke(main, ["domain", "analyze"])
        assert result.exit_code != 0
        assert "Extraction failed" in result.output


class TestDomainClear:
    def test_clear_wipes_table(self, runner: CliRunner, fake_cfg_and_db) -> None:
        from scripts.cli import main

        db, _, _ = fake_cfg_and_db
        db.save_domain_extraction(
            {
                "topics": ["x"],
                "methods": [], "materials": [],
                "adjacent_fields": [], "exclusions": [],
            }
        )
        result = runner.invoke(main, ["domain", "clear", "--yes"])
        assert result.exit_code == 0, result.output
        assert "Cleared 1 rows" in result.output
        assert db.list_domain_extraction()["topics"] == []
