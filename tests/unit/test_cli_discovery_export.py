"""P7: discovery export-md + render_digest tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner


class TestRenderDigest:
    def test_contains_title_doi_rationale(self):
        from scripts.output.digest_renderer import render_digest

        run = {"id": 1, "started_at": "2026-01-01T10:00:00", "total_ingested": 2}
        papers = [
            {
                "title": "Test Paper A",
                "doi": "10.1/a",
                "score": 0.91,
                "rationale": "Highly relevant.",
            },
        ]
        md = render_digest(run, papers)
        assert "Test Paper A" in md
        assert "10.1/a" in md
        assert "Highly relevant" in md

    def test_heading_present(self):
        from scripts.output.digest_renderer import render_digest

        md = render_digest({"id": 2, "started_at": "2026-05-01T09:00:00"}, [])
        assert md.startswith("#")
        assert "Discovery" in md

    def test_empty_papers_no_traceback(self):
        from scripts.output.digest_renderer import render_digest

        md = render_digest({"id": 3, "started_at": "2026-05-01"}, [])
        assert "no new papers" in md.lower() or "No new papers" in md

    def test_score_formatted_to_3_decimals(self):
        from scripts.output.digest_renderer import render_digest

        md = render_digest(
            {"id": 1, "started_at": "2026-01-01"},
            [{"title": "X", "doi": "10/y", "score": 0.567891, "rationale": ""}],
        )
        assert "0.568" in md


class TestExportMdCommand:
    @pytest.fixture
    def seeded_db(self, tmp_path, monkeypatch):
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run({})
        db.add_discovery_paper(
            run_id,
            doi="10.1/x",
            title="Paper X",
            score=0.88,
            rationale="Relevant",
            ingested=True,
        )
        db.finish_discovery_run(run_id, "success", 1, 1)
        fake_cfg = MagicMock()
        fake_cfg.state_db.path = str(tmp_path / "state.db")
        monkeypatch.setattr("scripts.core.config.get_config", lambda: fake_cfg)
        monkeypatch.setattr("scripts.cli.get_config", lambda: fake_cfg, raising=False)
        return tmp_path, run_id

    def test_writes_file(self, seeded_db):
        tmp_path, run_id = seeded_db
        out = tmp_path / "out.md"
        runner = CliRunner()
        from scripts.cli import main

        result = runner.invoke(main, ["discovery", "export-md", "--to", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        body = out.read_text()
        assert "Paper X" in body
        assert "10.1/x" in body

    def test_default_filename(self, seeded_db, monkeypatch):
        tmp_path, _ = seeded_db
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        from scripts.cli import main

        result = runner.invoke(main, ["discovery", "export-md"])
        assert result.exit_code == 0, result.output
        # Some Discovery_*.md file should now exist in cwd
        matches = list(tmp_path.glob("Discovery_*.md"))
        assert len(matches) == 1

    def test_unknown_run_id_exits_nonzero(self, seeded_db):
        runner = CliRunner()
        from scripts.cli import main

        result = runner.invoke(main, ["discovery", "export-md", "--run-id", "99999"])
        assert result.exit_code != 0
