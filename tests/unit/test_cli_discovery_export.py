"""P7: discovery export-md + render_digest tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner


class TestRenderDigest:
    def test_contains_title_doi_rationale(self):
        from lit_monitor.output.digest_renderer import render_digest

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
        from lit_monitor.output.digest_renderer import render_digest

        md = render_digest({"id": 2, "started_at": "2026-05-01T09:00:00"}, [])
        # DR1: format now starts with YAML frontmatter (---), heading follows.
        assert "# Discovery Digest" in md
        assert "Discovery" in md

    def test_empty_papers_no_traceback(self):
        from lit_monitor.output.digest_renderer import render_digest

        md = render_digest({"id": 3, "started_at": "2026-05-01"}, [])
        # DR1: empty-paper placeholder is "*No results.*" in the new unified format.
        assert "No results" in md or "no new papers" in md.lower()

    def test_score_formatted_to_3_decimals(self):
        from lit_monitor.output.digest_renderer import render_digest

        md = render_digest(
            {"id": 1, "started_at": "2026-01-01"},
            [{"title": "X", "doi": "10/y", "score": 0.567891, "rationale": ""}],
        )
        assert "0.568" in md

    def test_title_is_doi_link_when_doi_present(self):
        """A paper WITH a doi renders the title itself as a Markdown link to
        doi.org, and there is NO separate trailing `· [DOI](…)` suffix."""
        from lit_monitor.output.digest_renderer import render_digest

        md = render_digest(
            {"id": 1, "started_at": "2026-01-01"},
            [{"title": "Test Paper A", "doi": "10.1/a", "score": 0.9, "rationale": ""}],
        )
        assert "[**Test Paper A**](https://doi.org/10.1/a)" in md
        # The redundant separate DOI link must be gone.
        assert "· [DOI]" not in md
        assert "[DOI](https://doi.org/" not in md

    def test_title_plain_bold_when_no_doi(self):
        """A paper WITHOUT a doi renders a plain bold title (no link), and the
        line contains no https://doi.org/ for that paper."""
        from lit_monitor.output.digest_renderer import render_digest

        md = render_digest(
            {"id": 1, "started_at": "2026-01-01"},
            [{"title": "No DOI Paper", "doi": "", "score": 0.9, "rationale": ""}],
        )
        assert "**No DOI Paper**" in md
        assert "[**No DOI Paper**]" not in md
        assert "https://doi.org/" not in md

    def test_no_pipeline_run_summary_section(self):
        """render_digest output must never contain a Pipeline Run Summary section."""
        from lit_monitor.output.digest_renderer import render_digest

        md = render_digest(
            {"id": 1, "started_at": "2026-01-01"},
            [{"title": "X", "doi": "10/y", "score": 0.9, "rationale": ""}],
        )
        assert "## Pipeline Run Summary" not in md

    def test_render_digest_rejects_recent_runs_kwarg(self):
        """The recent_runs parameter has been removed from render_digest."""
        from lit_monitor.output.digest_renderer import render_digest

        with pytest.raises(TypeError):
            render_digest(
                {"id": 1, "started_at": "2026-01-01"},
                [],
                recent_runs=[{"run_type": "brain_build"}],
            )


class TestExportMdCommand:
    @pytest.fixture
    def seeded_db(self, tmp_path, monkeypatch):
        from lit_monitor.core.state_db import StateDB

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
        # Patch the source symbol only; "lit_monitor.cli.get_config" is never bound
        # (the old raising=False patch there was a silent no-op).
        monkeypatch.setattr("lit_monitor.core.config.get_config", lambda: fake_cfg)
        return tmp_path, run_id

    def test_writes_file(self, seeded_db):
        tmp_path, run_id = seeded_db
        out = tmp_path / "out.md"
        runner = CliRunner()
        from lit_monitor.cli import main

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
        from lit_monitor.cli import main

        result = runner.invoke(main, ["discovery", "export-md"])
        assert result.exit_code == 0, result.output
        # Some Discovery_*.md file should now exist in cwd
        matches = list(tmp_path.glob("Discovery_*.md"))
        assert len(matches) == 1

    def test_unknown_run_id_exits_nonzero(self, seeded_db):
        runner = CliRunner()
        from lit_monitor.cli import main

        result = runner.invoke(main, ["discovery", "export-md", "--run-id", "99999"])
        assert result.exit_code != 0
