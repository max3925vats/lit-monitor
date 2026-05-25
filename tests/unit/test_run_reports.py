"""
V-6 test backfill — I5.
Tests for write_brain_build_report().
All tests are marked @pytest.mark.unit.
No LLM or external services required.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_config(tmp_path: Path) -> MagicMock:
    """Return a minimal config mock with a vault_path."""
    cfg = MagicMock()
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    cfg.obsidian.vault_path = str(vault)
    cfg.obsidian.digests_folder = "Literature/Digests"
    cfg.brain_build.pass_strategy = "individual"
    return cfg


# ---------------------------------------------------------------------------
# I5 — brain build report structure
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_brain_build_report_has_seven_sections(tmp_path):
    """write_brain_build_report() must produce exactly 7 markdown sections (## N)."""
    from scripts.pipelines.brain_build import BuildSummary, write_brain_build_report

    cfg = _make_config(tmp_path)
    summary = BuildSummary(
        papers_processed=5,
        papers_skipped=2,
        papers_failed=1,
        errors=["10.1/x: extraction failed"],
        crossref_resolved=["10.1/resolved"],
        no_doi_items=[{"title": "Untitled", "zotero_key": "AAAA1111"}],
        started_at="2026-05-11T10:00:00+00:00",
    )
    path = write_brain_build_report(cfg, summary, model_str="phi4-mini")
    assert path, "report path must be non-empty"
    text = Path(path).read_text(encoding="utf-8")

    # Count numbered section headings: ## 1., ## 2., ..., ## 7.
    sections = re.findall(r"^## \d+\.", text, re.MULTILINE)
    assert len(sections) == 7, f"Expected 7 sections, found {len(sections)}: {sections}"


@pytest.mark.unit
def test_brain_build_report_crossref_and_no_doi_listed(tmp_path):
    """CrossRef resolutions and no-DOI items are rendered in the report."""
    from scripts.pipelines.brain_build import BuildSummary, write_brain_build_report

    cfg = _make_config(tmp_path)
    summary = BuildSummary(
        crossref_resolved=["10.1/resolved-a", "10.1/resolved-b"],
        no_doi_items=[{"title": "Mystery Paper", "zotero_key": "MYST0001"}],
        started_at="2026-05-11T10:00:00+00:00",
    )
    text = Path(write_brain_build_report(cfg, summary)).read_text(encoding="utf-8")

    assert "10.1/resolved-a" in text
    assert "10.1/resolved-b" in text
    assert "Mystery Paper" in text
    assert "MYST0001" in text




