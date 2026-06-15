"""
Smoke tests for scripts/pipelines/model_compare.py.

Primary purpose: prevent kwarg regressions (e.g. passes= vs phases=)
from silently breaking all extractions in a benchmark run.

All I/O is mocked; no real LLM or Zotero calls.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _hermetic_reference_papers(tmp_path):
    """E8: never let a test in this module read the user's REAL reference file.

    In production ``_load_reference_papers()`` prefers the gitignored,
    machine-specific ``docs/internal/reference_papers.yaml`` and falls back to
    the committed ``reference_papers.example.yaml``. Tests must be hermetic, so
    by default we redirect ``_REFERENCE_REAL`` to a NON-EXISTENT tmp path. With
    the real file out of the picture, ``_load_reference_papers()`` falls back to
    the committed ``_REFERENCE_EXAMPLE`` for every test — no test touches the
    user's real reference_papers.yaml.

    The A7 tests that ``patch.object(mc, "_REFERENCE_REAL", ...)`` inside the
    test body still win: their ``with patch.object`` runs after this fixture has
    set the module attr, so it overrides our default for the duration of that
    block (and restores it on exit).
    """
    from lit_monitor.pipelines import model_compare as mc

    with patch.object(mc, "_REFERENCE_REAL", tmp_path / "nonexistent_reference.yaml"):
        yield


def _make_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        zotero=SimpleNamespace(collection_name="lit-monitor"),
        brain_build=SimpleNamespace(timeout=30, temperature=0.1),
        comparison_models=[],
    )


def _make_state_db_with_one_paper() -> MagicMock:
    state_db = MagicMock()
    state_db.get_all_by_source_type.return_value = [
        {
            "doi": "10.1/test",
            "zotero_key": "ZKEY001",
            "title": "Test Paper",
        }
    ]
    return state_db


def _make_zotero_client() -> MagicMock:
    client = MagicMock()
    client.get_markdown_attachment.return_value = (
        "Ultrafiltration of IgG at 30 kDa membrane. Yield was 95%."
    )
    return client


# ---------------------------------------------------------------------------
# Critical regression: phases= kwarg must be accepted by extract_paper
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compare_models_no_extraction_failures(tmp_path):
    """
    Smoke test: run_model_comparison must produce items_failed == 0 for a
    mocked LLM that returns valid JSON.

    Previously broken (M7 regression): model_compare.py called
    extract_paper(..., passes=(1, 2, 3)) which raises TypeError because
    extract_paper() has no 'passes' parameter — every paper silently failed.
    This test would have caught that.
    """
    from lit_monitor.pipelines.model_compare import run_model_comparison

    config = _make_config(tmp_path)
    state_db = _make_state_db_with_one_paper()
    zotero_client = _make_zotero_client()

    mock_extraction = {
        "core_finding": "Flux decline above 20 g/L.",
        "core_finding_confidence": "explicit",
        "methods_summary": "TFF with 30 kDa PES membrane.",
        "methods_summary_confidence": "explicit",
        "_overall_confidence": 0.9,
    }

    # extract_paper is imported inside run_model_comparison — patch at its definition site.
    with (
        patch(
            "lit_monitor.pipelines.model_compare._resolve_output_dir",
            return_value=tmp_path / "comparison",
        ),
        patch(
            "lit_monitor.llm.extractor.extract_paper",
            return_value=mock_extraction,
        ),
        patch("lit_monitor.llm.llm_client.OllamaClient", return_value=MagicMock()),
    ):
        result = run_model_comparison(
            config=config,
            state_db=state_db,
            zotero_client=zotero_client,
            n_items=1,
            models=["ollama:qwen2.5:3b"],
            mode="paper",
        )

    assert len(result.scores) == 1
    score = result.scores[0]
    assert score.items_failed == 0, (
        f"Expected 0 extraction failures; got {score.items_failed}. "
        "This likely means extract_paper() was called with the wrong kwargs."
    )
    assert score.items_run == 1


@pytest.mark.unit
def test_extract_paper_rejects_passes_kwarg():
    """
    Defensive regression: extract_paper() must NOT accept a 'passes' keyword
    argument. If this test fails, the M3 hard-cut has been rolled back.
    """
    import inspect

    from lit_monitor.llm.extractor import extract_paper

    sig = inspect.signature(extract_paper)
    assert "passes" not in sig.parameters, (
        "extract_paper() must not have a 'passes' parameter — M3 hard-cut "
        "replaced it with 'phases'. Roll back this change."
    )
    assert "**" not in str(sig), (
        "extract_paper() must not accept **kwargs — that would hide kwarg drift."
    )


@pytest.mark.unit
def test_compare_models_writes_output_json(tmp_path):
    """run_model_comparison must write a summary JSON file to the output directory."""
    from lit_monitor.pipelines.model_compare import run_model_comparison

    config = _make_config(tmp_path)
    state_db = _make_state_db_with_one_paper()
    zotero_client = _make_zotero_client()

    mock_extraction = {
        "core_finding": "Result.",
        "core_finding_confidence": "explicit",
        "_overall_confidence": 0.8,
    }

    with (
        patch(
            "lit_monitor.pipelines.model_compare._resolve_output_dir",
            return_value=tmp_path / "comparison",
        ),
        patch(
            "lit_monitor.llm.extractor.extract_paper",
            return_value=mock_extraction,
        ),
        patch("lit_monitor.llm.llm_client.OllamaClient", return_value=MagicMock()),
    ):
        result = run_model_comparison(
            config=config,
            state_db=state_db,
            zotero_client=zotero_client,
            n_items=1,
            models=["ollama:qwen2.5:3b"],
            mode="paper",
        )

    out_dir = Path(result.output_dir)
    assert out_dir.exists(), f"Output directory not created: {out_dir}"
    # At least one JSON file written
    json_files = list(out_dir.rglob("*.json"))
    assert json_files, "No JSON output files written by run_model_comparison"


@pytest.mark.unit
def test_resolve_output_dir_is_under_user_config_home():
    """P1.2/P1.3: output dir resolves under ~/.config/lit-monitor, not into the
    install location (site-packages) or the repo checkout.

    Pre-fix, ``_OUTPUT_DIR = Path(__file__).parent.parent.parent / "comparison"``
    pointed at the package install root, which is read-only for wheel installs.
    """
    from lit_monitor.pipelines.model_compare import _resolve_output_dir

    out = _resolve_output_dir()
    expected = Path("~/.config/lit-monitor/comparison").expanduser()
    assert out == expected
    # Must live under the user config home, not under site-packages or the repo.
    config_home = Path("~/.config/lit-monitor").expanduser()
    assert str(out).startswith(str(config_home))
    assert "site-packages" not in str(out)
    # Not resolved relative to the source file's install tree.
    import lit_monitor.pipelines.model_compare as mc
    pkg_root = Path(mc.__file__).parent.parent.parent
    assert pkg_root not in out.parents


# ---------------------------------------------------------------------------
# A7: reference_papers ground-truth → null-honesty scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_reference_papers_real_overrides_example(tmp_path):
    """When the real reference_papers.yaml has entries, it is used over example."""
    import yaml

    from lit_monitor.pipelines import model_compare as mc

    real = tmp_path / "reference_papers.yaml"
    example = tmp_path / "reference_papers.example.yaml"
    real.write_text(yaml.safe_dump({"papers": {"10.1/real": {"expected_null": ["a"]}}}))
    example.write_text(yaml.safe_dump({"papers": {"10.1/example": {"expected_null": ["b"]}}}))

    with patch.object(mc, "_REFERENCE_REAL", real), patch.object(mc, "_REFERENCE_EXAMPLE", example):
        refs = mc._load_reference_papers()

    assert "10.1/real" in refs
    assert "10.1/example" not in refs


@pytest.mark.unit
def test_load_reference_papers_falls_back_to_example_when_real_empty(tmp_path):
    """When the real file is missing or has no paper entries, example is used."""
    import yaml

    from lit_monitor.pipelines import model_compare as mc

    real = tmp_path / "reference_papers.yaml"
    example = tmp_path / "reference_papers.example.yaml"
    # Real file exists but its papers map is empty (the all-commented template case).
    real.write_text(yaml.safe_dump({"papers": None}))
    example.write_text(yaml.safe_dump({"papers": {"10.1/example": {"expected_null": ["b"]}}}))

    with patch.object(mc, "_REFERENCE_REAL", real), patch.object(mc, "_REFERENCE_EXAMPLE", example):
        refs = mc._load_reference_papers()

    assert "10.1/example" in refs


@pytest.mark.unit
def test_null_honesty_violation_counted_for_fabricated_field():
    """A non-null value for an expected_null field increments the violation count."""
    from lit_monitor.pipelines.model_compare import _ModelScore, _score_extraction

    score = _ModelScore(model="test")
    reference = {"expected_null": ["statistical_methods", "data_availability"]}
    extraction = {
        "core_finding": "Something.",
        "statistical_methods": "ANOVA, p<0.05",  # fabricated — should be null
        "data_availability": None,                # correctly null
    }

    _score_extraction(score, extraction, "paper", reference=reference)

    assert score.null_honesty_violations == 1


@pytest.mark.unit
def test_null_honesty_zero_when_correctly_null():
    """When all expected_null fields are null/empty/absent, no violation is counted."""
    from lit_monitor.pipelines.model_compare import _ModelScore, _score_extraction

    score = _ModelScore(model="test")
    reference = {"expected_null": ["statistical_methods", "data_availability"]}
    extraction = {
        "core_finding": "Something.",
        "statistical_methods": None,
        "data_availability": "",  # empty string also counts as honest-null
        # software_code simply absent
    }

    _score_extraction(score, extraction, "paper", reference=reference)

    assert score.null_honesty_violations == 0


@pytest.mark.unit
def test_null_honesty_no_reference_contributes_zero():
    """A paper with no reference entry (reference=None) contributes 0 violations."""
    from lit_monitor.pipelines.model_compare import _ModelScore, _score_extraction

    score = _ModelScore(model="test")
    extraction = {"statistical_methods": "fabricated"}

    _score_extraction(score, extraction, "paper", reference=None)

    assert score.null_honesty_violations == 0


@pytest.mark.unit
def test_review_not_penalised_for_missing_study_type():
    """Reviews must be scored against the review schema (no study_type), not the paper schema.

    Audit-6 #4: _PAPER_REQUIRED includes study_type which was dropped from the review
    schema (R-10). A review extraction with the 3 review-schema required fields present
    should score 0 required_fields_missing — not 1 (as it would if scored against
    _PAPER_REQUIRED which includes study_type).
    """
    from lit_monitor.pipelines.model_compare import _ModelScore, _score_extraction

    review_extraction = {
        "core_finding": "x",
        "methods_summary": "y",
        "results_summary": "z",
        # study_type intentionally absent — it is NOT in the review schema (R-10)
    }
    score = _ModelScore(model="m")
    _score_extraction(score, review_extraction, "review", reference=None)
    assert score.required_fields_missing == 0, (
        f"Expected 0 missing required fields for a complete review extraction; "
        f"got {score.required_fields_missing}. "
        "Likely cause: _score_extraction is using _PAPER_REQUIRED (which includes "
        "study_type) instead of _REVIEW_REQUIRED when mode='review'."
    )


@pytest.mark.unit
def test_summary_includes_null_honesty_in_json_and_markdown(tmp_path):
    """scores.json carries null_honesty_violations and the markdown table has the column."""
    import json

    from lit_monitor.pipelines.model_compare import (
        _ComparisonResult,
        _ModelScore,
        _write_summary,
    )

    score = _ModelScore(model="ollama:qwen2.5:3b", items_run=2, null_honesty_violations=3)
    result = _ComparisonResult(mode="paper", n_items=2, run_date="2026-06-03_0000", scores=[score])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_summary(result, out_dir)

    scores_data = json.loads((out_dir / "scores.json").read_text())
    assert scores_data[0]["null_honesty_violations"] == 3

    md = (out_dir / "summary.md").read_text()
    assert "Null Violations" in md
    assert "| 3 |" in md
