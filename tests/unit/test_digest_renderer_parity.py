"""DR1: digest renderer parity — auto-write path unified with render_digest.

Two assertions:

1. Golden-file regression: render_digest(fixture) must byte-match
   tests/fixtures/digest_golden.md.  Any format change must update the golden
   file explicitly — the caller is forced to acknowledge user-visible drift.

2. Pipeline-write parity: _write_digest produces the same bytes that
   render_digest would for the same data.  After DR1 this is trivially true
   because _write_digest delegates to render_digest, but the test guards
   against future accidental divergence.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

# Fixed date: injected via render_digest(_today=...) and monkeypatched in
# _write_digest via date.today() override so output is deterministic.
_FIXTURE_DATE = "2026-01-15"

_FIXTURE_PAPERS: list[dict[str, Any]] = [
    {
        "doi": "10.1234/alpha",
        "title": "Alpha: a study of monoclonal antibodies",
        "authors": ["Smith J", "Jones K", "Lee A", "Wong B"],
        "year": 2024,
        "similarity_score": 0.912,
        "llm_rationale": "Highly relevant to UFDF; novel CEX method.",
    },
    {
        "doi": "10.1234/beta",
        "title": "Beta: cation exchange chromatography",
        "authors": ["Brown C"],
        "year": 2023,
        "similarity_score": 0.823,
        "llm_rationale": "Methods overlap with corpus; consider re-extracting.",
    },
    {
        "doi": "10.1234/gamma",
        "title": "Gamma: low-score irrelevant paper",
        "authors": [],
        "year": 2022,
        "similarity_score": 0.151,
        "llm_rationale": "",
    },
]

# 4 databases, threshold 0.5 → alpha + beta are "Top Results", gamma is not.
_FIXTURE_SIM_THRESHOLD = 0.5
_FIXTURE_N_DATABASES = 4

GOLDEN_PATH = Path(__file__).parent.parent / "fixtures" / "digest_golden.md"


# ---------------------------------------------------------------------------
# Test 1: golden-file regression
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_render_digest_matches_golden() -> None:
    """render_digest(fixture) must byte-match tests/fixtures/digest_golden.md.

    The golden file captures the current auto-write digest format.  Any change
    to the output format MUST update the golden — this test forces the caller
    to acknowledge user-visible output drift.
    """
    if not GOLDEN_PATH.exists():
        pytest.skip(
            f"Golden file missing at {GOLDEN_PATH}. "
            "Generate it once (see DR1 notes), then this test guards it."
        )
    from lit_monitor.output.digest_renderer import render_digest

    actual = render_digest(
        {},
        _FIXTURE_PAPERS,
        sim_threshold=_FIXTURE_SIM_THRESHOLD,
        n_databases=_FIXTURE_N_DATABASES,
        _today=_FIXTURE_DATE,
    )
    expected = GOLDEN_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        f"render_digest output drifted from golden file.\n"
        f"Update {GOLDEN_PATH} if the change is intentional.\n"
        f"len(actual)={len(actual)} vs len(expected)={len(expected)}"
    )


# ---------------------------------------------------------------------------
# Test 2: pipeline _write_digest → render_digest parity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pipeline_write_digest_delegates_to_render_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_write_digest(fixture) must produce the same bytes as render_digest(fixture).

    After DR1 this is guaranteed because _write_digest delegates to
    render_digest.  The test guards against future accidental divergence where
    someone adds logic in _write_digest that isn't reflected in render_digest.
    """
    import lit_monitor.pipelines.discovery as _disc_mod
    from lit_monitor.output.digest_renderer import render_digest

    # Build a minimal config so _write_digest can resolve the vault path.
    vault = tmp_path / "vault"
    (vault / "Literature" / "Digests").mkdir(parents=True)
    config = SimpleNamespace(
        obsidian=SimpleNamespace(
            vault_path=str(vault),
            digests_folder="Literature/Digests",
        ),
    )

    # Pin date.today() so the file name and frontmatter use _FIXTURE_DATE.
    import datetime as _dt

    class _FixedDate(_dt.date):
        @classmethod
        def today(cls) -> _FixedDate:  # type: ignore[override]
            return cls.fromisoformat(_FIXTURE_DATE)

    monkeypatch.setattr(_disc_mod, "date", _FixedDate)

    # Call the pipeline writer and read back what it wrote.
    written_path = _disc_mod._write_digest(
        ranked=_FIXTURE_PAPERS,
        config=config,
        sim_threshold=_FIXTURE_SIM_THRESHOLD,
        dry_run=False,
        n_databases=_FIXTURE_N_DATABASES,
        recent_runs=None,
    )
    written_content = Path(written_path).read_text(encoding="utf-8")

    # Call render_digest directly with the same inputs and fixed date.
    expected_content = render_digest(
        {},
        _FIXTURE_PAPERS,
        sim_threshold=_FIXTURE_SIM_THRESHOLD,
        n_databases=_FIXTURE_N_DATABASES,
        _today=_FIXTURE_DATE,
    )

    assert written_content == expected_content, (
        "_write_digest output diverged from render_digest.\n"
        "DR1 requires a single code path.  Check for extra logic in _write_digest."
    )
