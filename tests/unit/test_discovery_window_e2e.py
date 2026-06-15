"""Task 6 — frontier-based discovery search-window resolution + success-only advance.

Drives the REAL ``run_discovery`` (external search/embeddings/llm/zotero mocked, but a
REAL ``StateDB``) and asserts the window-resolution + ``coverage_until`` frontier-advance
contract:

  1. DEFAULT window starts at the ``coverage_until`` frontier and runs to today; a clean
     non-dry run advances the frontier to the run's covered_to.
  2. An explicit ``window_override`` with an OLDER ``until`` does NOT move the frontier
     (``advance_coverage_until`` is a max(), so it ignores the older value).
  3. A failed/partial search (non-empty search_errors) does NOT advance the frontier.
  4. ``dry_run=True`` does NOT advance the frontier and writes no ``last_run_date``.
  5. Cold start (no frontier, no ``last_run_date``) → default window since = today-14.

The mock pattern mirrors ``tests/unit/test_pipelines_and_tools.py`` (the existing
``run_discovery`` driver tests): patch ``run_searches`` / ``run_researcher_searches`` /
``filter_known_dois`` / ``rank_papers`` on the ``discovery`` module, real ``StateDB``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lit_monitor.pipelines.discovery import read_coverage_until, run_discovery
from lit_monitor.search.window import SearchWindow


# --------------------------------------------------------------------------- #
# Fixtures / helpers (mirror tests/unit/test_pipelines_and_tools.py)
# --------------------------------------------------------------------------- #
def _make_config(tmp_path: Path) -> SimpleNamespace:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Literature" / "Digests").mkdir(parents=True)
    (vault / "Literature" / "Papers").mkdir(parents=True)
    (vault / "Literature" / "Connections").mkdir(parents=True)
    return SimpleNamespace(
        zotero=SimpleNamespace(
            collection_name="lit-monitor",
            local_storage_path=str(tmp_path / "zotero"),
        ),
        obsidian=SimpleNamespace(
            vault_path=str(vault),
            papers_folder="Literature/Papers",
            digests_folder="Literature/Digests",
            connections_folder="Literature/Connections",
        ),
        topics=["filtration AND ProteinA"],
        researchers=[],
        wos_api_key=None,
        scopus_api_key=None,
        email="test@example.com",
    )


def _make_state_db(tmp_path: Path):
    from lit_monitor.core.state_db import StateDB

    return StateDB(tmp_path / "state.db")


def _mocks():
    """Standard set of MagicMock collaborators for a successful (empty-result) run."""
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    # First-run Zotero baseline path: get_current_version stored, nothing ingested.
    zotero_client.get_current_version.return_value = 100
    return embeddings_db, llm, zotero_client


def _run(config, state_db, *, dry_run=False, window_override=None,
         search_side_effect=None, search_return=None):
    """Drive the REAL run_discovery with the search layer mocked.

    ``search_side_effect`` lets a test inject errors/exceptions; ``search_return``
    sets a static return. Captures the window the search received.
    """
    embeddings_db, llm, zotero_client = _mocks()
    captured: dict = {}

    def _capture(cfg, window=None, **kwargs):
        captured["window"] = window
        if search_side_effect is not None:
            return search_side_effect(cfg, window=window, **kwargs)
        return [] if search_return is None else search_return

    with patch("lit_monitor.pipelines.discovery.run_searches", side_effect=_capture), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
        summary = run_discovery(
            config, state_db, zotero_client, embeddings_db, llm,
            dry_run=dry_run, window_override=window_override,
        )
    return summary, captured


# --------------------------------------------------------------------------- #
# 1. Default uses the frontier + advances on a clean non-dry run
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_default_window_from_frontier_and_advances_on_success(tmp_path):
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.set_kv("coverage_until", "2026-02-01")

    _, captured = _run(config, state_db, dry_run=False)

    # Default window starts at the seeded frontier, open-ended (today).
    assert captured["window"] == SearchWindow(since=date(2026, 2, 1), until=None)
    # Clean non-dry run advanced the frontier to the run's covered_to (today,
    # since until is None). >= the seeded value and == covered_to.
    advanced = read_coverage_until(state_db)
    assert advanced == date.today()
    assert advanced >= date(2026, 2, 1)


# --------------------------------------------------------------------------- #
# 2. An explicit OLDER override does not move the frontier
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_older_override_does_not_move_frontier(tmp_path):
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.set_kv("coverage_until", "2026-02-01")

    override = SearchWindow(since=date(2025, 8, 1), until=date(2025, 9, 1))
    _, captured = _run(config, state_db, dry_run=False, window_override=override)

    # The search received exactly the override window.
    assert captured["window"] == override
    # advance_coverage_until is a max(); covered_to=2025-09-01 < 2026-02-01 → no move.
    assert read_coverage_until(state_db) == date(2026, 2, 1)


# --------------------------------------------------------------------------- #
# 3. A failed/partial search does NOT advance the frontier
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_failed_search_does_not_advance_frontier(tmp_path):
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.set_kv("coverage_until", "2026-02-01")

    # run_searches returns a non-empty search_errors list (partial failure).
    # _run_discovery returns (raw_papers, search_errors); we mock _run_discovery
    # directly so we control the search_errors local that gates the advance.
    with patch("lit_monitor.pipelines.discovery._run_discovery",
               return_value=([], ["pubmed: timeout"])), \
         patch("lit_monitor.pipelines.discovery._run_exploration_searches",
               return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
        embeddings_db, llm, zotero_client = _mocks()
        run_discovery(
            config, state_db, zotero_client, embeddings_db, llm, dry_run=False,
        )

    # search_errors non-empty → frontier must stay put.
    assert read_coverage_until(state_db) == date(2026, 2, 1)


# --------------------------------------------------------------------------- #
# 4. Dry-run does NOT advance the frontier and writes no last_run_date
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_dry_run_does_not_advance_frontier(tmp_path):
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.set_kv("coverage_until", "2026-02-01")

    _run(config, state_db, dry_run=True)

    assert read_coverage_until(state_db) == date(2026, 2, 1)
    assert state_db.get_kv("last_run_date") is None


# --------------------------------------------------------------------------- #
# 5. Cold start: no frontier, no last_run_date → default since = today-14
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_cold_start_default_window_since_today_minus_14(tmp_path):
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    # No coverage_until, no last_run_date set.
    assert state_db.get_kv("coverage_until") is None
    assert state_db.get_kv("last_run_date") is None

    _, captured = _run(config, state_db, dry_run=False)

    assert captured["window"].since == date.today() - timedelta(days=14)
    assert captured["window"].until is None


# --------------------------------------------------------------------------- #
# 6. A FORWARD override advances the frontier to its until
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_forward_override_advances_frontier(tmp_path):
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.set_kv("coverage_until", "2026-02-01")

    override = SearchWindow(since=date(2026, 3, 1), until=date(2026, 4, 1))
    _, captured = _run(config, state_db, dry_run=False, window_override=override)

    assert captured["window"] == override
    # covered_to = override.until = 2026-04-01 > 2026-02-01 → frontier advances.
    assert read_coverage_until(state_db) == date(2026, 4, 1)


# --------------------------------------------------------------------------- #
# 7. Dry-run does NOT seed coverage_until even when last_run_date exists
#    (locks the seed_coverage_until dry-run gate where it actually bites)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_dry_run_does_not_seed_frontier_from_last_run_date(tmp_path):
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    # last_run_date present but coverage_until absent → seed WOULD fire on a real
    # run; a dry run must not, to preserve the read-only contract.
    state_db.set_kv("last_run_date", "2026-03-15")
    assert state_db.get_kv("coverage_until") is None

    _run(config, state_db, dry_run=True)

    assert state_db.get_kv("coverage_until") is None
