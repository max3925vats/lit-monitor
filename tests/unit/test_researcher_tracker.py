"""Unit tests for run_researcher_searches() with SearchWindow (Tasks 3-5).

Verifies that 'until' propagates through to _findpapers.search.
"""
from __future__ import annotations

import pytest


def _make_empty_search_result():
    from unittest.mock import MagicMock
    r = MagicMock()
    r.papers = []
    return r


@pytest.mark.unit
def test_run_researcher_searches_passes_until_to_findpapers():
    """run_researcher_searches with a bounded SearchWindow passes until to
    _findpapers.search.  Monkeypatches _findpapers.search to capture kwargs."""
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch

    from lit_monitor.search.researcher_tracker import run_researcher_searches
    from lit_monitor.search.window import SearchWindow

    window = SearchWindow(since=date(2025, 1, 1), until=date(2025, 2, 1))
    captured_kwargs: list[dict] = []

    def _fake_search(**kwargs):
        captured_kwargs.append(kwargs)

    config = SimpleNamespace(researchers=[{"name": "Jane Smith"}])
    with patch("lit_monitor.search.researcher_tracker._findpapers") as mock_fp:
        mock_fp.search.side_effect = _fake_search
        with patch(
            "lit_monitor.search.researcher_tracker._fp_load",
            return_value=_make_empty_search_result(),
        ):
            with patch(
                "lit_monitor.search.search_runner._load_api_secrets",
                return_value={},
                create=True,
            ):
                run_researcher_searches(config, window=window)

    assert len(captured_kwargs) == 1, "findpapers.search must be called once"
    assert captured_kwargs[0]["since"] == date(2025, 1, 1)
    assert captured_kwargs[0]["until"] == date(2025, 2, 1)
