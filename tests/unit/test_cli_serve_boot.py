"""Unit tests for `lit-monitor serve` boot output and browser auto-open.

Covers:
  1. Boot banner lines are printed before uvicorn.run().
  2. --no-browser suppresses the threading.Timer-based browser launch.
  3. Default (no flag, no [server].open_browser=False) triggers Timer.start().
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from lit_monitor.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.unit
def test_serve_emits_boot_lines(runner: CliRunner) -> None:
    """`serve --no-browser` emits the two `>>` banner lines before uvicorn.run."""
    with (
        patch("uvicorn.run") as mock_run,
        patch("lit_monitor.server.config_io.load_server_config", return_value={}),
    ):
        result = runner.invoke(
            main,
            ["serve", "--host", "127.0.0.1", "--port", "18999", "--no-browser"],
        )
    assert result.exit_code == 0, result.output
    assert ">> lit-monitor serve" in result.output
    assert ">> running at http://127.0.0.1:18999" in result.output
    mock_run.assert_called_once()


@pytest.mark.unit
def test_serve_no_browser_flag_suppresses_open(runner: CliRunner) -> None:
    """`--no-browser` must skip threading.Timer construction entirely."""
    with (
        patch("uvicorn.run"),
        patch("lit_monitor.server.config_io.load_server_config",
              return_value={"open_browser": True}),
        patch("threading.Timer") as mock_timer,
    ):
        result = runner.invoke(
            main,
            ["serve", "--host", "127.0.0.1", "--port", "18999", "--no-browser"],
        )
    assert result.exit_code == 0, result.output
    mock_timer.assert_not_called()


@pytest.mark.unit
def test_serve_opens_browser_by_default(runner: CliRunner) -> None:
    """Without --no-browser and no config opt-out, Timer.start() runs once."""
    fake_timer = MagicMock()
    with (
        patch("uvicorn.run"),
        patch("lit_monitor.server.config_io.load_server_config", return_value={}),
        patch("threading.Timer", return_value=fake_timer) as mock_timer_ctor,
    ):
        result = runner.invoke(
            main,
            ["serve", "--host", "127.0.0.1", "--port", "18999"],
        )
    assert result.exit_code == 0, result.output
    mock_timer_ctor.assert_called_once()
    fake_timer.start.assert_called_once()
