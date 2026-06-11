"""P4 Part B: tests for the ``lit-monitor feedback`` CLI command.

The command records one feedback event for a DOI via the SAME contract the
HTTP route uses (``StateDB.record_feedback_event``). It maps mutually-exclusive
flags to the closed signal vocabulary and validates inputs the same way the
route does (rating 1-5 only with ``--rating``; rating rejected otherwise).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from lit_monitor.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


def _patched(mock_config, mock_state_db):
    """Patch the config + state-db factories used by the feedback command."""
    return (
        patch("lit_monitor.cli._make_config", return_value=mock_config),
        patch("lit_monitor.cli._make_state_db", return_value=mock_state_db),
    )


def _make_config(tmp_path):
    cfg = MagicMock()
    cfg.state_db.path = str(tmp_path / "state.db")
    return cfg


class TestFeedbackRecordsSignals:
    @pytest.mark.parametrize(
        "flag,expected_signal",
        [
            ("--saved", "saved"),
            ("--dismissed", "dismissed"),
            ("--opened", "opened"),
            ("--thumbs-up", "thumbs_up"),
            ("--thumbs-down", "thumbs_down"),
        ],
    )
    def test_records_each_signal_type(self, runner, tmp_path, flag, expected_signal):
        cfg = _make_config(tmp_path)
        db = MagicMock()
        p_cfg, p_db = _patched(cfg, db)
        with p_cfg, p_db:
            result = runner.invoke(main, ["feedback", "10.1/abc", flag])
        assert result.exit_code == 0, result.output
        db.record_feedback_event.assert_called_once()
        args, kwargs = db.record_feedback_event.call_args
        # Positional contract: (doi, signal_type); rating + source are kwargs.
        assert args[0] == "10.1/abc"
        assert args[1] == expected_signal
        assert kwargs.get("rating") is None

    def test_records_rating(self, runner, tmp_path):
        cfg = _make_config(tmp_path)
        db = MagicMock()
        p_cfg, p_db = _patched(cfg, db)
        with p_cfg, p_db:
            result = runner.invoke(main, ["feedback", "10.1/abc", "--rating", "4"])
        assert result.exit_code == 0, result.output
        args, kwargs = db.record_feedback_event.call_args
        assert args[1] == "rated"
        assert kwargs.get("rating") == 4

    def test_source_tag_is_cli(self, runner, tmp_path):
        """Events recorded from the CLI carry a distinguishable source."""
        cfg = _make_config(tmp_path)
        db = MagicMock()
        p_cfg, p_db = _patched(cfg, db)
        with p_cfg, p_db:
            result = runner.invoke(main, ["feedback", "10.1/abc", "--saved"])
        assert result.exit_code == 0, result.output
        _, kwargs = db.record_feedback_event.call_args
        assert kwargs.get("source") == "cli"


class TestFeedbackValidation:
    def test_rejects_bad_rating(self, runner, tmp_path):
        cfg = _make_config(tmp_path)
        db = MagicMock()
        p_cfg, p_db = _patched(cfg, db)
        with p_cfg, p_db:
            result = runner.invoke(main, ["feedback", "10.1/abc", "--rating", "9"])
        assert result.exit_code != 0
        db.record_feedback_event.assert_not_called()

    def test_rejects_empty_doi(self, runner, tmp_path):
        cfg = _make_config(tmp_path)
        db = MagicMock()
        p_cfg, p_db = _patched(cfg, db)
        with p_cfg, p_db:
            result = runner.invoke(main, ["feedback", "   ", "--saved"])
        assert result.exit_code != 0
        db.record_feedback_event.assert_not_called()

    def test_rejects_no_signal(self, runner, tmp_path):
        """Exactly one signal flag (or --rating) is required."""
        cfg = _make_config(tmp_path)
        db = MagicMock()
        p_cfg, p_db = _patched(cfg, db)
        with p_cfg, p_db:
            result = runner.invoke(main, ["feedback", "10.1/abc"])
        assert result.exit_code != 0
        db.record_feedback_event.assert_not_called()

    def test_rejects_multiple_signals(self, runner, tmp_path):
        cfg = _make_config(tmp_path)
        db = MagicMock()
        p_cfg, p_db = _patched(cfg, db)
        with p_cfg, p_db:
            result = runner.invoke(
                main, ["feedback", "10.1/abc", "--saved", "--dismissed"]
            )
        assert result.exit_code != 0
        db.record_feedback_event.assert_not_called()


class TestFeedbackConfigGuard:
    def test_config_load_failure_exits_nonzero(self, runner):
        """A broken config must exit cleanly (no raw traceback)."""
        with patch("lit_monitor.cli._make_config", side_effect=RuntimeError("boom")):
            result = runner.invoke(main, ["feedback", "10.1/abc", "--saved"])
        assert result.exit_code != 0
        assert "Error" in result.output
