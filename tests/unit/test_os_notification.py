"""P2: OS notification tests.

These drive ``scripts.notify.os_notification`` directly by monkeypatching its
two module-level globals (``_plyer_notification`` / ``_PLYER_AVAILABLE``) rather
than reloading the module under a patched ``sys.modules``. The old reload
pattern left whatever plyer-availability state the *last* test set leaking into
later tests in the suite (the module stayed reloaded with a MagicMock plyer or
with plyer disabled). monkeypatch auto-reverts both globals at teardown, so the
module is left exactly as imported.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import lit_monitor.notify.os_notification as mod


def _install_mock_plyer(monkeypatch) -> MagicMock:
    """Make the module behave as if plyer is installed; return the notify mock.

    Reverts automatically at test teardown via monkeypatch.
    """
    mock_notify = MagicMock()
    mock_plyer_notification = MagicMock()
    mock_plyer_notification.notify = mock_notify
    monkeypatch.setattr(mod, "_plyer_notification", mock_plyer_notification)
    monkeypatch.setattr(mod, "_PLYER_AVAILABLE", True)
    return mock_notify


class TestNotifyEnabled:
    def test_calls_plyer_when_enabled(self, monkeypatch):
        mock_notify = _install_mock_plyer(monkeypatch)
        mod.notify_discovery_complete(
            run_id=1, paper_count=3,
            app_url="http://localhost:8765",
            enabled=True, on_zero_results=True,
        )
        mock_notify.assert_called_once()
        # Confirm key args
        call_kwargs = mock_notify.call_args.kwargs
        assert call_kwargs["app_name"] == "lit-monitor"
        assert "3" in call_kwargs["message"]


class TestNotifyDisabled:
    def test_skips_when_enabled_false(self, monkeypatch, caplog):
        mock_notify = _install_mock_plyer(monkeypatch)
        with caplog.at_level(logging.INFO):
            mod.notify_discovery_complete(
                run_id=1, paper_count=3,
                app_url="http://localhost:8765",
                enabled=False, on_zero_results=True,
            )
        mock_notify.assert_not_called()
        assert any("disabled" in r.message.lower() for r in caplog.records)


class TestZeroResultsSuppression:
    def test_skips_zero_results_when_configured(self, monkeypatch):
        mock_notify = _install_mock_plyer(monkeypatch)
        mod.notify_discovery_complete(
            run_id=1, paper_count=0,
            app_url="http://localhost:8765",
            enabled=True, on_zero_results=False,
        )
        mock_notify.assert_not_called()

    def test_p54_message_contains_resolved_url(self, monkeypatch):
        """P5.5/P5.4: the notification body shows the resolved click URL.

        plyer can't make notifications clickable cross-platform, so the URL
        must appear in the message text for the user to read/copy.
        """
        mock_notify = _install_mock_plyer(monkeypatch)
        mod.notify_discovery_complete(
            run_id=42, paper_count=3,
            app_url="http://localhost:8765",
            enabled=True, on_zero_results=True,
        )
        message = mock_notify.call_args.kwargs["message"]
        expected_url = "http://localhost:8765/discovery/notify-handler?run_id=42"
        assert expected_url in message

    def test_fires_zero_results_when_allowed(self, monkeypatch):
        mock_notify = _install_mock_plyer(monkeypatch)
        mod.notify_discovery_complete(
            run_id=1, paper_count=0,
            app_url="http://localhost:8765",
            enabled=True, on_zero_results=True,
        )
        mock_notify.assert_called_once()


class TestPlyerMissing:
    def test_silent_when_plyer_unavailable(self, monkeypatch, caplog):
        # Drive the not-installed path directly: plyer unavailable.
        monkeypatch.setattr(mod, "_plyer_notification", None)
        monkeypatch.setattr(mod, "_PLYER_AVAILABLE", False)
        with caplog.at_level(logging.INFO):
            # Must not raise
            mod.notify_discovery_complete(
                run_id=1, paper_count=3,
                app_url="http://localhost:8765",
                enabled=True, on_zero_results=True,
            )
        assert any("plyer" in r.message.lower() for r in caplog.records)


class TestPlyerFailure:
    def test_plyer_raise_logged_not_propagated(self, monkeypatch, caplog):
        mock_notify = _install_mock_plyer(monkeypatch)
        mock_notify.side_effect = RuntimeError("dbus down")
        with caplog.at_level(logging.WARNING):
            # Must not raise
            mod.notify_discovery_complete(
                run_id=1, paper_count=3,
                app_url="http://localhost:8765",
                enabled=True, on_zero_results=True,
            )
        assert any(
            "dbus down" in r.message.lower() or "failed" in r.message.lower()
            for r in caplog.records
        )


class TestPyprojectToml:
    def test_notify_extra_declared(self):
        """P2: pyproject.toml has [project.optional-dependencies] notify."""
        from pathlib import Path
        raw = Path("pyproject.toml").read_text()
        # Format may vary (TOML allows both inline and table styles).
        assert ("notify" in raw and "plyer" in raw), \
            "pyproject.toml missing [notify] optional extra with plyer"


class TestExtractionYamlBlock:
    def test_notify_block_present(self):
        """P2: extraction.example.yaml has a discovery.notify subsection."""
        from importlib.resources import files
        example = files("lit_monitor._data.config_examples") / "extraction.example.yaml"
        raw = example.read_text()
        assert "notify:" in raw
        assert "enabled" in raw
        assert "on_zero_results" in raw
