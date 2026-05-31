"""P2: OS notification tests."""
from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock, patch


class TestNotifyEnabled:
    def test_calls_plyer_when_enabled(self):
        mock_plyer_module = MagicMock()
        mock_notify = MagicMock()
        mock_plyer_module.notification.notify = mock_notify
        with patch.dict("sys.modules", {"plyer": mock_plyer_module}):
            # Force reload so module-level import picks up the mock
            import scripts.notify.os_notification as mod
            importlib.reload(mod)
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
    def test_skips_when_enabled_false(self, caplog):
        mock_plyer_module = MagicMock()
        mock_notify = MagicMock()
        mock_plyer_module.notification.notify = mock_notify
        with patch.dict("sys.modules", {"plyer": mock_plyer_module}):
            import scripts.notify.os_notification as mod
            importlib.reload(mod)
            with caplog.at_level(logging.INFO):
                mod.notify_discovery_complete(
                    run_id=1, paper_count=3,
                    app_url="http://localhost:8765",
                    enabled=False, on_zero_results=True,
                )
        mock_notify.assert_not_called()
        assert any("disabled" in r.message.lower() for r in caplog.records)


class TestZeroResultsSuppression:
    def test_skips_zero_results_when_configured(self):
        mock_plyer_module = MagicMock()
        mock_notify = MagicMock()
        mock_plyer_module.notification.notify = mock_notify
        with patch.dict("sys.modules", {"plyer": mock_plyer_module}):
            import scripts.notify.os_notification as mod
            importlib.reload(mod)
            mod.notify_discovery_complete(
                run_id=1, paper_count=0,
                app_url="http://localhost:8765",
                enabled=True, on_zero_results=False,
            )
        mock_notify.assert_not_called()

    def test_fires_zero_results_when_allowed(self):
        mock_plyer_module = MagicMock()
        mock_notify = MagicMock()
        mock_plyer_module.notification.notify = mock_notify
        with patch.dict("sys.modules", {"plyer": mock_plyer_module}):
            import scripts.notify.os_notification as mod
            importlib.reload(mod)
            mod.notify_discovery_complete(
                run_id=1, paper_count=0,
                app_url="http://localhost:8765",
                enabled=True, on_zero_results=True,
            )
        mock_notify.assert_called_once()


class TestPlyerMissing:
    def test_silent_when_plyer_unavailable(self, caplog):
        # Make `import plyer` raise ImportError inside the module.
        # Setting the key to None in sys.modules causes ImportError on import.
        with patch.dict("sys.modules", {"plyer": None}):
            import scripts.notify.os_notification as mod
            importlib.reload(mod)
            with caplog.at_level(logging.INFO):
                # Must not raise
                mod.notify_discovery_complete(
                    run_id=1, paper_count=3,
                    app_url="http://localhost:8765",
                    enabled=True, on_zero_results=True,
                )
        assert any("plyer" in r.message.lower() for r in caplog.records)


class TestPlyerFailure:
    def test_plyer_raise_logged_not_propagated(self, caplog):
        mock_plyer_module = MagicMock()
        mock_plyer_module.notification.notify.side_effect = RuntimeError("dbus down")
        with patch.dict("sys.modules", {"plyer": mock_plyer_module}):
            import scripts.notify.os_notification as mod
            importlib.reload(mod)
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
        from pathlib import Path
        raw = Path("config/extraction.example.yaml").read_text()
        assert "notify:" in raw
        assert "enabled" in raw
        assert "on_zero_results" in raw
