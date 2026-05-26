"""Tests for scripts.setup.health_check and scripts.setup.diagnose."""
from __future__ import annotations

from unittest.mock import patch

from scripts.setup.diagnose import run_diagnose
from scripts.setup.health_check import run_health_check


class TestRunHealthCheck:
    def test_run_health_check_returns_four_keys(self) -> None:
        """The structured result must group probes by service domain."""
        ok = {"probe": (True, "ok")}
        with (
            patch("scripts.setup.check_configured.check_configured", return_value=ok),
            patch("scripts.setup.check_ollama.check_ollama", return_value=ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=ok),
            patch("scripts.setup.check_vault.check_vault", return_value=ok),
            patch(
                "scripts.setup.health_check._get_configured_ollama_model",
                return_value=None,
            ),
        ):
            result = run_health_check()
        assert set(result.keys()) == {"config", "ollama", "zotero", "vault"}

    def test_run_health_check_preserves_check_tuples(self) -> None:
        """Probe results must pass through unchanged — same tuple, same message."""
        fake = {"foo": (False, "bar")}
        ok = {"ok": (True, "ok")}
        with (
            patch("scripts.setup.check_configured.check_configured", return_value=fake),
            patch("scripts.setup.check_ollama.check_ollama", return_value=ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=ok),
            patch("scripts.setup.check_vault.check_vault", return_value=ok),
            patch(
                "scripts.setup.health_check._get_configured_ollama_model",
                return_value=None,
            ),
        ):
            result = run_health_check()
        assert result["config"]["foo"] == (False, "bar")


class TestRunDiagnose:
    def test_run_diagnose_config_only_skips_service_probes(self) -> None:
        """config_only=True must not call run_health_check()."""
        with patch(
            "scripts.setup.diagnose.run_health_check",
        ) as mock_health:
            run_diagnose(config_only=True)
        mock_health.assert_not_called()

    def test_run_diagnose_full_invokes_health_check(self) -> None:
        """config_only=False merges health-check rows into the result dict.

        Sanity-check for the opposite branch — ensures the skip is gated
        on the flag rather than always-off.
        """
        with patch(
            "scripts.setup.diagnose.run_health_check",
            return_value={"config": {"x": (True, "y")}},
        ) as mock_health:
            result = run_diagnose(config_only=False)
        mock_health.assert_called_once()
        # Service-check rows are namespaced with the section prefix so
        # they cannot collide with config-file keys.
        assert result["config.x"] == (True, "y")
