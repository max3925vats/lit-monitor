"""Tests for scripts.setup.health_check and scripts.setup.diagnose."""
from __future__ import annotations

from unittest.mock import patch

from lit_monitor.setup.check_configured import CheckResult
from lit_monitor.setup.diagnose import run_diagnose
from lit_monitor.setup.health_check import run_health_check


class TestRunHealthCheck:
    def test_run_health_check_returns_four_keys(self) -> None:
        """The structured result must group probes by service domain."""
        cfg = {"probe": CheckResult(True, "ok", "ok")}
        ok = {"probe": (True, "ok")}
        with (
            patch("scripts.setup.check_configured.check_configured", return_value=cfg),
            patch("scripts.setup.check_ollama.check_ollama", return_value=ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=ok),
            patch("scripts.setup.check_vault.check_vault", return_value=ok),
            patch(
                "scripts.setup.check_graph.safe_graph_db",
                return_value=None,
            ),
            patch(
                "scripts.setup.health_check._get_configured_ollama_model",
                return_value=None,
            ),
        ):
            result = run_health_check()
        # FG-2 added the graph section as the 5th probe.
        assert set(result.keys()) == {"config", "ollama", "zotero", "vault", "graph"}

    def test_run_health_check_preserves_check_tuples(self) -> None:
        """Probe results must pass through unchanged — same shape, same message."""
        # check_configured returns CheckResult NamedTuples; the
        # service-probe mocks return plain 2-tuples (their real contract).
        fake = {"foo": CheckResult(False, "bar", "fail")}
        ok = {"ok": (True, "ok")}
        with (
            patch("scripts.setup.check_configured.check_configured", return_value=fake),
            patch("scripts.setup.check_ollama.check_ollama", return_value=ok),
            patch("scripts.setup.check_zotero.check_zotero", return_value=ok),
            patch("scripts.setup.check_vault.check_vault", return_value=ok),
            patch(
                "scripts.setup.check_graph.safe_graph_db",
                return_value=None,
            ),
            patch(
                "scripts.setup.health_check._get_configured_ollama_model",
                return_value=None,
            ),
        ):
            result = run_health_check()
        assert result["config"]["foo"] == CheckResult(False, "bar", "fail")


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
            return_value={"config": {"x": CheckResult(True, "y", "ok")}},
        ) as mock_health:
            result = run_diagnose(config_only=False)
        mock_health.assert_called_once()
        # Service-check rows are namespaced with the section prefix so
        # they cannot collide with config-file keys.
        assert result["config.x"] == (True, "y")

    def test_run_diagnose_handles_3_tuple_check_results(self) -> None:
        """Regression for Audit_28_May_2026 H2: run_diagnose must accept
        ``CheckResult`` 3-tuples in the ``config`` sub-dict without raising
        ``ValueError: too many values to unpack``.

        Pre-H2 ``diagnose.py:146`` did ``(ok, msg) = value``, which broke
        on real ``CheckResult`` NamedTuples (3-tuples) — the bug was only
        masked by test mocks that returned plain 2-tuples.
        """
        hc = {
            "config": {
                "secrets_file": CheckResult(True, "found", "ok"),
                "zotero.api_key": CheckResult(False, "missing", "fail"),
                "scopus.api_key": CheckResult(True, "skipped", "warn"),
            },
            # Service probes still return 2-tuples — must not regress.
            "ollama": {"reachable": (True, "ok")},
            "zotero": {"reachable": (False, "down")},
            "vault": {"vault_exists": (True, "ok")},
        }
        with patch(
            "scripts.setup.diagnose.run_health_check",
            return_value=hc,
        ):
            # Pre-H2 this line raised ValueError.
            result = run_diagnose(config_only=False)
        # Both shapes must have been merged into the flat result dict.
        assert result["config.secrets_file"] == (True, "found")
        assert result["config.zotero.api_key"] == (False, "missing")
        assert result["config.scopus.api_key"] == (True, "skipped")
        assert result["ollama.reachable"] == (True, "ok")
        assert result["zotero.reachable"] == (False, "down")
        assert result["vault.vault_exists"] == (True, "ok")

    def test_run_diagnose_config_only_includes_graph_row(self) -> None:
        """G13: run_diagnose(config_only=True) always returns a 'graph' key."""
        result = run_diagnose(config_only=True)
        assert "graph" in result, "'graph' row missing from run_diagnose output"
        ok, msg = result["graph"]
        assert isinstance(ok, bool)
        assert isinstance(msg, str) and msg  # non-empty message
