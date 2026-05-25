"""Unit tests for scripts.core.strict_mode."""
from __future__ import annotations

import logging

import pytest

import scripts.core.strict_mode as _sm


def _reset() -> None:
    """Reset module-level state between tests."""
    _sm._strict_override = None


# ---------------------------------------------------------------------------
# is_strict()
# ---------------------------------------------------------------------------

class TestIsStrict:
    def setup_method(self) -> None:
        _reset()

    def test_default_mode_off(self, monkeypatch) -> None:
        """is_strict() returns False when neither flag nor env var is set."""
        monkeypatch.delenv("LIT_MONITOR_STRICT", raising=False)
        assert _sm.is_strict() is False

    def test_env_var_activates_strict(self, monkeypatch) -> None:
        """LIT_MONITOR_STRICT=1 makes is_strict() True."""
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        assert _sm.is_strict() is True

    def test_env_var_true_string(self, monkeypatch) -> None:
        monkeypatch.setenv("LIT_MONITOR_STRICT", "true")
        assert _sm.is_strict() is True

    def test_env_var_yes_string(self, monkeypatch) -> None:
        monkeypatch.setenv("LIT_MONITOR_STRICT", "yes")
        assert _sm.is_strict() is True

    def test_env_var_other_value_is_off(self, monkeypatch) -> None:
        monkeypatch.setenv("LIT_MONITOR_STRICT", "0")
        assert _sm.is_strict() is False

    def test_set_strict_overrides_env(self, monkeypatch) -> None:
        """set_strict(False) wins over LIT_MONITOR_STRICT=1."""
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        _sm.set_strict(False)
        assert _sm.is_strict() is False

    def test_set_strict_true_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("LIT_MONITOR_STRICT", raising=False)
        _sm.set_strict(True)
        assert _sm.is_strict() is True


# ---------------------------------------------------------------------------
# strict_fallback()
# ---------------------------------------------------------------------------

class TestStrictFallback:
    def setup_method(self) -> None:
        _reset()

    def test_strict_fallback_logs_when_off(self, monkeypatch, caplog) -> None:
        """In non-strict mode, strict_fallback logs a WARNING and does not raise."""
        monkeypatch.delenv("LIT_MONITOR_STRICT", raising=False)
        log = logging.getLogger("test_strict_fallback_log")
        with caplog.at_level(logging.WARNING, logger="test_strict_fallback_log"):
            _sm.strict_fallback(log, "something went wrong", None)
        assert "something went wrong" in caplog.text

    def test_strict_fallback_raises_when_on(self, monkeypatch) -> None:
        """In strict mode, strict_fallback raises RuntimeError."""
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        log = logging.getLogger("test_strict_raises")
        with pytest.raises(RuntimeError, match="something went wrong"):
            _sm.strict_fallback(log, "something went wrong", None)

    def test_strict_fallback_chains_cause(self, monkeypatch) -> None:
        """Original exception is chained as __cause__ in strict mode."""
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        log = logging.getLogger("test_strict_cause")
        original = ValueError("original cause")
        with pytest.raises(RuntimeError) as exc_info:
            _sm.strict_fallback(log, "wrapper message", original)
        assert exc_info.value.__cause__ is original

    def test_strict_fallback_no_exc_raises_cleanly(self, monkeypatch) -> None:
        """strict_fallback without an exc arg raises RuntimeError without __cause__."""
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        log = logging.getLogger("test_strict_no_cause")
        with pytest.raises(RuntimeError, match="no exc"):
            _sm.strict_fallback(log, "no exc", None)
