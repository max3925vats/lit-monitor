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


class TestGetStrictOverride:
    def setup_method(self) -> None:
        _reset()

    def test_default_override_is_none(self) -> None:
        """Unset override is None (distinct from is_strict()'s bool)."""
        assert _sm.get_strict_override() is None

    def test_reflects_set_strict(self) -> None:
        _sm.set_strict(True)
        assert _sm.get_strict_override() is True
        _sm.set_strict(False)
        assert _sm.get_strict_override() is False

    def test_save_and_restore_roundtrip(self) -> None:
        """P6.16: a caller can force strict then restore the prior override."""
        _sm.set_strict(False)
        prev = _sm.get_strict_override()
        _sm.set_strict(True)  # temporarily force
        assert _sm.get_strict_override() is True
        _sm.set_strict(prev)  # restore
        assert _sm.get_strict_override() is False


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
        """Original exception is chained as __cause__ in strict mode.

        L3: when the original exception type is 1-arg-constructible, it is
        re-raised as the same type rather than wrapped in RuntimeError. Use
        the base ``Exception`` here to assert chaining regardless of which
        concrete type the wrapper picks.
        """
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        log = logging.getLogger("test_strict_cause")
        original = ValueError("original cause")
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            _sm.strict_fallback(log, "wrapper message", original)
        assert exc_info.value.__cause__ is original

    def test_strict_fallback_no_exc_raises_cleanly(self, monkeypatch) -> None:
        """strict_fallback without an exc arg raises RuntimeError without __cause__."""
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        log = logging.getLogger("test_strict_no_cause")
        with pytest.raises(RuntimeError, match="no exc"):
            _sm.strict_fallback(log, "no exc", None)

    def test_strict_fallback_preserves_exception_type(self, monkeypatch) -> None:
        """In strict mode, the original exception type is re-raised (not RuntimeError).

        L3: a JSONDecodeError-style downstream exception should still be
        catchable as ValueError so callers can write specific except clauses
        rather than catching the umbrella RuntimeError wrapper.
        """
        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        log = logging.getLogger("test_strict_preserve_type")
        original = ValueError("original cause")
        with pytest.raises(ValueError) as exc_info:
            _sm.strict_fallback(log, "wrapped message", original)
        # Type is preserved, message is rewritten, cause is chained.
        assert type(exc_info.value) is ValueError
        assert "wrapped message" in str(exc_info.value)
        assert exc_info.value.__cause__ is original

    def test_strict_fallback_falls_back_to_runtimeerror_when_type_not_constructible(
        self, monkeypatch,
    ) -> None:
        """Exception types that don't accept a single string arg fall back to RuntimeError.

        Some stdlib exceptions (OSError subclasses, custom multi-arg exceptions)
        cannot be reconstructed from a single message; strict_fallback must
        degrade gracefully to RuntimeError rather than raising TypeError.
        """
        class _MultiArgError(Exception):
            def __init__(self, a: str, b: str) -> None:
                super().__init__(f"{a}/{b}")
                self.a = a
                self.b = b

        monkeypatch.setenv("LIT_MONITOR_STRICT", "1")
        log = logging.getLogger("test_strict_multiarg_fallback")
        original = _MultiArgError("x", "y")
        with pytest.raises(RuntimeError) as exc_info:
            _sm.strict_fallback(log, "wrap msg", original)
        assert exc_info.value.__cause__ is original
