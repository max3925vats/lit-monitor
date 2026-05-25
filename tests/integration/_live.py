"""Shared helper for the LIT_MONITOR_LIVE env var convention.

Used by tests/integration/conftest.py (fixture-level skips) and by
individual integration tests that have additional in-body skip checks
(e.g., "model X is not pulled"). Centralised so every silent-skip site
honours the same `LIT_MONITOR_LIVE=1 → fail loudly` contract.
"""
from __future__ import annotations

import os

import pytest


def live_mode() -> bool:
    """True when the user wants missing services to fail loudly, not skip."""
    return os.environ.get("LIT_MONITOR_LIVE", "").lower() in ("1", "true", "yes")


def skip_or_fail(reason: str) -> None:
    """In live mode, fail loudly. Otherwise, skip silently (default).

    The point of live mode is that a "green" pytest run with services down
    is a lie — silently skipping integration tests gives the user no
    signal that anything actually ran.
    """
    if live_mode():
        pytest.fail(
            f"LIT_MONITOR_LIVE=1 set but a required service is unavailable:\n"
            f"  {reason}\n"
            f"Either bring the service up and re-run, or unset LIT_MONITOR_LIVE "
            f"to allow silent skips."
        )
    pytest.skip(reason)
