"""Global strict-mode helper.

When strict mode is on, silent-fallback patterns become hard errors.
Two activation paths:
  1. CLI: ``lit-monitor --strict <subcommand>`` (set via set_strict in main).
  2. Env: ``LIT_MONITOR_STRICT=1`` (useful for CI and test_everything.sh).

Default behaviour (strict off) is unchanged everywhere — the wrappers
all return their default value after logging a warning.
"""
from __future__ import annotations

import logging
import os
from typing import TypeVar

_strict_override: bool | None = None
T = TypeVar("T")


def set_strict(value: bool) -> None:
    """Called by Click main() when --strict appears on the command line."""
    global _strict_override
    _strict_override = value


def is_strict() -> bool:
    """True when --strict is on OR LIT_MONITOR_STRICT=1/true/yes is set."""
    if _strict_override is not None:
        return _strict_override
    return os.environ.get("LIT_MONITOR_STRICT", "").lower() in ("1", "true", "yes")


def strict_fallback(
    logger: logging.Logger,
    message: str,
    exc: BaseException | None = None,
) -> None:
    """Either raise (strict mode) or log a warning (default mode).

    Use at every silent-fallback site::

        try:
            return parse_yaml(path)
        except Exception as exc:
            strict_fallback(logger, f"Could not parse {path}: {exc}", exc)
            return {}  # only reached in non-strict mode

    Parameters
    ----------
    logger:
        Module-level logger from the calling module.
    message:
        Human-readable description of the fallback, including context (path,
        DOI, model name, etc.) so the error is actionable.
    exc:
        Original exception, if any.  Chained as ``__cause__`` in strict mode
        so the full traceback is visible.
    """
    if is_strict():
        if exc is not None:
            raise RuntimeError(message) from exc
        raise RuntimeError(message)
    logger.warning(message)
