"""Reusable health-check entry point for the lit-monitor CLI + web UI.

``run_health_check()`` returns the same per-check status that the CLI's
``check`` command prints, but suitable for direct consumption by the web
server (no Click side-effects, no exit codes — caller decides what to do).
"""
from __future__ import annotations

import logging

# Import the modules (not the functions) so monkeypatching at the source
# module path — e.g. ``patch("scripts.setup.check_configured.check_configured")``
# in the existing CLI test suite — continues to work transparently.
from scripts.setup import check_configured as _check_configured_mod
from scripts.setup import check_ollama as _check_ollama_mod
from scripts.setup import check_vault as _check_vault_mod
from scripts.setup import check_zotero as _check_zotero_mod

logger = logging.getLogger(__name__)


def _get_configured_ollama_model() -> str | None:
    """Read brain_build.model from config; return None on any failure.

    Mirrors the inline fallback used by the CLI's ``check`` command — a
    missing or unloadable config must not abort the health check.
    """
    try:
        from scripts.core.config import get_config
        cfg = get_config()
        return getattr(cfg.brain_build, "model", None)
    except Exception as exc:
        logger.debug("Could not read configured ollama model: %s", exc)
        return None


def run_health_check() -> dict[str, dict[str, tuple[bool, str]]]:
    """Run every health probe and return structured results.

    Returns:
        ``{"config": {...}, "ollama": {...}, "zotero": {...}, "vault": {...}}``

    Each sub-dict is the same ``{check_name: (ok, message)}`` shape the
    underlying probes already return. The grouping preserves per-check
    granularity so callers can render any subset (e.g., a 4-state badge
    rollup vs. a full diagnostic table).
    """
    model = _get_configured_ollama_model()
    return {
        "config": _check_configured_mod.check_configured(),
        "ollama": _check_ollama_mod.check_ollama(model=model),
        "zotero": _check_zotero_mod.check_zotero(),
        "vault": _check_vault_mod.check_vault(),
    }


__all__ = ["run_health_check"]
