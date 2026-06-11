"""Reusable health-check entry point for the lit-monitor CLI + web UI.

``run_health_check()`` returns the same per-check status that the CLI's
``check`` command prints, but suitable for direct consumption by the web
server (no Click side-effects, no exit codes — caller decides what to do).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

# Import the modules (not the functions) so monkeypatching at the source
# module path — e.g. ``patch("scripts.setup.check_configured.check_configured")``
# in the existing CLI test suite — continues to work transparently.
from lit_monitor.setup import check_configured as _check_configured_mod
from lit_monitor.setup import check_graph as _check_graph_mod
from lit_monitor.setup import check_ollama as _check_ollama_mod
from lit_monitor.setup import check_vault as _check_vault_mod
from lit_monitor.setup import check_zotero as _check_zotero_mod
from lit_monitor.setup.check_configured import CheckResult

logger = logging.getLogger(__name__)


def _get_configured_ollama_model() -> str | None:
    """Read brain_build.model from config; return None on any failure.

    Mirrors the inline fallback used by the CLI's ``check`` command — a
    missing or unloadable config must not abort the health check.
    """
    try:
        from lit_monitor.core.config import get_config
        cfg = get_config()
        return getattr(cfg.brain_build, "model", None)
    except Exception as exc:
        logger.debug("Could not read configured ollama model: %s", exc)
        return None


def run_health_check() -> dict[str, dict[str, Sequence[Any]]]:
    """Run every health probe and return structured results.

    Returns:
        ``{"config": {...}, "ollama": {...}, "zotero": {...}, "vault": {...},
        "graph": {"graph_indexed": CheckResult}}``

    The ``config`` and ``graph`` sub-dicts' values are :class:`CheckResult` NamedTuples
    (3-tuples of ok/message/severity). The ``ollama``/``zotero``/``vault``
    sub-dicts return plain ``(ok, msg)`` 2-tuples. Consumers must use
    indexed access (``value[0]``/``value[1]``) to handle both shapes —
    tuple unpacking via ``(ok, msg) = value`` will raise ``ValueError`` on
    the 3-tuple ``CheckResult`` rows.

    The grouping preserves per-check granularity so callers can render any
    subset (e.g., a 4-state badge rollup vs. a full diagnostic table).
    """
    model = _get_configured_ollama_model()
    return {
        "config": _check_configured_mod.check_configured(),
        "ollama": _check_ollama_mod.check_ollama(model=model),
        "zotero": _check_zotero_mod.check_zotero(),
        "vault": _check_vault_mod.check_vault(),
        "graph": _check_graph_section(),
    }


def _check_graph_section() -> dict[str, CheckResult]:
    """Wrap the single-row graph probe in a section dict.

    ``check_graph()`` returns one :class:`CheckResult`; the aggregator's
    contract is ``{section: {check_name: result}}`` (mirrors the ``config``
    section), so we key it under ``"graph_indexed"``. Guarded so a raising
    ``check_graph()`` (or a broken ``[graph]`` import) cannot kill the
    aggregator — the row degrades to a warn instead.
    """
    try:
        return {"graph_indexed": _check_graph_mod.check_graph()}
    except Exception as exc:  # noqa: BLE001 — aggregator must never crash
        logger.debug("check_graph() raised; degrading to warn: %s", exc)
        return {
            "graph_indexed": CheckResult(
                False, f"Graph check failed: {exc}", severity="warn"
            )
        }


__all__ = ["run_health_check"]
