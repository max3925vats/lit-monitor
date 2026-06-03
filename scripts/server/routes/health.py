"""Health badge endpoints — used by the topnav badge across all pages.

The badge is a small fragment polled by HTMX every 30s. Clicking it
lazy-loads a per-check detail panel below the topnav.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from html import escape
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# P6.23: short TTL cache around run_health_check().
# The badge is polled every 30s by HTMX on every open page, and each tab is a
# separate poller — without a cache, N tabs hammer Ollama+Zotero N times per
# interval. A 30s TTL collapses a burst of near-simultaneous polls (and the
# badge + its detail-panel open) into a single underlying check.
# ---------------------------------------------------------------------------
_HEALTH_CACHE_TTL_SECONDS: float = 30.0
_health_cache: dict[str, Any] | None = None
_health_cache_at: float = 0.0


def _cached_health_check() -> dict[str, dict[str, Sequence[Any]]]:
    """Return run_health_check() results, served from a TTL cache.

    Within ``_HEALTH_CACHE_TTL_SECONDS`` of the last real check, the cached
    result is returned and no underlying Ollama/Zotero probe runs.
    """
    global _health_cache, _health_cache_at
    from scripts.setup.health_check import run_health_check

    now = time.monotonic()
    if _health_cache is not None and (now - _health_cache_at) < _HEALTH_CACHE_TTL_SECONDS:
        return _health_cache

    _health_cache = run_health_check()
    _health_cache_at = now
    return _health_cache


def _severity_of(result) -> str:
    """Return the severity string for a check result.

    Backward-compatible with raw ``(ok, msg)`` tuples (no severity attr):
    fall back to ok→"ok" / !ok→"fail".
    """
    sev = getattr(result, "severity", None)
    if sev in ("ok", "warn", "fail"):
        return sev
    # CheckResult default is "auto" — fall through to ok/!ok inference.
    if sev == "auto" or sev is None:
        ok = result[0]
        return "ok" if ok else "fail"
    return "fail"


def _aggregate_state(results: dict[str, dict[str, Sequence[Any]]]) -> str:
    """Roll up the per-section dict into one of four badge states.

    Rule: fails drive the badge, warns only inform the detail panel.

    States:
      ``unconfigured`` — secrets file missing (gray)
      ``misconfigured`` — ≥2 sections contain a severity="fail" check (red)
      ``degraded`` — exactly 1 section contains a severity="fail" check (yellow)
      ``healthy`` — zero fails; warns are fine and never bump the badge (green)

    Severity is read from the new ``CheckResult`` NamedTuple; legacy
    ``(ok, msg)`` results still work via ``_severity_of``.

    Note: warn-severity rows still render with class="pill warning" in the
    detail panel — they signal "look at the detail" without dragging the
    overall health colour down. This means a setup where only an optional
    service (e.g. ``scopus.api_key``) is absent stays green.
    """
    # When the secrets file itself is missing, the config section comes back
    # with secrets_file=(False, ...). Treat that as the "not started" state.
    config_section = results.get("config", {})
    if any(
        name == "secrets_file" and not result[0]
        for name, result in config_section.items()
    ):
        return "unconfigured"

    # Count *sections* that contain at least one fail. Multiple fails inside a
    # single section still count as one section — we care about breadth of
    # breakage, not depth.
    sections_with_fail = 0
    for section in results.values():
        for result in section.values():
            if _severity_of(result) == "fail":
                sections_with_fail += 1
                break  # next section
    if sections_with_fail >= 2:
        return "misconfigured"
    if sections_with_fail == 1:
        return "degraded"
    return "healthy"


_STATE_LABELS = {
    "healthy": "Healthy",
    "degraded": "Degraded",
    "misconfigured": "Misconfigured",
    "unconfigured": "Setup needed",
}


_STATE_TOOLTIPS = {
    "healthy": "lit-monitor is healthy. Click for details.",
    "degraded": "One service check is failing. Click for details.",
    "misconfigured": "Multiple checks are failing. Click for details.",
    "unconfigured": "Setup not started — visit /setup.",
}


@router.get("/api/health/badge", response_class=HTMLResponse)
async def health_badge() -> str:
    """Return the badge HTML fragment for HTMX swap-in."""
    # Lazy import — keep boot resilient if config is missing.
    try:
        results = _cached_health_check()
        state = _aggregate_state(results)
    except Exception as exc:  # noqa: BLE001 — defensive boot fallback
        logger.warning("Health badge fell back to 'unconfigured' due to: %s", exc)
        state = "unconfigured"
    label = _STATE_LABELS[state]
    tooltip = _STATE_TOOLTIPS[state]
    # Clickable; toggles the detail panel and lazy-loads its content.
    return (
        f'<a class="status-badge {state}" href="#" title="{tooltip}" '
        f'hx-get="/api/health/badge/detail" hx-target="#health-detail" '
        f'hx-swap="innerHTML" '
        f'onclick="document.getElementById(\'health-detail\').toggleAttribute(\'hidden\'); return false;">'
        f'{label}</a>'
    )


@router.get("/api/health/badge/detail", response_class=HTMLResponse)
async def health_badge_detail() -> str:
    """Return the detail panel — full per-check table."""
    try:
        results = _cached_health_check()
    except Exception as exc:  # noqa: BLE001
        return f'<div class="health-detail-error">Could not compute health: {escape(str(exc))}</div>'
    rows = []
    for section_name, checks in results.items():
        for check_name, result in checks.items():
            message = result[1]
            sev = _severity_of(result)
            if sev == "warn":
                pill_class, symbol = "warning", "⚠"
            elif sev == "ok":
                pill_class, symbol = "success", "✓"
            else:
                pill_class, symbol = "danger", "✗"
            rows.append(
                f'<tr><td>{escape(section_name)}</td><td>{escape(check_name)}</td>'
                f'<td><span class="pill {pill_class}">{symbol} {escape(str(message))}</span></td></tr>'
            )
    return '<table class="health-detail-table"><tbody>' + ''.join(rows) + '</tbody></table>'


__all__ = ["router"]
