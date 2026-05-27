"""Health badge endpoints — used by the topnav badge across all pages.

The badge is a small fragment polled by HTMX every 30s. Clicking it
lazy-loads a per-check detail panel below the topnav.
"""
from __future__ import annotations

import logging
from html import escape

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()


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


def _aggregate_state(results: dict[str, dict[str, tuple[bool, str]]]) -> str:
    """Roll up the per-section dict into one of four badge states.

    States:
      ``unconfigured`` — secrets file missing (gray)
      ``misconfigured`` — any section contains a severity="fail" check (red)
      ``degraded`` — any section contains a severity="warn" check, no fails (yellow)
      ``healthy`` — every check is severity="ok" (green)

    Severity is read from the new ``CheckResult`` NamedTuple; legacy
    ``(ok, msg)`` results still work via ``_severity_of``.
    """
    # When the secrets file itself is missing, the config section comes back
    # with secrets_file=(False, ...). Treat that as the "not started" state.
    config_section = results.get("config", {})
    if any(
        name == "secrets_file" and not result[0]
        for name, result in config_section.items()
    ):
        return "unconfigured"

    has_fail = False
    has_warn = False
    for section in results.values():
        for result in section.values():
            sev = _severity_of(result)
            if sev == "fail":
                has_fail = True
            elif sev == "warn":
                has_warn = True
    if has_fail:
        return "misconfigured"
    if has_warn:
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
        from scripts.setup.health_check import run_health_check
        results = run_health_check()
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
        from scripts.setup.health_check import run_health_check
        results = run_health_check()
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
