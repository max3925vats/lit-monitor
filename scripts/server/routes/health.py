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


def _aggregate_state(results: dict[str, dict[str, tuple[bool, str]]]) -> str:
    """Roll up the per-section dict into one of four badge states.

    States:
      ``unconfigured`` — secrets file missing (gray)
      ``misconfigured`` — config exists but ≥2 sections have any failing check (red)
      ``degraded`` — exactly 1 section has any failing check (yellow)
      ``healthy`` — all checks pass (green)
    """
    # When the secrets file itself is missing, the config section comes back
    # with secrets_file=(False, ...). Treat that as the "not started" state.
    config_section = results.get("config", {})
    if any(name == "secrets_file" and not ok for name, (ok, _) in config_section.items()):
        return "unconfigured"
    failing_sections = sum(
        1 for section in results.values() if any(not ok for ok, _ in section.values())
    )
    if failing_sections >= 2:
        return "misconfigured"
    if failing_sections == 1:
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
        for check_name, (ok, message) in checks.items():
            pill_class = "success" if ok else "danger"
            symbol = "✓" if ok else "✗"
            rows.append(
                f'<tr><td>{escape(section_name)}</td><td>{escape(check_name)}</td>'
                f'<td><span class="pill {pill_class}">{symbol} {escape(message)}</span></td></tr>'
            )
    return '<table class="health-detail-table"><tbody>' + ''.join(rows) + '</tbody></table>'


__all__ = ["router"]
