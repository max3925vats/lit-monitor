"""Bundle H (v0.9): Advanced Settings page + per-section API.

Routes:
    GET  /settings                    — Advanced Settings HTML page
    POST /api/settings/{section}      — atomic per-section save to extraction.yaml
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from scripts.server.config_io import safe_save_settings_section

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_templates():
    """Lazy import to avoid circular dependency at module load."""
    from scripts.server.app import templates
    return templates


def _load_extraction_config() -> dict[str, Any]:
    """Return current extraction.yaml content as a dict (best-effort)."""
    try:
        from scripts.server.config_io import load_config
        return load_config("extraction")
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# GET /settings — HTML page
# ---------------------------------------------------------------------------

@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    """Post-onboarding Advanced Settings panel.

    Three collapsible sections:
      - ranking:    ranking weights and thresholds
      - clustering: k-means / HDBSCAN parameters
      - web_ui:     feedback buttons + other UI flags
    """
    cfg = _load_extraction_config()
    ranking_cfg = cfg.get("ranking", {})
    clustering_cfg = cfg.get("clustering", {})
    web_ui_cfg = cfg.get("web_ui", {})

    templates = _get_templates()
    return templates.TemplateResponse(
        request,
        "settings/index.html",
        {
            "ranking": ranking_cfg,
            "clustering": clustering_cfg,
            "web_ui": web_ui_cfg,
        },
    )


# ---------------------------------------------------------------------------
# POST /api/settings/{section} — per-section atomic save
# ---------------------------------------------------------------------------

@router.post("/api/settings/{section}")
async def save_settings_section(section: str, request: Request) -> dict:
    """Atomic write of one config section to extraction.yaml.

    Args:
        section: Top-level key to replace (must be in the allowed set).

    Request body:
        JSON object. The entire value of the section is replaced with this dict.

    Returns:
        {"ok": True, "section": section} on success.

    Raises:
        HTTPException(400): If section is not in the allowed set.
        HTTPException(422): If request body is not a JSON object.
        HTTPException(500): On file I/O failure.
    """
    try:
        data: Any = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")

    try:
        safe_save_settings_section(section, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.error("settings save failed for section %r: %s", section, exc)
        raise HTTPException(status_code=500, detail=f"File I/O error: {exc}") from exc

    logger.info("Bundle H: settings section %r saved", section)
    return {"ok": True, "section": section}
