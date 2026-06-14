"""Bundle H (v0.9): per-section Advanced Settings API.

Routes:
    POST /api/settings/{section}      — atomic per-section save to extraction.yaml

The former GET /settings HTML page was folded into the setup wizard as the
optional step-9 ("Tuning") — see lit_monitor/server/routes/setup.py and
templates/setup/step_tuning.html. Only the per-section save API lives here now.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from lit_monitor.server.config_io import (
    SettingsValidationError,
    safe_save_settings_section,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings"])


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
        HTTPException(422): If request body is not a JSON object, or (E3/B8)
            if it fails the section's schema (unknown keys / wrong types).
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
    except SettingsValidationError as exc:
        # E3/B8: payload failed per-section schema (unknown key / wrong type).
        # This is a malformed request body → 422, distinct from the 400 used
        # for an unknown section name below.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        # A3-5 info-leak guard: an OSError from the atomic write embeds the
        # absolute config path ([Errno 13] Permission denied: '/.../config/...').
        # Log it (with traceback) server-side; return a generic client detail.
        logger.error(
            "settings save failed for section %r", section, exc_info=True
        )
        raise HTTPException(status_code=500, detail="File I/O error") from exc

    logger.info("Bundle H: settings section %r saved", section)
    return {"ok": True, "section": section}
