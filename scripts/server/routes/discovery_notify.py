"""P3: /discovery/notify-handler — chooser or redirect based on preferred_viewer.

The plyer notification fired by P2 sets a click URL pointing here. This
handler either:
- redirects immediately to the chosen surface (browser / obsidian / none)
  if a viewer preference is already stored in extraction.yaml, or
- renders the chooser card so the user can pick (and optionally save) one.

POST /discovery/notify-handler/save-preference persists the choice via
P4's safe_save_preference when the "remember" flag is set.
"""
from __future__ import annotations

import logging
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from scripts.server.app import templates
from scripts.server.config_io import safe_save_preference

logger = logging.getLogger(__name__)
router = APIRouter(tags=["discovery_notify"])


def _get_preferred_viewer() -> str:
    """Read discovery.notify.preferred_viewer from extraction.yaml.

    Returns an empty string when the key is absent, the value is falsy,
    or any I/O / parse error occurs — so the caller always falls through
    to the chooser page rather than silently breaking.
    """
    try:
        from scripts.server.config_io import load_config

        data = load_config("extraction")
        viewer = (
            data.get("discovery", {})
            .get("notify", {})
            .get("preferred_viewer", "")
        )
        return (viewer or "").strip().lower()
    except Exception as exc:
        logger.warning("P3: _get_preferred_viewer failed: %s", exc)
        return ""


def _get_vault_name() -> str:
    """Read vault name from config for building the obsidian:// URI.

    Falls back to the generic string "vault" so the redirect is still
    well-formed even when paths.yaml is not yet configured.
    """
    try:
        from scripts.core.config import get_config

        cfg = get_config()
        vault_path = getattr(getattr(cfg, "obsidian", None), "vault_path", None)
        if vault_path:
            return str(vault_path).rstrip("/").split("/")[-1]
    except Exception as exc:
        logger.warning("P3: _get_vault_name failed: %s", exc)
    return "vault"


@router.get("/discovery/notify-handler")
async def notify_handler(
    request: Request,
    run_id: int = Query(..., ge=1),
) -> Response:
    """P3: chooser-or-redirect entrypoint for plyer notification clicks.

    Dispatches based on the stored preferred_viewer:
    - "browser"  → 302 /discovery/{run_id}
    - "obsidian" → 302 obsidian://open?vault=<name>
    - "none"     → 204 No Content
    - unset/""   → 200 chooser HTML (3 radio options + remember checkbox)
    """
    viewer = _get_preferred_viewer()

    if viewer == "browser":
        return RedirectResponse(url=f"/discovery/{run_id}", status_code=302)

    if viewer == "obsidian":
        vault = _get_vault_name()
        # Point at the vault root; P8 can refine the note path later.
        obsidian_url = f"obsidian://open?vault={quote(vault)}"
        return RedirectResponse(url=obsidian_url, status_code=302)

    if viewer == "none":
        return Response(status_code=204)

    # No preference set → render the chooser card.
    return templates.TemplateResponse(
        request,
        "discovery/chooser.html",
        {"run_id": run_id},
    )


class SavePreferenceBody(BaseModel):
    """Request body for the save-preference endpoint.

    Pydantic's Literal validator rejects any value outside the allowed set,
    causing FastAPI to return 422 automatically — no manual validation needed.
    """

    viewer: Literal["browser", "obsidian", "none"]
    remember: bool = False


@router.post("/discovery/notify-handler/save-preference")
async def save_preference_endpoint(body: SavePreferenceBody) -> JSONResponse:
    """P3: persist the user's chooser selection via P4's safe_save_preference.

    Calls safe_save_preference only when remember=True; a session-only choice
    (remember=False) still returns 200 so the front-end can route the user to
    /discovery/{run_id} without a separate round-trip.
    """
    if body.remember:
        try:
            safe_save_preference(body.viewer)
        except Exception:
            # A3-5 info-leak guard: log full exception server-side, return a
            # generic error to the client (str(exc) can embed config paths).
            logger.error("P3: safe_save_preference failed", exc_info=True)
            return JSONResponse(
                {"ok": False, "error": "Internal error"},
                status_code=500,
            )
    return JSONResponse(
        {"ok": True, "viewer": body.viewer, "remembered": body.remember}
    )
