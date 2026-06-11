"""Zotero credential check + collections list for the setup wizard.

Both endpoints run BEFORE the wizard has finished saving config, so they
must tolerate missing credentials, missing ``paths.yaml``, and network
failures. ``/collections`` returns 503 if creds are missing or the Zotero
API errors out — the wizard treats that as "user hasn't finished step 1".
``/test`` is the friendlier sibling: it always returns 200 with an
``{ok, message}`` body so the UI can render a status pill without an
exception path.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from lit_monitor.server.config_io import load_secrets

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/zotero", tags=["zotero"])


def _build_client():
    """Construct a ZoteroClient from current secrets, with config fallbacks.

    Returns ``None`` if no ``api_key`` is configured (the wizard hasn't run
    yet). Otherwise returns a ZoteroClient. ``library_id`` and ``api_key``
    come from secrets; ``library_type`` and ``local_storage_path`` come
    from the YAML config if available, otherwise fall back to pyzotero's
    defaults. The fallback path matters because the wizard's first call
    to ``/collections`` happens BEFORE ``paths.yaml`` is written, so
    ``runtime.config`` raises FileNotFoundError.
    """
    secrets = load_secrets()
    zot_secrets = secrets.get("zotero", {}) if isinstance(secrets, dict) else {}
    api_key = zot_secrets.get("api_key", "")
    library_id = zot_secrets.get("library_id", "")
    if not api_key or not library_id:
        return None

    # Defaults if config isn't loadable yet (wizard hasn't saved paths.yaml).
    library_type = "user"
    local_storage_path = "~/Zotero/storage"
    try:
        from lit_monitor.server.runtime import get_runtime

        cfg = get_runtime().config
        library_type = cfg.zotero.library_type or library_type
        local_storage_path = cfg.zotero.local_storage_path or local_storage_path
    except Exception as exc:
        # No paths.yaml yet, or schema mismatch — fine, use defaults.
        logger.debug("zotero route: using default config (runtime.config unavailable: %s)", exc)

    from lit_monitor.core.zotero_client import ZoteroClient

    return ZoteroClient(
        library_id=str(library_id),
        api_key=str(api_key),
        library_type=str(library_type),
        local_storage_path=str(local_storage_path),
    )


@router.get("/collections")
def collections() -> dict:
    """List Zotero collections for the wizard's dropdown.

    Returns ``{"collections": [{"key", "name"}, ...]}`` on success.
    Returns 503 if credentials are missing OR the Zotero API errors out,
    so the wizard can show a single "not ready" branch regardless of cause.
    """
    client = _build_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Zotero credentials not configured")
    try:
        rows = client.get_all_collections()
    except Exception as exc:
        logger.warning("Zotero collections fetch failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Zotero API error: {exc}") from exc
    return {"collections": [{"key": r["key"], "name": r["name"]} for r in rows]}


@router.get("/test")
def test() -> dict:
    """Quick credential check for the wizard's status pill.

    Always returns 200 with ``{ok, message}``. ``ok=False`` covers both
    "no credentials yet" and "credentials present but API rejected them".
    """
    client = _build_client()
    if client is None:
        return {"ok": False, "message": "credentials not configured"}
    try:
        # Cheapest probe pyzotero offers: a 1-item collections page.
        client._zot.collections(limit=1)
    except Exception:
        # Info-leak guard: the pyzotero exception can embed the API key,
        # library id, and request URL. Log it server-side; the client gets
        # a generic "check failed" so the wizard's status pill stays red
        # without exposing internals.
        logger.warning("Zotero /test probe failed", exc_info=True)
        return {"ok": False, "message": "credential check failed"}
    return {"ok": True, "message": "ok"}
