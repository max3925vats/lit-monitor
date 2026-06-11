"""Bundle G (v0.9): HTTP endpoints for domain focus extraction.

Endpoints
---------
GET    /api/domain                       — return current extraction (grouped by field_type)
POST   /api/domain/analyze               — trigger LLM extraction; persist result
POST   /api/domain/{row_id}/confirm      — mark one item user_confirmed=1
POST   /api/domain/{row_id}/reject       — mark one item user_confirmed=0
GET    /domain                           — render review UI (HTMX)

All persistent state lives in state.db::domain_focus_extracted.
"""
from __future__ import annotations

import logging
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from lit_monitor.domain.extract import analyze_domain
from lit_monitor.server import config_io
from lit_monitor.server.runtime import get_runtime

logger = logging.getLogger(__name__)

router = APIRouter()


def _read_domain_text() -> str:
    """Read the domain_focus paragraph from config/domain_context.yaml.

    Raises HTTPException(404) when the file is missing and HTTPException(422)
    when it loads but contains no usable text. Tolerates both the new
    ``domain_focus:`` key and a bare-string YAML body for forward-compat.
    """
    # A3-4: resolve against the project's config dir (config_io.CONFIG_DIR)
    # rather than a CWD-relative literal, so the route reads the same file the
    # rest of the server uses regardless of the process working directory.
    # Reference the attribute at call time so tests can monkeypatch CONFIG_DIR.
    path = config_io.CONFIG_DIR / "domain_context.yaml"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="domain_context.yaml not found",
        )
    data: Any
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"domain_context.yaml is not valid YAML: {exc}",
        ) from exc

    # Accept both {domain_focus: "..."} (current schema) and bare strings
    # (in case a user paste-edits the file).
    if isinstance(data, dict):
        text = data.get("domain_focus") or data.get("domain_context") or ""
    elif isinstance(data, str):
        text = data
    else:
        text = ""

    text = (text or "").strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="domain_context.yaml is empty — fill in 'domain_focus'.",
        )
    return text


@router.get("/api/domain")
def get_domain_extraction() -> dict:
    """Return current domain focus extraction grouped by plural field_type.

    Shape:
        {
          "topics":          [{id, value, user_confirmed, last_analyzed_at}, ...],
          "methods":         [...],
          "materials":       [...],
          "adjacent_fields": [...],
          "exclusions":      [...]
        }

    Always returns all five keys; lists may be empty.
    """
    return get_runtime().state_db.list_domain_extraction()


@router.post("/api/domain/analyze")
def post_domain_analyze() -> dict:
    """Run LLM extraction on the current domain_context.yaml; persist result.

    Returns:
        ``{"ok": True, "extraction": {...}}`` on success.

    Errors:
        404 — config/domain_context.yaml not found.
        422 — file present but empty / invalid YAML.
        502 — LLM extraction failed (see server logs for the breadcrumb).
    """
    text = _read_domain_text()
    extraction = analyze_domain(text)
    if extraction is None:
        raise HTTPException(
            status_code=502,
            detail="LLM extraction failed — check server logs",
        )

    get_runtime().state_db.save_domain_extraction(extraction)
    return {"ok": True, "extraction": extraction}


@router.post("/api/domain/{row_id}/confirm")
def post_confirm(row_id: int) -> dict:
    """Mark one extracted item as user_confirmed=1."""
    get_runtime().state_db.set_domain_extraction_confirmed(row_id, True)
    return {"ok": True, "row_id": row_id, "confirmed": True}


@router.post("/api/domain/{row_id}/reject")
def post_reject(row_id: int) -> dict:
    """Mark one extracted item as user_confirmed=0."""
    get_runtime().state_db.set_domain_extraction_confirmed(row_id, False)
    return {"ok": True, "row_id": row_id, "confirmed": False}


@router.get("/domain", response_class=HTMLResponse)
def get_domain_page(request: Request) -> HTMLResponse:
    """Render the domain review UI."""
    # Imported here so create_app() can wire `templates` first.
    from lit_monitor.server.app import templates

    extraction = get_runtime().state_db.list_domain_extraction()
    return templates.TemplateResponse(
        request,
        "domain/index.html",
        {"extraction": extraction},
    )


__all__ = ["router"]
