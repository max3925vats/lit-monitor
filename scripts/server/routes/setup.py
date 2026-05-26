"""Setup wizard routes.

Owns the ``/setup`` wizard landing and the per-step pages. F2.4 introduces
step 1 (credentials TOML editor) only; steps 2-8 are shown on the landing
as ``todo`` placeholders to be filled in by later F-series tasks.

All TemplateResponse calls use the Starlette 1.x modern signature
``templates.TemplateResponse(request, name, context)`` — the legacy
positional-dict form is rejected by Starlette 1.x (confirmed broken in F1.2).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from scripts.server.app import templates
from scripts.server.config_io import load_secrets, save_secrets

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/setup", tags=["setup"])

# Sentinel prefix used to render masked secret tails in the form's value
# attribute. ``_merge_credentials`` treats any submission starting with this
# prefix as "unchanged — keep the existing stored value".
_MASK_PREFIX = "•" * 8  # eight bullets


def _key_tail(value: str) -> str:
    """Return a masked display tail for a stored secret, or empty string.

    Empty / missing input → empty string so the input field renders empty
    and the user can type a fresh key without first deleting a placeholder.
    """
    if not value:
        return ""
    return _MASK_PREFIX + str(value)[-4:]


def _merge_credentials(existing: dict[str, Any], form: dict[str, str]) -> dict[str, Any]:
    """Merge form fields into existing secrets, preserving unchanged values.

    A form field is treated as "unchanged" when it is empty OR when it
    starts with the masked-tail prefix (because the form pre-fills masked
    tails and submitting the masked tail must NOT overwrite the real key).
    """
    out: dict[str, Any] = dict(existing)
    z = dict(out.get("zotero", {}))
    p = dict(out.get("pubmed", {}))
    o = dict(out.get("ollama", {}))

    def _maybe_set(section: dict, key: str, value: str) -> None:
        if not value or value.startswith(_MASK_PREFIX):
            return
        section[key] = value

    _maybe_set(z, "api_key", form["zotero_api_key"])
    _maybe_set(z, "library_id", form["zotero_library_id"])
    _maybe_set(p, "email", form["pubmed_email"])
    _maybe_set(o, "api_key", form["ollama_api_key"])

    if z:
        out["zotero"] = z
    if p:
        out["pubmed"] = p
    if o:
        out["ollama"] = o
    return out


def _validate_credentials(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors. Empty list means OK."""
    errors: list[str] = []
    z = data.get("zotero", {})
    if not z.get("api_key"):
        errors.append("Zotero API key is required.")
    if not z.get("library_id"):
        errors.append("Zotero library ID is required.")
    elif not str(z["library_id"]).isdigit():
        errors.append("Zotero library ID must be numeric.")
    if not data.get("pubmed", {}).get("email"):
        errors.append("PubMed email is required.")
    return errors


def _build_step_descriptors(checks: dict[str, tuple[bool, str]]) -> list[dict[str, Any]]:
    """Build the wizard-landing card list.

    Only step 1 has a real status mapping in F2.4; steps 2-8 are placeholders
    that future tasks fill in (rendered as ``todo``).
    """
    step1_ok = all(
        checks.get(name, (False, ""))[0]
        for name in (
            "secrets_file",
            "secrets_parse",
            "zotero.api_key",
            "zotero.library_id",
            "pubmed.email",
        )
    )
    return [
        {"num": 1, "title": "Credentials", "status": "ok" if step1_ok else "missing", "url": "/setup/step-1"},
        {"num": 2, "title": "Paths (vault + collection)", "status": "todo", "url": "/setup/step-2"},
        {"num": 3, "title": "Extraction (provider + model)", "status": "todo", "url": "/setup/step-3"},
        {"num": 4, "title": "Topics (weekly searches)", "status": "todo", "url": "/setup/step-4"},
        {"num": 5, "title": "Domain context", "status": "todo", "url": "/setup/step-5"},
        {"num": 6, "title": "Concepts (vocabulary)", "status": "todo", "url": "/setup/step-6"},
        {"num": 7, "title": "Researchers (optional)", "status": "todo", "url": "/setup/step-7"},
        {"num": 8, "title": "Item routing (advanced)", "status": "todo", "url": "/setup/step-8"},
    ]


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def landing(request: Request) -> HTMLResponse:
    """Wizard landing — one status card per step (8 total)."""
    from scripts.setup.check_configured import check_configured

    checks = check_configured()
    steps = _build_step_descriptors(checks)
    return templates.TemplateResponse(
        request,
        "setup/index.html",
        {"steps": steps, "current_step": 0},
    )


@router.get("/step-1", response_class=HTMLResponse)
def step_credentials_form(request: Request) -> HTMLResponse:
    """Step 1 form: credentials TOML editor.

    Pre-fills with masked tails for stored secrets so users can see "yes,
    this is set" without exposing the real key in the page source.
    """
    secrets_error: str | None
    try:
        secrets = load_secrets()
        secrets_error = None
    except PermissionError as exc:
        secrets = {}
        secrets_error = (
            f"Cannot read existing secrets file ({exc}). "
            "Saving will create a new one."
        )

    zotero = secrets.get("zotero", {}) if isinstance(secrets, dict) else {}
    pubmed = secrets.get("pubmed", {}) if isinstance(secrets, dict) else {}
    ollama = secrets.get("ollama", {}) if isinstance(secrets, dict) else {}

    ctx = {
        "z_api_key_tail": _key_tail(zotero.get("api_key", "")),
        "z_library_id": zotero.get("library_id", ""),
        "pubmed_email": pubmed.get("email", ""),
        "ollama_api_key_tail": _key_tail(ollama.get("api_key", "")),
        "secrets_error": secrets_error,
        "current_step": 1,
    }
    return templates.TemplateResponse(request, "setup/step_credentials.html", ctx)


@router.post("/api/credentials", response_class=HTMLResponse)
def save_credentials(
    request: Request,
    zotero_api_key: str = Form(""),
    zotero_library_id: str = Form(""),
    pubmed_email: str = Form(""),
    ollama_api_key: str = Form(""),
) -> HTMLResponse:
    """Save credentials TOML, then live-test against Zotero. Returns a partial."""
    try:
        existing = load_secrets()
    except PermissionError as exc:
        logger.warning("save_credentials: cannot read existing secrets: %s", exc)
        existing = {}

    new_secrets = _merge_credentials(
        existing,
        {
            "zotero_api_key": zotero_api_key.strip(),
            "zotero_library_id": zotero_library_id.strip(),
            "pubmed_email": pubmed_email.strip(),
            "ollama_api_key": ollama_api_key.strip(),
        },
    )

    errors = _validate_credentials(new_secrets)
    if errors:
        return templates.TemplateResponse(
            request,
            "setup/_credentials_result.html",
            {"ok": False, "errors": errors, "test_result": None},
        )

    try:
        save_secrets(new_secrets)
    except OSError as exc:
        logger.error("save_credentials: write failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup/_credentials_result.html",
            {"ok": False, "errors": [f"Could not save file: {exc}"], "test_result": None},
        )

    # Live test using the F2.3 /api/zotero/test helper directly. It never
    # raises — it always returns {ok, message}.
    from scripts.server.routes.zotero import test as zotero_test

    test_result = zotero_test()

    return templates.TemplateResponse(
        request,
        "setup/_credentials_result.html",
        {"ok": True, "errors": None, "test_result": test_result},
    )
