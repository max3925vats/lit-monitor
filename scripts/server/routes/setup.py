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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from scripts.server.app import templates
from scripts.server.config_io import load_config, load_secrets, save_config, save_secrets

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


def _merge_paths(existing: dict[str, Any], form: dict[str, str]) -> dict[str, Any]:
    """Merge form fields into existing paths.yaml, preserving unrelated keys.

    The form owns these five fields:
      - obsidian.vault_path
      - zotero.collection_name
      - zotero.library_id  (mirror of secrets, required in paths.yaml)
      - zotero.library_type
      - zotero.local_storage_path

    All other keys (papers_folder, books_folder, state_db.*, logs.*, etc.)
    are preserved verbatim from ``existing``, with sensible defaults
    backfilled only when missing.
    """
    out: dict[str, Any] = dict(existing)
    z = dict(out.get("zotero", {}))
    o = dict(out.get("obsidian", {}))

    o["vault_path"] = form["vault_path"]
    z["collection_name"] = form["collection_name"]
    z["library_id"] = form["library_id"]
    z["library_type"] = form["library_type"]
    z["local_storage_path"] = form["local_storage_path"]

    # Backfill the non-user-tunable obsidian sub-folder defaults if absent.
    o.setdefault("papers_folder", "Literature/Papers")
    o.setdefault("books_folder", "Literature/Books")
    o.setdefault("digests_folder", "Literature/Digests")
    o.setdefault("connections_folder", "Literature/Connections")

    out["zotero"] = z
    out["obsidian"] = o

    # Same defaulting for state_db and logs, only if missing.
    out.setdefault("state_db", {"path": "~/.config/lit-monitor/state.db"})
    out.setdefault("logs", {"path": "./logs", "retention_days": 90})

    return out


def _validate_paths(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors. Empty list means OK."""
    errors: list[str] = []
    o = data.get("obsidian", {})
    z = data.get("zotero", {})

    vault = o.get("vault_path", "")
    if not vault:
        errors.append("Obsidian vault path is required.")
    else:
        p = Path(vault).expanduser()
        if not p.exists():
            errors.append(f"Vault path does not exist: {vault}")
        elif not p.is_dir():
            errors.append(f"Vault path is not a directory: {vault}")

    if not z.get("library_id"):
        errors.append("Zotero library ID is required.")
    elif not str(z["library_id"]).isdigit():
        errors.append("Zotero library ID must be numeric.")

    if not z.get("collection_name"):
        errors.append("Zotero collection name is required.")

    if z.get("library_type") not in ("user", "group"):
        errors.append("Library type must be 'user' or 'group'.")

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
    step2_ok = False
    try:
        paths = load_config("paths")
        vault = paths.get("obsidian", {}).get("vault_path", "")
        if vault:
            p = Path(vault).expanduser()
            step2_ok = p.exists() and p.is_dir()
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 — defensive: yaml parse, etc.
        logger.debug("step2 status check failed: %s", exc)
    return [
        {"num": 1, "title": "Credentials", "status": "ok" if step1_ok else "missing", "url": "/setup/step-1"},
        {"num": 2, "title": "Paths (vault + collection)", "status": "ok" if step2_ok else "missing", "url": "/setup/step-2"},
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


@router.get("/step-2", response_class=HTMLResponse)
def step_paths_form(request: Request) -> HTMLResponse:
    """Step 2 form: paths.yaml editor (vault picker + collection dropdown)."""
    try:
        paths = load_config("paths")
    except FileNotFoundError:
        paths = {}

    z = paths.get("zotero", {}) if isinstance(paths, dict) else {}
    o = paths.get("obsidian", {}) if isinstance(paths, dict) else {}

    ctx = {
        "current_step": 2,
        "vault_path": o.get("vault_path", ""),
        "collection_name": z.get("collection_name", ""),
        "library_id": str(z.get("library_id", "")),
        "library_type": z.get("library_type", "user"),
        "local_storage_path": z.get("local_storage_path", "~/Zotero/storage"),
        "home": str(Path.home()),
    }
    return templates.TemplateResponse(request, "setup/step_paths.html", ctx)


@router.post("/api/paths", response_class=HTMLResponse)
def save_paths(
    request: Request,
    vault_path: str = Form(""),
    collection_name: str = Form(""),
    library_id: str = Form(""),
    library_type: str = Form("user"),
    local_storage_path: str = Form("~/Zotero/storage"),
) -> HTMLResponse:
    """Save paths.yaml, preserving non-form keys (papers_folder, state_db, etc.)."""
    try:
        existing = load_config("paths")
    except FileNotFoundError:
        existing = {}

    merged = _merge_paths(
        existing,
        {
            "vault_path": vault_path.strip(),
            "collection_name": collection_name.strip(),
            "library_id": library_id.strip(),
            "library_type": library_type.strip(),
            "local_storage_path": local_storage_path.strip(),
        },
    )

    errors = _validate_paths(merged)
    if errors:
        return templates.TemplateResponse(
            request,
            "setup/_paths_result.html",
            {"ok": False, "errors": errors},
        )

    try:
        save_config("paths", merged)
    except OSError as exc:
        logger.error("save_paths: write failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup/_paths_result.html",
            {"ok": False, "errors": [f"Could not save: {exc}"]},
        )

    return templates.TemplateResponse(
        request,
        "setup/_paths_result.html",
        {"ok": True, "errors": None},
    )


@router.get("/api/collections-options", response_class=HTMLResponse)
def collections_options(request: Request, current: str = "") -> HTMLResponse:
    """HTML-wrapped wrapper around /api/zotero/collections for the dropdown.

    On 503 (creds missing) or any other failure, falls back to a text
    input + banner telling the user to complete step 1 first.
    """
    from scripts.server.routes.zotero import collections as zotero_collections

    names: list[str] = []
    try:
        result = zotero_collections()
        names = [c["name"] for c in result.get("collections", [])]
    except (HTTPException, Exception):  # noqa: BLE001 — fallback path is sensible
        names = []

    from html import escape

    if not names:
        current_esc = escape(current, quote=True)
        html = (
            f'<input type="text" name="collection_name" value="{current_esc}">'
            '<div class="banner warning">Complete step 1 first to see your collections, '
            'or type a name manually.</div>'
        )
    else:
        opts = "\n".join(
            f'<option value="{escape(n, quote=True)}"'
            f'{" selected" if n == current else ""}>{escape(n)}</option>'
            for n in names
        )
        html = opts

    return HTMLResponse(html)


@router.get("/fs-modal", response_class=HTMLResponse)
def fs_modal(request: Request, path: str) -> HTMLResponse:
    """Render the directory browser modal for a given path."""
    from scripts.server.routes.fs import ls

    try:
        result = ls(path=path)
    except HTTPException as exc:
        return templates.TemplateResponse(
            request,
            "setup/_fs_modal.html",
            {"path": path, "parent": None, "entries": [], "error": exc.detail},
        )
    return templates.TemplateResponse(
        request,
        "setup/_fs_modal.html",
        result,  # {path, parent, entries}
    )


@router.get("/fs-modal-close", response_class=HTMLResponse)
def fs_modal_close(request: Request) -> HTMLResponse:
    """Return an empty fragment to close the modal."""
    return HTMLResponse("")
