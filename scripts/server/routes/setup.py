"""Setup wizard routes.

Owns the ``/setup`` wizard landing and the per-step pages. F2.4 introduces
step 1 (credentials TOML editor) only; steps 2-8 are shown on the landing
as ``todo`` placeholders to be filled in by later F-series tasks.

All TemplateResponse calls use the Starlette 1.x modern signature
``templates.TemplateResponse(request, name, context)`` — the legacy
positional-dict form is rejected by Starlette 1.x (confirmed broken in F1.2).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from scripts.server.app import templates
from scripts.server.config_io import load_config, load_secrets, save_config, save_secrets
from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/setup", tags=["setup"])

# F3.3: the build-vocabulary subprocess now lives in the shared process
# registry at ``runtime.processes["vocabulary"]``. The previous module-level
# globals (``_vocab_proc``, ``_vocab_lock``, ``_vocab_output``) are gone.

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
    s = dict(out.get("semantic_scholar", {}))

    def _maybe_set(section: dict, key: str, value: str) -> None:
        if not value or value.startswith(_MASK_PREFIX):
            return
        section[key] = value

    _maybe_set(z, "api_key", form["zotero_api_key"])
    _maybe_set(z, "library_id", form["zotero_library_id"])
    _maybe_set(p, "email", form["pubmed_email"])
    _maybe_set(o, "api_key", form["ollama_api_key"])
    # M3: optional Semantic Scholar API key (raises S2 rate limit from 100/5m → 5000/5m).
    # Form may omit this field for back-compat with older tests; default to "".
    _maybe_set(s, "api_key", form.get("s2_api_key", ""))

    if z:
        out["zotero"] = z
    if p:
        out["pubmed"] = p
    if o:
        out["ollama"] = o
    if s:
        out["semantic_scholar"] = s
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


# Extraction wizard (F2.6) — modes owned by the wizard and the only seven
# per-mode keys the wizard surfaces. All other keys (pass_strategy, chunk_chars,
# OCR settings, per-pass model overrides, ...) round-trip unchanged.
_WIZARD_MODES = ("brain_build", "ingestion", "build_vocabulary")

_MODE_DEFAULTS: dict[str, Any] = {
    "provider": "ollama",
    "ollama_host": "http://localhost:11434",
    "model": "",
    "temperature": 0.1,
    "timeout": 7200,
    "think": False,
    "litellm_model": "",
}


def _mode_view(existing: dict[str, Any], mode: str) -> dict[str, Any]:
    """Build the form ctx for one mode, falling back to defaults when missing."""
    section = existing.get(mode, {}) or {}
    return {k: section.get(k, v) for k, v in _MODE_DEFAULTS.items()}


def _merge_extraction(
    existing: dict[str, Any], form: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Merge per-mode form fields into existing extraction.yaml.

    Preserves all unrelated keys inside each mode block (pass_strategy,
    max_tokens_per_call, etc.) and every non-wizard top-level section
    (embeddings, reranker, comparison_models, ...).
    """
    out: dict[str, Any] = dict(existing)
    for mode in _WIZARD_MODES:
        cur = dict(out.get(mode, {}) or {})
        new = form.get(mode, {})
        cur["provider"] = new["provider"]
        cur["ollama_host"] = new["ollama_host"]
        cur["model"] = new["model"]
        cur["temperature"] = new["temperature"]
        cur["timeout"] = new["timeout"]
        cur["think"] = new["think"]
        if new["provider"] == "litellm":
            cur["litellm_model"] = new["litellm_model"]
        else:
            # When switching away from litellm, drop the stale key so loaders
            # don't pick up the previous litellm_model value.
            cur.pop("litellm_model", None)
        out[mode] = cur
    return out


def _validate_extraction(data: dict[str, dict[str, Any]]) -> list[str]:
    """Return a list of human-readable errors. Empty list means OK."""
    errors: list[str] = []
    for mode, section in data.items():
        prefix = f"[{mode}]"
        provider = section["provider"]
        if provider not in ("ollama", "litellm"):
            errors.append(f"{prefix} provider must be 'ollama' or 'litellm'.")
        if not section["model"]:
            errors.append(f"{prefix} model is required.")
        t = section["temperature"]
        if not (0.0 <= t <= 1.0):
            errors.append(f"{prefix} temperature must be in [0.0, 1.0].")
        if section["timeout"] <= 0:
            errors.append(f"{prefix} timeout must be > 0 seconds.")
        if provider == "litellm" and not section["litellm_model"]:
            errors.append(
                f"{prefix} litellm_model is required when provider=litellm."
            )
    return errors


# Topics + domain (F2.7, F2.8) — wizard-owned helpers.
_DATABASES = ("pubmed", "arxiv", "scopus")


def _merge_topics(form_searches: list[dict[str, Any]]) -> dict[str, Any]:
    """Build topics.yaml content from the form rows.

    Unlike paths/extraction, topics.yaml has no non-wizard keys to preserve —
    the wizard owns the whole file (the LLM-appended discovered_topics
    entries are still in `searches:`, so they'll round-trip if the form
    includes them).
    """
    return {"searches": form_searches}


def _validate_topics(searches: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not searches:
        errors.append("At least one search entry is required.")
        return errors
    seen: set[str] = set()
    for i, s in enumerate(searches, start=1):
        prefix = f"Search #{i}"
        name = s.get("name", "").strip()
        query = s.get("query", "").strip()
        databases = s.get("databases", [])
        if not name:
            errors.append(f"{prefix}: name is required.")
        elif name in seen:
            errors.append(f"{prefix}: name '{name}' is duplicated.")
        else:
            seen.add(name)
        if not query:
            errors.append(f"{prefix}: query is required.")
        if not databases:
            errors.append(f"{prefix}: at least one database is required.")
        for db in databases:
            if db not in _DATABASES:
                errors.append(f"{prefix}: invalid database '{db}'.")
    return errors


def _validate_domain(text: str) -> list[str]:
    errors: list[str] = []
    if not text.strip():
        errors.append("Domain description is required.")
    elif len(text.split()) > 800:
        # Hard cap — soft cap (400 words warning) is client-side only.
        errors.append("Domain description is too long (over 800 words).")
    return errors


def _load_concepts() -> tuple[dict[str, Any] | None, str | None]:
    """Try concepts.yaml first, then concepts_draft.yaml.

    Returns ``(data, source)`` where ``source`` is ``"concepts"`` or
    ``"concepts_draft"``, or ``(None, None)`` if neither file has any
    themes/unclustered content on disk yet.
    """
    for source in ("concepts", "concepts_draft"):
        try:
            data = load_config(source)
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001 — defensive: yaml parse, etc.
            logger.debug("Failed to load %s", source, exc_info=True)
            continue
        if data and (data.get("themes") or data.get("unclustered")):
            return data, source
    return None, None


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
    step3_ok = False
    try:
        ext = load_config("extraction")
        step3_ok = all(
            (ext.get(mode, {}) or {}).get("model") for mode in _WIZARD_MODES
        )
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 — defensive: yaml parse, etc.
        logger.debug("step3 status check failed: %s", exc)
    step4_ok = False
    try:
        topics = load_config("topics")
        searches = topics.get("searches", []) if isinstance(topics, dict) else []
        step4_ok = bool(searches)
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 — defensive: yaml parse, etc.
        logger.debug("step4 status check failed: %s", exc)
    step5_ok = False
    try:
        dom = load_config("domain_context")
        focus = dom.get("domain_focus", "") if isinstance(dom, dict) else ""
        step5_ok = bool(str(focus).strip())
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 — defensive: yaml parse, etc.
        logger.debug("step5 status check failed: %s", exc)
    step6_ok = False
    try:
        cdata, _csrc = _load_concepts()
        step6_ok = bool(cdata and cdata.get("themes"))
    except Exception as exc:  # noqa: BLE001 — defensive: yaml parse, etc.
        logger.debug("step6 status check failed: %s", exc)
    step7_ok = False
    try:
        rs = load_config("researchers")
        # Optional step: file exists + parses as a dict is enough (empty list OK).
        step7_ok = isinstance(rs, dict)
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 — defensive: yaml parse, etc.
        logger.debug("step7 status check failed: %s", exc)
    step8_ok = False
    try:
        ir = load_config("item_routing")
        step8_ok = isinstance(ir, dict) and "routes" in ir
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 — defensive: yaml parse, etc.
        logger.debug("step8 status check failed: %s", exc)
    return [
        {"num": 1, "title": "Credentials", "status": "ok" if step1_ok else "missing", "url": "/setup/step-1", "optional": False},
        {"num": 2, "title": "Paths (vault + collection)", "status": "ok" if step2_ok else "missing", "url": "/setup/step-2", "optional": False},
        {"num": 3, "title": "Extraction (provider + model)", "status": "ok" if step3_ok else "missing", "url": "/setup/step-3", "optional": False},
        {"num": 4, "title": "Topics (recurring searches)", "status": "ok" if step4_ok else "missing", "url": "/setup/step-4", "optional": False},
        {"num": 5, "title": "Domain context", "status": "ok" if step5_ok else "missing", "url": "/setup/step-5", "optional": False},
        {"num": 6, "title": "Concepts (vocabulary)", "status": "ok" if step6_ok else "missing", "url": "/setup/step-6", "optional": True},
        {"num": 7, "title": "Researchers", "status": "ok" if step7_ok else "missing", "url": "/setup/step-7", "optional": True},
        {"num": 8, "title": "Item routing (advanced)", "status": "ok" if step8_ok else "missing", "url": "/setup/step-8", "optional": False},
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
    s2 = secrets.get("semantic_scholar", {}) if isinstance(secrets, dict) else {}

    ctx = {
        "z_api_key_tail": _key_tail(zotero.get("api_key", "")),
        "z_library_id": zotero.get("library_id", ""),
        "pubmed_email": pubmed.get("email", ""),
        "ollama_api_key_tail": _key_tail(ollama.get("api_key", "")),
        "s2_api_key_tail": _key_tail(s2.get("api_key", "")),
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
    s2_api_key: str = Form(""),
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
            "s2_api_key": s2_api_key.strip(),
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


@router.post("/api/paths/collection", response_class=HTMLResponse)
def update_collection(
    request: Request, collection_name: str = Form("")
) -> HTMLResponse:
    """F3.4: rewrite only ``zotero.collection_name`` in paths.yaml.

    Used by the brain-build dashboard's collection switcher. All other
    keys in paths.yaml round-trip untouched via the atomic ``save_config``
    helper. Response carries ``HX-Refresh: true`` so HTMX does a full
    window reload — the dashboard's per-paper progress block must rebind
    to the new collection.
    """
    name = collection_name.strip()
    if not name:
        return HTMLResponse(
            '<div class="card danger">Collection name is required.</div>',
            status_code=400,
        )
    try:
        paths = load_config("paths")
    except FileNotFoundError:
        return HTMLResponse(
            '<div class="card danger">paths.yaml not configured yet. '
            'Complete setup step 2 first.</div>',
            status_code=400,
        )
    z = dict(paths.get("zotero", {}) or {})
    z["collection_name"] = name
    paths["zotero"] = z
    try:
        save_config("paths", paths)
    except OSError as exc:
        logger.error("update_collection: write failed: %s", exc)
        return HTMLResponse(
            f'<div class="card danger">Could not save: {exc}</div>',
            status_code=500,
        )
    from html import escape

    return HTMLResponse(
        f'<div class="card success">Collection updated to '
        f'<code>{escape(name)}</code>. Reloading…</div>',
        headers={"HX-Refresh": "true"},
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


@router.get("/step-3", response_class=HTMLResponse)
def step_extraction_form(request: Request) -> HTMLResponse:
    """Step 3 form: extraction.yaml editor (per-mode LLM provider/model/tuning)."""
    try:
        existing = load_config("extraction")
    except FileNotFoundError:
        existing = {}
    ctx = {
        "current_step": 3,
        "modes": [(mode, _mode_view(existing, mode)) for mode in _WIZARD_MODES],
    }
    return templates.TemplateResponse(request, "setup/step_extraction.html", ctx)


@router.post("/api/extraction", response_class=HTMLResponse)
def save_extraction(
    request: Request,
    # brain_build
    brain_build__provider: str = Form("ollama"),
    brain_build__ollama_host: str = Form(""),
    brain_build__model: str = Form(""),
    brain_build__temperature: float = Form(0.1),
    brain_build__timeout: int = Form(7200),
    brain_build__think: bool = Form(False),
    brain_build__litellm_model: str = Form(""),
    # ingestion
    ingestion__provider: str = Form("ollama"),
    ingestion__ollama_host: str = Form(""),
    ingestion__model: str = Form(""),
    ingestion__temperature: float = Form(0.1),
    ingestion__timeout: int = Form(7200),
    ingestion__think: bool = Form(False),
    ingestion__litellm_model: str = Form(""),
    # build_vocabulary
    build_vocabulary__provider: str = Form("ollama"),
    build_vocabulary__ollama_host: str = Form(""),
    build_vocabulary__model: str = Form(""),
    build_vocabulary__temperature: float = Form(0.1),
    build_vocabulary__timeout: int = Form(7200),
    build_vocabulary__think: bool = Form(False),
    build_vocabulary__litellm_model: str = Form(""),
) -> HTMLResponse:
    """Save extraction.yaml, preserving non-wizard keys + sections."""
    form_data: dict[str, dict[str, Any]] = {
        "brain_build": {
            "provider": brain_build__provider.strip(),
            "ollama_host": brain_build__ollama_host.strip(),
            "model": brain_build__model.strip(),
            "temperature": brain_build__temperature,
            "timeout": brain_build__timeout,
            "think": brain_build__think,
            "litellm_model": brain_build__litellm_model.strip(),
        },
        "ingestion": {
            "provider": ingestion__provider.strip(),
            "ollama_host": ingestion__ollama_host.strip(),
            "model": ingestion__model.strip(),
            "temperature": ingestion__temperature,
            "timeout": ingestion__timeout,
            "think": ingestion__think,
            "litellm_model": ingestion__litellm_model.strip(),
        },
        "build_vocabulary": {
            "provider": build_vocabulary__provider.strip(),
            "ollama_host": build_vocabulary__ollama_host.strip(),
            "model": build_vocabulary__model.strip(),
            "temperature": build_vocabulary__temperature,
            "timeout": build_vocabulary__timeout,
            "think": build_vocabulary__think,
            "litellm_model": build_vocabulary__litellm_model.strip(),
        },
    }

    errors = _validate_extraction(form_data)
    if errors:
        return templates.TemplateResponse(
            request,
            "setup/_extraction_result.html",
            {"ok": False, "errors": errors},
        )

    try:
        existing = load_config("extraction")
    except FileNotFoundError:
        existing = {}

    merged = _merge_extraction(existing, form_data)

    try:
        save_config("extraction", merged)
    except OSError as exc:
        logger.error("save_extraction: write failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup/_extraction_result.html",
            {"ok": False, "errors": [f"Could not save: {exc}"]},
        )

    return templates.TemplateResponse(
        request,
        "setup/_extraction_result.html",
        {"ok": True, "errors": None},
    )


@router.get("/api/ollama-test", response_class=HTMLResponse)
def ollama_test(request: Request, host: str = "") -> HTMLResponse:
    """Probe an Ollama host's /api/tags. Returns an inline HTMX status pill."""
    if not host.strip():
        return HTMLResponse('<span class="pill warning">no host provided</span>')

    import httpx

    url = host.strip().rstrip("/") + "/api/tags"
    try:
        r = httpx.get(url, timeout=5.0, headers={"User-Agent": "lit-monitor"})
    except httpx.RequestError as exc:
        return HTMLResponse(
            f'<span class="pill danger">unreachable: {type(exc).__name__}</span>'
        )

    if r.status_code == 200:
        try:
            n_models = len(r.json().get("models", []))
        except ValueError:
            n_models = 0
        return HTMLResponse(
            f'<span class="pill success">reachable — {n_models} models</span>'
        )
    return HTMLResponse(
        f'<span class="pill warning">HTTP {r.status_code}</span>'
    )


# --- F2.7: topics ---


@router.get("/step-4", response_class=HTMLResponse)
def step_topics_form(request: Request) -> HTMLResponse:
    """Step 4 form: topics.yaml row editor."""
    try:
        topics = load_config("topics")
    except FileNotFoundError:
        topics = {}
    searches = topics.get("searches", []) or []
    # Ensure each entry has the keys the template expects.
    for s in searches:
        if not s.get("name"):
            s["name"] = ""
        if not s.get("query"):
            s["query"] = ""
        if not s.get("databases"):
            s["databases"] = []
    ctx = {
        "current_step": 4,
        "searches": searches,
        "all_databases": _DATABASES,
    }
    return templates.TemplateResponse(request, "setup/step_topics.html", ctx)


@router.post("/api/topics", response_class=HTMLResponse)
async def save_topics(request: Request) -> HTMLResponse:
    """Topics POST is parsed manually because the row count is dynamic."""
    form = await request.form()
    # Find the row count by scanning name_N keys.
    indices: set[int] = set()
    for key in form.keys():
        if key.startswith("name_"):
            try:
                indices.add(int(key.split("_", 1)[1]))
            except ValueError:
                pass
    searches: list[dict[str, Any]] = []
    for i in sorted(indices):
        name = str(form.get(f"name_{i}", "")).strip()
        query = str(form.get(f"query_{i}", "")).strip()
        # Multi-checkbox: getlist returns all values for the key
        databases = [str(v) for v in form.getlist(f"databases_{i}")]
        # Skip rows that are entirely empty.
        if not (name or query or databases):
            continue
        searches.append({"name": name, "query": query, "databases": databases})

    errors = _validate_topics(searches)
    if errors:
        return templates.TemplateResponse(
            request,
            "setup/_topics_result.html",
            {"ok": False, "errors": errors},
        )

    try:
        save_config("topics", _merge_topics(searches))
    except OSError as exc:
        logger.error("save_topics: write failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup/_topics_result.html",
            {"ok": False, "errors": [f"Could not save: {exc}"]},
        )

    return templates.TemplateResponse(
        request,
        "setup/_topics_result.html",
        {"ok": True, "errors": None},
    )


@router.get("/api/topics/new-row", response_class=HTMLResponse)
def topics_new_row(request: Request, idx: int = 0) -> HTMLResponse:
    """Return the HTML fragment for a fresh row at the given index."""
    return templates.TemplateResponse(
        request,
        "setup/_topics_row.html",
        {
            "idx": idx,
            "search": {"name": "", "query": "", "databases": []},
            "all_databases": _DATABASES,
        },
    )


# --- F2.8: domain ---


@router.get("/step-5", response_class=HTMLResponse)
def step_domain_form(request: Request) -> HTMLResponse:
    """Step 5 form: domain_context.yaml single-textarea editor."""
    try:
        existing = load_config("domain_context")
    except FileNotFoundError:
        existing = {}
    ctx = {
        "current_step": 5,
        "domain_focus": existing.get("domain_focus", "") or "",
    }
    return templates.TemplateResponse(request, "setup/step_domain.html", ctx)


@router.post("/api/domain", response_class=HTMLResponse)
def save_domain(
    request: Request,
    domain_focus: str = Form(""),
) -> HTMLResponse:
    text = domain_focus.strip()
    errors = _validate_domain(text)
    if errors:
        return templates.TemplateResponse(
            request,
            "setup/_domain_result.html",
            {"ok": False, "errors": errors},
        )
    try:
        save_config("domain_context", {"domain_focus": text})
    except OSError as exc:
        logger.error("save_domain: write failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup/_domain_result.html",
            {"ok": False, "errors": [f"Could not save: {exc}"]},
        )
    return templates.TemplateResponse(
        request,
        "setup/_domain_result.html",
        {"ok": True, "errors": None},
    )


# --- F2.9: concepts (read-only view + Regenerate trigger) ---------------


@router.get("/step-6", response_class=HTMLResponse)
def step_concepts_form(request: Request) -> HTMLResponse:
    """Step 6: read-only concepts view + Regenerate trigger."""
    data, source = _load_concepts()
    themes = (data or {}).get("themes", []) or []
    unclustered = (data or {}).get("unclustered", []) or []
    slot = get_runtime().processes["vocabulary"]
    ctx = {
        "current_step": 6,
        "themes": themes,
        "unclustered": unclustered,
        "source": source,
        "is_running": slot.is_running(),
    }
    return templates.TemplateResponse(request, "setup/step_concepts.html", ctx)


@router.post("/api/build-vocabulary/start", response_class=HTMLResponse)
async def build_vocab_start(request: Request) -> HTMLResponse:
    """Spawn the build-vocabulary subprocess. Returns an HTMX-swappable card."""
    slot = get_runtime().processes["vocabulary"]
    async with slot.lock:
        if slot.is_running():
            return HTMLResponse(
                '<div class="card warning">'
                '<p>Vocabulary build already in progress '
                f'(pid {slot.process.pid}).</p>'
                '<p><a href="/setup/step-6/progress" class="button">'
                'View live output →</a></p>'
                '</div>'
            )
        slot.output = []  # reset per-run buffer
        # Repo root: scripts/server/routes/setup.py → parents[3] is repo root.
        repo_root = Path(__file__).resolve().parents[3]
        slot.process = await asyncio.create_subprocess_exec(
            "uv", "run", "lit-monitor", "build-vocabulary",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(repo_root),
        )
        slot.started_at = datetime.now(UTC).isoformat(timespec="seconds")
    return HTMLResponse(
        '<div class="card success">'
        f'<p>Vocabulary build started (pid {slot.process.pid}).</p>'
        '<p><a href="/setup/step-6/progress" class="button">'
        'View live output →</a></p>'
        '</div>'
    )


@router.get("/step-6/progress", response_class=HTMLResponse)
def step_concepts_progress(request: Request) -> HTMLResponse:
    """Live SSE-streamed stdout viewer for the running build-vocabulary subprocess."""
    slot = get_runtime().processes["vocabulary"]
    return templates.TemplateResponse(
        request,
        "setup/_concepts_progress.html",
        {
            "current_step": 6,
            "pid": slot.process.pid if slot.process else None,
        },
    )


@router.get("/api/build-vocabulary/stream")
async def build_vocab_stream(request: Request):
    """SSE stream of the build-vocabulary subprocess stdout.

    Emits ``progress`` events for each line and a final ``done`` event with
    the exit code. If no build is running, emits a single ``error`` event.
    """
    from sse_starlette.sse import EventSourceResponse

    slot = get_runtime().processes["vocabulary"]

    async def gen():
        proc = slot.process
        if proc is None or proc.stdout is None:
            yield {"event": "error", "data": "no build process running"}
            return
        while True:
            if await request.is_disconnected():
                return
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            slot.append_line(text)
            yield {"event": "progress", "data": text}
        rc = await proc.wait()
        yield {"event": "done", "data": f"exit code: {rc}"}

    return EventSourceResponse(gen())


@router.post("/api/build-vocabulary/stop", response_class=HTMLResponse)
async def build_vocab_stop() -> HTMLResponse:
    """SIGTERM the build-vocabulary subprocess if running; SIGKILL on timeout."""
    slot = get_runtime().processes["vocabulary"]
    stopped = await slot.stop(timeout=10.0)
    if not stopped:
        return HTMLResponse(
            '<div class="card warning">No vocabulary build in progress.</div>'
        )
    return HTMLResponse(
        '<div class="card success">Vocabulary build stopped.</div>'
    )


# --- F2.10: researchers (optional list editor) -------------------------


def _validate_researchers(researchers: list[dict[str, Any]]) -> list[str]:
    """Researchers list is optional — only flag truly malformed rows.

    No errors for empty list. Warnings (returned as errors) for rows with
    a name but no orcid/scopus_id (tracking won't work).
    """
    errors: list[str] = []
    for i, r in enumerate(researchers, start=1):
        name = (r.get("name") or "").strip()
        if not name:
            continue  # silent-skip empty rows
        orcid = (r.get("orcid") or "").strip()
        scopus_id = (r.get("scopus_id") or "").strip()
        if not orcid and not scopus_id:
            errors.append(
                f"Researcher #{i} ('{name}'): set orcid OR scopus_id to enable tracking."
            )
    return errors


@router.get("/step-7", response_class=HTMLResponse)
def step_researchers_form(request: Request) -> HTMLResponse:
    """Step 7 form: researchers.yaml row editor (optional)."""
    try:
        existing = load_config("researchers")
    except FileNotFoundError:
        existing = {}
    rs = existing.get("researchers", []) or []
    for r in rs:
        for key in ("name", "affiliation", "orcid", "scopus_id", "notes"):
            if not r.get(key):
                r[key] = ""
    return templates.TemplateResponse(
        request,
        "setup/step_researchers.html",
        {"current_step": 7, "researchers": rs},
    )


@router.post("/api/researchers", response_class=HTMLResponse)
async def save_researchers(request: Request) -> HTMLResponse:
    """Researchers POST is parsed manually because the row count is dynamic."""
    form = await request.form()
    indices: set[int] = set()
    for key in form.keys():
        if key.startswith("name_"):
            try:
                n = int(key.split("_", 1)[1])
                if n >= 0:
                    indices.add(n)
            except ValueError:
                pass
    out: list[dict[str, Any]] = []
    for i in sorted(indices):
        name = str(form.get(f"name_{i}", "")).strip()
        if not name:
            continue
        out.append({
            "name": name,
            "affiliation": str(form.get(f"affiliation_{i}", "")).strip(),
            "orcid": str(form.get(f"orcid_{i}", "")).strip(),
            "scopus_id": str(form.get(f"scopus_id_{i}", "")).strip(),
            "notes": str(form.get(f"notes_{i}", "")).strip(),
        })

    errors = _validate_researchers(out)
    # Researchers is optional — warnings are shown but save still proceeds.
    try:
        save_config("researchers", {"researchers": out})
    except OSError as exc:
        logger.error("save_researchers: write failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup/_researchers_result.html",
            {"ok": False, "errors": [f"Could not save: {exc}"], "warnings": []},
        )
    return templates.TemplateResponse(
        request,
        "setup/_researchers_result.html",
        {"ok": True, "errors": [], "warnings": errors, "count": len(out)},
    )


@router.get("/api/researchers/new-row", response_class=HTMLResponse)
def researchers_new_row(request: Request, idx: int = 0) -> HTMLResponse:
    """Return the HTML fragment for a fresh researcher row at the given index."""
    return templates.TemplateResponse(
        request,
        "setup/_researchers_row.html",
        {
            "idx": idx,
            "r": {"name": "", "affiliation": "", "orcid": "", "scopus_id": "", "notes": ""},
        },
    )


# --- F2.11: item_routing (read-only grid + advanced raw YAML edit) -----


def _routes_summary(data: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten the routes table for the read-only grid."""
    out: list[dict[str, str]] = []
    routes = (data or {}).get("routes", {}) or {}
    for item_type, cfg in sorted(routes.items()):
        out.append({
            "item_type": item_type,
            "pipeline": str(cfg.get("pipeline", "")),
            "default_schema": str(cfg.get("default_schema", "")),
            "source_type": str(cfg.get("source_type", "")),
        })
    return out


@router.get("/step-8", response_class=HTMLResponse)
def step_routing_form(request: Request) -> HTMLResponse:
    """Step 8 form: item_routing.yaml read-only grid + raw YAML edit."""
    try:
        data = load_config("item_routing")
    except FileNotFoundError:
        data = {}
    summary = _routes_summary(data)
    raw_yaml = yaml.safe_dump(data or {"routes": {}}, sort_keys=False)
    return templates.TemplateResponse(
        request,
        "setup/step_routing.html",
        {"current_step": 8, "rows": summary, "raw_yaml": raw_yaml},
    )


@router.post("/api/routing", response_class=HTMLResponse)
def save_routing(
    request: Request,
    raw_yaml: str = Form(""),
) -> HTMLResponse:
    """Lint the raw YAML; if it parses AND has a top-level 'routes' key, save it."""
    try:
        parsed = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        return templates.TemplateResponse(
            request,
            "setup/_routing_result.html",
            {"ok": False, "errors": [f"YAML parse error: {exc}"]},
        )
    if not isinstance(parsed, dict) or "routes" not in parsed:
        return templates.TemplateResponse(
            request,
            "setup/_routing_result.html",
            {"ok": False, "errors": ["Top-level 'routes:' key is required."]},
        )
    try:
        save_config("item_routing", parsed)
    except OSError as exc:
        logger.error("save_routing: write failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup/_routing_result.html",
            {"ok": False, "errors": [f"Could not save: {exc}"]},
        )
    return templates.TemplateResponse(
        request,
        "setup/_routing_result.html",
        {"ok": True, "errors": []},
    )


# --- F2.12: completion checklist ---------------------------------------


def _check_vault() -> dict[str, tuple[bool, str]]:
    """Synthetic check: paths.yaml exists AND vault_path is a real dir."""
    try:
        paths = load_config("paths")
    except FileNotFoundError:
        return {"paths_file": (False, "config/paths.yaml not found")}
    vault = paths.get("obsidian", {}).get("vault_path", "")
    if not vault:
        return {"vault_path": (False, "vault_path not set in paths.yaml")}
    p = Path(vault).expanduser()
    if not p.exists():
        return {"vault_path": (False, f"path does not exist: {vault}")}
    if not p.is_dir():
        return {"vault_path": (False, f"not a directory: {vault}")}
    return {"vault_path": (True, f"OK: {vault}")}


@router.get("/complete", response_class=HTMLResponse)
def step_complete(request: Request) -> HTMLResponse:
    """Setup completion landing — each row polls its own status endpoint."""
    return templates.TemplateResponse(
        request, "setup/complete.html", {"current_step": 8}
    )


@router.get("/api/check/{name}", response_class=HTMLResponse)
def setup_check(request: Request, name: str) -> HTMLResponse:
    """Return the inline status snippet for one of the four checks."""
    if name == "secrets":
        from scripts.setup.check_configured import check_configured
        results = check_configured()
    elif name == "ollama":
        from scripts.setup.check_ollama import check_ollama
        # Pass the configured brain_build model so the ollama check can probe it.
        model = None
        try:
            ext = load_config("extraction")
            model = (ext.get("brain_build") or {}).get("model")
        except Exception:  # noqa: BLE001 — defensive: any load failure
            pass
        results = check_ollama(model=model)
    elif name == "zotero":
        from scripts.setup.check_zotero import check_zotero
        results = check_zotero()
    elif name == "vault":
        results = _check_vault()
    else:
        return HTMLResponse(
            f'<span class="pill danger">unknown check: {name}</span>',
            status_code=404,
        )

    # Aggregate the sub-checks into one pill.
    all_ok = all(ok for ok, _msg in results.values())
    failing = [n for n, (ok, _msg) in results.items() if not ok]
    if all_ok:
        return HTMLResponse(
            f'<span class="pill success">all OK ({len(results)} checks)</span>'
        )
    return HTMLResponse(
        f'<span class="pill danger">{len(failing)} failing: '
        f'{", ".join(failing[:3])}'
        + (f' (+{len(failing) - 3} more)' if len(failing) > 3 else "")
        + '</span>'
    )
