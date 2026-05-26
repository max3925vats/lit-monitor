"""Weekly-run dashboard read-only view.

GET /weekly-lit-run — last run summary + recent runs table + today's digest viewer.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from scripts.server.app import templates
from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)
router = APIRouter(tags=["weekly"])


def _safe_runtime():
    try:
        return get_runtime()
    except Exception:
        return None


def _safe_db():
    r = _safe_runtime()
    if r is None:
        return None
    try:
        return r.state_db
    except Exception:
        return None


def _vault_root() -> Path | None:
    """Resolve the configured Obsidian vault path."""
    r = _safe_runtime()
    if r is None:
        return None
    try:
        vault = r.config.obsidian.vault_path
    except Exception:
        return None
    if not vault:
        return None
    return Path(vault).expanduser()


def _digests_folder_name() -> str:
    r = _safe_runtime()
    if r is None:
        return "Literature/Digests"
    try:
        return getattr(r.config.obsidian, "digests_folder", "Literature/Digests")
    except Exception:
        return "Literature/Digests"


def _todays_digest() -> tuple[Path | None, str | None]:
    """Locate today's digest file, if any. Returns (path, content_or_None).

    The weekly pipeline writes to ``{vault}/{digests_folder}/Discovery_{YYYY-MM-DD}.md``
    using ``date.today()`` (local timezone). Falls back to the most recent
    ``Discovery_*.md`` if today's exact filename is missing.
    """
    vault = _vault_root()
    if vault is None:
        return None, None
    folder = vault / _digests_folder_name()
    if not folder.exists() or not folder.is_dir():
        return None, None
    today = date.today().strftime("%Y-%m-%d")
    exact = folder / f"Discovery_{today}.md"
    if exact.exists():
        try:
            return exact, exact.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Cannot read digest %s", exact, exc_info=True)
            return exact, None
    # Fallback: newest Discovery_*.md in the folder.
    candidates = sorted(
        folder.glob("Discovery_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, None
    newest = candidates[0]
    try:
        return newest, newest.read_text(encoding="utf-8")
    except OSError:
        return newest, None


@router.get("/weekly-lit-run", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    db = _safe_db()
    recent_runs: list[dict] = []
    last_run: dict | None = None
    if db is not None:
        try:
            recent_runs = db.get_recent_runs_by_type("weekly", limit=10)
            if recent_runs:
                last_run = recent_runs[0]
        except Exception:
            logger.warning("get_recent_runs_by_type failed", exc_info=True)

    digest_path, digest_content = _todays_digest()
    return templates.TemplateResponse(
        request,
        "weekly/index.html",
        {
            "last_run": last_run,
            "recent_runs": recent_runs,
            "digest_path": str(digest_path) if digest_path else None,
            "digest_content": digest_content,
            "db_unavailable": db is None,
        },
    )
