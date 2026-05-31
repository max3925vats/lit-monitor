"""H2: POST /api/ingest — production ingest endpoint.

R28 invariant
-------------
papers.graph_indexed=1 ONLY after BOTH vector embed AND graph add succeed.
_process_paper (brain_build) enforces this. The HTTP layer merely wraps it.

R28 hardening
-------------
If _process_paper raises, the ingest_queue row is marked status='failed'
with the error text BEFORE re-raising — never left orphaned in 'queued'.
An orphaned 'queued' row would appear as an active job to H3's queue
listing and hide the real failure.

Duplicate handling
------------------
The ingest_queue PK is doi, so the duplicate check (SELECT before INSERT)
is the guard. Concurrent requests to the same DOI could race past the
SELECT and collide on the INSERT — SQLite's default serialised-writer
isolation means the second INSERT raises IntegrityError, which will
propagate as a 500 rather than a 409. Accepted limitation for v1; a
proper fix would use INSERT OR IGNORE + check rowcount.
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

router = APIRouter(tags=["ingest"])


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Ingest payload from the Zotero plugin or any web ingestor."""

    doi: str
    title: str
    authors: list[str] = []
    year: int | None = None
    abstract: str | None = None
    zotero_key: str | None = None

    @field_validator("doi")
    @classmethod
    def _validate_doi(cls, v: str) -> str:
        """Reject strings that don't look like a real DOI (must start with 10.)."""
        if not _DOI_RE.match(v.strip()):
            raise ValueError(f"invalid DOI: {v!r} — expected format '10.NNNN/...'")
        return v.strip()

    @field_validator("title")
    @classmethod
    def _validate_title(cls, v: str) -> str:
        """Reject empty or whitespace-only titles."""
        if not v or not v.strip():
            raise ValueError("title cannot be empty or whitespace-only")
        return v


# ---------------------------------------------------------------------------
# Pipeline wrapper (monkeypatched in unit tests)
# ---------------------------------------------------------------------------


def _process_paper(
    doi: str,
    title: str,
    authors: list[str],
    year: int | None,
    abstract: str | None,
) -> None:
    """Thin dispatch to brain_build._process_paper.

    Intentionally thin so unit tests can monkeypatch this symbol without
    importing the full brain_build module (which requires a configured
    runtime, Zotero client, LLM, etc.).

    Production note: brain_build._process_paper has a richer signature
    (zotero_key, item dict, config, state_db, embeddings_db, llm …).
    Wiring a full production call here is deferred to Phase 4d, which will
    add an async worker queue. For now the endpoint accepts and queues the
    request; the actual pipeline invocation is a no-op placeholder that
    will be replaced when the worker lands.
    """
    # Phase 4d: replace this with actual worker dispatch.
    # The R28 invariant is enforced inside brain_build._process_paper;
    # this layer MUST NOT write papers.graph_indexed directly.
    logger.info("H2: _process_paper placeholder called for doi=%s", doi)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/api/ingest", status_code=202)
def ingest(body: IngestRequest) -> JSONResponse:
    """Accept a paper ingest request and add it to the ingest_queue.

    Returns
    -------
    202  — queued successfully.
    409  — this DOI was already submitted (duplicate).
    422  — validation failed (bad DOI format or missing/blank title).
    500  — the ingest pipeline raised an unexpected exception.
    """
    state_db = get_runtime().state_db

    # --- Duplicate check ---
    # Use the _connect() context manager directly; StateDB has no public execute().
    with state_db._connect() as conn:
        existing = conn.execute(
            "SELECT doi FROM ingest_queue WHERE doi = ?",
            (body.doi,),
        ).fetchone()

    if existing is not None:
        return JSONResponse(
            {"status": "duplicate", "paper_id": body.doi},
            status_code=409,
        )

    # --- Insert queue row BEFORE calling pipeline ---
    # Inserting first means the failure path always has a row to update.
    # If the INSERT itself fails (e.g., race-condition duplicate), the
    # IntegrityError propagates as a 500 — acceptable for v1.
    queued_at = _utcnow()
    with state_db._connect() as conn:
        conn.execute(
            "INSERT INTO ingest_queue (doi, status, queued_at) VALUES (?, ?, ?)",
            (body.doi, "queued", queued_at),
        )

    # --- Invoke pipeline with R28 hardening ---
    try:
        _process_paper(body.doi, body.title, body.authors, body.year, body.abstract)
    except Exception as exc:
        # R28 hardening: mark queue row 'failed' BEFORE re-raising so the
        # row is never left orphaned in 'queued'. H3's queue listing reads
        # this table — an orphaned 'queued' row would look like an active job.
        logger.warning("H2: _process_paper failed for doi=%s: %s", body.doi, exc)
        try:
            with state_db._connect() as conn:
                conn.execute(
                    "UPDATE ingest_queue SET status = ?, error = ?, completed_at = ? "
                    "WHERE doi = ?",
                    ("failed", str(exc)[:1000], _utcnow(), body.doi),
                )
        except Exception:
            # Best-effort: if the UPDATE itself fails, log it but still re-raise
            # the original pipeline exception so the caller gets a 500.
            logger.exception(
                "H2: could not mark ingest_queue row failed for doi=%s", body.doi
            )
        raise HTTPException(
            status_code=500,
            detail=f"ingest pipeline failed: {exc}",
        )

    # --- Mark done on success ---
    with state_db._connect() as conn:
        conn.execute(
            "UPDATE ingest_queue SET status = ?, completed_at = ? WHERE doi = ?",
            ("done", _utcnow(), body.doi),
        )

    return JSONResponse(
        {"status": "queued", "paper_id": body.doi},
        status_code=202,
    )


# ---------------------------------------------------------------------------
# H3: read-only queue status endpoints
# ---------------------------------------------------------------------------

_QUEUE_COLS = ["doi", "status", "queued_at", "completed_at", "error"]
_QUEUE_SQL = (
    "SELECT doi, status, queued_at, completed_at, error "
    "FROM ingest_queue ORDER BY queued_at DESC LIMIT 100"
)


@router.get("/api/ingest/queue")
def list_queue() -> list[dict]:
    """H3: list up to 100 most-recent ingest_queue rows, newest first.

    Empty queue returns an empty list (200).  The explicit /queue path is
    registered BEFORE the {doi:path}/status wildcard so FastAPI's first-match
    routing does not swallow 'queue' as a DOI segment.
    """
    state_db = get_runtime().state_db
    with state_db._connect() as conn:
        rows = conn.execute(_QUEUE_SQL).fetchall()
    return [dict(zip(_QUEUE_COLS, r)) for r in rows]


@router.get("/api/ingest/{doi:path}/status")
def doi_status(doi: str) -> dict:
    """H3: single-DOI status lookup.

    Returns the ingest_queue row for *doi* or 404 if it is not in the queue.
    The /status suffix disambiguates this route from the bare /queue listing.
    """
    state_db = get_runtime().state_db
    with state_db._connect() as conn:
        row = conn.execute(
            "SELECT doi, status, queued_at, completed_at, error "
            "FROM ingest_queue WHERE doi = ?",
            (doi,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"DOI {doi!r} not in queue")
    return dict(zip(_QUEUE_COLS, row))


__all__ = ["router", "IngestRequest", "_process_paper"]
