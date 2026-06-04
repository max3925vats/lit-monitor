"""H2/A1: POST /api/ingest — production async ingest endpoint.

Architecture (A1)
-----------------
The endpoint validates the payload, inserts an ``ingest_queue`` row with
status ``"queued"``, schedules a FastAPI ``BackgroundTask``, and returns
``202`` immediately. The HTTP request is NEVER blocked on LLM extraction
(which takes 30–90 s). The background task does the real ingestion by
wiring ``scripts.pipelines.brain_build._process_paper`` with a fully
constructed runtime (config, state_db, embeddings_db, zotero_client, llm,
graph_db-or-None) and transitions the queue row:

    queued → processing → done | no_markdown | failed

The H3 status endpoints (``GET /api/ingest/queue``,
``GET /api/ingest/{doi}/status``) are the polling surface for this async
flow — clients submit, then poll for the terminal status.

R28 invariant
-------------
papers.graph_indexed=1 ONLY after BOTH vector embed AND graph add succeed.
brain_build._process_paper enforces this. The HTTP layer merely wraps it
and never writes papers.graph_indexed directly.

R28 hardening
-------------
If the background task raises, the ingest_queue row is marked
status='failed' with the error text — never orphaned in 'queued' or
'processing' by an *in-process* failure. An orphaned row would appear as
an active job to H3's queue listing and hide the real failure.

Known limitation (process-death window): this guarantee only covers
in-process failures caught by the task's try/except. If the worker process
is hard-killed (SIGKILL / OOM / restart) in the window between the
'processing' UPDATE and the terminal UPDATE, the row is stranded in
'processing' forever — no in-process handler can run during a hard kill.
The proper fix is a reaper / timeout sweep that re-fails rows stuck in
'processing' past a deadline; deferred to a future bundle.

no_markdown fallback
--------------------
``_process_paper`` returns ``(False, [])`` and marks the paper
``"no_markdown"`` in state_db when the Zotero item has no ``.md``
attachment yet (the plugin may push a paper before the user attaches the
markdown). This is EXPECTED, not an error — the queue row is set to
``"no_markdown"``, not ``"failed"``.

Duplicate handling
------------------
The ingest_queue PK is doi, so the duplicate check (SELECT before INSERT)
is the first-line guard. Concurrent requests to the same DOI can race past
the SELECT and collide on the INSERT — SQLite's serialised-writer isolation
means the loser's INSERT raises IntegrityError. A3-6: that IntegrityError is
caught around the INSERT and mapped to the same 409 the SELECT path returns,
so a concurrent duplicate is never surfaced as a 500.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

# P2.3: hard cap on title length. Real article titles are well under this;
# an oversized title is almost certainly malformed/abusive input and would
# bloat downstream storage, logs, and LLM prompts. Reject with 422.
_MAX_TITLE_LEN = 500

# Single source of truth for the ingest_queue state machine. These bare
# string literals were previously scattered across the route, the background
# task, and the tests; a typo like "no markdown" would silently create an
# unmatchable status. Reference these constants in all production code paths.
_STATUS_QUEUED = "queued"
_STATUS_PROCESSING = "processing"
_STATUS_DONE = "done"
_STATUS_NO_MARKDOWN = "no_markdown"
_STATUS_FAILED = "failed"

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
        """Reject empty/whitespace-only titles and titles over the length cap.

        The length check uses the stripped title so trailing whitespace can't
        be used to slip past the cap (or to pad an otherwise-empty title).
        """
        stripped = v.strip() if v else ""
        if not stripped:
            raise ValueError("title cannot be empty or whitespace-only")
        if len(stripped) > _MAX_TITLE_LEN:
            raise ValueError(
                f"title too long: {len(stripped)} chars (max {_MAX_TITLE_LEN})"
            )
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(UTC).isoformat()


def _duplicate_response(doi: str) -> JSONResponse:
    """409 returned for a DOI already in the queue.

    Single source of truth so the SELECT-found path and the INSERT-race path
    (A3-6) return byte-identical bodies — clients can't tell which guard fired.
    """
    return JSONResponse(
        {"status": "duplicate", "paper_id": doi},
        status_code=409,
    )


def _build_zotero_item(body: IngestRequest) -> dict[str, Any]:
    """Synthesize a Zotero-shaped item dict from an IngestRequest.

    brain_build._process_paper reads the item as
    ``{"key": <zotero_key>, "data": {...}}`` and pulls fields via
    ``data.get("title")``, ``ZoteroClient.extract_authors(data)``,
    ``_parse_year(data.get("date"))``, ``data.get("abstractNote")``, etc.

    The request gives ``authors`` as a flat ``list[str]`` of display names,
    but ``extract_authors`` expects Zotero ``creators`` with
    ``creatorType`` + ``lastName``. We round-trip each display name through
    the ``lastName`` field so ``extract_authors`` returns the exact strings
    the caller supplied (it emits "lastName, firstName".strip(", ") which,
    with an empty firstName, is just the lastName).
    """
    creators = [
        {"creatorType": "author", "firstName": "", "lastName": name}
        for name in body.authors
        if name and name.strip()
    ]
    # _parse_year scans "-"-split parts for a 4-digit year, so a bare year
    # string is sufficient. Empty string → _parse_year returns 0.
    date_str = str(body.year) if body.year is not None else ""
    return {
        "key": body.zotero_key or "",
        "data": {
            "itemType": "journalArticle",
            "title": body.title,
            "creators": creators,
            "date": date_str,
            "abstractNote": body.abstract or "",
            "DOI": body.doi,
        },
    }


# ---------------------------------------------------------------------------
# Pipeline wrapper (monkeypatched in unit tests)
# ---------------------------------------------------------------------------


def _process_paper(
    doi: str,
    item: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Real production wiring around brain_build._process_paper.

    Constructs the full runtime needed by brain_build's per-paper pipeline
    and dispatches to it. Returns ``(processed, discovered_topics)``:

      * ``processed=True``  — extracted and indexed → queue row ``"done"``.
      * ``processed=False`` — no ``.md`` attachment yet (brain_build marked
        the paper ``"no_markdown"`` in state_db) → queue row
        ``"no_markdown"``. Not an error.

    Raises on unrecoverable errors so the background task marks the queue
    row ``"failed"``.

    Kept as a module-level, monkeypatchable symbol: unit tests replace it
    with a stand-in so they can exercise the queue-status transitions
    without a configured runtime, Zotero client, or LLM.
    """
    # Local imports: keep module import cheap and avoid importing the full
    # brain_build stack (LLM, S2, graph) at server boot.
    from scripts.graph import safe_graph_db
    from scripts.llm.extractor import extract_paper
    from scripts.llm.llm_client import get_clients_for_passes
    from scripts.output.obsidian_writer import write_paper_note
    from scripts.pipelines.brain_build import _process_paper as brain_process_paper

    runtime = get_runtime()
    config = runtime.config
    secrets = runtime.secrets

    # Mirror the CLI: hydrate provider keys from config.toml so the LLM and
    # S2 enrichment inside brain_build can authenticate. Idempotent and
    # one-time per process (see _hydrate_provider_keys). Shell env wins.
    _hydrate_provider_keys(secrets)

    # Build the LLM client exactly as the brain_build CLI does.
    llm = get_clients_for_passes(config, mode="brain_build", think=True)

    # Resolve the graph DB; None when the [graph] extra isn't installed.
    # _process_paper accepts graph_db=None and runs vector-only.
    graph_db = safe_graph_db()

    pass_strategy = getattr(
        getattr(config, "brain_build", None), "pass_strategy", "individual"
    )
    zotero_key = item.get("key", "") or doi

    try:
        return brain_process_paper(
            doi=doi,
            zotero_key=zotero_key,
            item=item,
            config=config,
            state_db=runtime.state_db,
            embeddings_db=runtime.embeddings_db,
            llm=llm,
            zotero_client=runtime.zotero_client,
            source_type="paper",
            pass_strategy=pass_strategy,
            extract_paper_fn=extract_paper,
            write_paper_note_fn=write_paper_note,
            graph_db=graph_db,
        )
    finally:
        # Release the KuzuDB connection if one was opened. close() is a
        # no-op when graph_db is None and is safe to call multiple times.
        if graph_db is not None:
            try:
                graph_db.close()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("A1: graph_db.close() failed (non-fatal): %s", exc)


# Guards the one-time, process-wide env hydration below. Module-level so it
# persists across every background task in this process.
_KEYS_HYDRATED = False


def _hydrate_provider_keys(secrets: dict) -> None:
    """Inject OLLAMA_API_KEY / S2_API_KEY from config.toml if unset.

    Mirrors scripts.cli._maybe_set_ollama_key / _maybe_set_s2_key. Shell
    env vars always win over config.toml (we only set a key when
    ``os.environ.get(name)`` is empty).

    BLAST RADIUS — DELIBERATE: this is a *process-wide* mutation of
    ``os.environ``, not a request-scoped one. The keys remain set for the
    life of the server process and are inherited by any subprocess the
    server later spawns. That mirrors the CLI's behaviour and is intentional
    so brain_build's LLM/S2 calls can authenticate. The ``_KEYS_HYDRATED``
    guard makes this run AT MOST ONCE per process — it is idempotent and
    does NOT re-run on every background task.
    """
    global _KEYS_HYDRATED
    if _KEYS_HYDRATED:
        return

    import os

    if not os.environ.get("OLLAMA_API_KEY"):
        ollama_key = secrets.get("ollama", {}).get("api_key", "")
        if ollama_key:
            os.environ["OLLAMA_API_KEY"] = ollama_key
    if not os.environ.get("S2_API_KEY"):
        s2_key = secrets.get("semantic_scholar", {}).get("api_key", "")
        if s2_key:
            os.environ["S2_API_KEY"] = s2_key

    _KEYS_HYDRATED = True


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


def _run_ingest_task(doi: str, item: dict[str, Any]) -> None:
    """Background worker: transition the queue row and run real ingestion.

    Status transitions:
      queued → processing  (set at task start)
      processing → done         (_process_paper returned processed=True)
      processing → no_markdown  (_process_paper returned processed=False)
      processing → failed       (_process_paper raised)

    Wrapping _process_paper here (rather than inline) preserves the unit
    test seam: tests monkeypatch ingest._process_paper, and Starlette's
    TestClient runs background tasks synchronously after the response, so
    the monkeypatched stand-in still drives these transitions.
    """
    state_db = get_runtime().state_db

    # queued → processing
    try:
        with state_db._connect() as conn:
            conn.execute(
                "UPDATE ingest_queue SET status = ? WHERE doi = ?",
                (_STATUS_PROCESSING, doi),
            )
    except Exception:
        # If we can't even mark 'processing', log and continue — the real
        # work below still runs and will set a terminal status.
        logger.exception("A1: could not mark ingest_queue row processing for doi=%s", doi)

    try:
        processed, _topics = _process_paper(doi, item)
    except Exception as exc:
        # R28 hardening: terminal 'failed' with error text, never orphaned.
        logger.warning("A1: ingest task failed for doi=%s: %s", doi, exc)
        try:
            with state_db._connect() as conn:
                conn.execute(
                    "UPDATE ingest_queue SET status = ?, error = ?, completed_at = ? "
                    "WHERE doi = ?",
                    (_STATUS_FAILED, str(exc)[:1000], _utcnow(), doi),
                )
        except Exception:
            logger.exception(
                "A1: could not mark ingest_queue row failed for doi=%s", doi
            )
        return

    # processed=True → done; processed=False → no_markdown (not an error).
    terminal = _STATUS_DONE if processed else _STATUS_NO_MARKDOWN
    with state_db._connect() as conn:
        conn.execute(
            "UPDATE ingest_queue SET status = ?, completed_at = ? WHERE doi = ?",
            (terminal, _utcnow(), doi),
        )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/api/ingest", status_code=202)
def ingest(body: IngestRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    """Accept a paper ingest request, queue it, and schedule async ingestion.

    The HTTP request returns immediately (202) after inserting the queue
    row and scheduling the background task — it does NOT block on the
    30–90 s LLM extraction. Poll GET /api/ingest/{doi}/status for the
    terminal state (done / no_markdown / failed).

    Returns
    -------
    202  — queued; background ingestion scheduled.
    409  — this DOI was already submitted (duplicate).
    422  — validation failed (bad DOI format or missing/blank title).
    500  — the queue INSERT itself failed unexpectedly.
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
        return _duplicate_response(body.doi)

    # --- Insert queue row BEFORE scheduling the task ---
    # The row must exist so the background task always has something to
    # transition.
    #
    # A3-6: the SELECT above is NOT a sufficient guard against concurrency.
    # ingest_queue.doi is the PK, so two simultaneous requests for the same DOI
    # can both pass the SELECT (neither row exists yet) and then race on the
    # INSERT — the loser hits a UNIQUE/PK violation. Treat that as a duplicate
    # (the SELECT path's outcome) rather than letting it surface as a 500.
    queued_at = _utcnow()
    try:
        with state_db._connect() as conn:
            conn.execute(
                "INSERT INTO ingest_queue (doi, status, queued_at) VALUES (?, ?, ?)",
                (body.doi, _STATUS_QUEUED, queued_at),
            )
    except sqlite3.IntegrityError:
        # Concurrent same-DOI insert won the race. The DOI is already queued,
        # so this request is a duplicate — return the identical 409 the SELECT
        # path returns. The winning request owns the row and its background
        # task; we do NOT schedule a second one.
        logger.info("A3-6: ingest INSERT lost race for doi=%s; treating as duplicate", body.doi)
        return _duplicate_response(body.doi)

    # --- Schedule real async ingestion ---
    # Synthesize the Zotero-shaped item the background task feeds to
    # brain_build._process_paper.
    item = _build_zotero_item(body)
    background_tasks.add_task(_run_ingest_task, body.doi, item)

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
