"""Bundle F: embedding provider migration paths.

Two public entry points:
  switch_provider() — switch to a different embedding provider/model,
                      optionally preserving the old collection.
  rebuild_active()  — re-embed using current config (delegates to switch_provider).

Safety invariants:
  - WITHOUT --keep-old: requires --confirm (destructive, replaces current collection).
  - WITH --keep-old: creates a new `papers_v<N>_<provider>` collection alongside;
    old collection remains untouched; provenance table updated.
  - Pre-flight: user sees paper count + estimated time + cost BEFORE any rebuild.
  - Atomicity: provenance is recorded ONLY after the rebuild loop completes;
    if interrupted mid-rebuild, state.db still points at the old collection.
  - B3 swap-order: BOTH paths build into a FRESH versioned collection first.
    The destructive path drops the old collection ONLY after the new one is
    built and marked current (build → record → set-current → drop-old). A crash
    mid-rebuild leaves the old collection intact + current; the half-built new
    collection is best-effort dropped so no partial collection leaks.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def switch_provider(
    *,
    provider: str,
    model: str,
    dim: int | None,
    keep_old: bool,
    confirm: bool,
) -> None:
    """Switch the active embedding provider/model.

    Parameters
    ----------
    provider:
        'ollama' or 'litellm'.
    model:
        Model string (e.g. 'mxbai-embed-large', 'text-embedding-3-large').
    dim:
        Embedding dimension.  When None, uses the provider default.
    keep_old:
        True → build a new collection alongside; old collection preserved.
        False → destructive; requires confirm=True.
    confirm:
        Required when keep_old=False.  Guards against accidental data loss.
    """
    import click

    from scripts.core.config import get_config
    from scripts.core.state_db import StateDB

    cfg = get_config()
    state_db = StateDB(Path(cfg.state_db.path).expanduser())

    rows = state_db.list_embedding_provenance()
    # The default collection is 'lit_monitor_v1'; this is the fallback when
    # no provenance row has been recorded yet (fresh install).
    current_collection = next(
        (r["collection_name"] for r in rows if r["is_current"]),
        "lit_monitor_v1",
    )

    # Guard against destructive operation without --confirm.
    if not keep_old and not confirm:
        click.echo(
            "ERROR: Switching embedding provider without --keep-old is DESTRUCTIVE "
            "(the existing ChromaDB collection will be deleted). "
            "Add --confirm to proceed, or use --keep-old to preserve the "
            "existing collection and build the new one alongside it.",
            err=True,
        )
        raise click.exceptions.Exit(2)

    # Resolve effective dim using provider defaults when not supplied.
    effective_dim = dim or _default_dim(provider)

    # Pre-flight: show paper count + estimated cost before doing anything.
    n_papers = _count_papers(state_db)
    _show_preflight(provider, model, effective_dim, n_papers)
    if not click.confirm("Continue?"):
        click.echo("Aborted.")
        raise click.exceptions.Exit(1)

    # Always build into a FRESH versioned collection — never overwrite the
    # current one in place.  (B3) This eliminates the destruction window: even
    # on the destructive path we build new first, then swap on success, so a
    # crash mid-rebuild leaves the OLD collection fully intact and current.
    next_v = _next_collection_version(rows)
    new_collection = f"papers_v{next_v}_{provider}"

    persist_dir = str(Path(cfg.state_db.path).expanduser().parent / "chroma")

    # Build the new collection by re-embedding all papers.  If this raises
    # (e.g. the user kills it mid-embed), best-effort clean up the half-built
    # new collection and re-raise — the OLD collection is never touched here,
    # so it remains current and recoverable.
    try:
        _rebuild_collection(
            collection_name=new_collection,
            provider=provider,
            model=model,
            dim=effective_dim,
            persist_dir=persist_dir,
            state_db=state_db,
            cfg=cfg,
        )
    except Exception:
        # Drop the orphaned, partially-populated new collection so we don't
        # leak it.  NEVER drop the old collection on this path.
        try:
            _drop_collection(new_collection, persist_dir=persist_dir)
        except Exception as cleanup_exc:
            logger.warning(
                "Failed to clean up half-built collection %r after a failed "
                "rebuild: %s",
                new_collection, cleanup_exc,
            )
        raise

    # Rebuild succeeded.  Record provenance + mark the new collection current.
    # Only AFTER the swap do we drop the old collection on the destructive path.
    state_db.record_embedding_provenance(new_collection, provider, model, effective_dim)
    state_db.set_current_embedding_collection(new_collection)

    if not keep_old and current_collection != new_collection:
        # Destructive path: now that the new collection is built and current,
        # it is safe to drop the old one.
        _drop_collection(current_collection, persist_dir=persist_dir)

    click.echo(
        f"\nDone. New embedding collection: {new_collection!r} "
        f"(provider={provider}, model={model}, dim={effective_dim})."
    )
    if keep_old and current_collection != new_collection:
        click.echo(
            f"Old collection {current_collection!r} is preserved. "
            "To use it again, run: lit-monitor embeddings switch --provider <old-provider> "
            "--model <old-model> --keep-old"
        )
    click.echo(
        "\nConfig diff (not auto-written — update config/extraction.yaml manually):\n"
        f"  embedding:\n"
        f"    provider: {provider}\n"
    )


def rebuild_active(*, keep_old: bool, confirm: bool) -> None:
    """Re-embed the active collection using the current config.

    Useful when the user manually edited the embedding model in config
    and needs to rebuild the ChromaDB collection to match.

    Parameters
    ----------
    keep_old:
        True → preserve the old collection alongside the new one.
    confirm:
        Required when keep_old=False (destructive).
    """
    from scripts.core.config import get_config
    cfg = get_config()

    active_provider = getattr(cfg.embedding, "provider", "ollama")
    provider_cfg = getattr(cfg.embedding, active_provider, None)
    active_model = getattr(provider_cfg, "model", _default_model(active_provider))
    active_dim: int | None = getattr(provider_cfg, "dim", None)

    switch_provider(
        provider=active_provider,
        model=active_model,
        dim=active_dim,
        keep_old=keep_old,
        confirm=confirm,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_papers(state_db: Any) -> int:
    """Count paper rows in the state DB (for pre-flight estimate)."""
    with state_db._connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
    return row[0] if row else 0


def _default_model(provider: str) -> str:
    if provider == "litellm":
        return "text-embedding-3-large"
    return "mxbai-embed-large"


def _default_dim(provider: str) -> int:
    if provider == "litellm":
        return 3072
    return 1024


def _show_preflight(provider: str, model: str, dim: int, n_papers: int) -> None:
    """Print estimated cost and duration to stdout."""
    import click

    # Very rough estimates — useful order-of-magnitude guidance.
    if provider == "litellm" and "text-embedding-3" in model:
        # OpenAI text-embedding-3-large: ~$0.13 / 1M tokens; ~300 tokens/abstract
        est_cost = f"~${n_papers * 300 * 0.00000013:.2f} (OpenAI text-embedding-3-large estimate)"
    elif provider == "litellm":
        est_cost = "~varies (check provider pricing)"
    else:
        est_cost = "free (local Ollama, no API cost)"

    # mxbai-embed-large: ~0.08 s/abstract on M2 Mac; text-embedding API: ~0.05 s
    secs_per_paper = 0.08 if provider == "ollama" else 0.05
    est_minutes = max(1, int(n_papers * secs_per_paper / 60))

    click.echo(f"\nSwitching to: provider={provider!r}  model={model!r}  dim={dim}")
    click.echo(f"  Papers to re-embed : {n_papers}")
    click.echo(f"  Estimated time     : ~{est_minutes} min")
    click.echo(f"  Estimated cost     : {est_cost}")


def _next_collection_version(rows: list[dict]) -> int:
    """Return the next free papers_v<N> version index."""
    used: set[int] = set()
    for r in rows:
        match = re.match(r"^papers_v(\d+)", r["collection_name"])
        if match:
            used.add(int(match.group(1)))
    # Default collection 'lit_monitor_v1' doesn't match papers_v<N> pattern —
    # if no versioned collections exist yet, start at v2 (v1 = the default).
    return max(used) + 1 if used else 2


def _drop_collection(collection_name: str, persist_dir: str) -> None:
    """Delete a ChromaDB collection (no-op if it does not exist)."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(collection_name)
        logger.info("Deleted ChromaDB collection %r", collection_name)
    except Exception as exc:
        logger.debug("drop_collection %r: %s (likely absent, continuing)", collection_name, exc)


def _rebuild_collection(
    *,
    collection_name: str,
    provider: str,
    model: str,
    dim: int,
    persist_dir: str,
    state_db: Any,
    cfg: Any,
) -> None:
    """Re-embed all papers into the named ChromaDB collection.

    Iterates all papers in state_db, embeds each using the new provider,
    and upserts into the new collection.  Progress is logged at INFO level.

    The rebuild is NOT atomic at the SQLite level — if interrupted, the
    new collection will be partially populated but provenance will not have
    been recorded yet (provenance is written by the caller AFTER this returns),
    so the old collection remains the active one.  A subsequent re-run will
    restart the rebuild from scratch.
    """
    from scripts.output.embeddings import EmbeddingsDB

    # Determine ollama_host from config if the provider is ollama.
    ollama_host = "http://localhost:11434"
    if provider == "ollama":
        ollama_host = getattr(
            getattr(cfg, "embedding", None) and getattr(cfg.embedding, "ollama", None),
            "host",
            getattr(getattr(cfg, "brain_build", None), "ollama_host", "http://localhost:11434"),
        )

    embed_db = EmbeddingsDB(
        persist_dir=persist_dir,
        ollama_host=ollama_host,
        provider=provider,
        model=model,
        dim=dim,
        papers_collection=collection_name,
    )

    papers = state_db.get_all_by_source_type("paper")
    papers += state_db.get_all_by_source_type("review")
    logger.info("Rebuild: %d papers to re-embed into %r", len(papers), collection_name)

    ok = 0
    failed = 0
    for i, paper in enumerate(papers, 1):
        doi = paper.get("doi", "")
        if not doi:
            continue
        extraction_json = paper.get("extraction_json")
        if not extraction_json:
            logger.debug("Skip %s: no extraction_json", doi)
            continue
        import json as _json
        try:
            ext = _json.loads(extraction_json)
        except Exception:
            logger.debug("Skip %s: bad extraction_json", doi)
            continue

        # Build the embedding text using the same format as the ingest pipeline.
        title = paper.get("title") or ext.get("title", "")
        abstract = ext.get("abstract", "")
        core_finding = ext.get("core_finding", "")
        text = " ".join(filter(None, [title, abstract, core_finding]))
        if not text.strip():
            logger.debug("Skip %s: empty embed text", doi)
            continue

        try:
            meta: dict = {
                "source_type": paper.get("source_type", "paper"),
                "title": title,
                "year": paper.get("year") or 0,
            }
            embed_db.add_paper(doi=doi, text=text, metadata=meta)
            ok += 1
        except Exception as exc:
            logger.warning("Rebuild: failed to embed %s: %s", doi, exc)
            failed += 1

        if i % 100 == 0:
            logger.info("Rebuild progress: %d / %d (ok=%d failed=%d)", i, len(papers), ok, failed)

    logger.info("Rebuild complete: %d embedded, %d failed", ok, failed)
