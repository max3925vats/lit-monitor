"""
Embeddings management using ChromaDB.
Two collections:
  - ``lit_monitor_v1``       — paper-level (title + abstract + core_finding).
    PK: doi.  Used by ranker and relink.
  - ``lit_monitor_chunks_v1`` — chunk-level passages split from full markdown.
    PK: ``{doi}#chunk-{n}``.  Used by synthesize for passage-level RAG.

Embedding model: configured via ``extraction.yaml`` (``embeddings.model``); default
  ``mxbai-embed-large`` (1024-dim, ~670 MB).  ``nomic-embed-text`` (768-dim,
  ~270 MB) is the documented smaller alternative.  Changing the model triggers a
  full re-embed of all content via ``check_embed_model_change()`` which clears
  both ChromaDB collections and resets ``embeddings_indexed = 0`` in the state DB.
  - 1024-dimensional (mxbai-embed-large default)
  - Cosine distance
  - Served at the Ollama host configured in extraction.yaml
"""
from __future__ import annotations

import logging
import os
from typing import Any

# Disable ChromaDB's anonymous telemetry before the module is imported.
# The Settings flag alone doesn't suppress the posthog capture() call in all
# ChromaDB versions, leading to a noisy "Failed to send telemetry" warning.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb

from scripts.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)
_COLLECTION_NAME = "lit_monitor_v1"
_CHUNKS_COLLECTION_NAME = "lit_monitor_chunks_v1"


class _EmbedContextLengthError(Exception):
    """Internal sentinel — ``_embed_call`` raises this when Ollama returns 400
    with a ``context length`` body so ``_embed`` knows to truncate-and-retry.
    Callers outside this module should never see it; ``_embed`` either retries
    successfully or re-raises the original ``urllib.error.HTTPError``.
    """

    def __init__(self, body: str, original: Exception):
        super().__init__(body)
        self.body = body
        self.original = original
class EmbeddingsDB:
    """
    Thin wrapper around a single ChromaDB collection.
    All add_* methods are idempotent — re-adding an existing ID
    updates the document and metadata without creating a duplicate.
    """
    def __init__(
        self,
        persist_dir: str,
        ollama_host: str = "http://localhost:11434",
        embed_model: str = "mxbai-embed-large",
    ) -> None:
        # Telemetry already suppressed via os.environ above; no Settings object needed.
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ollama_host = ollama_host.rstrip("/")
        self._embed_model = embed_model  # configurable — change triggers full re-embed
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_collection = self._client.get_or_create_collection(
            name=_CHUNKS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    # ------------------------------------------------------------------
    # Add / update
    # ------------------------------------------------------------------
    def add_paper(self, doi: str, text: str, metadata: dict[str, Any]) -> None:
        """
        Add or update a paper embedding.
        text: title + " " + abstract + " " + core_finding
        metadata must include at least: source_type, title, year.
        """
        meta = dict(metadata)
        meta["source_type"] = "paper"
        embedding = self._embed(text)
        self._collection.upsert(
            ids=[doi],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],

        )
        logger.debug("Upserted paper embedding: %s", doi)

    def add_chunks(self, doi: str, chunks: list) -> None:
        """
        Index chunk-level passages for a paper (delete-then-add).

        Deletes all existing ``{doi}#chunk-*`` entries first so stale chunks
        from a prior run are never mixed with the fresh set.  Idempotent —
        calling again with new chunks produces a clean, up-to-date index.

        Parameters
        ----------
        doi:
            Paper DOI — used to scope the delete and build chunk_ids.
        chunks:
            List of ``Chunk`` objects from ``scripts.core.chunker.chunk_markdown()``.
            Empty list is a no-op (stale chunks are still deleted).
        """
        # Remove all previous chunks for this DOI before re-indexing.
        try:
            existing = self._chunks_collection.get(
                where={"doi": doi},
                include=[],
            )
            stale_ids = existing.get("ids", [])
            if stale_ids:
                self._chunks_collection.delete(ids=stale_ids)
                logger.debug("Deleted %d stale chunks for %s", len(stale_ids), doi)
        except Exception as exc:
            logger.debug("Could not delete stale chunks for %s: %s", doi, exc)

        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {**c.metadata, "doi": doi, "source_type": "chunk"}
            for c in chunks
        ]
        embeddings = [self._embed(t) for t in texts]
        self._chunks_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.debug("Indexed %d chunks for %s", len(chunks), doi)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def find_similar_to_text(
        self,
        text: str,
        top_k: int = 10,
        exclude_id: str | None = None,
        source_types: list[str] | None = None,
        rerank_with_query: str | None = None,
        reranker_config=None,
    ) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to the given text.
        Parameters
        ----------
        text:
            Query text (will be embedded on the fly).
        top_k:
            Number of results to return.
        exclude_id:
            Optional document ID to exclude (e.g., the source doc itself).
        source_types:
            Optional list to filter by source_type metadata.
            e.g. ['paper'], ['chapter', 'book'].
        rerank_with_query:
            When provided and reranker is enabled, retrieve ``top_k *
            candidate_multiplier`` candidates by cosine first, then
            cross-encoder rerank to ``top_k``.  Requires N19 config block.
        reranker_config:
            The ``config.reranker`` namespace (or None to skip reranking).

        Returns
        -------
        list[dict]
            Each dict: {id, score, document, metadata}.
            Sorted by similarity descending (highest first).
        """
        where: dict | None = None
        if source_types:
            if len(source_types) == 1:
                where = {"source_type": source_types[0]}
            else:
                where = {"source_type": {"$in": source_types}}
        # Fetch extra results to allow for exclusion and reranking headroom
        multiplier = _reranker_multiplier(reranker_config, rerank_with_query)
        n_results = (top_k * multiplier) + (1 if exclude_id else 0)
        # Can't query more than collection size
        n_in_collection = self._collection.count()
        if n_in_collection == 0:
            return []
        n_results = min(n_results, n_in_collection)
        query_embedding = self._embed(text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        output: list[dict[str, Any]] = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        for doc_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
            if exclude_id and doc_id == exclude_id:
                continue
            # cosine distance → similarity: 1 - distance
            score = round(1.0 - dist, 4)
            output.append({
                "id": doc_id,
                "score": score,
                "document": doc,
                "metadata": meta,
            })
        if rerank_with_query and _reranker_enabled(reranker_config):
            output = _apply_reranker(rerank_with_query, output, top_k, reranker_config)
        else:
            output = output[:top_k]
        return output
    def find_similar_chunks(
        self,
        text: str,
        top_k: int = 30,
        dedupe_by_paper: bool = True,
        rerank_with_query: str | None = None,
        reranker_config=None,
    ) -> list[dict[str, Any]]:
        """
        Find the top_k most similar chunks to the given text.

        Parameters
        ----------
        text:
            Query text (will be embedded on the fly).
        top_k:
            Number of results to return.
        dedupe_by_paper:
            When True, at most one chunk per DOI is returned — the one with
            the highest similarity score.  The returned list is still up to
            top_k entries but each entry comes from a distinct paper.
        rerank_with_query:
            When provided and reranker is enabled, retrieve a larger cosine
            candidate set, then cross-encoder rerank to ``top_k``.
        reranker_config:
            The ``config.reranker`` namespace (or None to skip reranking).

        Returns
        -------
        list[dict]
            Each dict: {id, score, document, metadata}.
            ``metadata`` includes ``doi``, ``section_heading``, ``chunk_index``.
            Sorted by similarity descending (highest first).
        """
        n_in_collection = self._chunks_collection.count()
        if n_in_collection == 0:
            return []

        # Fetch extra results to allow for deduplication and reranking headroom.
        multiplier = _reranker_multiplier(reranker_config, rerank_with_query)
        fetch_k = min((top_k * max(multiplier, 3)) if dedupe_by_paper else top_k * multiplier,
                      n_in_collection)
        query_embedding = self._embed(text)
        results = self._chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=max(fetch_k, 1),
            include=["documents", "metadatas", "distances"],
        )

        candidates: list[dict[str, Any]] = []
        seen_dois: set[str] = set()
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for chunk_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
            score = round(1.0 - dist, 4)
            doi = meta.get("doi", "")
            if dedupe_by_paper:
                if doi in seen_dois:
                    continue
                seen_dois.add(doi)
            candidates.append({
                "id": chunk_id,
                "score": score,
                "document": doc,
                "metadata": meta,
            })

        if rerank_with_query and _reranker_enabled(reranker_config):
            return _apply_reranker(rerank_with_query, candidates, top_k, reranker_config)
        return candidates[:top_k]

    def get_collection_stats(self) -> dict:
        """Return basic collection stats dict."""
        return {"count": self.count(), "chunk_count": self._chunks_collection.count()}

    def search_similar(self, query: str, top_k: int = 10) -> list:
        """Similarity search returning dicts with doi, score, metadata keys."""
        results = self.find_similar_to_text(query, top_k=top_k)
        return [{"doi": r["id"], "score": r["score"], "metadata": r["metadata"]} for r in results]

    def count(self) -> int:
        """Return total number of documents in the collection."""
        return self._collection.count()

    @property
    def embed_model(self) -> str:
        """The embedding model name in use (single source of truth for model change checks)."""
        return self._embed_model

    def clear(self) -> None:
        """Delete and recreate both ChromaDB collections, discarding all embeddings.

        Called when the embedding model changes so vectors from the old model
        are not mixed with vectors from the new one. Clears both the paper-level
        and chunk-level collections. Safe to call on a fresh install.
        """
        for name in (_COLLECTION_NAME, _CHUNKS_COLLECTION_NAME):
            try:
                self._client.delete_collection(name)
            except Exception as exc:
                # ChromaDB raises ValueError or InvalidCollectionException if absent
                logger.debug("delete_collection raised (likely collection absent): %s", exc)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_collection = self._client.get_or_create_collection(
            name=_CHUNKS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collections '%s' and '%s' cleared.",
            _COLLECTION_NAME,
            _CHUNKS_COLLECTION_NAME,
        )

    def clear_chunks(self) -> None:
        """Delete and recreate the chunk-level collection only.

        Used by ``rechunk-all`` and C4 hot-swap (model change clears both via
        ``clear()``; this method handles the rechunk case where the paper-level
        index is still valid).
        """
        try:
            self._client.delete_collection(_CHUNKS_COLLECTION_NAME)
        except Exception as exc:
            logger.debug("delete_collection raised (likely chunks collection absent): %s", exc)
        self._chunks_collection = self._client.get_or_create_collection(
            name=_CHUNKS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB chunks collection '%s' cleared.", _CHUNKS_COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    # Belt-and-braces ceiling for /api/embed payloads.  ``mxbai-embed-large``
    # has a hard 512-token context; ``nomic-embed-text`` has 8192.  Choosing
    # 1400 chars caps a chunk at ~400 real tokens for scientific text — safely
    # under mxbai's ceiling — and is comfortably below nomic's too.
    _EMBED_TRUNCATE_CHARS = 1400
    # Floor below which we stop progressively truncating.  If a 200-char input
    # *still* exceeds the model's context, something is structurally wrong
    # (e.g. the wrong model is configured) — re-raise rather than loop.
    _EMBED_TRUNCATE_FLOOR_CHARS = 200

    def _embed(self, text: str) -> list[float]:
        """Call Ollama embedding API and return the embedding vector.

        On HTTPError ``context length exceeded``, transparently retry with
        progressively shorter input — halving each round — until either the
        embed succeeds or the input drops below ``_EMBED_TRUNCATE_FLOOR_CHARS``.
        Each truncation step logs a WARNING so the user knows data was dropped.

        Other 400s (model not found, malformed input) are NOT retried; they
        propagate after capturing the response body in the ERROR log.
        """
        # Start at full text; cap subsequent rounds via the truncate ladder.
        truncated_to: int | None = None
        attempt = 0
        while True:
            attempt += 1
            current = text if truncated_to is None else text[:truncated_to]
            try:
                return self._embed_call(current)
            except _EmbedContextLengthError as ctx_err:
                # In strict mode, raise immediately on the first context-length error
                # rather than silently truncating the embedding input.
                from scripts.core.strict_mode import is_strict
                if is_strict():
                    raise RuntimeError(
                        f"Ollama /api/embed: input {len(current)} chars exceeded "
                        f"{self._embed_model} context window. "
                        "Run without --strict to enable automatic truncation. "
                        f"Body={ctx_err.body!r}"
                    ) from ctx_err.original
                # Compute next attempt size: first hit goes to the cap; subsequent
                # hits halve until we hit the floor.
                if truncated_to is None:
                    next_size = min(self._EMBED_TRUNCATE_CHARS, len(text))
                else:
                    next_size = truncated_to // 2
                if next_size < self._EMBED_TRUNCATE_FLOOR_CHARS:
                    logger.error(
                        "Ollama /api/embed: input STILL exceeds %s context at "
                        "%d chars (attempt %d) — giving up below floor of %d chars. "
                        "Likely a model/config mismatch.  Body=%r",
                        self._embed_model, truncated_to or len(text), attempt,
                        self._EMBED_TRUNCATE_FLOOR_CHARS, ctx_err.body,
                    )
                    raise ctx_err.original
                logger.warning(
                    "Ollama /api/embed: input %d chars exceeded %s context "
                    "(attempt %d) — retrying with %d chars. Body=%r",
                    len(current), self._embed_model, attempt, next_size, ctx_err.body,
                )
                truncated_to = next_size

    def _embed_call(self, text: str) -> list[float]:
        """One Ollama /api/embed call.  Raises _EmbedContextLengthError on 400
        ``context length`` errors so the caller can decide whether to retry;
        propagates all other HTTPErrors with the response body in the log.
        """
        import json
        import urllib.error
        import urllib.request
        payload = json.dumps({"model": self._embed_model, "input": text}).encode()
        req = urllib.request.Request(
            url=f"{self._ollama_host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            if exc.code == 400 and "context length" in err_body.lower():
                raise _EmbedContextLengthError(err_body, exc) from exc
            head = text[:80].replace("\n", "\\n")
            tail = text[-80:].replace("\n", "\\n") if len(text) > 80 else ""
            logger.error(
                "Ollama /api/embed HTTP %d for model=%s len=%d chars head=%r tail=%r body=%r",
                exc.code, self._embed_model, len(text), head, tail, err_body,
            )
            raise
        embeddings = body.get("embeddings") or body.get("embedding")
        if not embeddings:
            raise ValueError(f"Empty or missing embeddings in Ollama response: {list(body.keys())}")
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings


# ---------------------------------------------------------------------------
# Reranker helpers (N19)
# ---------------------------------------------------------------------------

def _reranker_enabled(reranker_config) -> bool:
    """Return True if reranking is configured and enabled."""
    if reranker_config is None:
        return False
    return bool(getattr(reranker_config, "enabled", False))


def _reranker_multiplier(reranker_config, rerank_with_query: str | None) -> int:
    """Return the cosine candidate multiplier (default 1 → no extra fetching)."""
    if rerank_with_query and _reranker_enabled(reranker_config):
        return int(getattr(reranker_config, "candidate_multiplier", 3))
    return 1


def _apply_reranker(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    reranker_config,
) -> list[dict[str, Any]]:
    """Rerank candidates with the cross-encoder and return top_k results."""
    try:
        from scripts.output.reranker import get_reranker
        model_name = getattr(reranker_config, "model", "mixedbread-ai/mxbai-rerank-large-v2")
        device = getattr(reranker_config, "device", None)
        reranker = get_reranker(model_name=model_name, device=device)
        return reranker.rerank(query, candidates, top_k=top_k)
    except Exception as exc:
        strict_fallback(
            logger,
            f"Reranking failed (falling back to cosine order): {exc}. "
            "Check that the reranker model is installed and the device setting is correct.",
            exc,
        )
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Embedding model change detection
# ---------------------------------------------------------------------------
def check_embed_model_change(
    state_db: Any,
    embed_db: EmbeddingsDB,
    configured_model: str,
) -> bool:
    """Detect whether the embedding model has changed since the last run.

    On first run (no prior model stored in kv_store) the model name is recorded
    and the function returns False — nothing needs to be re-embedded.

    If the model has changed:
    - Clears the ChromaDB collection (vectors from the old model are discarded)
    - Resets ``embeddings_indexed = 0`` for all rows in the state DB so every
      item is re-embedded on the next pipeline run
    - Persists the new model name in kv_store

    Parameters
    ----------
    state_db:
        A ``StateDB`` instance (or any object with ``get_kv``/``set_kv``/
        ``reset_embeddings_indexed`` methods).
    embed_db:
        The active ``EmbeddingsDB`` instance.
    configured_model:
        The embedding model name from the current config (e.g. ``"mxbai-embed-large"``).

    Returns
    -------
    bool
        ``True`` if a model change was detected and a reset was performed,
        ``False`` otherwise.
    """
    prior_model = state_db.get_kv("embed_model")
    if prior_model is None:
        # First run — record the model; nothing to reset yet.
        state_db.set_kv("embed_model", configured_model)
        logger.info("Embedding model recorded for the first time: %s", configured_model)
        return False
    if prior_model == configured_model:
        return False
    # Model has changed — full reset required so old vectors don't pollute the index.
    logger.warning(
        "Embedding model changed: '%s' → '%s'. "
        "Clearing ChromaDB collection and resetting embeddings_indexed for all rows.",
        prior_model,
        configured_model,
    )
    embed_db.clear()
    state_db.reset_embeddings_indexed()
    state_db.set_kv("embed_model", configured_model)
    return True
