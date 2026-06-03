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
from collections import OrderedDict
from typing import Any

import numpy as np

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

    Bundle F: accepts provider/model/dim kwargs so users can route
    through any LiteLLM-compatible model in addition to local Ollama.
    When a collection already has provenance recorded in state.db,
    those values OVERRIDE the constructor kwargs — this prevents
    dimension-mismatch corruption when the user changes config but
    re-opens an existing (dim-locked) ChromaDB collection.
    """

    # Default values for the ollama provider (backward-compat).
    _DEFAULT_OLLAMA_MODEL = "mxbai-embed-large"
    _DEFAULT_OLLAMA_DIM = 1024
    _DEFAULT_LITELLM_MODEL = "text-embedding-3-large"
    _DEFAULT_LITELLM_DIM = 3072
    # Max number of distinct texts cached per instance by embed_text().
    _EMBED_TEXT_CACHE_MAXSIZE = 32

    def __init__(
        self,
        persist_dir: str,
        ollama_host: str = "http://localhost:11434",
        # Legacy positional kwarg preserved for backward compat with existing call sites.
        embed_model: str | None = None,
        *,
        papers_collection: str = _COLLECTION_NAME,
        chunks_collection: str = _CHUNKS_COLLECTION_NAME,
        # Bundle F: provider abstraction
        provider: str = "ollama",
        model: str | None = None,
        dim: int | None = None,
        # Optional state_db path for provenance lookup.  When None, no
        # provenance override is attempted (safe default for tests).
        state_db_path: str | None = None,
    ) -> None:
        # Validate provider before touching ChromaDB.
        if provider not in ("ollama", "litellm"):
            raise ValueError(
                f"provider must be 'ollama' or 'litellm'; got {provider!r}"
            )

        # Bundle F: provenance override — if this collection already has a
        # recorded provider/model/dim, use those values instead of the
        # constructor args.  ChromaDB collections are dim-locked on creation;
        # overriding with the stored provenance prevents corrupt inserts.
        resolved_provider, resolved_model, resolved_dim = self._resolve_provenance(
            collection_name=papers_collection,
            state_db_path=state_db_path,
            constructor_provider=provider,
            constructor_model=model or embed_model,  # embed_model is the legacy kwarg
            constructor_dim=dim,
        )

        self.provider = resolved_provider
        self.model = resolved_model
        self.dim = resolved_dim
        self._ollama_host = ollama_host.rstrip("/")
        # Keep _embed_model for backward compat with check_embed_model_change() +
        # the .embed_model property used by callers.
        self._embed_model = self.model

        # Telemetry already suppressed via os.environ above; no Settings object needed.
        self._client = chromadb.PersistentClient(path=persist_dir)
        # Collection names are instance-bound so the same class can be reused
        # for the production indices AND the /dev sandbox (which uses _dev_*).
        # Defaults preserve historic behaviour exactly.
        self._papers_collection_name = papers_collection
        self._chunks_collection_name = chunks_collection
        self._collection = self._client.get_or_create_collection(
            name=papers_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_collection = self._client.get_or_create_collection(
            name=chunks_collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Bundle F: provider resolution helpers (class-level, no ChromaDB)
    # ------------------------------------------------------------------
    @classmethod
    def _resolve_provenance(
        cls,
        *,
        collection_name: str,
        state_db_path: str | None,
        constructor_provider: str,
        constructor_model: str | None,
        constructor_dim: int | None,
    ) -> tuple[str, str, int]:
        """Return (provider, model, dim) after applying provenance override.

        If state_db_path is given and has a provenance row for collection_name,
        those values win.  Otherwise falls back to the constructor args (with
        sensible defaults per provider).
        """
        if state_db_path is not None:
            try:
                from scripts.core.state_db import StateDB
                sdb = StateDB(state_db_path)
                prov = sdb.get_embedding_provenance(collection_name)
                if prov is not None:
                    stored = (prov["provider"], prov["model"], prov["dim"])
                    given = (constructor_provider, constructor_model, constructor_dim)
                    if given != stored:
                        logger.warning(
                            "Bundle F: collection %r has stored provenance "
                            "(provider=%r model=%r dim=%d); ignoring constructor args %r.",
                            collection_name, prov["provider"], prov["model"],
                            prov["dim"], given,
                        )
                    return prov["provider"], prov["model"], prov["dim"]
            except Exception as exc:
                logger.debug(
                    "Bundle F: provenance lookup failed (%s); using constructor args.", exc
                )

        # No stored provenance — use constructor args with per-provider defaults.
        provider = constructor_provider
        if provider == "ollama":
            model = constructor_model or cls._DEFAULT_OLLAMA_MODEL
            dim = constructor_dim or cls._DEFAULT_OLLAMA_DIM
        else:  # litellm
            model = constructor_model or cls._DEFAULT_LITELLM_MODEL
            dim = constructor_dim or cls._DEFAULT_LITELLM_DIM
        return provider, model, dim
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
        for name in (self._papers_collection_name, self._chunks_collection_name):
            try:
                self._client.delete_collection(name)
            except Exception as exc:
                # ChromaDB raises ValueError or InvalidCollectionException if absent
                logger.debug("delete_collection raised (likely collection absent): %s", exc)
        self._collection = self._client.get_or_create_collection(
            name=self._papers_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_collection = self._client.get_or_create_collection(
            name=self._chunks_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collections '%s' and '%s' cleared.",
            self._papers_collection_name,
            self._chunks_collection_name,
        )

    def clear_chunks(self) -> None:
        """Delete and recreate the chunk-level collection only.

        Used by ``rechunk-all`` and C4 hot-swap (model change clears both via
        ``clear()``; this method handles the rechunk case where the paper-level
        index is still valid).
        """
        try:
            self._client.delete_collection(self._chunks_collection_name)
        except Exception as exc:
            logger.debug("delete_collection raised (likely chunks collection absent): %s", exc)
        self._chunks_collection = self._client.get_or_create_collection(
            name=self._chunks_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB chunks collection '%s' cleared.", self._chunks_collection_name)

    # ------------------------------------------------------------------
    # Bundle A: arbitrary-text embedding helper
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """Embed an arbitrary string using the current provider/model.

        Bundle A integration point: called once per discovery run to embed
        the ``domain_context`` paragraph, then reused for cosine scoring
        against each candidate.

        Cache design note
        -----------------
        Results are memoized in a per-instance bounded LRU
        (``self._embed_text_cache``, an OrderedDict capped at
        ``_EMBED_TEXT_CACHE_MAXSIZE``).  This deliberately replaces an earlier
        ``@lru_cache`` decorator: ``lru_cache`` on a bound method stores entries
        in a single *class-level* cache keyed by ``(self, text)``, which keeps a
        strong reference to every EmbeddingsDB instance ever embedded — a memory
        leak in long-running processes (the instance, its ChromaDB client, and
        its vectors can never be garbage-collected).  An instance-owned dict ties
        the cache lifetime to the instance, so it is freed when the instance is.
        The cache is created lazily so ``__new__``-constructed test doubles that
        skip ``__init__`` still work.

        Bundle F: routes through _embed_via_provider() so the configured
        provider (ollama or litellm) is used.  Bundle A's contract is
        preserved: returns np.ndarray float32.

        Parameters
        ----------
        text:
            The string to embed.  Must be non-empty.

        Returns
        -------
        np.ndarray
            1-D float32 vector matching the configured embedding model's
            dimensionality (1024 for mxbai-embed-large, 768 for nomic-embed-text).

        Raises
        ------
        ValueError
            If ``text`` is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError(
                "embed_text requires non-empty text; got an empty or whitespace-only string."
            )
        # Lazily create the per-instance cache (setdefault keeps __new__-built
        # test doubles working) and serve a hit, refreshing LRU recency.
        cache: OrderedDict[str, np.ndarray] = self.__dict__.setdefault(
            "_embed_text_cache", OrderedDict()
        )
        if text in cache:
            cache.move_to_end(text)
            return cache[text]
        # Bundle F: route through provider abstraction instead of hardcoded Ollama.
        # _embed_via_provider returns np.ndarray; for LiteLLM it is already float32.
        # For Ollama it goes through _embed (which handles truncation/retry) and
        # returns list[float] — we wrap in np.array here for uniform return type.
        vector = self._embed_via_provider(text)
        cache[text] = vector
        cache.move_to_end(text)
        if len(cache) > self._EMBED_TEXT_CACHE_MAXSIZE:
            cache.popitem(last=False)  # evict least-recently-used
        return vector

    # ------------------------------------------------------------------
    # Bundle F: provider dispatch
    # ------------------------------------------------------------------
    def _embed_via_provider(self, text: str) -> np.ndarray:
        """Bundle F: dispatch an embedding call to the configured provider.

        Returns a 1-D float32 numpy array.  Called by embed_text() and
        (for non-Ollama providers) by _embed().

        Uses getattr with defaults so __new__-constructed test instances
        without a full __init__ still work via the Ollama path.
        """
        provider = getattr(self, "provider", "ollama")
        if provider != "litellm":
            # Ollama path: route through _embed so the truncation/retry loop
            # (context-length handling) still applies, and so existing tests
            # that patch _embed continue to intercept the call correctly.
            raw: list[float] = self._embed(text)
            return np.array(raw, dtype=np.float32)
        else:
            from scripts.llm.embedding_client import embed_via_litellm
            model = getattr(self, "model", self._DEFAULT_LITELLM_MODEL)
            return embed_via_litellm(text, model=model)

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
        """Embed text using the configured provider and return the vector.

        For the 'ollama' provider: applies truncation/retry logic on context-
        length errors (same behavior as before Bundle F).

        For the 'litellm' provider: delegates directly to embed_via_litellm
        (no truncation — LiteLLM providers generally support longer contexts).

        On HTTPError ``context length exceeded`` (Ollama only), transparently
        retries with progressively shorter input — halving each round — until
        either the embed succeeds or the input drops below
        ``_EMBED_TRUNCATE_FLOOR_CHARS``.  Each truncation step logs a WARNING.

        Other 400s (model not found, malformed input) are NOT retried; they
        propagate after capturing the response body in the ERROR log.
        """
        # Bundle F: LiteLLM path — skip Ollama-specific truncation/retry.
        # Use getattr so __new__-constructed test instances (no __init__) default
        # to the Ollama path and preserve pre-Bundle-F behavior.
        if getattr(self, "provider", "ollama") != "ollama":
            return self._embed_via_provider(text).tolist()

        # Ollama path: truncation + retry loop (pre-Bundle-F behavior preserved).
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

        Reads OLLAMA_API_KEY from the env and adds a Bearer Authorization
        header when present — required for cloud Ollama embed models (e.g.
        ``mxbai-embed-large:cloud``). Local Ollama ignores the extra header,
        so this is strictly additive.
        """
        import json
        import os
        import urllib.error
        import urllib.request
        payload = json.dumps({"model": self._embed_model, "input": text}).encode()
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OLLAMA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            url=f"{self._ollama_host}/api/embed",
            data=payload,
            headers=headers,
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
        # Idempotency / crash-recovery guard: kv_store says we're already on the
        # configured model, but the ChromaDB collection is empty. That can only
        # happen if a previous run completed clear() but crashed before
        # reset_embeddings_indexed() + set_kv() — except set_kv is now LAST, so
        # an interrupted prior run will leave kv_store at the OLD model and we
        # would enter the reset branch instead. The case we recover here is the
        # narrower one where the collection is independently empty (manual wipe,
        # fresh chromadb dir) while state_db still flags papers as indexed —
        # bring state_db back in sync so the next run re-embeds.
        try:
            collection_empty = embed_db.count() == 0
        except Exception as exc:
            logger.debug("embed_db.count() raised during idempotency check: %s", exc)
            return False
        if collection_empty:
            logger.warning(
                "Embed model unchanged ('%s') but ChromaDB collection is empty. "
                "Resetting embeddings_indexed so papers are re-embedded.",
                configured_model,
            )
            state_db.reset_embeddings_indexed()
            # set_kv last so a crash here is re-runnable (still no-op next time).
            state_db.set_kv("embed_model", configured_model)
            return True
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
    # set_kv LAST: if a crash interrupts us before this point, kv_store still
    # holds the old model name so the next run re-enters this branch and
    # completes the work. The clear() step is idempotent (deleting an
    # already-empty collection is a no-op).
    state_db.set_kv("embed_model", configured_model)
    return True
