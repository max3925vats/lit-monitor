"""
GraphDB — thin wrapper around a KuzuDB embedded database.

Mirrors the shape of ``scripts.output.embeddings.EmbeddingsDB``:
  - ``__init__(self, persist_dir: str)``
  - lazy import of kuzu (only at instantiation, not at module load)
  - idempotent init via DDL ``IF NOT EXISTS`` guards in ``apply_schema``

Schema versioning (G14)
-----------------------
The current SCHEMA_VERSION is stored as plain text in a sentinel file whose
name is derived from the KuzuDB path:

    <persist_dir>.schema_version

For example: ``~/.config/lit-monitor/graph.kuzu.schema_version``

On ``__init__``:
1. ``apply_schema`` runs the full DDL (IF NOT EXISTS — idempotent on
   already-initialised databases).
2. The sentinel file is read; if absent, version is assumed to be 1
   (the G1 baseline — database existed before G14 shipped).
3. ``apply_migrations`` is called with the persisted version; any pending
   migrations run and the new version is returned.
4. The sentinel file is written back with the current SCHEMA_VERSION.

The kuzu import is deferred so that ``import scripts.graph`` succeeds even
when kuzu is not installed.  Only calling ``GraphDB(...)`` will raise an
``ImportError`` in that case — non-graph users see no disruption.
"""
from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path
from typing import Any

from lit_monitor.graph.migrations import apply_migrations, apply_schema
from lit_monitor.graph.normalizer import EntityNormalizer
from lit_monitor.graph.relationship_validator import VALID_PREDICATES

logger = logging.getLogger(__name__)

# Queries slower than this threshold emit a WARN-level "slow query" line.
_SLOW_QUERY_THRESHOLD_S: float = 0.5

# Sentinel value written into NOT-NULL-ish STRING and INT columns when the
# caller's tuple has Python None (Kuzu CREATE does not accept NULL bindings on
# columns without a DEFAULT for nullability).  These sentinels are inert for
# downstream queries — span_start/end of -1 means "no character offset" and
# matches the Phase 2 NER convention.
_NULL_STRING = ""
_NULL_INT = -1


def _coerce_authors(authors: Any) -> str:
    """Normalise the authors value from paper_metadata into a semicolon-separated string.

    Accepts:
    - str  → returned as-is (caller may use "; " or ";" as separator)
    - list → joined with "; "
    - None / falsy → empty string

    Used by _upsert_paper_node (Bundle D: Paper.authors column v5).
    """
    if not authors:
        return ""
    if isinstance(authors, list):
        return "; ".join(str(a) for a in authors if a)
    return str(authors)


class GraphDB:
    """Embedded KuzuDB knowledge graph for lit-monitor.

    Parameters
    ----------
    persist_dir:
        Path to the KuzuDB file.  Parent directory is created (including
        parents) if absent.  Use ``~/.config/lit-monitor/graph.kuzu`` in
        production.

    Raises
    ------
    ImportError
        When ``kuzu`` is not importable (i.e. the ``[graph]`` optional extra
        has not been installed).  Install with ``uv sync --extra graph``.
    """

    def __init__(self, persist_dir: str) -> None:
        # Expand ~ and resolve so the path is always absolute.
        # ``persist_dir`` is the path passed to kuzu.Database().  KuzuDB stores
        # its data at this path as a file (not a directory).  We ensure only
        # the PARENT directory exists — kuzu.Database() creates the file itself.
        self._persist_dir = Path(persist_dir).expanduser()
        self._persist_dir.parent.mkdir(parents=True, exist_ok=True)

        # Sentinel file that persists the schema version across process restarts.
        # Stored alongside the DB file: <persist_dir>.schema_version
        self._version_file = Path(str(self._persist_dir) + ".schema_version")

        # Lazy kuzu import — deferred so non-graph users never hit ImportError
        # at module-load time.  Only instantiation requires kuzu to be present.
        try:
            import kuzu  # noqa: PLC0415  (intentional lazy import)
        except ImportError as exc:
            raise ImportError(
                "kuzu is required for GraphDB but is not installed. "
                "Install with: uv sync --extra graph"
            ) from exc

        logger.debug("Opening KuzuDB at %s", self._persist_dir)
        self._db = kuzu.Database(str(self._persist_dir))
        self._conn = kuzu.Connection(self._db)

        # Step 1: Apply the full schema DDL — idempotent on re-open.
        apply_schema(self._conn)

        # Step 2: Read persisted schema version (absent → G1 baseline = 1).
        persisted_version = self._read_schema_version()

        # Step 3: Run any pending migrations.
        new_version = apply_migrations(self._conn, current_version=persisted_version)

        # Step 4: Persist the updated version so future opens skip migrations.
        if new_version != persisted_version:
            self._write_schema_version(new_version)
            logger.info(
                "GraphDB migrated: schema v%d → v%d at %s",
                persisted_version,
                new_version,
                self._persist_dir,
            )

        # G7: query-time EntityNormalizer cache.  None = not yet built.
        # P3.2: add_paper() now calls invalidate_query_cache() on success, so
        # resolve_query_entity sees newly-added entities in the same process
        # without a manual invalidate.
        self._query_normalizer: EntityNormalizer | None = None

        logger.debug(
            "GraphDB ready (schema v%d): %s", new_version, self._persist_dir
        )

    # ------------------------------------------------------------------
    # Schema version helpers
    # ------------------------------------------------------------------

    def _read_schema_version(self) -> int:
        """Return the persisted schema version, or 1 if the sentinel is absent.

        Version 1 is the G1 baseline: a database created before G14 shipped
        and therefore lacking the three provenance columns.
        """
        if not self._version_file.exists():
            # Pre-G14 database — treat as v1 so migrations run on first open.
            return 1
        try:
            return int(self._version_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError) as exc:
            logger.warning(
                "Could not read schema version from %s (%s); defaulting to 1.",
                self._version_file,
                exc,
            )
            return 1

    def _write_schema_version(self, version: int) -> None:
        """Write the schema version integer to the sentinel file.

        Logs a warning on OSError rather than raising — by the time this is
        called, schema migrations have already committed to the database, so
        failing the whole __init__ would leave the user with no way to use
        the graph at all. Stale sentinel is recoverable; an unconstructable
        GraphDB is not.
        """
        try:
            self._version_file.write_text(str(version), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Could not write schema version to %s (%s); "
                "migration will re-run on next open.",
                self._version_file,
                exc,
            )

    def close(self) -> None:
        """Release the KuzuDB connection and database handle deterministically.

        Explicitly calls ``close()`` on the kuzu Connection then Database (both
        expose it as of kuzu 0.11) so file locks and buffer-pool memory are freed
        immediately rather than waiting on CPython refcounting / GC. References are
        then dropped. Safe to call multiple times and tolerant of partially-built
        instances (e.g. ``__new__``-constructed test doubles missing these fields).
        """
        # Close connection before database — the connection holds a handle into
        # the database's buffer pool. getattr() guards instances that never ran
        # __init__; the broad except keeps close() idempotent if kuzu raises on a
        # double-close or an already-released handle.
        for attr in ("_conn", "_db"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:  # noqa: BLE001 — best-effort teardown
                    logger.debug("GraphDB %s.close() raised: %s", attr, exc)
        self._conn = None  # type: ignore[assignment]
        self._db = None    # type: ignore[assignment]
        logger.debug("GraphDB closed: %s", self._persist_dir)

    def __enter__(self) -> GraphDB:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # G6 — atomic write entrypoint
    # ------------------------------------------------------------------

    def add_paper(
        self,
        doi: str,
        entities: list[Any],
        relationships: list[Any],
        paper_metadata: dict[str, Any],
        *,
        prompt_version: str = "phase1.0",
        extracted_at: _dt.datetime | None = None,
    ) -> None:
        """G6: atomically write one paper + its entities + its typed relationships.

        Wraps the Paper-node UPSERT, all Entity-node UPSERTs, all MENTIONS
        edges and all typed predicate edges in a single Kuzu transaction.
        On any exception during the writes, the transaction is ROLLed BACK
        and the exception is re-raised so the R28 ingest helper can leave
        ``papers.graph_indexed = 0``.

        Idempotency: each CREATE is guarded by a count-based existence check
        — re-running ``add_paper`` for the same (doi, entity, relationship)
        triple is a no-op for the duplicate rows.  Kuzu's MERGE support is
        too limited in 0.11.x to use for this; explicit existence checks
        within a single transaction are the most portable path.

        Edge provenance: every written MENTIONS and typed predicate edge
        explicitly sets ``prompt_version`` so the G14 provenance columns are
        guaranteed populated even if a future Kuzu version drops the column
        DEFAULTs.  ``confidence`` is taken from each RelationshipTuple.

        Parameters
        ----------
        doi:
            Source paper's DOI (Paper primary key).
        entities:
            List of ``EntityTuple`` (from ``scripts.graph.entity_extractor``).
        relationships:
            List of validated ``RelationshipTuple`` (from
            ``scripts.graph.relationship_extractor``).  Each tuple's
            ``predicate`` MUST be in ``VALID_PREDICATES`` — the validator
            upstream guarantees this; we re-check defensively because the
            predicate name is interpolated into the Cypher REL label and a
            regression dropping validation must NOT permit injection.
        paper_metadata:
            Dict with ``title``, ``year`` (int), ``journal`` keys.  Missing
            keys default to empty string / 0 to avoid CREATE failures.
        prompt_version:
            Provenance tag written onto every edge created in this call.
            Defaults to ``"phase1.0"`` (the Phase 1 tag).
        extracted_at:
            Wall-clock extraction/ingest timestamp stamped onto every edge
            (P3.1). When ``None`` (default) the current time is captured once
            at call start and applied to every edge in this call, so all edges
            from one ``add_paper`` share a consistent timestamp. Passing an
            explicit value records that instant instead of relying on the DDL
            ``current_timestamp()`` DEFAULT (which would record *migration*
            time on some Kuzu paths, not extraction time).

        Raises
        ------
        ValueError
            If any relationship's predicate is not in VALID_PREDICATES (the
            f-string-into-Cypher safety check).
        Exception
            Any Kuzu-side failure during the txn.  ROLLBACK is attempted
            before the exception propagates.
        """
        conn = self._conn
        if conn is None:
            raise RuntimeError("GraphDB is closed; cannot add_paper")

        # P3.1: capture extraction time once so every edge in this call shares
        # a single, consistent timestamp instead of relying on the per-edge
        # DDL current_timestamp() DEFAULT (which can record migration time).
        if extracted_at is None:
            extracted_at = _dt.datetime.now()

        # Defence-in-depth: predicates are interpolated into Cypher REL labels,
        # so the validator upstream is the only thing standing between user
        # input and Cypher injection.  Re-check here so a regression in the
        # validator cannot reach Kuzu.
        for rel in relationships:
            if rel.predicate not in VALID_PREDICATES:
                raise ValueError(
                    f"add_paper: predicate {rel.predicate!r} not in VALID_PREDICATES; "
                    f"refusing to interpolate into Cypher REL label."
                )

        conn.execute("BEGIN TRANSACTION")
        try:
            self._upsert_paper_node(doi, paper_metadata)
            for ent in entities:
                self._upsert_entity_and_mentions(
                    doi, ent, prompt_version, extracted_at
                )
            for rel in relationships:
                self._upsert_typed_edge(rel, prompt_version, extracted_at)
            conn.execute("COMMIT")
            # A1: invalidate schema cache after any successful write so that
            # subsequent describe_schema calls reflect the current graph state.
            # Best-effort: an import failure here must never break the R28
            # ingest invariant — schema cache staleness is benign.
            try:
                from lit_monitor.graph.schema_describer import (  # noqa: PLC0415
                    invalidate_schema_cache,
                )
                invalidate_schema_cache(self)
            except Exception:  # pragma: no cover — defensive
                pass
            # P3.2: drop the cached query-time EntityNormalizer so newly-added
            # entities/aliases resolve in subsequent queries within the same
            # process without a manual invalidate_query_cache() call.
            self.invalidate_query_cache()
        except Exception as exc:
            # Best-effort rollback — if the rollback itself raises, log and
            # re-raise the ORIGINAL exception so the caller sees the real cause.
            try:
                conn.execute("ROLLBACK")
            except Exception as rb_exc:  # pragma: no cover — defensive
                logger.error(
                    "add_paper: ROLLBACK also failed for %s: %s "
                    "(original: %s)", doi, rb_exc, exc,
                )
            logger.warning(
                "add_paper rolled back for doi=%s: %s", doi, exc,
            )
            raise

    # ------------------------------------------------------------------
    # G6 — internal helpers (one Cypher op per method)
    # ------------------------------------------------------------------

    def _upsert_paper_node(
        self, doi: str, paper_metadata: dict[str, Any]
    ) -> None:
        """Create the Paper node iff missing.  Title/year/journal/authors coerced safely.

        Bundle D (v5): authors stored as a semicolon-separated STRING so
        ``get_graph_signals_for_candidate`` can compute author overlap without
        leaving the graph.  Empty string when caller provides no authors key.
        """
        conn = self._conn
        res = conn.execute(
            "MATCH (p:Paper {doi: $d}) RETURN count(p)",
            {"d": doi},
        )
        if int(res.get_next()[0]) > 0:
            return
        year_val = paper_metadata.get("year") or 0
        try:
            year_int = int(year_val)
        except (TypeError, ValueError):
            year_int = 0
        conn.execute(
            "CREATE (p:Paper {doi: $d, title: $t, year: $y, journal: $j, authors: $a})",
            {
                "d": doi,
                "t": str(paper_metadata.get("title") or ""),
                "y": year_int,
                "j": str(paper_metadata.get("journal") or ""),
                # authors: accept list or semicolon-separated string; normalise to string
                "a": _coerce_authors(paper_metadata.get("authors")),
            },
        )

    def _upsert_entity_and_mentions(
        self,
        doi: str,
        ent: Any,
        prompt_version: str,
        extracted_at: _dt.datetime,
    ) -> None:
        """Create Entity node and MENTIONS edge iff missing.

        MENTIONS uniqueness is (paper_doi, entity_canonical_id, field) — the
        same entity surfacing in two different extraction fields produces two
        edges, matching G3's dedup key.

        N4 duck-typing: accepts both ``EntityTuple`` (canonical_id, no source)
        and ``MentionEdge`` (entity_key, source, confidence).  Fields not
        present on EntityTuple default to 'schema' / 1.0 so the existing
        schema-only path is unchanged.
        """
        conn = self._conn

        # N4: MentionEdge uses entity_key; EntityTuple uses canonical_id.
        # getattr fallback keeps the call-site identical for both types.
        cid: str = getattr(ent, "entity_key", None) or ent.canonical_id
        # N4: MentionEdge carries per-source provenance; EntityTuple defaults
        # to 'schema' / 1.0 (unchanged from G6 behaviour).
        source: str = getattr(ent, "source", "schema")
        confidence: float = float(getattr(ent, "confidence", 1.0))

        # 1. Entity node — create if missing.
        res = conn.execute(
            "MATCH (e:Entity {canonical_id: $cid}) RETURN count(e)",
            {"cid": cid},
        )
        if int(res.get_next()[0]) == 0:
            conn.execute(
                "CREATE (e:Entity {canonical_id: $cid, type: $t, surface: $s})",
                {
                    "cid": cid,
                    "t": str(ent.type),
                    "s": str(ent.surface),
                },
            )

        # 2. MENTIONS edge — keyed on (doi, canonical_id, field).
        # Kuzu STRING columns on CREATE want a real value, so None -> "".
        field_val = ent.field if ent.field is not None else _NULL_STRING
        res = conn.execute(
            "MATCH (p:Paper {doi: $d})-[m:MENTIONS]->(e:Entity {canonical_id: $cid}) "
            "WHERE m.field = $f RETURN count(m)",
            {"d": doi, "cid": cid, "f": field_val},
        )
        if int(res.get_next()[0]) > 0:
            return
        span_start = ent.span_start if ent.span_start is not None else _NULL_INT
        span_end = ent.span_end if ent.span_end is not None else _NULL_INT
        conn.execute(
            "MATCH (p:Paper {doi: $d}), (e:Entity {canonical_id: $cid}) "
            "CREATE (p)-[m:MENTIONS {"
            "source: $src, surface: $s, field: $f, "
            "span_start: $ss, span_end: $se, "
            "confidence: $conf, prompt_version: $pv, extracted_at: $ea"
            "}]->(e)",
            {
                "d": doi,
                "cid": cid,
                "src": source,
                "s": str(ent.surface),
                "f": field_val,
                "ss": int(span_start),
                "se": int(span_end),
                "conf": confidence,
                "pv": prompt_version,
                # P3.1: stamp extraction time explicitly, not migration time.
                "ea": extracted_at,
            },
        )

    def _upsert_typed_edge(
        self,
        rel: Any,
        prompt_version: str,
        extracted_at: _dt.datetime,
    ) -> None:
        """G6+R3: write one typed predicate edge with multi-source dedup.

        R3 invariants (multi-source):
        1. Same (subject, predicate, target) from different prompt_version values
           (e.g. "phase1.0_schema" vs "phase3.0_llm") produces TWO distinct edges,
           one per source.
        2. Same (subject, predicate, target) re-run with the SAME prompt_version
           is idempotent (no duplicate).
        3. Different (subject, predicate, target) combinations always produce
           distinct edges regardless of source.
        4. Phase 1 schema-only path: works unchanged for callers that use the
           default prompt_version="phase1.0".

        Predicate label is interpolated from ``rel.predicate`` — caller has
        already validated against VALID_PREDICATES; ``add_paper`` re-checks.
        Idempotency key is (source_doi, predicate, target_id, prompt_version).
        """
        conn = self._conn
        pred = rel.predicate

        if rel.target_kind == "Entity":
            target_match = "(t:Entity {canonical_id: $tid})"
        else:
            target_match = "(t:Paper {doi: $tid})"

        # Existence check — count edges with this predicate between endpoints
        # AND matching prompt_version so schema + LLM sources produce distinct edges.
        check_q = (
            f"MATCH (s:Paper {{doi: $sd}})-[r:{pred}]->{target_match} "
            "WHERE r.prompt_version = $pv "
            "RETURN count(r)"
        )
        res = conn.execute(check_q, {"sd": rel.source_doi, "tid": rel.target_id, "pv": prompt_version})
        if int(res.get_next()[0]) > 0:
            return

        create_q = (
            f"MATCH (s:Paper {{doi: $sd}}), {target_match} "
            f"CREATE (s)-[r:{pred} {{"
            "evidence: $ev, confidence: $cf, prompt_version: $pv, "
            "extracted_at: $ea"
            "}]->(t)"
        )
        conn.execute(
            create_q,
            {
                "sd": rel.source_doi,
                "tid": rel.target_id,
                "ev": str(rel.evidence or ""),
                "cf": float(rel.confidence),
                "pv": prompt_version,
                # P3.1: stamp extraction time explicitly, not migration time.
                "ea": extracted_at,
            },
        )

    # ------------------------------------------------------------------
    # G7 — slow-query helper
    # ------------------------------------------------------------------

    def _slow_log(self, method: str, elapsed: float) -> None:
        """Emit WARN if a read query exceeded the 500 ms threshold."""
        if elapsed > _SLOW_QUERY_THRESHOLD_S:
            logger.warning(
                "GraphDB.%s: slow query (%.3fs > %.3fs)",
                method,
                elapsed,
                _SLOW_QUERY_THRESHOLD_S,
            )

    def invalidate_query_cache(self) -> None:
        """G7: drop the cached query-time EntityNormalizer.

        P3.2: add_paper() calls this automatically on a successful write, so
        resolve_query_entity sees newly-added entities/aliases in the same
        process.  Still exposed for callers that mutate the graph by other
        means and need to force a rebuild.
        """
        self._query_normalizer = None

    # ------------------------------------------------------------------
    # G7 — read-side query API
    # ------------------------------------------------------------------

    def find_similar_papers(
        self,
        doi: str,
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """G7: papers sharing the most MENTIONS with the given paper.

        Traverses Entity nodes shared between the source paper and every
        other paper, groups by neighbour paper and counts shared entities,
        then returns the top-k by count descending.

        Returns
        -------
        list[tuple[str, float]]
            ``[(doi, score), ...]`` ranked by shared-entity count desc.
            Self is always excluded.  Empty list when the source DOI has no
            entities or does not exist in the graph.
        """
        cypher = (
            "MATCH (a:Paper {doi: $doi})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(b:Paper) "
            "WHERE a.doi <> b.doi "
            "RETURN b.doi AS doi, count(DISTINCT e) AS score "
            "ORDER BY score DESC LIMIT $k"
        )
        t0 = time.perf_counter()
        res = self._conn.execute(cypher, {"doi": doi, "k": int(k)})
        elapsed = time.perf_counter() - t0
        self._slow_log("find_similar_papers", elapsed)

        out: list[tuple[str, float]] = []
        while res.has_next():
            row = res.get_next()
            out.append((str(row[0]), float(row[1])))
        return out

    def resolve_query_entity(
        self,
        text: str,
        type_: str | None = None,
    ) -> str | None:
        """G7: normalize a query string through G2's EntityNormalizer + per-type vocab.

        Uses the query-time fuzzy threshold (``EntityNormalizer.DEFAULT_QUERY_RATIO``
        = 85), which is looser than the ingest threshold (90).  This lets user
        typos and minor spelling variations resolve to an existing canonical ID.

        The per-type vocabulary is seeded from the live Entity nodes on the first
        call and then cached on the ``GraphDB`` instance in ``_query_normalizer``.

        .. note::
            P3.2: ``add_paper`` invalidates ``_query_normalizer`` on every
            successful write, so entities added via ``add_paper`` are visible to
            the fuzzy matcher on the next resolve call within the same process.
            The cache is rebuilt lazily from the live Entity table on first use
            after invalidation.

        Parameters
        ----------
        text:
            Raw query string (surface form) to resolve.
        type_:
            Restrict resolution to this entity type.  ``None`` tries all known
            types (slower; not recommended for hot paths).

        Returns
        -------
        str | None
            The matched canonical_id, or ``None`` if no match found within scope.
        """
        from lit_monitor.graph.aliases import load_aliases

        t0 = time.perf_counter()

        # Build (and cache) the normalizer seeded with live Entity nodes.
        # P3.2: add_paper() invalidates this cache on write, so the rebuild
        # below picks up newly-added entities automatically.
        if self._query_normalizer is None:
            normalizer: EntityNormalizer = EntityNormalizer(aliases=load_aliases())
            res = self._conn.execute(
                "MATCH (e:Entity) RETURN e.canonical_id, e.type"
            )
            while res.has_next():
                row = res.get_next()
                # add_to_vocab(type_, canonical) — row order: [canonical_id, type]
                normalizer.add_to_vocab(str(row[1]), str(row[0]))
            self._query_normalizer = normalizer

        normalizer = self._query_normalizer

        if type_ is not None:
            canonical, _via = normalizer.normalize(
                text,
                type_=type_,
                min_ratio=EntityNormalizer.DEFAULT_QUERY_RATIO,
            )
            elapsed = time.perf_counter() - t0
            self._slow_log("resolve_query_entity", elapsed)
            # Only return when the result is actually in the vocab for this type —
            # identity passthrough (no alias/fuzzy hit) still returns the basic-
            # normalized form, but that form may not exist in the graph.
            existing = normalizer._vocab.get(type_, set())
            return canonical if canonical in existing else None

        # type_=None: try all known types in a fixed priority order, return first hit.
        for t in ("topic", "method", "material", "author", "journal", "keyword"):
            canonical, _via = normalizer.normalize(
                text,
                type_=t,
                min_ratio=EntityNormalizer.DEFAULT_QUERY_RATIO,
            )
            existing = normalizer._vocab.get(t, set())
            if canonical in existing:
                elapsed = time.perf_counter() - t0
                self._slow_log("resolve_query_entity", elapsed)
                return canonical

        elapsed = time.perf_counter() - t0
        self._slow_log("resolve_query_entity", elapsed)
        return None

    def find_papers_by_entities(
        self,
        canonical_ids: list[str],
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """G7: papers ranked by MENTIONS overlap into a given entity set.

        For each paper, counts how many of its MENTIONS edges point into the
        provided ``canonical_ids`` set, then returns the top-k by count desc.
        Designed to accept the output of :meth:`resolve_query_entity` calls and
        feed the RRF scorer in G8.

        Parameters
        ----------
        canonical_ids:
            List of canonical entity IDs to match against.  Empty list returns
            ``[]`` without touching the database.
        k:
            Maximum number of results.

        Returns
        -------
        list[tuple[str, float]]
            ``[(doi, score), ...]`` ranked by overlap count desc.
        """
        if not canonical_ids:
            return []

        cypher = (
            "MATCH (p:Paper)-[:MENTIONS]->(e:Entity) "
            "WHERE e.canonical_id IN $ids "
            "RETURN p.doi AS doi, count(DISTINCT e) AS score "
            "ORDER BY score DESC LIMIT $k"
        )
        t0 = time.perf_counter()
        res = self._conn.execute(
            cypher, {"ids": list(canonical_ids), "k": int(k)}
        )
        elapsed = time.perf_counter() - t0
        self._slow_log("find_papers_by_entities", elapsed)

        out: list[tuple[str, float]] = []
        while res.has_next():
            row = res.get_next()
            out.append((str(row[0]), float(row[1])))
        return out
