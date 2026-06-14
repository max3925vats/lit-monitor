"""Bundle E (v0.9): library-activity ("trending") concept detection.

Surfaces the concepts you've been adding to your library most actively in the
recent window. It identifies entities whose MENTIONS count grew significantly
over the recent 60-day window vs the prior 60-day window, where the window is
keyed on each edge's ``extracted_at`` timestamp.

IMPORTANT — what "trending" means here. ``extracted_at`` is the wall-clock
INGEST/extraction time the MENTIONS edge was written to the graph (it defaults
to ``current_timestamp()`` in the Kuzu DDL and ``add_paper`` never overrides
it). So the 60-day windows bucket papers by WHEN THEY WERE ADDED TO YOUR
LIBRARY, not by when they were published. This feature therefore measures
ingest-recency / library-activity ("what am I reading and adding a lot of
lately"), NOT field-level publication trends. Brain-build papers carry only a
publication ``year``, so true publication-recency windows are not buildable
here anyway.

Results are suggestion-only — never auto-applied to topics.yaml (per locked
decision #4).

Defensive perimeter: find_trending_concepts() never raises. All failures
are logged at WARNING level and an empty list is returned.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from lit_monitor.core.path_utils import resolve_path as _resolve_path
from lit_monitor.graph.db import GraphDB

logger = logging.getLogger(__name__)

# Module-level *logical* config path; the read below routes it through
# resolve_path() so a wheel run from an arbitrary CWD still finds the user's
# topics.yaml (the dev/editable case still resolves ./config). Tests may
# monkeypatch this for isolation.
_TOPICS_PATH = Path("config/topics.yaml")

# Window lengths in days for growth-rate calculation.
_WINDOW_DAYS = 60


def find_trending_concepts(
    graph_db: Any,
    state_db: Any,
    cfg: Any,
    *,
    threshold_growth_rate: float | None = None,
    min_recent_mentions: int | None = None,
    cooldown_days: int | None = None,
) -> list[dict]:
    """Bundle E: detect concepts recently active in the user's library.

    Counts MENTIONS edges whose extracted_at falls within the recent 60-day
    window, compares to the prior 60-day window, and filters by growth-rate
    threshold and minimum noise floor. Cooldown prevents re-suggesting concepts
    dismissed within the configured window.

    Because extracted_at is INGEST time (see module docstring), this measures
    library-activity recency — concepts you've been adding to your library a lot
    lately — not field-level publication trends.

    Args:
        graph_db:              GraphDB instance (must expose ._conn.execute).
        state_db:              StateDB instance (for cooldown lookup + persistence).
        cfg:                   Config namespace; reads cfg.trending_concepts.*.
        threshold_growth_rate: Override config threshold (default 0.3 = 30%).
        min_recent_mentions:   Override config noise floor (default 5).
        cooldown_days:         Override config cooldown window (default 60 days).

    Returns:
        List of dicts: {concept_text, concept_type, n_mentions_new,
        n_mentions_prev, growth_rate, in_existing_topics}.
        Returns [] on any failure.
    """
    # --- Resolve config ---
    cfg_t = getattr(cfg, "trending_concepts", None)
    if threshold_growth_rate is None:
        threshold_growth_rate = float(getattr(cfg_t, "threshold_growth_rate", 0.3))
    if min_recent_mentions is None:
        min_recent_mentions = int(getattr(cfg_t, "min_recent_mentions", 5))
    if cooldown_days is None:
        cooldown_days = int(getattr(cfg_t, "cooldown_days_after_dismiss", 60))

    try:
        # --- Fetch raw mention counts from graph ---
        raw_rows = _query_mention_counts(graph_db)
        if not raw_rows:
            return []

        # --- Resolve dismissed concepts for cooldown ---
        dismissed = set()
        try:
            dismissed = set(state_db.get_dismissed_trending_concepts(cooldown_days))
        except Exception as exc:
            logger.info("E: could not load dismissed concepts (non-fatal): %s", exc)

        # --- Build result list ---
        results: list[dict] = []
        for row in raw_rows:
            concept_text: str = row["concept_text"]
            concept_type: str = row["concept_type"]
            n_new: int = row["n_mentions_new"]
            n_prev: int = row["n_mentions_prev"]

            # Noise floor: skip if too few recent mentions
            if n_new < min_recent_mentions:
                continue

            # Growth-rate filter
            growth = _compute_growth_rate(n_new, n_prev)
            if growth < threshold_growth_rate:
                continue

            # Cooldown: skip if concept was recently dismissed
            if concept_text.lower() in dismissed:
                continue

            results.append(
                {
                    "concept_text": concept_text,
                    "concept_type": concept_type,
                    "n_mentions_new": n_new,
                    "n_mentions_prev": n_prev,
                    "growth_rate": growth,
                    "in_existing_topics": _check_in_existing_topics(concept_text),
                }
            )

        return results

    except Exception as exc:
        logger.warning("E: trending detection failed: %s", exc)
        return []


def detect_and_persist_trending(cfg: Any, state_db: Any) -> int:
    """Open the graph, detect trending concepts, persist them, return the count.

    Shared code path for the CLI ``trending suggest`` command and the web
    ``POST /api/trending/detect`` route, so both run identical detection +
    persistence (DRY). Opens the Kuzu graph at
    ``cfg.retrieval.graph_db.persist_dir``, runs ``find_trending_concepts``, and
    persists each result via ``state_db.persist_trending_suggestion``.

    This helper does NOT gate on ``cfg.trending_concepts.enabled`` — callers are
    responsible for that feature gate. It is also deliberately *not* defensive
    about opening the graph: if the graph DB cannot be opened, the underlying
    exception propagates so the caller can surface a real error rather than a
    misleading count of 0.

    Args:
        cfg:      Config namespace (reads ``cfg.retrieval.graph_db.persist_dir``).
        state_db: StateDB instance used for cooldown lookup + persistence.

    Returns:
        The number of suggestions persisted.

    Raises:
        Exception: Whatever the GraphDB constructor raises if the graph cannot
            be opened (e.g. graph not built / DB error).
    """
    graph_path = Path(cfg.retrieval.graph_db.persist_dir).expanduser()
    # GraphDB construction may raise (missing extra, graph not built, corrupt
    # store) — let it propagate; the caller decides how to surface it.
    graph_db = GraphDB(str(graph_path))
    with graph_db:
        suggestions = find_trending_concepts(graph_db, state_db, cfg)

    persisted = 0
    for s in suggestions:
        state_db.persist_trending_suggestion(
            concept_text=s["concept_text"],
            concept_type=s["concept_type"],
            n_mentions_new=s["n_mentions_new"],
            n_mentions_prev=s["n_mentions_prev"],
            growth_rate=s["growth_rate"],
        )
        persisted += 1
    return persisted


def _query_mention_counts(graph_db: Any) -> list[dict]:
    """Query MENTIONS edges grouped by entity, split into recent vs prior window.

    Uses extracted_at TIMESTAMP on MENTIONS edges to assign each mention to
    one of two 60-day buckets relative to today. Returns a list of dicts with
    keys: concept_text, concept_type, n_mentions_new, n_mentions_prev.

    NOTE: extracted_at is the INGEST/extraction time the edge was written, not
    the paper's publication date — so these windows bucket by library-activity
    recency (when papers were added), not by publication recency.

    Kuzu supports TIMESTAMP arithmetic and interval() — we use date subtraction
    via string comparison on ISO timestamps (Kuzu stores TIMESTAMP as ISO string
    internally for comparison purposes, matching SQLite/Cypher patterns).
    """
    # Kuzu TIMESTAMP comparisons work with ISO-8601 string literals.
    # Interval syntax: interval('60 days') is the Kuzu way to express durations.
    # We compute both windows relative to current_timestamp().
    # NOTE: m.extracted_at is INGEST time (DEFAULT current_timestamp() in the DDL,
    # never overridden by add_paper), so these windows measure when papers were
    # ADDED to the library, i.e. ingest-recency — not publication-recency.
    cypher = (
        "MATCH (p:Paper)-[m:MENTIONS]->(e:Entity) "
        "WHERE m.extracted_at IS NOT NULL "
        "WITH e, m, "
        "  current_timestamp() AS now "
        "WITH e, m, now, "
        "  (now - interval('60 days')) AS cutoff_recent, "
        "  (now - interval('120 days')) AS cutoff_prior "
        "WITH e, "
        "  count(CASE WHEN m.extracted_at >= cutoff_recent THEN 1 END) AS n_new, "
        "  count(CASE WHEN m.extracted_at >= cutoff_prior "
        "              AND m.extracted_at < cutoff_recent THEN 1 END) AS n_prev "
        "WHERE n_new > 0 "
        "RETURN e.surface, e.type, n_new, n_prev "
        "ORDER BY n_new DESC "
        "LIMIT 500"
    )
    result = graph_db._conn.execute(cypher)
    rows: list[dict] = []
    while result.has_next():
        row = result.get_next()
        rows.append(
            {
                "concept_text": str(row[0]) if row[0] else "",
                "concept_type": str(row[1]) if row[1] else "keyword",
                "n_mentions_new": int(row[2]) if row[2] is not None else 0,
                "n_mentions_prev": int(row[3]) if row[3] is not None else 0,
            }
        )
    return [r for r in rows if r["concept_text"]]


def _compute_growth_rate(n_new: int, n_prev: int) -> float:
    """Compute the growth rate as (n_new / max(n_prev, 1)) - 1.

    A result of 0.0 means no change; 1.0 means doubled; 9.0 means 10× growth.
    Negative values indicate decline. n_prev=0 is treated as 1 to avoid
    division by zero while preserving the "went from nothing to something"
    signal.

    Args:
        n_new:  Mention count in the recent window.
        n_prev: Mention count in the prior window (treated as 1 if 0).

    Returns:
        float growth rate.
    """
    denom = max(n_prev, 1)
    return (n_new / denom) - 1.0


def _load_existing_topic_terms() -> set[str]:
    """Read topics.yaml and return a lowercase set of all name/query strings.

    Used by _check_in_existing_topics() for the in_existing_topics flag.
    Returns an empty set on any failure (missing file, malformed YAML).
    """
    # Route the logical config/ path through resolve_path so a wheel run finds
    # the user config dir / packaged default (dev/editable still resolves
    # ./config). An absolute override (e.g. a test monkeypatch pointing at a
    # tmp file) is honoured as-is — resolve_path's read-fallback tiers must NOT
    # rescue a deliberately-nonexistent absolute path back to the real config.
    if _TOPICS_PATH.is_absolute():
        path = _TOPICS_PATH
        if not path.exists():
            return set()
    else:
        try:
            path = _resolve_path(_TOPICS_PATH)
        except FileNotFoundError:
            return set()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        terms: set[str] = set()
        for search in data.get("searches") or []:
            for key in ("name", "query"):
                val = search.get(key)
                if isinstance(val, str) and val.strip():
                    terms.add(val.strip().lower())
        return terms
    except Exception as exc:
        logger.debug("E: could not load topics.yaml for term check: %s", exc)
        return set()


def _check_in_existing_topics(concept_text: str) -> bool:
    """Return True if concept_text is a substring (case-insensitive) of any topic term.

    A concept "biorefinery" would match topics whose name or query contains
    "biorefinery" as a substring (e.g. "Biorefinery Design").

    Args:
        concept_text: The entity surface text to check.

    Returns:
        True if any existing topic name/query contains concept_text as a substring.
    """
    needle = concept_text.strip().lower()
    if not needle:
        return False
    terms = _load_existing_topic_terms()
    return any(needle in term for term in terms)
