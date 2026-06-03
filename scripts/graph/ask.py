"""A2-A5 (Phase 4a): lit-monitor `ask` pipeline.

A2 (this file's first function — ``generate_cypher``):
    NL question → Cypher query, via a single cloud-Ollama call.

Defensive perimeter — the entry point NEVER raises:
    - On any LLM / config / prompt / validator failure, returns ``None``
      and emits an INFO (validator) or WARNING (infra) log line.
    - Subsequent bundles (A3 execution, A4 summary, A5 CLI) will add
      functions to this file. Keep the defensive perimeter discipline.

The prompt is the long-lived asset: see
``config/prompts/ask_cypher.example.yaml``. The Python here is mechanical.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np  # Bundle I: cosine similarity for cluster-aware ask

from scripts.llm.llm_client import strip_markdown_fences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validator regexes.
#
# Mutation gate — word-boundary, case-insensitive. Rejects the 8 Cypher
# mutation keywords. The \b anchors mean lowercase property names like
# "creation_date" or "creator_name" do NOT match the keyword CREATE (because
# the next char 'a' / 'o' is a word char and breaks the \b boundary).
# Uppercase property names that happen to start with a forbidden keyword
# (e.g. literal "CREATED") WILL match — accepted false-positive cost in
# exchange for never letting a write query through.
# ---------------------------------------------------------------------------
_MUTATION_RE = re.compile(
    r"\b(?:CREATE|DELETE|DROP|MERGE|SET|REMOVE|ALTER|LOAD\s+CSV)\b",
    re.IGNORECASE,
)

# Query-shape gate — output must START with one of the read-only Cypher
# keywords. Defends against prose-prefixed responses ("Here is your query: …")
# and against bare aggregations that bypass MATCH entirely is permitted via
# RETURN/WITH (those are valid in Cypher).
_VALID_START_RE = re.compile(
    r"^\s*(?:OPTIONAL\s+MATCH|MATCH|CALL|RETURN|WITH)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Indirection — tests monkeypatch this to control client construction.
# ---------------------------------------------------------------------------
def _maybe_construct_client(cfg: Any = None) -> Any | None:
    """Construct an OllamaClient iff OLLAMA_API_KEY is set + config loads.

    Args:
        cfg: optional pre-loaded config (test injection). When ``None``,
            ``load_config()`` is called at runtime.

    Returns ``None`` on any failure (no key, missing config, import error).
    Lazy imports keep the LLM client off the module-level import surface.
    """
    if not os.environ.get("OLLAMA_API_KEY"):
        return None
    try:
        from scripts.llm.llm_client import OllamaClient  # noqa: PLC0415

        if cfg is None:
            from scripts.core.config import load_config  # noqa: PLC0415
            cfg = load_config()
        # Resolution order for the model:
        #   1. graph.ask.model (explicit ask-pipeline override)
        #   2. graph.ner.cloud_model (shared cloud-Ollama default)
        #   3. literal fallback 'gemma2:27b-cloud'
        cloud_model = (
            getattr(
                getattr(getattr(cfg, "graph", None), "ask", None),
                "model",
                None,
            )
            or getattr(
                getattr(getattr(cfg, "graph", None), "ner", None),
                "cloud_model",
                "gemma2:27b-cloud",
            )
        )
        cloud_host = getattr(
            getattr(getattr(cfg, "graph", None), "ner", None),
            "cloud_host",
            "https://ollama.com",
        )
        return OllamaClient(model=cloud_model, host=cloud_host)
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("A2: client construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_cypher(
    question: str,
    schema_text: str,
    *,
    client: Any = None,
    model: str | None = None,
    cfg: Any = None,
    prompt: Any = None,
) -> str | None:
    """A2: translate a natural-language question into read-only Cypher.

    Args:
        question: user's natural-language question.
        schema_text: live KuzuDB schema markdown (from A1's
            ``describe_schema``). Injected into the prompt so the LLM can
            see the current node/REL/property surface.
        client: optional pre-built LLM client (test injection). When
            ``None``, the function constructs an ``OllamaClient`` ONLY if
            ``OLLAMA_API_KEY`` is set.
        model: optional model-id override; mutates ``client.model`` if
            set. Useful for ad-hoc model switching in tests.
        cfg: optional pre-loaded config (test injection). Forwarded to
            ``_maybe_construct_client`` so tests can bypass filesystem
            lookup.
        prompt: optional pre-loaded ``Prompt`` instance (test injection).
            When ``None``, loaded via
            ``prompt_registry.load_prompt('ask_cypher')``.

    Returns:
        The validated Cypher string, or ``None`` on any failure (LLM
        error, empty response, validator rejection). NEVER raises.

    The function emits:
        - INFO log on validator rejection or empty input/response.
        - WARNING log on infra failure (client construction, LLM call,
          prompt load).
    """
    # Cheap-out on empty input — don't burn an LLM call.
    if not question or not question.strip():
        logger.info("A2: empty question; nothing to generate")
        return None

    # Construct client if not injected.
    if client is None:
        client = _maybe_construct_client(cfg=cfg)
        if client is None:
            return None

    # Apply model override if explicitly passed.
    if model is not None and hasattr(client, "model"):
        client.model = model

    # Load the prompt (registry or override).
    if prompt is None:
        try:
            from scripts.llm.prompt_registry import load_prompt  # noqa: PLC0415

            prompt = load_prompt("ask_cypher")
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("A2: prompt load failed: %s", exc)
            return None

    # Render placeholders. `examples` lives on the Prompt as an
    # ``extra: allow`` field (Pydantic) — fall back to "" if missing.
    examples = getattr(prompt, "examples", "") or ""
    try:
        system_msg = prompt.system
        # C1 review: `question` is operator-supplied (the user querying their
        # OWN graph) — not untrusted text in the threat model. Intentionally
        # NOT sanitized: scrubbing would corrupt legitimate queries that
        # contain ``` or literally ask about role markers.
        user_msg = prompt.render_user(
            question=question,
            schema=schema_text,
            examples=examples,
        )
    except (KeyError, AttributeError, TypeError) as exc:
        logger.warning("A2: prompt render failed: %s", exc)
        return None

    # ONE LLM call.
    try:
        max_tokens = int(getattr(prompt, "max_tokens", 400))
        response = client.complete(
            system=system_msg, user=user_msg, max_tokens=max_tokens
        )
    except TypeError:
        # Tolerate simple test clients whose .complete() lacks max_tokens.
        try:
            response = client.complete(system=system_msg, user=user_msg)
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("A2: client.complete failed: %s", exc)
            return None
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("A2: client.complete failed: %s", exc)
        return None

    if not response or not str(response).strip():
        logger.info("A2: empty response from LLM")
        return None

    # Strip markdown fences if present.
    cypher = strip_markdown_fences(response)
    if not cypher:
        logger.info("A2: empty response after fence stripping")
        return None

    # Mutation gate.
    mutation = _MUTATION_RE.search(cypher)
    if mutation is not None:
        logger.info(
            "A2: rejected by validator: forbidden keyword %r",
            mutation.group(0),
        )
        return None

    # Query-shape gate.
    if not _VALID_START_RE.match(cypher):
        logger.info(
            "A2: rejected by validator: must start with "
            "MATCH/OPTIONAL MATCH/CALL/RETURN/WITH"
        )
        return None

    return cypher


# =============================================================================
# A3: execute_cypher + render_rows
# =============================================================================

import datetime as _dt  # noqa: E402
from typing import Any as _Any  # noqa: E402


def _coerce_jsonable(value: _Any) -> _Any:
    """A3: coerce Kuzu return values to JSON-serializable Python types.

    Mirrors the helper in scripts/api/queries.py — kept local here to avoid
    a cross-package import cycle between graph and api.

    Handles: None, primitives, list/tuple, dict, datetime/date/time,
    bytes/bytearray, and a catch-all str() fallback.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    # Catch-all: str() so json.dumps never chokes on unknown Kuzu types.
    return str(value)


def execute_cypher(
    graph_db: _Any,
    cypher: str,
    *,
    row_cap: int = 100,
) -> list[dict] | None:
    """A3: execute a (presumed read-only — validated by A2) Cypher query.

    Args:
        graph_db: a ``GraphDB`` instance (or any object exposing ``_conn``).
        cypher: the Cypher string to execute.
        row_cap: hard upper limit on rows consumed. Defense-in-depth against
            LLMs that forget LIMIT in their generated query.

    Returns:
        A list of dicts (one per result row) on success; ``None`` + an INFO
        log on any failure. NEVER raises.

    Each dict's keys come from Kuzu's ``get_column_names()`` for the result.
    Values are JSON-serializable: datetime → ISO string, bytes → decoded/hex,
    unknown objects → str().
    """
    if not cypher or not cypher.strip():
        logger.info("A3: empty cypher; nothing to execute")
        return None

    try:
        conn = graph_db._conn
        result = conn.execute(cypher)
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.info("A3: execute failed: %s", exc)
        return None

    try:
        col_names = result.get_column_names()
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.info("A3: get_column_names failed: %s", exc)
        return None

    rows: list[dict] = []
    try:
        while result.has_next():
            if len(rows) >= row_cap:
                logger.info("A3: row_cap=%d reached; truncating result", row_cap)
                break
            raw = result.get_next()
            row = {col_names[i]: _coerce_jsonable(raw[i]) for i in range(len(col_names))}
            rows.append(row)
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.info("A3: row iteration failed: %s", exc)
        return None

    return rows


def render_rows(rows: list[dict], *, max_rows: int = 20) -> str:
    """A3: render result rows as a markdown table for human display.

    Args:
        rows: list of dicts as returned by ``execute_cypher``.
        max_rows: maximum number of data rows to include. When the total
            exceeds this, a ``_(showing first M of N rows)_`` footnote is
            appended.

    Returns:
        A markdown table string. Empty list → ``"_(no results)_"``.

    Rendering rules:
        - Header keys: union of all row keys, stable insertion order
          (first row's keys, then any new keys from later rows).
        - ``None`` values render as an empty cell (not the string "None").
        - Cell values longer than 80 chars are truncated at 79 + "…".
        - Pipe characters inside cell values are backslash-escaped.
        - Newlines inside cell values are collapsed to a space.
    """
    if not rows:
        return "_(no results)_"

    # Build header — union of keys across all rows, stable insertion order.
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    total = len(rows)
    shown_rows = rows[:max_rows]

    lines: list[str] = []
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")

    for row in shown_rows:
        cells = []
        for k in keys:
            val = row.get(k)
            if val is None:
                cells.append("")
            else:
                s = str(val)
                if len(s) > 80:
                    s = s[:79] + "…"
                # Escape pipes and collapse newlines to keep table valid.
                s = s.replace("|", "\\|").replace("\n", " ")
                cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")

    if total > max_rows:
        lines.append("")
        lines.append(f"_(showing first {max_rows} of {total} rows)_")

    return "\n".join(lines)


# =============================================================================
# Bundle I: cluster-aware ask helpers
# =============================================================================
#
# _identify_relevant_cluster embeds the user's question and finds the cluster
# whose centroid is most cosine-similar. Returns a small dict for injection
# into the summarize_results prompt, or None when no cluster is relevant.
#
# Graceful-degradation guarantees:
#   - missing state_db / embeddings_db → returns None, no exception
#   - empty cluster table → returns None
#   - similarity below threshold → returns None
#   - any unexpected exception → warns + returns None


def _deserialize_centroid(blob: bytes | None) -> np.ndarray | None:
    """Deserialise a float32 centroid BLOB stored by Bundle C.

    Args:
        blob: raw bytes stored in clusters.centroid_blob (np.float32 array).

    Returns:
        1-D float32 ndarray, or ``None`` if ``blob`` is empty / unparseable.
    """
    if not blob:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.debug("I: centroid deserialise failed: %s", exc)
        return None


def _identify_relevant_cluster(
    question: str,
    state_db: Any,
    embeddings_db: Any,
    *,
    threshold: float = 0.3,
) -> dict | None:
    """Bundle I: find the cluster whose centroid is most similar to the question.

    Embeds the question with the configured embedding provider and computes
    cosine similarity against every active cluster centroid.  Returns a dict
    with display_name, n_papers, and similarity if the best score is above
    ``threshold``; otherwise returns ``None``.

    Args:
        question: the user's raw natural-language question.
        state_db: a live ``StateDB`` instance (not None — caller checks).
        embeddings_db: a live ``EmbeddingsDB`` instance (not None — caller
            checks).
        threshold: minimum cosine similarity to annotate the response.
            Default 0.3 avoids spurious cluster labels on very weak matches.

    Returns:
        ``{"display_name": str, "n_papers": int, "similarity": float}`` when
        a cluster matches above threshold, else ``None``.  NEVER raises.
    """
    try:
        clusters = state_db.list_active_clusters()
        if not clusters:
            return None

        # Embed the question through the configured provider (Ollama or LiteLLM).
        question_emb = embeddings_db.embed_text(question)

        best_cluster: dict | None = None
        best_score = -1.0

        for cluster in clusters:
            centroid = _deserialize_centroid(cluster.get("centroid_blob"))
            if centroid is None:
                continue
            # Guard against dimension mismatch (schema drift between embed models).
            if centroid.shape != question_emb.shape:
                logger.debug(
                    "I: centroid shape %s != question_emb shape %s; skipping cluster %s",
                    centroid.shape,
                    question_emb.shape,
                    cluster.get("id"),
                )
                continue
            norm_c = np.linalg.norm(centroid)
            norm_q = np.linalg.norm(question_emb)
            if norm_c < 1e-9 or norm_q < 1e-9:
                continue  # zero vectors — skip
            sim = float(np.dot(centroid, question_emb) / (norm_c * norm_q))
            if sim > best_score:
                best_score = sim
                best_cluster = cluster

        if best_cluster is not None and best_score >= threshold:
            return {
                "display_name": best_cluster.get("display_name") or "Cluster",
                "n_papers": best_cluster.get("n_papers", 0),
                "similarity": round(best_score, 3),
            }

    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("I: cluster-aware ask failed: %s", exc)

    return None


# =============================================================================
# A4: summarize_results
# =============================================================================
#
# Turn the Cypher result rows (markdown table from render_rows) into 1-3
# paragraphs of prose that answer the user's natural-language question.
# Single LLM call. Defensive perimeter — returns None on any failure;
# never raises. The A5 CLI is expected to fall back to "I ran this Cypher
# but couldn't summarize — here's the raw table" on a None return.
#
# Empty-result short-circuit: when render_rows returned `_(no results)_`,
# we skip the LLM call entirely and emit a deterministic
# "No matching results were found for: {question}." message. This saves a
# token round-trip and — more importantly — prevents the model from
# hallucinating rows that aren't there.
#
# The prompt YAML at config/prompts/ask_summarize.example.yaml is the
# long-lived asset; the code here is mechanical.

# Token used by render_rows() to signal an empty result set. Kept as a
# module-level constant so the short-circuit check has a single canonical
# source. If render_rows() ever changes its empty-set sentinel, update
# both call sites together.
_EMPTY_RESULTS_TOKEN = "_(no results)_"


def summarize_results(
    question: str,
    cypher: str,
    rendered_rows: str,
    *,
    client: _Any = None,
    model: str | None = None,
    cfg: _Any = None,
    prompt: _Any = None,
    cluster_context: dict | None = None,
) -> str | None:
    """A4: turn Cypher result rows into a 1-3 paragraph prose answer.

    Args:
        question: the user's original natural-language question.
        cypher: the Cypher query that A3 executed (passed to the LLM for
            situational awareness only — the prompt forbids the model
            from explaining or referencing it in the prose output).
        rendered_rows: the markdown table returned by ``render_rows``.
            The special token ``"_(no results)_"`` triggers a
            deterministic short-circuit response WITHOUT an LLM call.
        client: optional pre-built LLM client (test injection). When
            ``None``, the function constructs an ``OllamaClient`` ONLY
            if ``OLLAMA_API_KEY`` is set.
        model: optional model-id override; mutates ``client.model`` if
            set. Useful for ad-hoc model switching in tests.
        cfg: optional pre-loaded config (test injection). Forwarded to
            ``_maybe_construct_client``. When set, the summarize-specific
            model knob ``cfg.graph.ask.summarize_model`` is preferred
            over the generic ``cfg.graph.ask.model``.
        prompt: optional pre-loaded ``Prompt`` instance (test injection).
            When ``None``, loaded via
            ``prompt_registry.load_prompt('ask_summarize')``.
        cluster_context: optional dict from
            :func:`_identify_relevant_cluster` containing
            ``display_name``, ``n_papers``, and ``similarity``.
            When provided, a cluster-framing sentence is injected into the
            prompt so the LLM opens its answer referencing the dominant
            theme.  ``None`` → plain summarisation, unchanged behavior.

    Returns:
        The prose answer (1-3 paragraphs), or ``None`` on any failure
        (LLM error, empty response, prompt load error, empty question).
        NEVER raises.

    The function emits:
        - INFO log on empty input / empty response / LLM call failure /
          prompt load/render failure.
    """
    # Cheap-out on empty input — don't burn an LLM call.
    if not question or not question.strip():
        logger.info("A4: empty question; nothing to summarize")
        return None

    # Empty-result short-circuit. A3 emits the literal token
    # "_(no results)_" when the query returned zero rows. We skip the
    # LLM entirely and return a deterministic message — saves a token
    # round-trip and avoids the model hallucinating content for an
    # empty table. The .strip() tolerates accidental padding by callers.
    if rendered_rows is None or rendered_rows.strip() == _EMPTY_RESULTS_TOKEN:
        return f"No matching results were found for: {question}."

    # Construct client if not injected.
    if client is None:
        client = _maybe_construct_client(cfg=cfg)
        if client is None:
            return None

    # Model resolution:
    #   1. explicit `model=` kwarg (highest precedence — test injection)
    #   2. cfg.graph.ask.summarize_model (summarize-specific override)
    #   3. whatever the client was constructed with (which already
    #      honored cfg.graph.ask.model → cfg.graph.ner.cloud_model →
    #      'gemma2:27b-cloud' via _maybe_construct_client).
    if model is not None and hasattr(client, "model"):
        client.model = model
    elif cfg is not None and hasattr(client, "model"):
        summarize_model = getattr(
            getattr(getattr(cfg, "graph", None), "ask", None),
            "summarize_model",
            None,
        )
        if summarize_model:
            client.model = summarize_model

    # Load the prompt (registry or override).
    if prompt is None:
        try:
            from scripts.llm.prompt_registry import load_prompt  # noqa: PLC0415

            prompt = load_prompt("ask_summarize")
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.info("A4: prompt load failed: %s", exc)
            return None

    # Bundle I: build optional cluster-framing blurb.
    # Empty string when no cluster context — the {cluster_context} placeholder
    # in the YAML is always filled; the LLM simply sees nothing extra.
    cluster_blurb = ""
    if cluster_context:
        cluster_blurb = (
            f"The user's question is most aligned with their "
            f"'{cluster_context['display_name']}' theme "
            f"({cluster_context['n_papers']} papers). "
            f"Begin your response with: "
            f"'Based on your {cluster_context['display_name']} theme:'"
        )

    # Render placeholders. Mirrors A2's render path.
    try:
        system_msg = prompt.system
        # C1 review: `question` is operator-supplied (the user querying their
        # OWN graph) — not untrusted text in the threat model. Intentionally
        # NOT sanitized (see A2 render path above).
        user_msg = prompt.render_user(
            question=question,
            cypher=cypher,
            rows=rendered_rows,
            cluster_context=cluster_blurb,
        )
    except (KeyError, AttributeError, TypeError) as exc:
        logger.info("A4: prompt render failed: %s", exc)
        return None

    # ONE LLM call.
    try:
        max_tokens = int(getattr(prompt, "max_tokens", 500))
        response = client.complete(
            system=system_msg, user=user_msg, max_tokens=max_tokens
        )
    except TypeError:
        # Tolerate simple test clients whose .complete() lacks max_tokens.
        try:
            response = client.complete(system=system_msg, user=user_msg)
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.info("A4: client.complete failed: %s", exc)
            return None
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.info("A4: client.complete failed: %s", exc)
        return None

    if not response or not str(response).strip():
        logger.info("A4: empty response from LLM")
        return None

    return str(response).strip()


# =============================================================================
# H8: shared pipeline runner
# =============================================================================
#
# run_pipeline() wires together A2 generate_cypher → A3 execute_cypher +
# render_rows → A4 summarize_results into a single callable consumed by:
#   - A5 CLI (can refactor to use it in a follow-up)
#   - H8 HTTP route (POST /api/ask)
#
# Shape parity guarantee: both callers see the same AskResult dataclass, so
# CLI and HTTP responses can never silently diverge.
#
# Defensive perimeter: run_pipeline NEVER raises. Each stage that fails
# results in None / [] / fallback prose in the returned AskResult.

from dataclasses import dataclass  # noqa: E402


@dataclass
class AskResult:
    """Result shape returned by run_pipeline; consumed by A5 CLI + H8 HTTP.

    Attributes:
        cypher:   The Cypher query generated by A2, or ``None`` if A2 failed.
        rows:     Raw result rows from A3 execution; empty list on failure.
        rendered: Markdown table from render_rows, or ``"_(no results)_"``.
        prose:    A4 prose answer, or ``None`` if summarize=False / A4 failed.
    """

    cypher: str | None
    rows: list[dict]
    rendered: str
    prose: str | None


def run_pipeline(
    question: str,
    *,
    graph_db: _Any = None,
    cfg: _Any = None,
    model: str | None = None,
    summarize: bool = True,
    max_rows: int = 20,
    state_db: _Any = None,
    embeddings_db: _Any = None,
) -> AskResult:
    """H8: end-to-end ask pipeline returning a structured AskResult.

    Used by both A5 CLI and H8 HTTP route to ensure shape parity.

    Args:
        question:      Natural-language question from the user.
        graph_db:      Optional pre-built GraphDB instance (test injection).
                       When ``None``, acquired via ``safe_graph_db()``.
        cfg:           Optional pre-loaded config (test injection).
                       When ``None``, loaded via ``get_config()``.
        model:         Optional model-id override forwarded to
                       generate_cypher and summarize_results.
        summarize:     When ``False``, A4 is skipped and ``prose`` is
                       ``None``.
        max_rows:      Passed to ``render_rows`` as the display row cap.
        state_db:      Optional pre-built ``StateDB`` instance (Bundle I).
                       When ``None`` AND clusters cannot be loaded, cluster
                       annotation is silently skipped.
        embeddings_db: Optional pre-built ``EmbeddingsDB`` instance
                       (Bundle I).  Required alongside ``state_db`` for
                       cluster-aware annotation; missing either → no
                       annotation, no exception.

    Returns:
        An :class:`AskResult` with partial values on any stage failure.
        NEVER raises.
    """
    # --- config / graph_db acquisition ----------------------------------------
    if cfg is None:
        try:
            from scripts.core.config import get_config  # noqa: PLC0415

            cfg = get_config()
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("H8 run_pipeline: get_config failed: %s", exc)
            cfg = None

    if graph_db is None:
        try:
            from scripts.graph import safe_graph_db  # noqa: PLC0415

            graph_db = safe_graph_db()
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("H8 run_pipeline: safe_graph_db failed: %s", exc)
            graph_db = None

    if graph_db is None:
        return AskResult(
            cypher=None,
            rows=[],
            rendered="_(no results)_",
            prose="Graph backend unavailable. Run 'lit-monitor graph build' first.",
        )

    # --- schema introspection --------------------------------------------------
    try:
        from scripts.graph.schema_describer import describe_schema  # noqa: PLC0415

        schema_text = describe_schema(graph_db)
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("H8 run_pipeline: describe_schema failed: %s", exc)
        schema_text = ""

    if not schema_text or not schema_text.strip():
        return AskResult(
            cypher=None,
            rows=[],
            rendered="_(no results)_",
            prose="Failed to introspect graph schema.",
        )

    # --- A2: NL → Cypher -------------------------------------------------------
    cypher = generate_cypher(question, schema_text, cfg=cfg, model=model)
    if cypher is None:
        return AskResult(
            cypher=None,
            rows=[],
            rendered="_(no results)_",
            prose="I couldn't translate your question into a Cypher query.",
        )

    # --- A3: execute + render --------------------------------------------------
    rows = execute_cypher(graph_db, cypher)
    if rows is None:
        return AskResult(
            cypher=cypher,
            rows=[],
            rendered="_(no results)_",
            prose="I generated a query but it failed to execute.",
        )

    rendered = render_rows(rows, max_rows=max_rows)

    # --- Bundle I: cluster-aware context injection ----------------------------
    # Identify which theme cluster is most similar to the user's question.
    # Requires both state_db and embeddings_db; degrades gracefully on either
    # missing. Only runs when summarize=True (A4 will consume the context).
    cluster_context: dict | None = None
    if summarize and state_db is not None and embeddings_db is not None:
        cluster_context = _identify_relevant_cluster(
            question, state_db, embeddings_db
        )

    # --- A4: summarize ---------------------------------------------------------
    prose: str | None = None
    if summarize:
        prose = summarize_results(
            question,
            cypher,
            rendered,
            cfg=cfg,
            model=model,
            cluster_context=cluster_context,
        )

    return AskResult(cypher=cypher, rows=rows, rendered=rendered, prose=prose)
