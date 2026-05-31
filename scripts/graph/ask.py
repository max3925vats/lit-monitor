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
# Helpers
# ---------------------------------------------------------------------------
def _strip_fences(raw: str) -> str:
    """Tolerate ```cypher / bare ``` fenced output.

    Returns the inner text with fences and surrounding whitespace stripped.
    If the response is fence-only or empty, returns "".
    """
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("```"):
        # Drop the opening fence line (```cypher / ```sql / ```)
        first_nl = text.find("\n")
        text = text[first_nl + 1 :] if first_nl != -1 else text[3:]
        # Drop trailing fence if present
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text.strip()


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
    cypher = _strip_fences(response)
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
