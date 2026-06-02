"""Bundle G (v0.9 engine plan): LLM extraction of structured focus areas.

Bridges the free-text ``config/domain_context.yaml::domain_focus`` paragraph
to structured lists usable by:

  - the ranker's entity-overlap signal (KuzuDB entity matches against
    topics / methods / materials → bonus; exclusion match → penalty),
  - the trending-concept suggester (Bundle E) — gap analysis between
    user focus and topics.yaml.

Single LLM call per analysis. Defensive perimeter: ``analyze_domain``
NEVER raises. Returns ``None`` on:

  - empty / whitespace-only input,
  - LLM construction failure,
  - prompt load/render failure,
  - LLM call exception,
  - empty or malformed JSON response,
  - missing required schema keys.

The long-lived asset is the prompt YAML at
``config/prompts/domain_extraction.yaml`` (with example fallback). The
code here is mechanical — refine extraction quality by editing the YAML.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Schema enforcement constants. The keys here MUST match the schema spelled
# out in the prompt's system message — adding a field requires editing both
# this set AND the prompt YAML.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"topics", "methods", "materials", "adjacent_fields", "exclusions"}
)

# Cap on items per list. Mirrors the cap in the prompt — defense-in-depth
# in case the model ignores the instruction.
_MAX_ITEMS_PER_LIST: int = 8


def analyze_domain(
    text: str,
    *,
    llm: Any = None,
    cfg: Any = None,
) -> dict[str, list[str]] | None:
    """Bundle G: extract structured focus areas from a domain_context paragraph.

    Single LLM call. Returns the parsed dict on success, ``None`` on any
    failure (with INFO-level log breadcrumb). Never raises.

    Args:
        text: The free-text domain_context paragraph (typically the
            ``domain_focus`` field of ``config/domain_context.yaml``).
        llm: Optional pre-built LLMClient. When ``None``, a client is
            constructed via ``get_client(cfg, mode="ingestion")`` — the
            same path Bundle F's LiteLLM/Ollama provider abstraction uses.
        cfg: Optional config object. When ``None``, ``get_config()`` is
            called. Only needed to construct the default LLM client.

    Returns:
        ``{"topics": [...], "methods": [...], "materials": [...],
          "adjacent_fields": [...], "exclusions": [...]}`` on success.
        ``None`` on any failure.
    """
    if not text or not text.strip():
        logger.info("G/analyze_domain: empty input; nothing to extract")
        return None

    # ---- Build LLM client if caller didn't supply one ------------------
    if llm is None:
        try:
            from scripts.core.config import get_config
            from scripts.llm.llm_client import get_client

            cfg = cfg if cfg is not None else get_config()
            llm = get_client(cfg, mode="ingestion")
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.info("G/analyze_domain: LLM client construction failed: %s", exc)
            return None

    # ---- Load + render prompt ------------------------------------------
    try:
        from scripts.llm.prompt_registry import load_prompt
        from scripts.llm.prompt_safety import sanitize_for_prompt

        prompt = load_prompt("domain_extraction")
        # C1: the domain_focus paragraph is free-text user config; scrub role
        # markers before rendering. Short paragraph → default fence-stripping.
        rendered_user = prompt.render_user(
            domain_context=sanitize_for_prompt(text)
        )
        system = prompt.system
        max_tokens = prompt.max_tokens
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.info("G/analyze_domain: prompt load/render failed: %s", exc)
        return None

    # ---- Single LLM call -----------------------------------------------
    try:
        raw = llm.complete(system=system, user=rendered_user, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.info("G/analyze_domain: LLM call failed: %s", exc)
        return None

    if not raw or not raw.strip():
        logger.info("G/analyze_domain: empty LLM response")
        return None

    # ---- Strip fences + thinking blocks + parse JSON --------------------
    try:
        from scripts.llm.llm_client import strip_markdown_fences

        cleaned = strip_markdown_fences(raw)
    except Exception:  # noqa: BLE001 — fall back to in-module stripper
        cleaned = _local_strip_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.info(
            "G/analyze_domain: JSON parse failed: %s; raw=%r",
            exc,
            cleaned[:200],
        )
        return None

    if not isinstance(parsed, dict):
        logger.info("G/analyze_domain: LLM output is not a JSON object: %r", type(parsed).__name__)
        return None

    # ---- Schema validation: require ALL five keys -----------------------
    if not _REQUIRED_KEYS.issubset(parsed.keys()):
        missing = _REQUIRED_KEYS - parsed.keys()
        logger.info("G/analyze_domain: missing required keys: %s", sorted(missing))
        return None

    # ---- Normalize: coerce to list-of-string, dedup, cap ----------------
    result: dict[str, list[str]] = {}
    for key in _REQUIRED_KEYS:
        val = parsed.get(key, [])
        if not isinstance(val, list):
            logger.info(
                "G/analyze_domain: field %r is not a list (got %r); treating as empty",
                key,
                type(val).__name__,
            )
            val = []
        # Coerce items → str, strip whitespace, drop empties
        items: list[str] = []
        for v in val:
            s = str(v).strip() if v is not None else ""
            if s:
                items.append(s)
        # Dedup case-insensitive, preserving first-seen order
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            key_lc = item.lower()
            if key_lc not in seen:
                seen.add(key_lc)
                deduped.append(item)
        result[key] = deduped[:_MAX_ITEMS_PER_LIST]

    return result


def _local_strip_fences(text: str) -> str:
    """Minimal markdown-fence stripper used when llm_client's helper is
    unavailable (shouldn't happen at runtime — only protects tests that
    monkeypatch the llm_client module).
    """
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()
