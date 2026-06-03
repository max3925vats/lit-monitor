"""N2: cloud-Ollama long-tail NER + low-confidence validation.

Single LLM round-trip per paper.

Input  : the paper's body text + BioBERT low-confidence spans (as JSON).
Output : { "new_entities": [...], "validations": [...] }
         - new_entities: biopharm-specific entities BioBERT missed.
         - validations: keep/reject decisions for the low-confidence spans.

The prompt is THE long-lived asset of this bundle — see
``config/prompts/long_tail_ner.example.yaml``.  Refine the prompt over time
as the entity-layer quality reveals gaps; the Python here is mechanical.

Defensive contract:
  - exactly ONE Ollama call per paper — no retries, no nested calls.
  - the function NEVER raises; failures degrade with a WARNING log.
  - DESIGN DECISION: cloud-Ollama is ENRICHMENT, not a gate.  When the
    cloud is unavailable (disabled flag, no API key, parse failure,
    network error), ``low_conf_entities`` are PRESERVED as ``validations``
    entries with ``keep=True`` and a descriptive ``reason``.  A network
    blip therefore does NOT silently drop BioBERT NER results — the
    caller should treat empty ``new_entities`` + all-keep ``validations``
    as "BioBERT result accepted unchanged".
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from scripts.llm.prompt_safety import sanitize_for_prompt

logger = logging.getLogger(__name__)


# Stable sentinel — every disabled / error path returns a *copy* of this so the
# caller can never mutate the module-level dict.
def _empty_result() -> dict[str, list]:
    return {"new_entities": [], "validations": []}


def _preserve_low_conf(
    low_conf_entities: list[dict[str, Any]] | None, reason: str
) -> dict[str, list]:
    """When cloud is unavailable, preserve low-conf BioBERT spans as keep=True.

    Phase 2 design decision: cloud-Ollama is enrichment, not a gate. A
    network blip should not silently drop NER results that already came
    from a working local model. The caller treats this shape — empty
    ``new_entities`` + all-keep ``validations`` — as "BioBERT result
    accepted unchanged, no cloud enrichment available".
    """
    if not low_conf_entities:
        return {"new_entities": [], "validations": []}
    validations: list[dict[str, Any]] = [
        {
            "surface": str(e.get("surface", "")),
            "keep": True,
            "reason": reason,
        }
        for e in low_conf_entities
        if isinstance(e, dict) and e.get("surface")
    ]
    return {"new_entities": [], "validations": validations}


# Truncation cap on the paper text we pass to the LLM.  Matches the "~6000
# chars" hint in the prompt's user_template comment.  Keeps the single round
# trip affordable for long papers.
_MAX_TEXT_CHARS = 6000


# ---------------------------------------------------------------------------
# Indirection layer — tests monkeypatch these to control the gate.
# ---------------------------------------------------------------------------
def _load_runtime_config() -> Any:
    """Indirection so tests can monkeypatch config lookup."""
    from scripts.core.config import load_config  # noqa: PLC0415

    return load_config()


def _is_enabled(config: Any) -> bool:
    """Both the config flag AND OLLAMA_API_KEY must be present."""
    try:
        flag = bool(config.graph.ner.cloud_long_tail_enabled)
    except (AttributeError, TypeError):
        flag = False
    if not flag:
        return False
    if not os.environ.get("OLLAMA_API_KEY"):
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate_text(text: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """Cap input text at a whitespace boundary to control LLM input cost."""
    if len(text) <= max_chars:
        return text
    # Cut at the last space before the cap so we don't slice a word in half.
    snippet = text[:max_chars]
    last_space = snippet.rfind(" ")
    return snippet if last_space == -1 else snippet[:last_space]


def _strip_fences(raw: str) -> str:
    """Tolerate the LLM wrapping JSON in ```json ... ``` fences."""
    text = raw.strip()
    if not text.startswith("```"):
        return text
    # Drop the opening fence (may be ```json\n or just ```\n).
    first_nl = text.find("\n")
    if first_nl != -1:
        text = text[first_nl + 1 :]
    # Drop the closing fence.
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _validate_response(
    parsed: Any, *, source_text: str = ""
) -> dict[str, list]:
    """Validate response shape + per-item structural integrity.

    Drops items that fail validation (does NOT drop the whole response).
    The prompt promises ``text[span_start:span_end] == surface`` but LLMs
    routinely lie about offsets — we therefore re-check every span against
    ``source_text`` and silently skip the bad ones. Each drop is logged at
    WARNING with a short content prefix so debugging is possible.
    """
    if not isinstance(parsed, dict):
        logger.warning(
            "long_tail: response is not a dict (%s)", type(parsed).__name__
        )
        return _empty_result()
    if "new_entities" not in parsed or "validations" not in parsed:
        logger.warning(
            "long_tail: response missing required keys (got %s)",
            sorted(parsed.keys()),
        )
        return _empty_result()
    if not isinstance(parsed["new_entities"], list) or not isinstance(
        parsed["validations"], list
    ):
        logger.warning("long_tail: new_entities/validations are not lists")
        return _empty_result()

    # ---- Per-item check on new_entities: offset contract must hold.
    clean_new: list[dict[str, Any]] = []
    for item in parsed["new_entities"]:
        if not isinstance(item, dict):
            continue
        required = {"surface", "type", "span_start", "span_end"}
        if not required.issubset(item):
            logger.warning(
                "long_tail: dropping new_entity missing keys: %r", item
            )
            continue
        surf = item["surface"]
        s = item["span_start"]
        e = item["span_end"]
        if not (
            isinstance(surf, str) and isinstance(s, int) and isinstance(e, int)
        ):
            logger.warning(
                "long_tail: dropping new_entity with bad types: %r", item
            )
            continue
        if not surf:
            logger.warning("long_tail: dropping new_entity with empty surface")
            continue
        if not (0 <= s < e <= len(source_text)):
            logger.warning(
                "long_tail: dropping span out of range s=%d e=%d len=%d",
                s, e, len(source_text),
            )
            continue
        if source_text[s:e] != surf:
            logger.warning(
                "long_tail: dropping span — text[%d:%d]=%r != surface=%r",
                s, e, source_text[s:e][:60], surf[:60],
            )
            continue
        clean_new.append(item)

    # ---- Per-item check on validations: must have surface(str) + keep(bool).
    clean_val: list[dict[str, Any]] = []
    for v in parsed["validations"]:
        if not isinstance(v, dict):
            continue
        surf = v.get("surface")
        keep = v.get("keep")
        if not isinstance(surf, str) or not surf:
            continue
        if not isinstance(keep, bool):
            continue
        clean_val.append(v)

    return {"new_entities": clean_new, "validations": clean_val}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract_long_tail_and_validate(
    text: str,
    low_conf_entities: list[dict[str, Any]],
    *,
    client: Any = None,
    prompt: Any = None,
    cfg: Any = None,  # inject for tests; None → _load_runtime_config() at runtime
) -> dict[str, list]:
    """N2: single-LLM-call long-tail NER + low-confidence validation.

    Args:
        text: the paper's body text (truncated to ~6000 chars before sending).
        low_conf_entities: list of ``{"surface": ..., "confidence": ..., ...}``
            dicts for BioBERT spans below the confidence threshold. The LLM
            decides keep/reject for each.
        client: optional pre-built LLM client (test injection point). When
            ``None``, the function constructs an ``OllamaClient`` ONLY if both
            the config flag (``graph.ner.cloud_long_tail_enabled``) and the
            ``OLLAMA_API_KEY`` env var are present.
        prompt: optional pre-loaded ``Prompt`` instance (test injection).
            When ``None``, loaded via ``prompt_registry.load_prompt('long_tail_ner')``.
        cfg: optional pre-loaded config (test injection). When ``None``,
            ``_load_runtime_config()`` is called at runtime (the existing
            monkeypatch hook for back-compat). Injecting a ``MagicMock``
            bypasses the filesystem lookup without touching any monkeypatches.

    Returns:
        ``{"new_entities": [...], "validations": [...]}``.

    Note: ``low_conf_entities`` are PRESERVED with ``keep=True`` when the
    cloud is unavailable (disabled, no key, parse failure, network error).
    The cloud LLM is enrichment, not a gate — caller should treat empty
    ``new_entities`` + all-keep ``validations`` as "BioBERT result
    accepted unchanged".

    The function NEVER raises.  Failures are logged at WARNING and degrade
    to the preserve-low-conf result.
    """
    # ---- Gate: only construct a client when explicitly enabled AND key present.
    # When the caller passed a client, skip the gate (this is the test injection
    # path and the future "called from a higher-level pipeline that already
    # built the client" path).
    if client is None:
        try:
            # Use injected cfg when available; fall back to the existing
            # _load_runtime_config() indirection so tests that monkeypatch
            # that function keep working unchanged.
            config = cfg if cfg is not None else _load_runtime_config()
            if not _is_enabled(config):
                return _preserve_low_conf(
                    low_conf_entities, reason="cloud disabled"
                )
            from scripts.llm.llm_client import OllamaClient  # noqa: PLC0415

            cloud_model = getattr(
                config.graph.ner, "cloud_model", "gemma2:27b-cloud"
            )
            cloud_host = getattr(
                config.graph.ner, "cloud_host", "https://ollama.com"
            )
            client = OllamaClient(model=cloud_model, host=cloud_host)
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("long_tail: client construction failed: %s", exc)
            return _preserve_low_conf(
                low_conf_entities, reason="client construction failed"
            )

    # ---- Load the prompt (registry or override).
    if prompt is None:
        try:
            from scripts.llm.prompt_registry import load_prompt  # noqa: PLC0415

            prompt = load_prompt("long_tail_ner")
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("long_tail: prompt load failed: %s", exc)
            return _preserve_low_conf(
                low_conf_entities, reason="prompt load failed"
            )

    # ---- Render placeholders.
    # C1 review: fulltext is full-document text (same injection class as
    # relationship_llm) → preserve legitimate code fences, strip only role
    # markers. The low_conf entities carry extraction-derived surface strings
    # (originating from fulltext); json.dumps does NOT neutralize role markers
    # inside string values, so sanitize each string field before serializing.
    # Short fields → default fence-stripping.
    truncated_text = sanitize_for_prompt(
        _truncate_text(text), strip_code_fences=False
    )
    safe_entities = [
        {
            k: (sanitize_for_prompt(v) if isinstance(v, str) else v)
            for k, v in entity.items()
        }
        for entity in (low_conf_entities or [])
    ]
    low_conf_json = json.dumps(safe_entities)
    try:
        system_msg = prompt.system
        user_msg = prompt.render_user(
            text=truncated_text, low_conf_json=low_conf_json
        )
    except (KeyError, AttributeError, TypeError) as exc:
        logger.warning("long_tail: prompt render failed: %s", exc)
        return _preserve_low_conf(
            low_conf_entities, reason="prompt render failed"
        )

    # ---- ONE LLM call.
    try:
        max_tokens = int(getattr(prompt, "max_tokens", 1500))
        response = client.complete(
            system=system_msg, user=user_msg, max_tokens=max_tokens
        )
    except TypeError:
        # Allow simple test clients whose .complete() signature is just
        # (system, user) without the max_tokens kw.
        try:
            response = client.complete(system=system_msg, user=user_msg)
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("long_tail: LLM call failed: %s", exc)
            return _preserve_low_conf(
                low_conf_entities, reason="LLM call failed"
            )
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("long_tail: LLM call failed: %s", exc)
        return _preserve_low_conf(
            low_conf_entities, reason="LLM call failed"
        )

    # ---- Parse the JSON response.
    try:
        cleaned = _strip_fences(response)
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning(
            "long_tail: JSON parse failed: %s (response prefix: %r)",
            exc,
            str(response)[:120],
        )
        return _preserve_low_conf(
            low_conf_entities, reason="JSON parse failed"
        )

    return _validate_response(parsed, source_text=truncated_text)
