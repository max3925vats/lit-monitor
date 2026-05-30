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
  - any failure (no API key, disabled flag, malformed JSON, network error)
    returns the empty result and logs WARNING.  Never raises to the caller.
  - exactly ONE Ollama call per paper — no retries, no nested calls.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Stable sentinel — every disabled / error path returns a *copy* of this so the
# caller can never mutate the module-level dict.
def _empty_result() -> dict[str, list]:
    return {"new_entities": [], "validations": []}


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


def _validate_response(parsed: Any) -> dict[str, list]:
    """Return ``parsed`` if shape is correct; else ``_empty_result()`` (and log)."""
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
    return parsed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract_long_tail_and_validate(
    text: str,
    low_conf_entities: list[dict[str, Any]],
    *,
    client: Any = None,
    prompt: Any = None,
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

    Returns:
        ``{"new_entities": [...], "validations": [...]}`` — possibly both empty
        on disabled / missing key / parse failure / network error.

    The function NEVER raises.  Failures are logged at WARNING and degrade
    to the empty result.
    """
    # ---- Gate: only construct a client when explicitly enabled AND key present.
    # When the caller passed a client, skip the gate (this is the test injection
    # path and the future "called from a higher-level pipeline that already
    # built the client" path).
    if client is None:
        try:
            config = _load_runtime_config()
            if not _is_enabled(config):
                return _empty_result()
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
            return _empty_result()

    # ---- Load the prompt (registry or override).
    if prompt is None:
        try:
            from scripts.llm.prompt_registry import load_prompt  # noqa: PLC0415

            prompt = load_prompt("long_tail_ner")
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("long_tail: prompt load failed: %s", exc)
            return _empty_result()

    # ---- Render placeholders.
    truncated_text = _truncate_text(text)
    low_conf_json = json.dumps(low_conf_entities or [])
    try:
        system_msg = prompt.system
        user_msg = prompt.render_user(
            text=truncated_text, low_conf_json=low_conf_json
        )
    except (KeyError, AttributeError, TypeError) as exc:
        logger.warning("long_tail: prompt render failed: %s", exc)
        return _empty_result()

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
            return _empty_result()
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("long_tail: LLM call failed: %s", exc)
        return _empty_result()

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
        return _empty_result()

    return _validate_response(parsed)
