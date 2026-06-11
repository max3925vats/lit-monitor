"""
Bundle C: LLM-based cluster naming.

Single call to produce a short human-readable theme name (≤4 words) for a
cluster, given a sample of paper titles from that cluster.

Defensive perimeter — never raises; returns None on any failure.
Caller falls back to f"Cluster {cluster_id}" on None.
"""
from __future__ import annotations

import logging

# Imported at module level so tests can patch scripts.clustering.naming.load_prompt.
# Delayed-import guard: if prompt_registry isn't importable (e.g. bare test env
# without full install), we fall back to None and name_cluster returns None gracefully.
try:
    from lit_monitor.llm.prompt_registry import load_prompt
except Exception:  # pragma: no cover
    load_prompt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MAX_WORDS = 4
_MAX_CHARS = 60
_MAX_SAMPLE_TITLES = 5


def name_cluster(
    sample_dois: list[str],
    sample_titles: list[str],
    *,
    llm=None,
    cfg=None,
) -> str | None:
    """Generate a short human-readable name for a cluster.

    Args:
        sample_dois: DOIs of representative papers (informational, not sent to LLM).
        sample_titles: Paper titles to send to the LLM as cluster samples.
        llm: Optional LLM client. Built from config when None.
        cfg: Optional config for LLM client construction.

    Returns:
        Short name string (≤4 words) or None on any failure.
    """
    if not sample_titles:
        return None

    # Lazily build the LLM client if not supplied
    if llm is None:
        try:
            from lit_monitor.llm.llm_client import build_llm_client
            llm = build_llm_client()
        except Exception as exc:
            logger.info("C: LLM client construction failed: %s", exc)
            return None

    # Load the cluster_naming prompt
    try:
        if load_prompt is None:
            return None
        prompt = load_prompt("cluster_naming")
        sample_text = "\n".join(
            f"- {t}" for t in sample_titles[:_MAX_SAMPLE_TITLES]
        )
        # user_template contains {paper_samples}
        rendered_user = prompt.user_template.format(paper_samples=sample_text)
        system_msg = prompt.system
    except Exception as exc:
        logger.info("C: prompt load/render failed: %s", exc)
        return None

    # Call the LLM
    try:
        max_tokens = getattr(prompt, "max_tokens", 30)
        raw = llm.complete(system_msg, rendered_user, max_tokens=max_tokens)
    except Exception as exc:
        logger.info("C: LLM call failed: %s", exc)
        return None

    if not raw or not raw.strip():
        return None

    name = _sanitize_name(raw)
    return name or None


def _sanitize_name(raw: str) -> str:
    """Strip punctuation, cap at 4 words / 60 chars, return clean string."""
    # Strip outer whitespace and common punctuation the LLM might add
    cleaned = raw.strip().strip("\"'.,!?-").strip()
    # Collapse internal whitespace
    words = cleaned.split()
    words = words[:_MAX_WORDS]
    result = " ".join(words)[:_MAX_CHARS]
    return result
