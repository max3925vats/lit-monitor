"""R2 (Phase 3): LLM-driven relationship extractor.

Single Ollama call per paper. Returns validated ``RelationshipTuple`` list
across the 9 closed predicates: 7 schema-source (MENTIONS, CITES, COMPARES_TO,
DEPENDS_ON, PROPOSES, LIMITED_BY, INTRODUCES, RAISES_QUESTION) PLUS the two
LLM-only predicates EXTENDS and CONTRADICTS.

Defensive perimeter (mirrors N2/N6):
  - exactly ONE Ollama call per paper, no retries, no nested calls.
  - the function NEVER raises; failures degrade with a WARNING log + ``[]``.
  - per-triple validation drops bad rows but keeps the good ones.
  - out-of-vocab predicates are dropped with an INFO log so the operator can
    see what the LLM tried.
  - the prompt is loaded from YAML via ``prompt_registry.load_prompt``; the
    long-lived asset lives in
    ``config/prompts/relationship_extraction.example.yaml``.

R3 (multi-source merge) identifies LLM-extracted edges via
``RelationshipTuple.field == "llm_extracted"``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from scripts.graph.relationship_extractor import RelationshipTuple
from scripts.graph.relationship_validator import VALID_PREDICATES

logger = logging.getLogger(__name__)

# DOI regex: "10." + 1+ digits + "/" + at least one non-whitespace char.
# Matches the CrossRef public DOI shape (typically 4-9 digit registrant
# codes; allow short test DOIs like 10.0/example too). Tighter than just
# startswith("10.") so vague strings like "10. previous work" are rejected.
_DOI_RE = re.compile(r"^10\.\d+/\S+$")

# Predicates whose Paper -> Paper form must carry a DOI-shaped object.
# EXTENDS / CONTRADICTS / COMPARES_TO are the high-stakes "claim about another
# paper" edges — false positives are sticky in the graph.
# CITES is intentionally NOT here: it is G4's domain (citation_edges produces
# CITES). The R2 prompt does not list CITES; this set is the validator's gate
# and must mirror the prompt vocabulary to avoid silent surface divergence.
_PAPER_PREDICATES: frozenset[str] = frozenset(
    {"EXTENDS", "CONTRADICTS", "COMPARES_TO"}  # CITES is G4's domain
)

# R2: minimum confidence floor. Prompt says "do NOT emit anything below 0.5".
# We enforce it in code too so prompt drift cannot silently lower the bar.
_MIN_CONFIDENCE = 0.5

# Truncation cap on the paper text we pass to the LLM. Matches the "~8000
# chars" hint in the prompt's user_template comment. Keeps the single round
# trip affordable for long papers.
_MAX_FULLTEXT_CHARS = 8000

# Hard upper bound on emitted triples per paper. The prompt also asks the
# model to cap at 25 — this enforces the bound on parse output regardless of
# what the model returned.
_MAX_TRIPLES = 25

# Evidence excerpt cap (matches the relationship_extractor convention).
_EVIDENCE_MAX_CHARS = 200


def _normalize_doi(s: str) -> str:
    """Strip trailing punctuation that LLMs often emit on DOIs mid-sentence.

    LLMs frequently echo DOIs with a trailing period/comma/paren when the
    DOI appears at the end of a sentence (e.g. ``"10.1002/btpr.123."``).
    Normalize before regex match AND before storing as ``target_id`` so the
    downstream graph lookup uses the canonical form.
    """
    return s.strip().rstrip(".,;:)]}")


def _is_doi(s: Any) -> bool:
    """R2: True iff `s` looks like a CrossRef-shaped DOI (10.X/Y) after
    trailing-punctuation normalization.
    """
    if not isinstance(s, str):
        return False
    return bool(_DOI_RE.match(_normalize_doi(s)))


def _truncate_text(text: str, max_chars: int = _MAX_FULLTEXT_CHARS) -> str:
    """Cap input text at a whitespace boundary to control LLM input cost."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    last_space = snippet.rfind(" ")
    return snippet if last_space == -1 else snippet[:last_space]


def _truncate_evidence(text: str) -> str:
    """Cap evidence at _EVIDENCE_MAX_CHARS at a word boundary, append ellipsis."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if len(text) <= _EVIDENCE_MAX_CHARS:
        return text
    snippet = text[:_EVIDENCE_MAX_CHARS]
    last_space = snippet.rfind(" ")
    base = snippet if last_space == -1 else snippet[:last_space]
    return base + "…"


def _summarize_extraction(extraction_json: dict[str, Any] | None) -> str:
    """Compress extraction_json into a short context block for the prompt.

    Avoids dumping the whole extraction (some fields are huge) — picks the
    high-signal keys for relationship extraction. Returns a short string
    suitable for inlining in the user_template.
    """
    if not isinstance(extraction_json, dict) or not extraction_json:
        return "(no structured extraction available)"
    keys = (
        "novelty_statement",
        "comparison_to_prior",
        "limitations",
        "future_work",
        "assumptions",
        "open_questions",
        "methods_summary",
        "materials_systems",
    )
    parts: list[str] = []
    for k in keys:
        v = extraction_json.get(k)
        if not v:
            continue
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v[:5])
        parts.append(f"  {k}: {str(v)[:300]}")
    return "\n".join(parts) if parts else "(no structured extraction available)"


def _strip_fences(raw: str) -> str:
    """Tolerate markdown-fenced JSON output from the LLM.

    More permissive than long_tail's variant: extracts the first balanced
    ``{...}`` block from anywhere in the response so a chatty LLM with prose
    around the JSON still parses cleanly.
    """
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _validate_triple(
    triple: Any,
    *,
    paper_doi: str,
) -> RelationshipTuple | None:
    """Per-triple validation. Returns RelationshipTuple or None.

    Drops (with INFO log) on any of:
      - not a dict.
      - subject != paper_doi (case-insensitive).
      - predicate not in VALID_PREDICATES.
      - empty/non-string object.
      - object_kind not in {'Paper', 'Entity'}.
      - Paper -> Paper edge whose object is not DOI-shaped (after
        trailing-punctuation normalization).
      - confidence not in [0.0, 1.0] or wrong type.
      - confidence below ``_MIN_CONFIDENCE`` (mirrors prompt rule).
    """
    if not isinstance(triple, dict):
        logger.info("R2: dropped non-dict triple: %r", type(triple).__name__)
        return None

    subject = triple.get("subject")
    predicate = triple.get("predicate")
    obj = triple.get("object")
    object_kind = triple.get("object_kind", "Entity")
    evidence = triple.get("evidence", "")
    confidence = triple.get("confidence")

    # Subject gate: must equal paper_doi (case-insensitive — CrossRef DOIs
    # are officially case-insensitive, and LLMs occasionally re-case them).
    # The LLM occasionally echoes a cited paper's DOI here — drop those
    # silently.
    if (
        not isinstance(subject, str)
        or subject.strip().lower() != paper_doi.strip().lower()
    ):
        logger.info(
            "R2: dropped predicate=%s subject=%r object=%r (subject mismatch with paper_doi=%r)",
            predicate, subject, obj, paper_doi,
        )
        return None

    # Predicate gate: closed vocabulary only.
    if not isinstance(predicate, str) or predicate not in VALID_PREDICATES:
        logger.info(
            "R2: dropped predicate=%r subject=%s object=%r (out-of-vocab)",
            predicate, subject, obj,
        )
        return None

    # Object presence + type.
    if not isinstance(obj, str) or not obj.strip():
        logger.info(
            "R2: dropped predicate=%s subject=%s (empty/non-string object)",
            predicate, subject,
        )
        return None
    obj = obj.strip()

    # object_kind must be exactly one of two values.
    if object_kind not in ("Paper", "Entity"):
        logger.info(
            "R2: dropped predicate=%s subject=%s object=%r (bad object_kind=%r)",
            predicate, subject, obj, object_kind,
        )
        return None

    # Paper -> Paper sticky-edge defense: object MUST be DOI-shaped.
    # Without this gate the LLM happily emits ``object="Smith et al."`` for
    # EXTENDS, which would silently degrade every downstream graph query.
    # Normalize trailing punctuation BEFORE the regex check AND store the
    # normalized form as target_id so downstream lookups don't fail on the
    # LLM-emitted "10.X/Y." form.
    if predicate in _PAPER_PREDICATES and object_kind == "Paper":
        normalized = _normalize_doi(obj)
        if not _DOI_RE.match(normalized):
            logger.info(
                "R2: dropped predicate=%s subject=%s object=%r "
                "(Paper->Paper requires DOI-shaped object)",
                predicate, subject, obj,
            )
            return None
        obj = normalized  # store the clean form on the tuple

    # Confidence range gate.
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        logger.info(
            "R2: dropped predicate=%s subject=%s (confidence bad type: %r)",
            predicate, subject, type(confidence).__name__,
        )
        return None
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        logger.info(
            "R2: dropped predicate=%s subject=%s (confidence=%s out of [0,1])",
            predicate, subject, confidence,
        )
        return None
    # Sticky-edge defense extends to the confidence dimension. The prompt
    # already asks for >= 0.5; enforcing in code prevents prompt drift from
    # silently lowering the bar.
    if confidence < _MIN_CONFIDENCE:
        logger.info(
            "R2: dropped predicate=%s subject=%s (confidence=%s below floor %s)",
            predicate, subject, confidence, _MIN_CONFIDENCE,
        )
        return None

    return RelationshipTuple(
        source_doi=paper_doi,
        predicate=predicate,
        target_id=obj,
        target_kind=object_kind,
        evidence=_truncate_evidence(evidence),
        confidence=confidence,
        # Marker so R3 (multi-source merge) can identify LLM-sourced edges.
        field="llm_extracted",
    )


# ---------------------------------------------------------------------------
# Indirection — tests monkeypatch these to control client construction.
# ---------------------------------------------------------------------------
def _maybe_construct_client(cfg: Any = None) -> Any | None:
    """Construct an OllamaClient iff OLLAMA_API_KEY is set + config loads.

    Args:
        cfg: optional pre-loaded config (test injection). When ``None``,
            ``load_config()`` is called at runtime. Injecting a ``MagicMock``
            bypasses the filesystem lookup.

    Returns ``None`` on any failure (no key, missing config, import error).
    Lazy imports keep the LLM client off the module-level import surface
    (the G11 "no LLM imports in module" test continues to pass).
    """
    if not os.environ.get("OLLAMA_API_KEY"):
        return None
    try:
        from scripts.llm.llm_client import OllamaClient  # noqa: PLC0415

        if cfg is None:
            from scripts.core.config import load_config  # noqa: PLC0415
            cfg = load_config()
        # Prefer graph.relationships.cloud_model if the user set one;
        # otherwise reuse the NER cloud model so a single env tuning
        # controls all phase-2/phase-3 cloud call sites.
        cloud_model = (
            getattr(
                getattr(getattr(cfg, "graph", None), "relationships", None),
                "cloud_model",
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
        logger.warning("R2: client construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract_llm_relationships(
    fulltext: str,
    paper_doi: str,
    extraction_json: dict[str, Any] | None = None,
    *,
    client: Any = None,
    prompt: Any = None,
    cfg: Any = None,  # inject for tests; None → load_config() inside _maybe_construct_client
) -> list[RelationshipTuple]:
    """R2: extract typed relationships via cloud-Ollama (single call per paper).

    Returns a list of validated ``RelationshipTuple`` covering up to 25 triples
    across the 9 closed predicates. Returns ``[]`` on any failure (no key,
    malformed JSON, network error, missing config). Never raises.

    Args:
        fulltext: paper body text (truncated to ~8000 chars before sending).
        paper_doi: this paper's DOI; the subject of every emitted triple.
            Triples whose ``subject`` field does not match this DOI are
            dropped.
        extraction_json: structured extraction (G3-era), summarized into a
            short prompt context block. May be ``None`` or empty.
        client: optional pre-built LLM client (test injection). When
            ``None``, the function constructs an ``OllamaClient`` ONLY if
            ``OLLAMA_API_KEY`` is set.
        prompt: optional pre-loaded ``Prompt`` instance (test injection).
            When ``None``, loaded via
            ``prompt_registry.load_prompt('relationship_extraction')``.
        cfg: optional pre-loaded config (test injection). Forwarded to
            ``_maybe_construct_client`` so tests can bypass filesystem lookup.

    Returns:
        ``list[RelationshipTuple]`` with ``field="llm_extracted"`` on every
        item. Empty list on any failure.

    The function NEVER raises. Failures are logged at WARNING and degrade
    to an empty list.
    """
    extraction_json = extraction_json or {}

    # Gate: construct client only when explicitly enabled (key present).
    # When the caller passed a client, skip the gate (this is the test
    # injection path and the future "higher-level pipeline already built
    # the client" path).
    if client is None:
        client = _maybe_construct_client(cfg=cfg)
        if client is None:
            return []

    # Load the prompt (registry or override).
    if prompt is None:
        try:
            from scripts.llm.prompt_registry import load_prompt  # noqa: PLC0415

            prompt = load_prompt("relationship_extraction")
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("R2: prompt load failed: %s", exc)
            return []

    # Render placeholders.
    truncated_text = _truncate_text(fulltext)
    extraction_summary = _summarize_extraction(extraction_json)
    try:
        system_msg = prompt.system
        user_msg = prompt.render_user(
            fulltext=truncated_text,
            paper_doi=paper_doi,
            extraction_summary=extraction_summary,
        )
    except (KeyError, AttributeError, TypeError) as exc:
        logger.warning("R2: prompt render failed: %s", exc)
        return []

    # ONE LLM call.
    try:
        max_tokens = int(getattr(prompt, "max_tokens", 2500))
        response = client.complete(
            system=system_msg, user=user_msg, max_tokens=max_tokens
        )
    except TypeError:
        # Tolerate simple test clients whose .complete() takes (system, user)
        # without the max_tokens kwarg.
        try:
            response = client.complete(system=system_msg, user=user_msg)
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("R2: LLM call failed: %s", exc)
            return []
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("R2: LLM call failed: %s", exc)
        return []

    # Parse the JSON response.
    try:
        cleaned = _strip_fences(response)
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning(
            "R2: JSON parse failed: %s (response prefix: %r)",
            exc, str(response)[:200],
        )
        return []

    if not isinstance(parsed, dict):
        logger.warning(
            "R2: parsed response not a dict (got %s)", type(parsed).__name__
        )
        return []
    if "triples" not in parsed:
        logger.warning(
            "R2: response missing 'triples' key (got %s)",
            sorted(parsed.keys()),
        )
        return []
    triples = parsed.get("triples")
    if not isinstance(triples, list):
        logger.warning(
            "R2: 'triples' is not a list (got %s)", type(triples).__name__
        )
        return []

    # Per-triple validation. Cap at _MAX_TRIPLES BEFORE validation so a
    # runaway model that returns 200 items doesn't burn CPU on validation.
    out: list[RelationshipTuple] = []
    for triple in triples[:_MAX_TRIPLES]:
        rel = _validate_triple(triple, paper_doi=paper_doi)
        if rel is not None:
            out.append(rel)

    return out
