"""
Extraction logic — two-phase LLM extraction for papers and reviews (M3).

Simple phase: end-matter stripped, chunked via A6, regurgitative fields.
Complex phase: single full-text call, references kept, synthesis/inferential
  fields (novelty_statement, key_citations, discovered_topics, etc.).

Long documents are split in the simple phase so no single LLM call exceeds
the model's safe context window. Per-chunk results are merged with
confidence-weighted selection (first chunk wins on ties).

Field definitions, prompt guidance, valid enum values, phase labels, and all
prompt text live in config/extraction_schema.yaml and config/review_schema.yaml.
Editing those YAMLs takes effect without any code change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from scripts.core.markdown_processor import strip_end_matter, strip_references_section
from scripts.core.strict_mode import strict_fallback
from scripts.llm import extraction_schema as _es
from scripts.llm.llm_client import LLMClient, parse_llm_json
from scripts.llm.prompt_safety import sanitize_for_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public field-list constants — computed from YAML at import time.
# ---------------------------------------------------------------------------
PAPER_SIMPLE_FIELDS: list[str] = _es.fields_for_phase("paper", "simple")
PAPER_COMPLEX_FIELDS: list[str] = _es.fields_for_phase("paper", "complex")

# ALL_PAPER_FIELDS: union of simple + complex fields (includes key_citations etc.)
ALL_PAPER_FIELDS: list[str] = _es.all_fields_for_schema("paper")

VALID_CONFIDENCE: set[str] = _es.confidence_values()


# ---------------------------------------------------------------------------
# PassResult — return type for paper extraction pass functions (I3).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PassResult:
    """Result of a single extraction pass, carrying extracted fields and chunking metadata.

    Attributes
    ----------
    fields:
        Merged extraction dict — same content as the plain dict the pass functions
        previously returned.  Callers use ``result.fields`` where they previously
        used the raw dict.
    n_chunks:
        Number of A6 text chunks the pass produced.  1 means no chunking was needed.
        Used by I3 to flag ``_extraction_quality = "degraded_high_chunking"`` when
        any pass required more than 3 chunks.
    """
    fields: dict[str, Any]
    n_chunks: int

# Numeric weight for each confidence level (used by compute_confidence_score)
_CONFIDENCE_WEIGHT: dict[str, float] = {"explicit": 1.0, "inferred": 0.5, "absent": 0.0}

# ---------------------------------------------------------------------------
# A6 — Context window chunking for long documents
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN: int = 4            # rule of thumb: English scientific text
_SAFETY_FACTOR: float = 0.75         # fraction of ctx to budget for input
_OUTPUT_RESERVE_TOKENS: int = 2048   # realistic extraction output size per pass
_SYSTEM_PROMPT_RESERVE_TOKENS: int = 500   # system prompt overhead estimate
_FALLBACK_CTX: int = 16384           # conservative fallback (phi4-mini default)
_MIN_CHUNK_CHARS: int = 4000         # floor: never split into < ~1 000 token chunks
# Confidence level → numeric rank for merge decisions
_CONFIDENCE_RANK: dict[str, int] = {"explicit": 2, "inferred": 1, "absent": 0}


def _available_input_chars(
    llm: LLMClient,
    output_reserve: int = _OUTPUT_RESERVE_TOKENS,
) -> int:
    """Return the safe character budget for a single LLM call's input text.

    Uses the LLM's num_ctx attribute if present; falls back to _FALLBACK_CTX.
    Accounts for system prompt overhead and realistic output size, then
    applies _SAFETY_FACTOR as a buffer for models with inaccurate token counts.

    Parameters
    ----------
    output_reserve:
        Token budget reserved for the LLM's output.  Default 2048 is correct
        for individual passes 1–3 (output ≈ 2000 tokens).  Pass-all callers
        (K1, pass_strategy="all") must pass the mode's actual max_tokens here
        (~8400 for 21 fields × ~400 tokens each) to prevent over-use of the
        input budget.
    """
    ctx = getattr(llm, "num_ctx", None) or _FALLBACK_CTX
    chars = int(
        (ctx * _SAFETY_FACTOR - output_reserve - _SYSTEM_PROMPT_RESERVE_TOKENS)
        * _CHARS_PER_TOKEN
    )
    return max(chars, _MIN_CHUNK_CHARS)


def _split_for_extraction(text: str, chunk_chars: int) -> list[str]:
    """Split text into chunks of at most chunk_chars characters.

    Splits at paragraph breaks (\\n\\n) when possible, falling back to any
    line break (\\n).  Hard character cuts are a last resort.

    The search window for a natural boundary is the last 10% of each chunk
    (minimum 200 chars) to avoid pathologically short trailing pieces.
    """
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    search_window = max(chunk_chars // 10, 200)
    while start < len(text):
        end = start + chunk_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        split_pos = text.rfind("\n\n", end - search_window, end)
        if split_pos != -1:
            split_pos += 2  # include both newlines in the outgoing chunk
        else:
            split_pos = text.rfind("\n", end - search_window, end)
            if split_pos != -1:
                split_pos += 1
            else:
                split_pos = end  # hard cut — no newline in search window
        chunks.append(text[start:split_pos])
        start = split_pos
    return chunks


def _merge_pass_results(results: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    """Merge extraction results from multiple text chunks into one dict.

    Per field: take the result with the highest confidence level.
    Ties go to the first chunk — intro/abstract comes first and typically
    contains the most concise statement of each extractable field.
    """
    merged: dict[str, Any] = {}
    for field in fields:
        conf_key = f"{field}_confidence"
        best_value: Any = None
        best_conf = "absent"
        best_rank = -1
        for result in results:
            value = result.get(field)
            conf = result.get(conf_key, "absent")
            rank = _CONFIDENCE_RANK.get(conf, 0)
            if rank > best_rank:  # strictly higher → replace; ties keep first chunk
                best_value = value
                best_conf = conf
                best_rank = rank
        merged[field] = best_value
        merged[conf_key] = best_conf
    return merged


# ---------------------------------------------------------------------------
# C1 — Extraction confidence score
# ---------------------------------------------------------------------------
def compute_confidence_score(extraction: dict[str, Any]) -> float:
    """Compute an overall extraction quality score from per-field confidence values.

    Scans every key ending in '_confidence', maps values to weights
    (explicit=1.0, inferred=0.5, absent=0.0), and returns the mean.
    Returns 0.0 for empty or confidence-free extractions.
    """
    scores: list[float] = [
        _CONFIDENCE_WEIGHT.get(v, 0.0)
        for k, v in extraction.items()
        if k.endswith("_confidence") and isinstance(v, str)
    ]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system_prompt(
    content_type: str,
    fields: list[str],
    pass_label_str: str | None = None,
    domain_ctx: str | None = None,
) -> str:
    """Unified system prompt builder for paper and review extractions.

    All text content (role, null instruction, JSON directive, domain context,
    per-field guidance, enum constraints) comes from the YAML schema — nothing
    is hardcoded in this function body.

    Parameters
    ----------
    content_type:
        One of "paper", "review".
    fields:
        Ordered list of field names to extract in this call.
    pass_label_str:
        Human-readable pass label (e.g. "Core"). None for single-call extraction.
    domain_ctx:
        Optional override for the domain context paragraph.
        When None, uses extraction_schema.domain_context().
    """
    _domain = domain_ctx if domain_ctx is not None else _es.domain_context()
    _json_hdr = _es.json_format_instruction()
    _role = _es.system_role(content_type)
    _null = _es.null_instruction(content_type)
    _conf_vals = ", ".join(sorted(_es.confidence_values()))

    from scripts.llm.prompt_registry import load_extraction_prompts
    _ep = load_extraction_prompts()

    pass_section = (
        _ep.pass_label_prefix.format(pass_label=pass_label_str) + "\n\n"
        if pass_label_str
        else ""
    )

    field_lines: list[str] = []
    confidence_lines: list[str] = []
    for f in fields:
        guidance = _es.field_prompt(content_type, f)
        valid_vals = _es.field_valid_values(content_type, f)
        suffix = ""
        if valid_vals:
            suffix = f" Must be one of: {', '.join(sorted(valid_vals))}."
        field_lines.append(f"  - {f}: {guidance}{suffix}")
        confidence_lines.append(f"  - {f}_confidence")

    # null examples appear only for paper prompts (they reference paper field names)
    examples_section = ""
    if content_type == "paper":
        examples = _es.null_examples()
        if examples:
            examples_section = f"\n\n{examples}"

    _conf_footer = _ep.confidence_footer.format(conf_vals=_conf_vals)

    return (
        f"{_json_hdr}\n\n"
        f"{_role} {pass_section}"
        f"{_null}{examples_section}\n\n"
        f"{_domain}\n\n"
        f"{_ep.fields_header}\n"
        + "\n".join(field_lines)
        + f"\n\n{_ep.confidence_header}\n"
        + "\n".join(confidence_lines)
        + f"\n\n{_conf_footer}"
    )


def _build_extraction_user_prompt(text: str, ocr_heavy: bool) -> str:
    from scripts.llm.prompt_registry import load_extraction_prompts
    _ep = load_extraction_prompts()
    # C1: sanitize the paper fulltext once at this boundary. Paper text comes
    # from a Zotero markdown attachment (user-controllable) and could carry
    # ChatML/Llama role markers to break out of the prompt. strip_code_fences
    # is False so legitimate methods/algorithm code blocks survive — fence
    # removal here would degrade extraction quality on code-bearing papers.
    text = sanitize_for_prompt(text, strip_code_fences=False)
    parts = [_ep.user_prefix, "", text]
    if ocr_heavy and _es.ocr_warning():
        parts.extend(["", _es.ocr_warning().strip()])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------

def _normalise_extraction(
    raw_result: dict[str, Any],
    fields: list[str],
    content_type: str = "paper",
) -> dict[str, Any]:
    """Ensure all expected fields and confidence companions are present.

    Missing fields default to null/absent.
    Out-of-vocabulary enum values are logged as WARNING and dropped to None.

    Parameters
    ----------
    content_type:
        Used to look up valid_values constraints per field. Defaults to "paper"
        for backward compatibility with callers that don't pass it explicitly.
    """
    result: dict[str, Any] = {}
    for field in fields:
        conf_key = f"{field}_confidence"
        value = raw_result.get(field)
        confidence = raw_result.get(conf_key, "absent" if value is None else "explicit")

        # Coerce empty string to None
        if value == "":
            value = None

        # Enum validation: drop out-of-vocabulary values rather than crashing
        valid_vals = _es.field_valid_values(content_type, field)
        if value is not None and valid_vals is not None and value not in valid_vals:
            strict_fallback(
                logger,
                f"Field {field!r} got out-of-vocabulary value {value!r}; "
                f"expected one of {sorted(valid_vals)}. Dropping to None. "
                "Check that extraction_schema.yaml valid_values match the LLM output.",
            )
            value = None

        # Validate confidence
        # If value is None, confidence must be absent
        if value is None:
            confidence = "absent"
        elif confidence not in VALID_CONFIDENCE:
            confidence = "explicit"

        result[field] = value
        result[conf_key] = confidence
    return result


# ---------------------------------------------------------------------------
# Paper extraction
# ---------------------------------------------------------------------------

def extract_simple(
    fulltext: str,
    llm: LLMClient,
    content_type: str = "paper",
    domain_context: str | None = None,
    ocr_heavy: bool = False,
    max_tokens: int = 12288,
    chunk_chars_override: int | None = None,
) -> PassResult:
    """Run simple-phase extraction: end-matter stripped, chunked, regurgitative fields.

    Applied before the complex phase.  Strips references, acknowledgements, and
    other end-matter (strip_end_matter) so the LLM is not distracted by them
    when extracting core/method/metadata fields.  Chunked via A6 with an
    optional per-mode override (M5).

    Parameters
    ----------
    chunk_chars_override:
        When set, overrides the A6 auto-computed chunk size.  Used by M5 to
        expose a per-mode ``chunk_chars`` knob in extraction.yaml.
    """
    fields = _es.fields_for_phase(content_type, "simple")
    stripped = strip_end_matter(fulltext)
    if len(stripped) != len(fulltext):
        logger.debug(
            "Simple phase: stripped end-matter (%d → %d chars)",
            len(fulltext), len(stripped),
        )
    chunk_chars = chunk_chars_override or _available_input_chars(llm)
    chunks = _split_for_extraction(stripped, chunk_chars)
    if len(chunks) > 1:
        logger.info(
            "Simple phase: %d chunks (%d chars, ctx=%d, chunk_chars=%d)",
            len(chunks), len(stripped),
            getattr(llm, "num_ctx", None) or _FALLBACK_CTX,
            chunk_chars,
        )
    system = _build_system_prompt(content_type, fields, "Simple", domain_context)
    chunk_results: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        user = _build_extraction_user_prompt(chunk, ocr_heavy)
        raw = llm.complete(system, user, max_tokens=max_tokens)
        parsed = parse_llm_json(raw)
        chunk_results.append(_normalise_extraction(parsed, fields, content_type))
        if len(chunks) > 1:
            logger.debug("Simple phase chunk %d/%d complete", i + 1, len(chunks))
    if len(chunk_results) == 1:
        return PassResult(fields=chunk_results[0], n_chunks=1)
    return PassResult(
        fields=_merge_pass_results(chunk_results, fields), n_chunks=len(chunks)
    )


def extract_complex(
    fulltext: str,
    llm: LLMClient,
    content_type: str = "paper",
    domain_context: str | None = None,
    ocr_heavy: bool = False,
    max_tokens: int = 12288,
) -> PassResult:
    """Run complex-phase extraction: single full-text call, references kept, inferential fields.

    The full text (including bibliography) is sent in one call so that
    key_citations substantiveness judgment can span the whole document.
    No chunking is applied — if the text exceeds the model's context window
    a WARNING is logged and quality may be reduced.
    """
    fields = _es.fields_for_phase(content_type, "complex")
    ctx = getattr(llm, "num_ctx", None) or _FALLBACK_CTX
    input_budget = int(ctx * _SAFETY_FACTOR * _CHARS_PER_TOKEN)
    if len(fulltext) > input_budget:
        logger.warning(
            "Complex phase: fulltext (%d chars) may exceed input budget "
            "(%d chars at ctx=%d). Citation extraction quality may be reduced.",
            len(fulltext), input_budget, ctx,
        )
    system = _build_system_prompt(content_type, fields, "Complex", domain_context)
    user = _build_extraction_user_prompt(fulltext, ocr_heavy)
    raw = llm.complete(system, user, max_tokens=max_tokens)
    parsed = parse_llm_json(raw)
    normalized = _normalise_extraction(parsed, fields, content_type)
    extracted_count = sum(1 for f in fields if normalized.get(f) is not None)
    logger.info(
        "Complex phase complete (%d/%d fields extracted, model=%s)",
        extracted_count, len(fields), getattr(llm, "model", "?"),
    )
    return PassResult(fields=normalized, n_chunks=1)


def extract_paper(
    fulltext: str,
    llm: LLMClient,
    domain_context: str | None = None,
    ocr_heavy: bool = False,
    phases: tuple[str, ...] = ("simple", "complex"),
    existing_extraction: dict[str, Any] | None = None,
    max_tokens: int = 12288,
    content_type: str = "paper",
    chunk_chars_override: int | None = None,
) -> dict[str, Any]:
    """Run extraction on a paper and return merged results.

    Parameters
    ----------
    phases:
        Ordered sequence of phases to run: "simple", "complex", or both.
    content_type:
        Schema to use: "paper" or "review".
    chunk_chars_override:
        Forwarded to ``extract_simple()`` to override the A6 auto-computed
        chunk size (M5).
    existing_extraction:
        Partial results from a previous run; merged before new phases run.
    """
    llm_single: LLMClient = llm
    merged = dict(existing_extraction) if existing_extraction else {}
    max_n_chunks = 1

    if "simple" in phases:
        try:
            sr = extract_simple(
                fulltext, llm_single, content_type=content_type,
                domain_context=domain_context, ocr_heavy=ocr_heavy,
                max_tokens=max_tokens, chunk_chars_override=chunk_chars_override,
            )
            merged.update(sr.fields)
            max_n_chunks = max(max_n_chunks, sr.n_chunks)
            logger.info(
                "Simple phase complete (%d fields, model=%s)",
                len(sr.fields), getattr(llm_single, "model", "?"),
            )
        except Exception as exc:
            logger.error("Simple phase failed: %s", exc)
            merged["_simple_error"] = str(exc)

    if "complex" in phases:
        try:
            cr = extract_complex(
                fulltext, llm_single, content_type=content_type,
                domain_context=domain_context, ocr_heavy=ocr_heavy,
                max_tokens=max_tokens,
            )
            merged.update(cr.fields)
            logger.info(
                "Complex phase complete (%d fields, model=%s)",
                len(cr.fields), getattr(llm_single, "model", "?"),
            )
        except Exception as exc:
            logger.error("Complex phase failed: %s", exc)
            merged["_complex_error"] = str(exc)

    merged["_overall_confidence"] = compute_confidence_score(merged)
    merged["_max_n_chunks"] = max_n_chunks
    return merged


# ---------------------------------------------------------------------------
# E3 — Targeted field extraction (subset re-extraction without full passes)
# ---------------------------------------------------------------------------

def extract_fields(
    fulltext: str,
    fields: list[str],
    llm: LLMClient,
    domain_context: str | None = None,
    ocr_heavy: bool = False,
    max_tokens: int = 4096,
    output_reserve: int = _OUTPUT_RESERVE_TOKENS,
    content_type: str = "paper",
) -> dict[str, Any]:
    """
    Extract a targeted subset of fields in a single focused LLM call.

    Unlike running a full phase (which extracts all fields for that phase),
    this builds a prompt containing only the requested fields and their
    confidence companions.  Use for iterating on one or two specific fields
    without paying for a full-phase LLM call.

    Applies A5 reference stripping and A6 context chunking.

    Parameters
    ----------
    fulltext:
        Raw paper text (bibliography will be stripped before prompting).
    fields:
        Non-empty list of field names to extract.  Must all belong to the
        schema for ``content_type``; raises ``ValueError`` otherwise.
    llm:
        Any ``LLMClient`` instance.
    domain_context:
        Domain context string. When None, loads from config/domain_context.yaml.
    ocr_heavy:
        If True, appends the OCR degradation warning to the user prompt.
    max_tokens:
        Maximum output token budget.  Default 4096 — smaller than full-pass
        default (12288) since fewer fields are requested.
    content_type:
        Schema to use for field validation and prompt construction.  Defaults to
        ``"paper"`` for backward compatibility with E3 re-extract callers.
        Pass ``"review"`` for review-schema extraction (20 fields, paper minus
        study_type) per H4 design.
    """
    if not fields:
        raise ValueError("fields must be a non-empty list.")
    # V-7: validate against the schema-specific field set, not hardcoded ALL_PAPER_FIELDS.
    _valid_fields = _es.all_fields_for_schema(content_type)
    unknown = [f for f in fields if f not in _valid_fields]
    if unknown:
        _cite_fields = {"key_citations", "comparison_to_prior"}
        _cite_hint = (
            " For key_citations / comparison_to_prior use --pass 4 (requires bibliography)."
            if _cite_fields.intersection(unknown) else ""
        )
        raise ValueError(
            f"Unknown field(s) for schema {content_type!r}: {unknown!r}. "
            f"Valid fields: {_valid_fields}.{_cite_hint}"
        )

    # A5: strip bibliography before prompting — same as passes 1–3.
    stripped = strip_references_section(fulltext)
    if stripped is not fulltext:
        logger.debug(
            "extract_fields: stripped reference section (%d → %d chars)",
            len(fulltext), len(stripped),
        )

    system = _build_system_prompt(content_type, fields, "Targeted Re-extraction", domain_context)

    # A6: chunk long documents so no single call exceeds the context budget.
    # output_reserve is passed through for K1 pass-all callers (H2 alignment).
    chunk_chars = _available_input_chars(llm, output_reserve)
    chunks = _split_for_extraction(stripped, chunk_chars)
    if len(chunks) > 1:
        logger.info(
            "extract_fields: %d chunks (target %d chars/chunk) for fields %s",
            len(chunks), chunk_chars, fields,
        )

    results: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            logger.info("extract_fields: LLM call %d/%d ...", i, len(chunks))
        user = _build_extraction_user_prompt(chunk, ocr_heavy)
        raw = llm.complete(system, user, max_tokens=max_tokens)
        parsed = parse_llm_json(raw)
        results.append(_normalise_extraction(parsed, fields, content_type))

    merged = _merge_pass_results(results, fields) if len(results) > 1 else results[0]
    merged["_max_n_chunks"] = len(chunks)
    merged["_overall_confidence"] = compute_confidence_score(merged)
    return merged

