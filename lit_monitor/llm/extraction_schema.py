"""
Schema loader for extraction field definitions.

Reads YAML files at import time and exposes a unified accessor surface:
  - fields_for_pass(content_type, pass_num)    → ordered list of field names (legacy)
  - fields_for_phase(content_type, phase)      → fields for "simple" or "complex" phase (M3)
  - field_prompt(content_type, name)           → per-field guidance (domain vars rendered)
  - field_valid_values(content_type, name)     → set[str] | None
  - pass_label(content_type, pass_num)         → human-readable label
  - domain_context()                           → domain_focus string for prompt body
  - domain_context_values()                    → full domain context dict
  - null_instruction(content_type)             → null-honesty instruction string
  - null_examples()                            → null-honesty examples string (papers only)
  - ocr_warning()                              → OCR degradation notice string
  - json_format_instruction()                  → JSON-only header string
  - confidence_values()                        → set of valid confidence strings
  - system_role(content_type)                  → system role sentence for the LLM
  - load_schema(content_type)                  → raw schema object for the given type (H4)
  - schema_max_pass(content_type)              → max pass number for the given type (H4)

Source files (resolved relative to cwd, then project root):
  config/extraction_schema.yaml     — paper schema
  config/review_schema.yaml         — review article schema (H4)
  config/domain_context.yaml        — domain context text

The schema is loaded once and cached; call _reset_schema_cache() in tests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from lit_monitor.core.path_utils import resolve_path as _resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_PAPER_SCHEMA_PATH = Path("config/extraction_schema.yaml")
_REVIEW_SCHEMA_PATH = Path("config/review_schema.yaml")          # H4
_DOMAIN_CONTEXT_PATH = Path("config/domain_context.yaml")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class _FieldDef(BaseModel):
    id: str
    label: str
    prompt: str = Field(min_length=1)
    pass_num: int = Field(alias="pass", ge=1)
    complexity: str = "simple"  # "simple" | "complex" (M3)
    valid_values: list[str] | None = None
    required: bool = False
    extract: bool = True

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("valid_values")
    @classmethod
    def _non_empty_if_present(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("valid_values, if present, must be non-empty")
        return v

    @field_validator("complexity")
    @classmethod
    def _valid_complexity(cls, v: str) -> str:
        if v not in ("simple", "complex"):
            raise ValueError(f"complexity must be 'simple' or 'complex', got {v!r}")
        return v


class _ExtractionSchema(BaseModel):
    """Paper extraction schema (config/extraction_schema.yaml)."""
    system_role: str
    json_format_instruction: str
    null_instruction: str
    null_examples: str = ""
    ocr_warning: str
    confidence_values: list[str]
    pass_labels: dict[int, str]
    fields: list[_FieldDef]

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _validate_passes(self) -> _ExtractionSchema:
        seen = {f.pass_num for f in self.fields}
        if not seen:
            raise ValueError("Paper schema defines no fields (no passes present)")
        # Derive the required pass set from the schema itself rather than
        # hardcoding {1, 2, 3}: passes must form a contiguous range
        # 1..max(seen) so a 2-pass schema is legal but a gap (e.g. {1, 3})
        # is still rejected.
        max_pass = max(seen)
        expected = set(range(1, max_pass + 1))
        missing_passes = expected - seen
        if missing_passes:
            raise ValueError(
                f"Paper schema missing fields for pass(es): {sorted(missing_passes)}"
            )
        missing_labels = seen - set(self.pass_labels)
        if missing_labels:
            raise ValueError(
                f"pass_labels missing entries for pass(es): {sorted(missing_labels)}"
            )
        return self

    def fields_for_pass(self, pass_num: int) -> list[str]:
        return [f.id for f in self.fields if f.pass_num == pass_num]

    def fields_for_phase(self, phase: str) -> list[str]:
        """Return field names (in YAML order) for the given phase ("simple" or "complex")."""
        return [f.id for f in self.fields if f.complexity == phase]

    def get_field(self, field_id: str) -> _FieldDef | None:
        return next((f for f in self.fields if f.id == field_id), None)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_paper_schema: _ExtractionSchema | None = None
_review_schema: _ExtractionSchema | None = None          # H4
_domain_ctx: dict[str, Any] | None = None


def _reset_schema_cache() -> None:
    """Test hook — forces the next accessor call to re-read from disk."""
    global _paper_schema, _review_schema, _domain_ctx
    _paper_schema = None
    _review_schema = None
    _domain_ctx = None


def _get_paper_schema(paper_path: Path | None = None) -> _ExtractionSchema:
    global _paper_schema
    if _paper_schema is None:
        path = _resolve_path(paper_path or _PAPER_SCHEMA_PATH)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        _paper_schema = _ExtractionSchema.model_validate(data)
        logger.info(
            "Loaded paper extraction schema: %d fields across passes 1/2/3",
            len(_paper_schema.fields),
        )
    return _paper_schema


def _get_review_schema(review_path: Path | None = None) -> _ExtractionSchema:
    global _review_schema
    if _review_schema is None:
        path = _resolve_path(review_path or _REVIEW_SCHEMA_PATH)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        _review_schema = _ExtractionSchema.model_validate(data)
        logger.info(
            "Loaded review extraction schema: %d fields across passes 1/2/3",
            len(_review_schema.fields),
        )
    return _review_schema


def _get_domain_ctx(ctx_path: Path | None = None) -> dict[str, Any]:
    global _domain_ctx
    if _domain_ctx is None:
        path = _resolve_path(ctx_path or _DOMAIN_CONTEXT_PATH)
        _domain_ctx = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _domain_ctx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fields_for_pass(content_type: str, pass_num: int) -> list[str]:
    """Return field names (in YAML order) assigned to (content_type, pass_num)."""
    if content_type == "paper":
        return _get_paper_schema().fields_for_pass(pass_num)
    if content_type == "review":
        return _get_review_schema().fields_for_pass(pass_num)
    raise ValueError(f"Unknown content_type: {content_type!r}")


def fields_for_phase(content_type: str, phase: str) -> list[str]:
    """Return field names (in YAML order) for the given complexity phase.

    phase must be "simple" or "complex".  M3 replaces pass-based grouping with
    phase-based grouping — use this instead of fields_for_pass() for new code.
    """
    if phase not in ("simple", "complex"):
        raise ValueError(f"phase must be 'simple' or 'complex', got {phase!r}")
    if content_type == "paper":
        return _get_paper_schema().fields_for_phase(phase)
    if content_type == "review":
        return _get_review_schema().fields_for_phase(phase)
    raise ValueError(f"Unknown content_type: {content_type!r}")


def field_prompt(content_type: str, name: str) -> str:
    """Return per-field extraction guidance, with {domain_focus} rendered."""
    ctx = _get_domain_ctx()
    if content_type == "paper":
        fdef = _get_paper_schema().get_field(name)
        if fdef is None:
            raise KeyError(f"Field {name!r} not found in paper schema")
        return fdef.prompt.strip().format_map(ctx)
    if content_type == "review":
        fdef = _get_review_schema().get_field(name)
        if fdef is None:
            raise KeyError(f"Field {name!r} not found in review schema")
        return fdef.prompt.strip().format_map(ctx)
    raise ValueError(f"Unknown content_type: {content_type!r}")


def field_valid_values(content_type: str, name: str) -> set[str] | None:
    """Return valid enum values for a field, or None if the field has no constraint."""
    if content_type == "paper":
        fdef = _get_paper_schema().get_field(name)
        if fdef and fdef.valid_values:
            return set(fdef.valid_values)
    if content_type == "review":
        fdef = _get_review_schema().get_field(name)
        if fdef and fdef.valid_values:
            return set(fdef.valid_values)
    return None


def pass_label(content_type: str, pass_num: int) -> str:
    """Return the human-readable label for (content_type, pass_num)."""
    if content_type == "paper":
        return _get_paper_schema().pass_labels[pass_num]
    if content_type == "review":
        return _get_review_schema().pass_labels[pass_num]
    raise ValueError(f"Unknown content_type: {content_type!r}")


def domain_context() -> str:
    """Return the domain_focus text for use in the system prompt body."""
    return _get_domain_ctx()["domain_focus"].strip()


def domain_context_values() -> dict[str, Any]:
    """Return the full domain context dict (for format_map rendering)."""
    return _get_domain_ctx()


def null_instruction(content_type: str = "paper") -> str:
    """Return the null-honesty instruction for the given content type."""
    if content_type == "paper":
        return _get_paper_schema().null_instruction.strip()
    if content_type == "review":
        return _get_review_schema().null_instruction.strip()
    return _get_paper_schema().null_instruction.strip()


def null_examples() -> str:
    """Return null-honesty examples string (papers only; empty string for others)."""
    return _get_paper_schema().null_examples.strip()


def ocr_warning() -> str:
    """Return the OCR degradation warning appended to user prompts for OCR-heavy docs."""
    return _get_paper_schema().ocr_warning.strip()


def json_format_instruction() -> str:
    """Return the JSON-only format directive prepended to every system prompt."""
    return _get_paper_schema().json_format_instruction.strip()


def confidence_values() -> set[str]:
    """Return the set of valid confidence strings."""
    return set(_get_paper_schema().confidence_values)


def system_role(content_type: str) -> str:
    """Return the system role sentence for the given content type."""
    if content_type == "paper":
        return _get_paper_schema().system_role
    if content_type == "review":
        return _get_review_schema().system_role
    raise ValueError(f"Unknown content_type: {content_type!r}")


def load_schema(content_type: str) -> Any:
    """Return the loaded schema object for a given content type.

    Used by callers that need direct schema access (e.g. I1 routing, I4 pass tracking).
    Prefer the targeted accessor functions (fields_for_pass, field_prompt, etc.) when
    only specific information is needed.

    Returns one of: _ExtractionSchema.
    """
    if content_type == "paper":
        return _get_paper_schema()
    if content_type == "review":
        return _get_review_schema()
    raise ValueError(f"Unknown content_type: {content_type!r}")


def schema_max_pass(content_type: str) -> int:
    """Return the highest pass number defined for a given content type.

    Derived from the loaded schema's field definitions (max ``field.pass_num``)
    rather than a hardcoded literal, so it tracks the YAML automatically if a
    pass is ever added or removed. Both shipped schemas span passes 1/2/3, so
    this returns 3 today. Used by I4 to decide when a paper/review is complete.
    """
    if content_type == "paper":
        schema = _get_paper_schema()
    elif content_type == "review":
        schema = _get_review_schema()
    else:
        raise ValueError(f"Unknown content_type: {content_type!r}")
    return max(f.pass_num for f in schema.fields)


def all_fields_for_schema(content_type: str) -> list[str]:
    """Return all extractable fields for a content_type (simple + complex phases).

    Returns fields in YAML order (simple fields first, complex fields last), deduped.
    Used by K1 pass-all callers and field validation.
    """
    simple = fields_for_phase(content_type, "simple")
    complex_ = fields_for_phase(content_type, "complex")
    seen: set[str] = set()
    result: list[str] = []
    for f in simple + complex_:
        if f not in seen:
            result.append(f)
            seen.add(f)
    return result
