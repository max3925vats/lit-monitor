"""
Unit tests for scripts/llm/extraction_schema.py.
Validates that the YAML schemas load correctly and the public API
returns the expected values without any hardcoded strings in Python.
"""
from __future__ import annotations

import pytest

from lit_monitor.llm.extraction_schema import (
    _reset_schema_cache,
    confidence_values,
    domain_context,
    domain_context_values,
    field_prompt,
    field_valid_values,
    fields_for_pass,
    json_format_instruction,
    load_schema,
    null_examples,
    null_instruction,
    ocr_warning,
    pass_label,
    schema_max_pass,
    system_role,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure each test starts with a clean schema cache."""
    _reset_schema_cache()
    yield
    _reset_schema_cache()


# ---------------------------------------------------------------------------
# fields_for_pass
# ---------------------------------------------------------------------------

# Canonical field-per-pass sets for the paper schema. If the schema changes
# (field added / removed / renamed / moved to another pass), exactly one of
# the three assertions below fails, and the diff says which field changed.
# Update these sets in the same commit as the schema YAML change.
_PAPER_PASS1_FIELDS = {
    "conclusions",
    "core_finding",
    "methods_summary",
    "results_summary",
    "study_type",
}
_PAPER_PASS2_FIELDS = {
    "assumptions",
    "background_motivation",
    "experimental_conditions",
    "key_parameters",
    "limitations",
    "materials_systems",
    "scale",
    "statistical_methods",
}
_PAPER_PASS3_FIELDS = {
    "actionable_insights",
    "comparison_to_prior",
    "data_availability",
    "discovered_topics",
    "funding_conflicts",
    "future_work",
    "key_citations",
    "novelty_statement",
    "open_questions",
    "relevance_to_domain",
    "software_code",
}


@pytest.mark.unit
def test_paper_pass1_exact_field_set():
    assert set(fields_for_pass("paper", 1)) == _PAPER_PASS1_FIELDS


@pytest.mark.unit
def test_paper_pass2_exact_field_set():
    assert set(fields_for_pass("paper", 2)) == _PAPER_PASS2_FIELDS


@pytest.mark.unit
def test_paper_pass3_exact_field_set():
    assert set(fields_for_pass("paper", 3)) == _PAPER_PASS3_FIELDS


@pytest.mark.unit
def test_passes_are_disjoint():
    """No field should appear in more than one paper pass."""
    p1 = set(fields_for_pass("paper", 1))
    p2 = set(fields_for_pass("paper", 2))
    p3 = set(fields_for_pass("paper", 3))
    assert not (p1 & p2), f"Pass 1 and 2 overlap: {p1 & p2}"
    assert not (p1 & p3), f"Pass 1 and 3 overlap: {p1 & p3}"
    assert not (p2 & p3), f"Pass 2 and 3 overlap: {p2 & p3}"


# ---------------------------------------------------------------------------
# field_prompt
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_field_prompt_non_empty_for_core_finding():
    prompt = field_prompt("paper", "core_finding")
    assert isinstance(prompt, str) and len(prompt) > 0


@pytest.mark.unit
def test_field_prompt_raises_for_unknown_field():
    with pytest.raises(KeyError):
        field_prompt("paper", "nonexistent_field_xyz")


# ---------------------------------------------------------------------------
# field_valid_values
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_study_type_has_valid_values():
    vals = field_valid_values("paper", "study_type")
    assert vals is not None
    assert "experimental" in vals
    assert "computational" in vals


@pytest.mark.unit
def test_scale_has_valid_values():
    vals = field_valid_values("paper", "scale")
    assert vals is not None
    assert "lab" in vals


@pytest.mark.unit
def test_core_finding_has_no_valid_values():
    vals = field_valid_values("paper", "core_finding")
    assert vals is None


@pytest.mark.unit
def test_unknown_field_valid_values_is_none():
    vals = field_valid_values("paper", "completely_unknown_field")
    assert vals is None


# ---------------------------------------------------------------------------
# pass_label
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_paper_pass1_label():
    label = pass_label("paper", 1)
    assert isinstance(label, str) and len(label) > 0


# ---------------------------------------------------------------------------
# Shared text accessors
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_null_instruction_covers_full_contract():
    """The null instruction must tell the LLM about null-honesty AND the three
    confidence values. A short or partial instruction would silently degrade
    extraction quality across every paper."""
    instr = null_instruction("paper").lower()
    assert "null" in instr
    # Mentions every confidence enum value so the LLM knows the vocabulary.
    assert "explicit" in instr
    assert "inferred" in instr
    assert "absent" in instr
    # Tells the LLM not to hallucinate from background knowledge.
    assert "not infer" in instr or "do not infer" in instr


@pytest.mark.unit
def test_null_examples_contain_each_confidence_value():
    """The null_examples block must demonstrate all three confidence values
    via concrete JSON snippets, not just say the words. Without a worked
    example for each value, the LLM is more likely to default to one of them."""
    ex = null_examples()
    assert isinstance(ex, str) and ex.strip()
    assert '"absent"' in ex
    assert '"explicit"' in ex
    assert '"inferred"' in ex
    # Each example uses the {field: value, field_confidence: value} pairing.
    assert "_confidence" in ex


@pytest.mark.unit
def test_ocr_warning_mentions_ocr():
    warning = ocr_warning()
    assert "OCR" in warning or "ocr" in warning.lower()


@pytest.mark.unit
def test_json_format_instruction_mentions_json():
    instr = json_format_instruction()
    assert "json" in instr.lower() or "JSON" in instr


@pytest.mark.unit
def test_confidence_values_contains_expected():
    vals = confidence_values()
    assert "explicit" in vals
    assert "inferred" in vals
    assert "absent" in vals


@pytest.mark.unit
def test_system_role_paper_non_empty():
    role = system_role("paper")
    assert isinstance(role, str) and len(role) > 0


# ---------------------------------------------------------------------------
# domain_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_domain_context_non_empty():
    ctx = domain_context()
    assert isinstance(ctx, str) and len(ctx) > 0


@pytest.mark.unit
def test_domain_context_values_is_dict():
    vals = domain_context_values()
    assert isinstance(vals, dict)
    assert "domain_focus" in vals


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cache_returns_consistent_result_on_second_call():
    """fields_for_pass should return equal lists on repeated calls (schema is cached)."""
    first = fields_for_pass("paper", 1)
    second = fields_for_pass("paper", 1)
    assert first == second


@pytest.mark.unit
def test_reset_cache_allows_reload():
    """After reset, a fresh load should still return valid data."""
    _ = fields_for_pass("paper", 1)
    _reset_schema_cache()
    reloaded = fields_for_pass("paper", 1)
    assert "core_finding" in reloaded


# ---------------------------------------------------------------------------
# H4 — review schema via fields_for_pass / field_prompt / system_role
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_review_pass1_has_core_finding_but_not_study_type():
    """Review pass 1 has core_finding but NOT study_type (dropped in H4)."""
    f = fields_for_pass("review", 1)
    assert "core_finding" in f
    assert "study_type" not in f


@pytest.mark.unit
def test_review_has_23_fields_total():
    """Review schema has exactly 23 fields (paper minus study_type).

    Field count grew from 20 → 23 during M3 (simple/complex phase redesign added
    background_motivation, scale, software_code) and M4 (discovered_topics).
    """
    all_fields = (
        fields_for_pass("review", 1)
        + fields_for_pass("review", 2)
        + fields_for_pass("review", 3)
    )
    assert len(all_fields) == 23, f"Expected 23 fields, got {len(all_fields)}: {all_fields}"


@pytest.mark.unit
def test_review_methods_summary_prompt_mentions_prisma():
    """Review schema overrides methods_summary to mention PRISMA compliance."""
    prompt = field_prompt("review", "methods_summary")
    assert "PRISMA" in prompt or "prisma" in prompt.lower()


@pytest.mark.unit
def test_review_experimental_conditions_prompt_mentions_inclusion_criteria():
    """Review schema overrides experimental_conditions to describe review scope."""
    prompt = field_prompt("review", "experimental_conditions")
    assert "inclusion" in prompt.lower() or "scope" in prompt.lower()


@pytest.mark.unit
def test_review_materials_systems_prompt_mentions_primary_studies():
    """Review schema overrides materials_systems to describe breadth of primary studies."""
    prompt = field_prompt("review", "materials_systems")
    assert "primary stud" in prompt.lower()


@pytest.mark.unit
def test_review_system_role_mentions_review():
    """Review schema system_role is distinct from the paper schema system_role."""
    review_role = system_role("review")
    paper_role = system_role("paper")
    assert review_role != paper_role
    assert "review" in review_role.lower()


@pytest.mark.unit
def test_review_null_instruction_non_empty():
    """Review schema null_instruction is a non-empty string."""
    instr = null_instruction("review")
    assert isinstance(instr, str)
    assert len(instr) > 10


@pytest.mark.unit
def test_review_pass_labels_defined():
    """Review schema has pass labels for passes 1, 2, and 3."""
    for p in (1, 2, 3):
        label = pass_label("review", p)
        assert isinstance(label, str) and len(label) > 0


# ---------------------------------------------------------------------------
# H4 — schema_max_pass()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_schema_max_pass_paper():
    assert schema_max_pass("paper") == 3


@pytest.mark.unit
def test_schema_max_pass_review():
    assert schema_max_pass("review") == 3


@pytest.mark.unit
def test_schema_max_pass_unknown_raises():
    with pytest.raises(ValueError, match="Unknown content_type"):
        schema_max_pass("nonexistent")


# ---------------------------------------------------------------------------
# H4 — load_schema()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_schema_paper_returns_object_with_fields():
    """load_schema('paper') returns an object with a fields attribute."""
    s = load_schema("paper")
    assert hasattr(s, "fields")
    assert len(s.fields) > 0


@pytest.mark.unit
def test_load_schema_review_returns_object_without_study_type():
    """load_schema('review') fields do not include study_type."""
    s = load_schema("review")
    field_ids = [f.id for f in s.fields]
    assert "study_type" not in field_ids


@pytest.mark.unit
def test_load_schema_unknown_raises():
    with pytest.raises(ValueError, match="Unknown content_type"):
        load_schema("nonexistent")


# ---------------------------------------------------------------------------
# V-7 — all_fields_for_schema()
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("content_type,expected_count", [
    ("paper",   24),   # 14 simple-phase fields + 10 complex-phase fields (post-M3 + M4 discovered_topics)
    ("review",  23),   # paper minus study_type
])
def test_all_fields_for_schema_per_content_type(content_type, expected_count):
    """
    all_fields_for_schema(ct) returns the union of all fields for the schema.
    The field count must match the expected value per content type.
    """
    from lit_monitor.llm.extraction_schema import all_fields_for_schema

    fields = all_fields_for_schema(content_type)
    assert isinstance(fields, list)
    assert len(fields) > 0, f"all_fields_for_schema({content_type!r}) returned empty list"
    assert len(fields) == expected_count, (
        f"all_fields_for_schema({content_type!r}): expected {expected_count} fields, "
        f"got {len(fields)}: {fields}"
    )
    # No duplicates
    assert len(fields) == len(set(fields)), (
        f"all_fields_for_schema({content_type!r}) contains duplicate fields"
    )


# ---------------------------------------------------------------------------
# schema_max_pass — derived from the loaded schema (E3), not hardcoded.
# (paper/review/unknown cases already covered above; this pins the derivation.)
# ---------------------------------------------------------------------------
def test_schema_max_pass_is_derived_from_schema(monkeypatch):
    """Regression guard for E3: the value must come from the schema's field
    definitions, not a hardcoded 3. If a hypothetical 4th pass were added to the
    paper schema, schema_max_pass('paper') must reflect it.
    """
    from types import SimpleNamespace

    import lit_monitor.llm.extraction_schema as es

    # Fake schema whose fields span passes 1..4 (the real _validate_passes
    # requires 1/2/3 present; 4 is an allowed extra).
    fake = SimpleNamespace(
        fields=[SimpleNamespace(pass_num=p) for p in (1, 2, 3, 4)]
    )
    monkeypatch.setattr(es, "_get_paper_schema", lambda: fake)

    assert es.schema_max_pass("paper") == 4


# ---------------------------------------------------------------------------
# P6.19: _validate_passes derives the required pass set from the schema, so a
# 2-pass schema is legal but a pass gap is still rejected.
# ---------------------------------------------------------------------------
def _make_schema(pass_nums, pass_labels):
    """Build an _ExtractionSchema with one field per pass number."""
    from lit_monitor.llm.extraction_schema import _ExtractionSchema

    fields = [
        {
            "id": f"f{p}",
            "label": f"Field {p}",
            "prompt": f"Extract field {p}.",
            "pass": p,
        }
        for p in pass_nums
    ]
    return _ExtractionSchema(
        system_role="x",
        json_format_instruction="x",
        null_instruction="x",
        ocr_warning="x",
        confidence_values=["high", "low"],
        pass_labels=pass_labels,
        fields=fields,
    )


def test_two_pass_schema_is_valid():
    """A schema whose fields only span passes 1 and 2 must validate."""
    schema = _make_schema([1, 2], {1: "Quick", 2: "Deep"})
    assert {f.pass_num for f in schema.fields} == {1, 2}


def test_pass_gap_is_rejected():
    """A non-contiguous pass set (1, 3 with 2 missing) must raise."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"missing fields for pass"):
        _make_schema([1, 3], {1: "a", 3: "c"})
