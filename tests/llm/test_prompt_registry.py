"""
Unit tests for scripts/llm/prompt_registry.py.
Validates that prompts load from YAML, render correctly, and respect the cache.
"""
from __future__ import annotations

import pytest

from scripts.llm.prompt_registry import (
    _reset_prompt_cache,
    load_clustering_prompts,
    load_extraction_prompts,
    load_prompt,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure each test starts with a clean prompt cache."""
    _reset_prompt_cache()
    yield
    _reset_prompt_cache()


# ---------------------------------------------------------------------------
# load_prompt — rationale and synthesis
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_rationale_prompt_returns_prompt():
    prompt = load_prompt("rationale")
    # System prompt must instruct JSON output.
    assert "JSON" in prompt.system, "system prompt must instruct JSON output"
    # user_template must accept a 'papers' placeholder and render it.
    rendered = prompt.render_user(papers="TEST_PAPERS")
    assert "TEST_PAPERS" in rendered, "papers argument should be rendered in user template"


@pytest.mark.unit
def test_load_synthesis_prompt_returns_prompt():
    prompt = load_prompt("synthesis")
    # System prompt must include synthesis-related guidance.
    assert "synthesis" in prompt.system.lower(), "system prompt should mention synthesis"
    # user_template must accept 'topic', 'n_sources', and 'sources' placeholders.
    rendered = prompt.render_user(topic="X", n_sources=3, sources="Y")
    assert "X" in rendered and "Y" in rendered and "3" in rendered, (
        "all template variables should be rendered"
    )


@pytest.mark.unit
def test_rationale_render_user_inserts_papers():
    prompt = load_prompt("rationale")
    rendered = prompt.render_user(papers="DOI: 10.1/test\nTitle: A Paper")
    assert "10.1/test" in rendered
    assert "A Paper" in rendered


@pytest.mark.unit
def test_synthesis_render_user_inserts_all_vars():
    prompt = load_prompt("synthesis")
    rendered = prompt.render_user(
        topic="TopicX fouling",
        n_sources=3,
        sources="1. [[Note]] — key finding",
    )
    assert "TopicX fouling" in rendered
    assert "3" in rendered
    assert "Note" in rendered


@pytest.mark.unit
def test_load_prompt_caches_on_second_call():
    first = load_prompt("rationale")
    second = load_prompt("rationale")
    assert first is second


@pytest.mark.unit
def test_reset_cache_allows_reload():
    _ = load_prompt("rationale")
    _reset_prompt_cache()
    reloaded = load_prompt("rationale")
    assert reloaded.system


@pytest.mark.unit
def test_load_nonexistent_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist_xyz")


# ---------------------------------------------------------------------------
# load_clustering_prompts — ClusteringPrompts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_clustering_prompts_has_all_system_keys():
    cp = load_clustering_prompts()
    assert cp.system
    assert cp.remediation_system
    assert cp.refinement_system
    assert cp.user_template


@pytest.mark.unit
def test_clustering_render_user_contains_keywords():
    cp = load_clustering_prompts()
    rendered = cp.render_user(["topicx", "system fouling", "tff"])
    assert "topicx" in rendered
    assert "system fouling" in rendered


@pytest.mark.unit
def test_clustering_render_remediation_user():
    cp = load_clustering_prompts()
    rendered = cp.render_remediation_user(
        theme_names=["Mock Domain", "Bioprocessing"],
        chunk=["hollow fiber", "diafiltration"],
    )
    assert "Mock Domain" in rendered
    assert "hollow fiber" in rendered
    assert "diafiltration" in rendered


@pytest.mark.unit
def test_clustering_render_refinement_user():
    cp = load_clustering_prompts()
    merged = {
        "themes": [
            {"name": "Mock Domain", "keywords": ["topicx", "tff"]},
        ],
        "unclustered": ["flux decline"],
    }
    rendered = cp.render_refinement_user(merged)
    assert "Mock Domain" in rendered
    assert "topicx" in rendered
    assert "flux decline" in rendered


@pytest.mark.unit
def test_load_clustering_prompts_cached():
    first = load_clustering_prompts()
    second = load_clustering_prompts()
    assert first is second


@pytest.mark.unit
def test_load_clustering_prompts_reload_after_reset():
    _ = load_clustering_prompts()
    _reset_prompt_cache()
    reloaded = load_clustering_prompts()
    assert reloaded.system


@pytest.mark.unit
def test_rationale_prompt_has_paper_card_template():
    """ranker depends on the rationale prompt declaring paper_card_template."""
    prompt = load_prompt("rationale")
    assert prompt.paper_card_template is not None
    # Must accept the variables ranker.py interpolates.
    rendered = prompt.render_paper_card(doi="X", title="Y", abstract="Z")
    assert "X" in rendered and "Y" in rendered and "Z" in rendered


@pytest.mark.unit
def test_render_paper_card_raises_when_template_missing():
    """If a prompt YAML doesn't declare paper_card_template, calling
    render_paper_card() must raise ValueError, not return None."""
    from scripts.llm.prompt_registry import Prompt
    p = Prompt(system="x", user_template="y")
    with pytest.raises(ValueError, match="paper_card_template"):
        p.render_paper_card(doi="x", title="y", abstract="z")


# ---------------------------------------------------------------------------
# load_extraction_prompts — ExtractionPrompts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_extraction_prompts():
    from scripts.llm.prompt_registry import load_extraction_prompts
    prompts = load_extraction_prompts()
    # Required fields must be non-empty.
    assert prompts.user_prefix
    assert prompts.fields_header
    assert prompts.confidence_header
    assert prompts.confidence_footer
    # {domain_focus} should be rendered at load time (or absent from these fields).
    assert "{domain_focus}" not in prompts.user_prefix


@pytest.mark.unit
def test_load_extraction_prompts_cached():
    from scripts.llm.prompt_registry import load_extraction_prompts
    first = load_extraction_prompts()
    second = load_extraction_prompts()
    assert first is second


@pytest.mark.unit
def test_load_extraction_prompts_reload_after_reset():
    from scripts.llm.prompt_registry import load_extraction_prompts
    _ = load_extraction_prompts()
    _reset_prompt_cache()
    reloaded = load_extraction_prompts()
    assert reloaded.user_prefix


# ---------------------------------------------------------------------------
# _render — A6 regression tests for JSON-brace handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_render_substitutes_known_placeholder():
    """A simple {variable} should be replaced with the supplied value."""
    from scripts.llm.prompt_registry import _render
    out = _render("Hello {name}!", {"name": "world"})
    assert out == "Hello world!"


@pytest.mark.unit
def test_render_preserves_literal_json_braces():
    """Literal JSON inside the prompt must pass through unchanged while
    Python-identifier placeholders are still substituted."""
    from scripts.llm.prompt_registry import _render
    text = 'Respond with: {"<doi>": "<rationale>"}. Domain: {domain_focus}.'
    out = _render(text, {"domain_focus": "TEST_DOMAIN"})
    # The JSON example must be intact.
    assert '{"<doi>": "<rationale>"}' in out, f"JSON braces were mangled: {out!r}"
    # AND the placeholder must be substituted.
    assert "TEST_DOMAIN" in out, f"domain_focus not substituted: {out!r}"
    assert "{domain_focus}" not in out, f"placeholder still present: {out!r}"


@pytest.mark.unit
def test_render_unknown_placeholder_left_intact():
    """Unknown variable names should be left as-is, not raise."""
    from scripts.llm.prompt_registry import _render
    out = _render("Hello {unknown_var}!", {"name": "world"})
    assert "{unknown_var}" in out, f"unknown placeholder was removed: {out!r}"


@pytest.mark.unit
def test_render_warns_on_unrendered_placeholder(caplog):
    """L4: unrendered placeholders should log a WARNING listing the typos so
    domain_context.yaml mistakes are visible instead of silently leaking
    {placeholder} tokens into the live prompt.
    """
    import logging

    from scripts.llm.prompt_registry import _render

    with caplog.at_level(logging.WARNING, logger="scripts.llm.prompt_registry"):
        out = _render(
            "text with {typo_field} and {known}",
            {"known": "value"},
            prompt_name="rationale",
        )

    # Unmatched placeholder is preserved in the rendered string.
    assert "{typo_field}" in out
    assert "value" in out
    # And a WARNING was emitted naming both the prompt and the typo.
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "Expected a WARNING about unrendered placeholders"
    msg = warning_records[-1].getMessage()
    assert "typo_field" in msg, f"placeholder name missing from warning: {msg!r}"
    assert "rationale" in msg, f"prompt name missing from warning: {msg!r}"


@pytest.fixture()
def _restore_strict():
    """Save/restore the process-wide strict override around a test (A5)."""
    from scripts.core.strict_mode import get_strict_override, set_strict
    prior = get_strict_override()
    try:
        yield set_strict
    finally:
        set_strict(prior)


@pytest.mark.unit
def test_render_strict_raises_on_unknown_placeholder(_restore_strict):
    """A5: under strict mode, a genuinely-unknown placeholder raises instead of
    warning."""
    from scripts.llm.prompt_registry import _render

    _restore_strict(True)
    with pytest.raises(Exception) as exc_info:
        _render(
            "text with {typo_field}",
            {"known": "value"},
            prompt_name="rationale",
        )
    assert "typo_field" in str(exc_info.value)


@pytest.mark.unit
def test_render_strict_does_not_raise_on_deferred_placeholder(_restore_strict):
    """A5 critical nuance: an intentionally-deferred placeholder (listed in
    allowed_extra, e.g. {cluster_context}) must NOT raise even under strict —
    it is preserved verbatim for downstream rendering."""
    from scripts.llm.prompt_registry import _render

    _restore_strict(True)
    # cluster_context is a downstream-render placeholder for ask_summarize.
    out = _render(
        "Summarize. {cluster_context}",
        {"known": "value"},
        prompt_name="ask_summarize",
        allowed_extra={"cluster_context"},
    )
    assert "{cluster_context}" in out


@pytest.mark.unit
def test_render_strict_does_not_raise_on_json_braces(_restore_strict):
    """A5 nuance: literal JSON braces never match _VAR_PLACEHOLDER, so strict
    mode must not raise on them."""
    from scripts.llm.prompt_registry import _render

    _restore_strict(True)
    out = _render(
        'Respond with: {"<doi>": "<rationale>"}. Domain: {domain_focus}.',
        {"domain_focus": "TEST_DOMAIN"},
        prompt_name="rationale",
    )
    assert '{"<doi>": "<rationale>"}' in out
    assert "TEST_DOMAIN" in out


@pytest.mark.unit
def test_render_non_strict_warns_not_raises(_restore_strict, caplog):
    """A5: with strict explicitly OFF, the unknown placeholder still only WARNs
    (prior behaviour preserved)."""
    import logging

    from scripts.llm.prompt_registry import _render

    _restore_strict(False)
    with caplog.at_level(logging.WARNING, logger="scripts.llm.prompt_registry"):
        out = _render(
            "text with {typo_field}",
            {"known": "value"},
            prompt_name="rationale",
        )
    assert "{typo_field}" in out
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "Expected a WARNING (non-strict) about unrendered placeholders"
    assert "typo_field" in warning_records[-1].getMessage()


@pytest.mark.unit
def test_loaded_rationale_prompt_has_rendered_domain_focus():
    """End-to-end: domain_focus must be rendered in the loaded rationale prompt."""
    prompt = load_prompt("rationale")
    # No literal {domain_focus} placeholder remaining.
    assert "{domain_focus}" not in prompt.system, (
        f"domain_focus not rendered; prompt.system = {prompt.system!r}"
    )
    # The JSON example must have survived intact.
    assert '"<doi>"' in prompt.system, (
        f"JSON example was mangled by the renderer: {prompt.system!r}"
    )


@pytest.mark.unit
def test_load_prompt_emits_no_false_positive_warnings(caplog):
    """L4 follow-up: shipped YAML files must not trigger 'unrendered placeholder'
    warnings when loaded. Tokens like {doi}, {title}, {abstract}, {conf_vals},
    and {pass_label} are downstream-format placeholders rendered later by
    ranker/extractor code paths — they are NOT typos in domain_context.yaml.

    Regression for the false-positive WARNING spam on every CLI startup.
    """
    import logging

    _reset_prompt_cache()
    with caplog.at_level(logging.WARNING, logger="scripts.llm.prompt_registry"):
        load_prompt("rationale")
        load_extraction_prompts()

    noise = [
        r for r in caplog.records if "Unrendered placeholders" in r.getMessage()
    ]
    assert noise == [], (
        f"False-positive WARNINGs on prompt load: "
        f"{[r.getMessage() for r in noise]}"
    )


# ---------------------------------------------------------------------------
# ask_summarize — Bundle I cluster_context placeholder
#
# Relocated from tests/integration/test_v09_e2e.py (Audit-2 A2): these are
# pure-unit prompt-registry checks that need no live services, so they belong
# in the unit/llm suite where they run unconditionally — not gated behind a
# live-Ollama integration fixture.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prompt_registry_ask_summarize_has_cluster_context():
    """prompt_registry must declare cluster_context as required for
    ask_summarize after the Bundle I cluster-aware-ask update."""
    from scripts.llm.prompt_registry import required_placeholders

    placeholders = required_placeholders("ask_summarize")
    assert "cluster_context" in placeholders, (
        "prompt_registry.ask_summarize must list cluster_context as required "
        "after Bundle I"
    )


@pytest.mark.unit
def test_ask_summarize_yaml_renders_with_four_placeholders():
    """The ask_summarize YAML round-trip must succeed with the four Bundle I
    placeholders (question, cypher, rows, cluster_context)."""
    prompt = load_prompt("ask_summarize")
    rendered = prompt.render_user(
        question="find methods",
        cypher="MATCH (p) RETURN p",
        rows="| doi |\n|---|\n| 10.0/a |",
        cluster_context="Based on your Protein A theme:",
    )
    assert "find methods" in rendered
    assert "Based on your Protein A theme:" in rendered
