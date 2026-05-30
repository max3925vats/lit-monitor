"""
Prompt registry for non-extraction LLM call sites.

Loads YAML prompt files from config/prompts/ and exposes typed access with
domain-context variable rendering and Pydantic validation.  Mirrors the
singleton-cache pattern of extraction_schema.py.

Public API:
  load_prompt(name)               → Prompt  (.system, .render_user(**kw), .max_tokens)
  load_clustering_prompts()       → ClusteringPrompts
  load_extraction_prompts()       → ExtractionPrompts

Source files:
  config/prompts/rationale.yaml   — ranker.py
  config/prompts/clustering.yaml  — clusterer.py
  config/prompts/synthesis.yaml   — synthesize.py
  config/prompts/extraction.yaml  — extractor.py (connective scaffolding only)

Call _reset_prompt_cache() in tests to clear the singleton caches.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from scripts.core.path_utils import resolve_path as _resolve_path
from scripts.llm.extraction_schema import domain_context_values

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path("config/prompts")


def _load_yaml(name: str) -> dict[str, Any]:
    """Load a prompt YAML by base name.

    Resolves through ``scripts.core.path_utils.resolve_path``, which transparently
    falls back to a sibling ``{name}.example.yaml`` when the live ``{name}.yaml``
    is absent.  This keeps the package working on a fresh clone before any user
    customization, and lets new prompts (e.g. ``long_tail_ner``) ship a single
    tracked ``.example.yaml`` without committing a duplicate ``.yaml``.
    """
    path = _resolve_path(_PROMPTS_DIR / f"{name}.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# Matches only Python-identifier placeholders like {domain_focus} — never
# JSON-style content like {"<doi>": ...} because those contain non-identifier
# chars (<, >, ", :) that the pattern excludes.
_VAR_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _render(
    text: str,
    values: dict[str, Any] | None = None,
    prompt_name: str | None = None,
    allowed_extra: Iterable[str] = (),
) -> str:
    """Substitute {name}-style placeholders in prompt text.

    Only Python-identifier patterns match (e.g. {domain_focus}), so literal
    JSON blocks like {"<doi>": "<rationale>"} pass through untouched.
    Unknown placeholder names are left intact rather than raising, but a
    WARNING is logged so typos in domain_context.yaml are visible.

    Args:
        text: Raw prompt text that may contain {placeholder} tokens.
        values: Substitution dict. Defaults to domain_context_values().
        prompt_name: Optional source identifier used in the warning message
            when unrendered placeholders are found.
        allowed_extra: Placeholders intentionally left for downstream
            ``.format(...)`` calls (e.g. ``{doi}``, ``{title}`` in
            ``paper_card_template``). These pass through silently and are
            NOT reported as unrendered.
    """
    if values is None:
        values = domain_context_values()

    allowed = set(allowed_extra)
    unmatched: list[str] = []

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        if key in allowed:
            # Downstream-render placeholder; preserve verbatim, no warning.
            return match.group(0)
        unmatched.append(key)
        return match.group(0)  # leave unknown placeholders as-is

    rendered = _VAR_PLACEHOLDER.sub(_replace, text)
    if unmatched:
        logger.warning(
            "Unrendered placeholders in prompt %r: %s",
            prompt_name or "<unknown>",
            sorted(set(unmatched)),
        )
    return rendered


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Prompt(BaseModel):
    """A simple two-part prompt with an optional max_tokens cap."""
    system: str = Field(min_length=1)
    user_template: str = Field(min_length=1)
    max_tokens: int = Field(default=4096, gt=0)
    paper_card_template: str | None = None
    paper_card_separator: str = "\n\n---\n\n"

    model_config = {"extra": "allow"}

    def render_user(self, **kwargs: Any) -> str:
        """Render the user_template with the supplied keyword arguments."""
        return self.user_template.format_map(kwargs)

    def render_paper_card(self, **kwargs: Any) -> str:
        """Render one paper card. Raises if paper_card_template is not set."""
        if self.paper_card_template is None:
            raise ValueError(
                "paper_card_template not set in this prompt's YAML — "
                "cannot render paper card"
            )
        return self.paper_card_template.format_map(kwargs)


class ClusteringPrompts(BaseModel):
    """Prompts for the three vocabulary clustering passes."""
    system: str = Field(min_length=1)
    remediation_system: str = Field(min_length=1)
    refinement_system: str = Field(min_length=1)
    user_template: str = Field(min_length=1)

    model_config = {"extra": "allow"}

    def render_user(self, keywords: list[str]) -> str:
        """Render the clustering user prompt for one keyword chunk."""
        kw_list = "\n".join(f"  - {kw}" for kw in keywords)
        return self.user_template.format_map({"keywords": kw_list})

    def render_remediation_user(
        self, theme_names: list[str], chunk: list[str]
    ) -> str:
        """Build user prompt for one remediation chunk."""
        return (
            "Existing themes:\n"
            + "\n".join(f"  - {name}" for name in theme_names)
            + "\n\nKeywords to assign:\n"
            + "\n".join(f"  - {kw}" for kw in chunk)
            + "\n\nAssign each keyword to one or more existing themes, "
            "or list it under 'still_unclustered' if it fits nowhere."
        )

    def render_refinement_user(self, merged: dict[str, Any]) -> str:
        """Build user prompt for the global cross-theme refinement pass."""
        lines: list[str] = ["Current clustering:\n"]
        for theme in merged.get("themes", []):
            lines.append(f"Theme: {theme['name']}")
            for kw in theme.get("keywords", []):
                lines.append(f"  - {kw}")
        if merged.get("unclustered"):
            lines.append("\nUnclustered:")
            for kw in merged["unclustered"]:
                lines.append(f"  - {kw}")
        lines.append(
            "\nIdentify cross-theme assignments that are missing. "
            "Output additions only."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registered prompts and their required user_template placeholders.
# Used at load time to surface typos in shipped YAMLs and at call sites to
# document the contract.  Extend when adding new prompts.
# ---------------------------------------------------------------------------
_REQUIRED_PLACEHOLDERS: dict[str, frozenset[str]] = {
    # N2: cloud-Ollama long-tail NER + low-confidence validation.
    "long_tail_ner": frozenset({"text", "low_conf_json"}),
}


def required_placeholders(name: str) -> frozenset[str]:
    """Return the documented set of required placeholders for a prompt.

    Empty frozenset for prompts that haven't opted in.  Callers can use this
    to validate ``.render_user(**kwargs)`` argument sets at startup.
    """
    return _REQUIRED_PLACEHOLDERS.get(name, frozenset())


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_cache: dict[str, Any] = {}


def _reset_prompt_cache() -> None:
    """Test hook — clears the singleton caches."""
    _cache.clear()


def load_prompt(name: str) -> Prompt:
    """Load and cache a named prompt from config/prompts/{name}.yaml.

    Domain-context variables in system and user_template are rendered at
    load time via domain_context_values().
    """
    cache_key = f"prompt:{name}"
    if cache_key not in _cache:
        raw = _load_yaml(name)
        # Render domain context into all string fields that may reference it
        raw["system"] = _render(raw.get("system", ""), prompt_name=name)
        raw["user_template"] = raw.get("user_template", "")
        if raw.get("paper_card_template") is not None:
            # {doi}, {title}, {abstract} are downstream .format(...) args
            # rendered per-paper by ranker._get_rationales — not typos.
            raw["paper_card_template"] = _render(
                raw["paper_card_template"],
                prompt_name=name,
                allowed_extra={"doi", "title", "abstract"},
            )
        if raw.get("paper_card_separator") is not None:
            raw["paper_card_separator"] = _render(
                raw["paper_card_separator"], prompt_name=name
            )
        loaded = Prompt.model_validate(raw)
        # Verify any registered required placeholders are actually present in
        # the user_template — catches typos in the YAML or in
        # _REQUIRED_PLACEHOLDERS at load time rather than at the call site.
        required = _REQUIRED_PLACEHOLDERS.get(name, frozenset())
        if required:
            present = set(_VAR_PLACEHOLDER.findall(loaded.user_template))
            missing = required - present
            if missing:
                logger.warning(
                    "Prompt %r is missing required placeholders %s in user_template",
                    name, sorted(missing),
                )
        _cache[cache_key] = loaded
        logger.debug("Loaded prompt: %s", name)
    return _cache[cache_key]


def load_clustering_prompts() -> ClusteringPrompts:
    """Load and cache the clustering prompt set from config/prompts/clustering.yaml."""
    cache_key = "clustering"
    if cache_key not in _cache:
        raw = _load_yaml("clustering")
        # Render domain context into all system prompts
        for key in ("system", "remediation_system", "refinement_system"):
            if key in raw:
                raw[key] = _render(raw[key], prompt_name=f"clustering.{key}")
        _cache[cache_key] = ClusteringPrompts.model_validate(raw)
        logger.debug("Loaded clustering prompts")
    return _cache[cache_key]


class ExtractionPrompts(BaseModel):
    """Connective scaffolding strings for the extractor system and user prompts.

    All substantive content (role, null instruction, JSON directive, domain
    context, per-field guidance, OCR warning) lives in extraction_schema.yaml
    and review_schema.yaml.  Only the literal connector text that was
    previously hardcoded in extractor.py lives here.
    """

    # User-message prefix placed before the paper/review text.
    user_prefix: str = Field(min_length=1)
    # Introduces the field list in the system prompt.
    fields_header: str = Field(min_length=1)
    # Introduces the confidence companion list.
    confidence_header: str = Field(min_length=1)
    # Closing sentence(s) after confidence list; may contain {conf_vals}.
    confidence_footer: str = Field(min_length=1)
    # Multi-pass label prefix; may contain {pass_label}.
    pass_label_prefix: str = Field(default="Pass: {pass_label}.")

    model_config = {"extra": "allow"}


def load_extraction_prompts() -> ExtractionPrompts:
    """Singleton-cached loader for config/prompts/extraction.yaml."""
    key = "extraction"
    if key not in _cache:
        raw = _load_yaml("extraction")
        # Per-field downstream placeholders that should pass through _render
        # untouched (they are later rendered by extractor.py at use time).
        downstream_extras: dict[str, set[str]] = {
            "confidence_footer": {"conf_vals"},
            "pass_label_prefix": {"pass_label"},
        }
        rendered = {
            k: (
                _render(
                    v,
                    prompt_name=f"extraction.{k}",
                    allowed_extra=downstream_extras.get(k, set()),
                )
                if isinstance(v, str)
                else v
            )
            for k, v in raw.items()
        }
        _cache[key] = ExtractionPrompts.model_validate(rendered)
        logger.debug("Loaded extraction prompts")
    return _cache[key]
