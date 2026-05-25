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
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from scripts.llm.extraction_schema import domain_context_values

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path("config/prompts")


def _resolve_path(path: Path) -> Path:
    """Resolve a config path relative to cwd, then ancestor directories.

    Falls back to a sibling `*.example.yaml` if the real file is missing.
    This keeps the package working on a fresh clone (CI, new contributors)
    before install.sh has seeded the personal configs.
    """
    candidates = [path]
    if path.suffix == ".yaml" and not path.name.endswith(".example.yaml"):
        candidates.append(path.with_suffix(".example.yaml"))

    if path.exists():
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in candidates:
            full = parent / cand
            if full.exists():
                return full
    raise FileNotFoundError(
        f"Cannot find {path} — searched cwd and ancestors of {here}"
    )


def _load_yaml(name: str) -> dict[str, Any]:
    path = _resolve_path(_PROMPTS_DIR / f"{name}.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# Matches only Python-identifier placeholders like {domain_focus} — never
# JSON-style content like {"<doi>": ...} because those contain non-identifier
# chars (<, >, ", :) that the pattern excludes.
_VAR_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _render(text: str, values: dict[str, Any] | None = None) -> str:
    """Substitute {name}-style placeholders in prompt text.

    Only Python-identifier patterns match (e.g. {domain_focus}), so literal
    JSON blocks like {"<doi>": "<rationale>"} pass through untouched.
    Unknown placeholder names are left intact rather than raising.

    Args:
        text: Raw prompt text that may contain {placeholder} tokens.
        values: Substitution dict. Defaults to domain_context_values().
    """
    if values is None:
        values = domain_context_values()

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        return match.group(0)  # leave unknown placeholders as-is

    return _VAR_PLACEHOLDER.sub(_replace, text)


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
        raw["system"] = _render(raw.get("system", ""))
        raw["user_template"] = raw.get("user_template", "")
        if raw.get("paper_card_template") is not None:
            raw["paper_card_template"] = _render(raw["paper_card_template"])
        if raw.get("paper_card_separator") is not None:
            raw["paper_card_separator"] = _render(raw["paper_card_separator"])
        _cache[cache_key] = Prompt.model_validate(raw)
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
                raw[key] = _render(raw[key])
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
        # Render domain-context variables into all string fields.
        rendered = {
            k: _render(v) if isinstance(v, str) else v
            for k, v in raw.items()
        }
        _cache[key] = ExtractionPrompts.model_validate(rendered)
        logger.debug("Loaded extraction prompts")
    return _cache[key]
