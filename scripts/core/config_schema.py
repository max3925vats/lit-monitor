"""
Pydantic v2 schema models for lit-monitor configuration files.

These models validate the raw YAML dicts loaded by config.py immediately
after YAML parsing, so misconfigured fields are caught at startup rather
than surfacing as confusing runtime errors mid-pipeline.

Design decisions
----------------
- ``extra="allow"`` on every model: unknown keys are silently accepted so
  user-added keys and future features never cause validation breakage.
- Business-logic checks (vault path must not start with ~, LiteLLM model
  required when provider is "litellm") remain in config.py / llm_client.py
  where they already live.  Schema validation handles structural and type
  constraints only.
- ``ValidationError`` propagates unchanged from ``Config.__init__``; callers
  see Pydantic's field-level error messages directly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# paths.yaml
# ---------------------------------------------------------------------------

class ZoteroConfig(BaseModel):
    """paths.yaml → zotero section."""

    model_config = ConfigDict(extra="allow")

    library_type: Literal["user", "group"] = "user"
    library_id: str = ""
    local_storage_path: str = "~/Zotero/storage"
    collection_name: str = "lit-monitor"


class ObsidianPathsConfig(BaseModel):
    """paths.yaml → obsidian section.

    ``vault_path`` is required; empty-string and tilde-prefix checks are
    handled by ``_validate_vault_path()`` in config.py after _Namespace
    construction.  Here we only assert the key is present in the YAML dict.
    """

    model_config = ConfigDict(extra="allow")

    vault_path: str  # required — no default; ValidationError if key absent
    papers_folder: str = "Literature/Papers"
    books_folder: str = "Literature/Books"
    digests_folder: str = "Literature/Digests"
    connections_folder: str = "Literature/Connections"


class StateDbConfig(BaseModel):
    """paths.yaml → state_db section."""

    model_config = ConfigDict(extra="allow")

    path: str = "~/.config/lit-monitor/state.db"


class LogsConfig(BaseModel):
    """paths.yaml → logs section."""

    model_config = ConfigDict(extra="allow")

    path: str = "./logs"
    retention_days: int = Field(default=90, gt=0)


class PathsConfig(BaseModel):
    """Top-level schema for config/paths.yaml."""

    model_config = ConfigDict(extra="allow")

    zotero: ZoteroConfig = Field(default_factory=ZoteroConfig)
    obsidian: ObsidianPathsConfig  # required — raises ValidationError if absent
    state_db: StateDbConfig = Field(default_factory=StateDbConfig)
    logs: LogsConfig = Field(default_factory=LogsConfig)


# ---------------------------------------------------------------------------
# extraction.yaml
# ---------------------------------------------------------------------------

class LLMProviderConfig(BaseModel):
    """Schema for any LLM mode section in config/extraction.yaml.

    Key constraints
    ---------------
    - ``provider`` must be ``"ollama"`` or ``"litellm"``; any other value
      raises ``ValidationError`` immediately rather than failing obscurely
      inside ``get_client()``.
    - ``temperature`` must be in [0.0, 1.0]; out-of-range values are caught
      before passing them to an LLM and producing undefined behaviour.
    - All other fields are optional with sensible defaults; extra keys
      (``ocr_dpi``, ``pass1_model``, ``pass1_litellm_model``, etc.) are
      allowed for forward compatibility.
    """

    model_config = ConfigDict(extra="allow")

    provider: Literal["ollama", "litellm"] = "ollama"
    model: str | None = None
    ollama_host: str = "http://localhost:11434"
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    timeout: int | None = Field(default=None, gt=0)
    max_tokens_per_call: int | None = Field(default=None, gt=0)
    think: bool | None = None


class EmbeddingsConfig(BaseModel):
    """extraction.yaml → embeddings section."""

    model_config = ConfigDict(extra="allow")

    model: str = "mxbai-embed-large"
    ollama_host: str = "http://localhost:11434"


class ComparisonModelEntry(BaseModel):
    """Schema for one entry in extraction.yaml → comparison_models.

    Catches typos in the entry shape (missing ``provider`` or ``model`` key)
    at config-load time instead of surfacing as a confusing KeyError later
    inside the model-comparison pipeline.  ``extra="allow"`` lets each entry
    carry provider-specific extras (e.g. ``litellm_model``, ``timeout``).
    """

    model_config = ConfigDict(extra="allow")

    provider: str
    model: str


class ExtractionConfig(BaseModel):
    """Top-level schema for config/extraction.yaml."""

    model_config = ConfigDict(extra="allow")

    brain_build: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    ingestion: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    build_vocabulary: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    # Each comparison_models entry must declare provider + model; extras allowed.
    comparison_models: list[ComparisonModelEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# E3 / B8: per-section validation schemas for POST /api/settings/{section}
#
# These are DELIBERATELY DISTINCT from the config-load models above. The
# config-load models all use ``extra="allow"`` so that forward-compat keys in
# a user's on-disk YAML never break startup. The Advanced Settings HTTP route
# is the opposite situation: a POST body is untrusted input that gets written
# verbatim into the named section, so unknown keys / wrong types must be
# REJECTED (HTTP 422) instead of corrupting the file. Hence every model below
# sets ``extra="forbid"`` at every nesting level.
#
# Shapes are derived from config/extraction.example.yaml and the reads in
# scripts/core/config.py. ``feedback`` and ``web_ui`` have no shape anywhere
# in the loader; their shapes are documented at their class definitions.
# ---------------------------------------------------------------------------

class _RankingWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_context: float = 0.0
    cluster_centroid: float = 0.0
    graph_entity_overlap: float = 0.0
    graph_citation: float = 0.0
    graph_shared_authors: float = 0.0


class _RankingDomainFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    threshold: float = 0.35
    minimum_off_domain_slots_pct: float = 0.05


class _RankingS2Cap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_relevance: float = 0.4


class RankingSettings(BaseModel):
    """settings → ranking section (config.py Bundle A/C/D reads)."""

    model_config = ConfigDict(extra="forbid")

    weights: _RankingWeights = Field(default_factory=_RankingWeights)
    domain_filter: _RankingDomainFilter = Field(default_factory=_RankingDomainFilter)
    s2_supplement_cap: _RankingS2Cap = Field(default_factory=_RankingS2Cap)


class _ClusteringPriors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    collection_names: list[str] = Field(default_factory=list)


class _ClusteringWriteBackTags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    namespace: str = "lm"


class _ClusteringWriteBackCollections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    parent_collection: str = "lit-monitor"


class _ClusteringWriteBack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: _ClusteringWriteBackTags = Field(default_factory=_ClusteringWriteBackTags)
    collections: _ClusteringWriteBackCollections = Field(
        default_factory=_ClusteringWriteBackCollections
    )


class ClusteringSettings(BaseModel):
    """settings → clustering section (config.py Bundle C reads)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_papers_threshold: int = 100
    k_min: int = 5
    k_max: int = 15
    recompute_frequency: str = "nightly"
    use_existing_collections_as_priors: _ClusteringPriors = Field(
        default_factory=_ClusteringPriors
    )
    write_back: _ClusteringWriteBack = Field(default_factory=_ClusteringWriteBack)


class TrendingConceptsSettings(BaseModel):
    """settings → trending_concepts section (config.py Bundle E reads)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    threshold_growth_rate: float = 0.3
    min_recent_mentions: int = 5
    cooldown_days_after_dismiss: int = 60


class QueryExpansionSettings(BaseModel):
    """settings → query_expansion section (config.py Bundle E reads)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    top_k_co_entities: int = 3


class ResearcherGatingSettings(BaseModel):
    """settings → researcher_gating section (config.py Bundle E reads)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_graph_overlap: int = 1


class _EmbeddingOllama(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "http://localhost:11434"
    model: str = "mxbai-embed-large"
    dim: int = 1024


class _EmbeddingLiteLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "text-embedding-3-large"
    dim: int = 3072


class EmbeddingSettings(BaseModel):
    """settings → embedding section (config.py Bundle F reads).

    NOTE: this is the ``embedding`` (singular) provider-selection block from
    extraction.example.yaml, NOT the ``embeddings`` block validated by
    ``EmbeddingsConfig`` above — different shape, hence a separate model.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "litellm"] = "ollama"
    ollama: _EmbeddingOllama = Field(default_factory=_EmbeddingOllama)
    litellm: _EmbeddingLiteLLM = Field(default_factory=_EmbeddingLiteLLM)


class _DiscoveryNotify(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    on_zero_results: bool = False
    min_papers_to_notify: int = 1
    preferred_viewer: str = ""
    asked_user: bool = False


class _DiscoveryNotes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_write_per_paper: bool = True


class _DiscoveryScoreDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show_in_md: bool = False
    show_in_web_ui: bool = True


class _DiscoveryDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_write: bool = True
    score_decomposition: _DiscoveryScoreDecomposition = Field(
        default_factory=_DiscoveryScoreDecomposition
    )


class DiscoverySettings(BaseModel):
    """settings → discovery section (config.py P10/P10b + example.yaml reads)."""

    model_config = ConfigDict(extra="forbid")

    notify: _DiscoveryNotify = Field(default_factory=_DiscoveryNotify)
    notes: _DiscoveryNotes = Field(default_factory=_DiscoveryNotes)
    digest: _DiscoveryDigest = Field(default_factory=_DiscoveryDigest)


class WebUiSettings(BaseModel):
    """settings → web_ui section.

    DESIGN CALL (no shape defined in the loader). The ONLY web_ui key read
    anywhere in the codebase is ``web_ui.show_feedback_buttons`` (bool):
      - scripts/server/routes/discovery.py:195
      - scripts/server/routes/themes.py:261
      - scripts/server/templates/settings/index.html:83  (the checkbox)
      - scripts/server/templates/_partials/paper_card.html:12,58
      - tests/unit/test_routes_settings.py (web_ui payloads)
    So the model allows exactly that one key, ``extra="forbid"``.
    """

    model_config = ConfigDict(extra="forbid")

    show_feedback_buttons: bool = False


class FeedbackSettings(BaseModel):
    """settings → feedback section.

    DESIGN CALL (no shape defined anywhere). Grepping the codebase finds NO
    top-level ``feedback`` config key read at all — the only ``feedback``
    matches are the ``feedback_events`` DB table, the /api/feedback route, and
    feedback CSS/JS. The user-facing feedback toggle lives under
    ``web_ui.show_feedback_buttons``, not a ``feedback`` section.

    Since no key can be determined from real usage, this model uses
    ``extra="allow"`` for THIS section ONLY (documented escalation) rather than
    guessing a wrong shape and rejecting legitimate-but-undiscoverable keys.
    A non-dict body is still rejected upstream (caller's isinstance check).
    """

    model_config = ConfigDict(extra="allow")
