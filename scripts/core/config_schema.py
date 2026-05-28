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
