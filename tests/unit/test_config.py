"""
Unit tests for D1 — Pydantic config schema validation.

Tests cover:
- Schema-level validation (PathsConfig, ExtractionConfig, LLMProviderConfig)
  tested directly, independently of Config().
- Config() integration: ValidationError fires on bad extraction.yaml.
- Backward compatibility: existing valid YAML still loads without error.

Note: Config-level checks (vault_path empty / tilde) remain in test_core.py
because they go through _validate_vault_path() and raise ValueError, not
ValidationError — that split is intentional (see ARCHITECTURE.md D1 note).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers (mirrors pattern from test_core.py)
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w") as fh:
        yaml.dump(data, fh)


def _make_paths_yaml(tmp_path: Path, vault_path: str | None = None) -> Path:
    p = tmp_path / "paths.yaml"
    _vault = str(tmp_path / "vault") if vault_path is None else vault_path
    _write_yaml(p, {
        "zotero": {
            "library_type": "user",
            "library_id": "12345",
            "local_storage_path": str(tmp_path / "zotero_storage"),
            "collection_name": "lit-monitor",
        },
        "obsidian": {
            "vault_path": _vault,
            "papers_folder": "Literature/Papers",
            "books_folder": "Literature/Books",
            "digests_folder": "Literature/Digests",
            "connections_folder": "Literature/Connections",
        },
        "state_db": {"path": str(tmp_path / "state.db")},
        "logs": {"path": str(tmp_path / "logs"), "retention_days": 30},
    })
    return p


def _make_extraction_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "extraction.yaml"
    _write_yaml(p, {
        "brain_build": {"provider": "ollama", "model": "phi4-mini", "timeout": 300},
        "ingestion": {"provider": "ollama", "model": "qwen2.5:3b", "timeout": 300},
        "build_vocabulary": {"provider": "ollama", "model": "qwen2.5:3b"},
        "embeddings": {"model": "nomic-embed-text", "ollama_host": "http://localhost:11434"},
        "comparison_models": [],
    })
    return p


# ---------------------------------------------------------------------------
# PathsConfig — schema-level tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_paths_config_valid_full():
    """A well-formed paths dict validates without error."""
    from scripts.core.config_schema import PathsConfig
    PathsConfig.model_validate({
        "zotero": {"library_type": "user", "library_id": "123"},
        "obsidian": {"vault_path": "/absolute/path/to/vault"},
        "state_db": {"path": "/tmp/state.db"},
        "logs": {"path": "/tmp/logs", "retention_days": 90},
    })  # should not raise


@pytest.mark.unit
def test_config_validation_rejects_missing_vault_path():
    """PathsConfig raises ValidationError when vault_path key is absent."""
    from scripts.core.config_schema import PathsConfig
    with pytest.raises(ValidationError, match="vault_path"):
        PathsConfig.model_validate({
            "zotero": {"library_type": "user", "library_id": "123"},
            "obsidian": {},  # vault_path key entirely missing
            "state_db": {"path": "/tmp/state.db"},
            "logs": {"path": "/tmp/logs", "retention_days": 90},
        })


@pytest.mark.unit
def test_paths_config_rejects_missing_obsidian_section():
    """PathsConfig raises ValidationError when the obsidian section is absent."""
    from scripts.core.config_schema import PathsConfig
    with pytest.raises(ValidationError, match="obsidian"):
        PathsConfig.model_validate({
            "zotero": {"library_type": "user", "library_id": "123"},
            "state_db": {"path": "/tmp/state.db"},
            "logs": {"path": "/tmp/logs", "retention_days": 90},
        })


@pytest.mark.unit
def test_paths_config_rejects_invalid_library_type():
    """PathsConfig raises ValidationError when library_type is not 'user' or 'group'."""
    from scripts.core.config_schema import PathsConfig
    with pytest.raises(ValidationError, match="library_type"):
        PathsConfig.model_validate({
            "zotero": {"library_type": "organisation", "library_id": "123"},
            "obsidian": {"vault_path": "/vault"},
        })


@pytest.mark.unit
def test_paths_config_allows_extra_keys():
    """Unknown keys in paths.yaml are silently accepted (extra='allow')."""
    from scripts.core.config_schema import PathsConfig
    PathsConfig.model_validate({
        "zotero": {"library_type": "user", "library_id": "123"},
        "obsidian": {"vault_path": "/vault"},
        "future_section": {"some_key": "some_value"},  # forward-compat key
    })  # should not raise


# ---------------------------------------------------------------------------
# LLMProviderConfig / ExtractionConfig — schema-level tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extraction_config_valid_full():
    """A well-formed extraction dict validates without error."""
    from scripts.core.config_schema import ExtractionConfig
    ExtractionConfig.model_validate({
        "brain_build": {
            "provider": "ollama", "model": "phi4-mini",
            "temperature": 0.1, "timeout": 7200, "max_tokens_per_call": 12288,
            "think": False,
        },
        "ingestion": {"provider": "ollama", "model": "qwen2.5:3b"},
        "build_vocabulary": {"provider": "ollama", "model": "qwen2.5:3b"},
        "embeddings": {"model": "nomic-embed-text"},
        "comparison_models": [],
    })  # should not raise


@pytest.mark.unit
def test_config_validation_rejects_invalid_model_name():
    """LLMProviderConfig raises ValidationError for an unknown provider string.

    'Model name' in D1 context means the provider discriminator — 'openai'
    is not a valid value; only 'ollama' and 'litellm' are accepted.
    """
    from scripts.core.config_schema import LLMProviderConfig
    with pytest.raises(ValidationError, match="provider"):
        LLMProviderConfig.model_validate({"provider": "openai", "model": "gpt-4"})


@pytest.mark.unit
def test_llm_provider_rejects_temperature_above_1():
    """LLMProviderConfig raises ValidationError for temperature > 1.0."""
    from scripts.core.config_schema import LLMProviderConfig
    with pytest.raises(ValidationError, match="temperature"):
        LLMProviderConfig.model_validate({"provider": "ollama", "temperature": 1.5})


@pytest.mark.unit
def test_llm_provider_rejects_temperature_below_0():
    """LLMProviderConfig raises ValidationError for temperature < 0.0."""
    from scripts.core.config_schema import LLMProviderConfig
    with pytest.raises(ValidationError, match="temperature"):
        LLMProviderConfig.model_validate({"provider": "ollama", "temperature": -0.1})


@pytest.mark.unit
def test_llm_provider_accepts_litellm_provider():
    """LLMProviderConfig accepts provider='litellm' without error."""
    from scripts.core.config_schema import LLMProviderConfig
    cfg = LLMProviderConfig.model_validate({
        "provider": "litellm",
        "litellm_model": "anthropic/claude-haiku-4-5",
        "temperature": 0.0,
    })
    assert cfg.provider == "litellm"


@pytest.mark.unit
def test_config_comparison_models_malformed_entry_raises():
    """Malformed comparison_models entry (missing provider/model) raises ValidationError.

    A typo or missing key in one of the entries — e.g. ``[{"foo": "bar"}]`` —
    must fail validation rather than silently passing as a free-form dict and
    blowing up later in the model-comparison pipeline.
    """
    from scripts.core.config_schema import ExtractionConfig
    with pytest.raises(ValidationError):
        ExtractionConfig.model_validate({
            "comparison_models": [{"foo": "bar"}],
        })


@pytest.mark.unit
def test_extraction_config_allows_extra_keys():
    """Per-pass model keys (pass1_model etc.) are extra keys — should not raise."""
    from scripts.core.config_schema import ExtractionConfig
    ExtractionConfig.model_validate({
        "brain_build": {
            "provider": "ollama",
            "model": "phi4-mini",
            "pass1_model": "qwen2.5:7b",   # F15 per-pass key — extra field
            "ocr_dpi": 400,                 # pipeline config extra field
        },
        "ingestion": {"provider": "ollama", "model": "qwen2.5:3b"},
        "build_vocabulary": {"provider": "ollama", "model": "qwen2.5:3b"},
        "embeddings": {"model": "nomic-embed-text"},
    })  # should not raise


# ---------------------------------------------------------------------------
# Config() integration — ValidationError fires on bad extraction.yaml
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_config_init_raises_on_invalid_extraction_provider(tmp_path):
    """Config() raises ValidationError when a mode section has provider='anthropic'."""
    from pydantic import ValidationError as PydanticValidationError

    from scripts.core.config import Config

    paths = _make_paths_yaml(tmp_path)
    extraction = tmp_path / "extraction.yaml"
    _write_yaml(extraction, {
        "brain_build": {"provider": "anthropic", "model": "claude-3"},  # invalid
        "ingestion": {"provider": "ollama", "model": "qwen2.5:3b"},
        "build_vocabulary": {"provider": "ollama", "model": "qwen2.5:3b"},
        "embeddings": {"model": "nomic-embed-text"},
        "comparison_models": [],
    })
    with pytest.raises(PydanticValidationError):
        Config(paths_yaml=paths, extraction_yaml=extraction)


@pytest.mark.unit
def test_config_init_accepts_valid_yaml(tmp_path):
    """Config() loads cleanly when both YAML files are valid — no regression."""
    from scripts.core.config import Config

    paths = _make_paths_yaml(tmp_path)
    extraction = _make_extraction_yaml(tmp_path)
    cfg = Config(paths_yaml=paths, extraction_yaml=extraction)
    assert cfg.brain_build.provider == "ollama"
    assert cfg.brain_build.model == "phi4-mini"
    assert cfg.embeddings.model == "nomic-embed-text"
