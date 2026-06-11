"""
Integration test fixtures.
All fixtures in this file require live services (Ollama, Zotero, filesystem).

Default behavior (no env var set):
    pytest tests/integration/ -m integration -v
    → Missing services cause tests to SKIP silently. Useful for ad-hoc local runs
      where you don't want a hard failure just because Ollama isn't running.

Strict / "live" mode (recommended for the release gate):
    LIT_MONITOR_LIVE=1 pytest tests/integration/ -m integration -v
    → Missing services cause tests to FAIL loudly with a clear message.
      Catches the case where someone thought integration was working but
      services were silently down.

Prerequisites:
    - lit-monitor check  must pass (Ollama running, Zotero creds set, vault exists)
    - Ollama has the ingestion model pulled (default: qwen2.5:3b)
    - Ollama has mxbai-embed-large pulled (for ChromaDB embeddings)
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tests.integration._live import skip_or_fail as _skip_or_fail

_SECRETS_PATH = Path.home() / ".config" / "lit-monitor" / "config.toml"
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # lit-monitor/


# ---------------------------------------------------------------------------
# Credential loader
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def secrets() -> dict:
    """Load raw secrets from config.toml. Skip entire session if absent."""
    if not _SECRETS_PATH.exists():
        _skip_or_fail(
            f"Secrets file not found: {_SECRETS_PATH}\n"
            "Create it with [zotero] api_key and library_id before running integration tests."
        )
    with _SECRETS_PATH.open("rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_config(secrets):
    """
    Loaded Config object.
    Skips if vault_path doesn't exist (vault not mounted / path not set).
    """
    from lit_monitor.core.config import get_config
    try:
        cfg = get_config(force_reload=True)
    except Exception as exc:
        _skip_or_fail(f"Config failed to load: {exc}")

    vault = Path(cfg.obsidian.vault_path)
    if not vault.exists():
        _skip_or_fail(
            f"Obsidian vault not accessible at {vault}. "
            "Set obsidian.vault_path in config/paths.yaml to a full absolute path."
        )
    return cfg


# ---------------------------------------------------------------------------
# Ollama fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_llm(real_config):
    """
    OllamaClient for the ingestion model.
    Skips if Ollama is not running or the model is not pulled.
    """
    import requests as _requests

    from lit_monitor.llm.llm_client import get_client

    host = getattr(real_config.ingestion, "ollama_host", "http://localhost:11434")
    try:
        resp = _requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        _skip_or_fail(f"Ollama not reachable at {host}: {exc}")

    model = getattr(real_config.ingestion, "model", "qwen2.5:3b")
    tags = resp.json().get("models", [])
    pulled = [m.get("name", "") for m in tags]
    if not any(model in p for p in pulled):
        _skip_or_fail(
            f"Ollama model '{model}' is not pulled. "
            f"Run: ollama pull {model}"
        )

    return get_client(real_config, mode="ingestion")


# ---------------------------------------------------------------------------
# Zotero fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_zotero(real_config, secrets):
    """ZoteroClient authenticated with real credentials."""
    from lit_monitor.core.zotero_client import ZoteroClient
    try:
        zot = ZoteroClient(
            library_id=str(secrets["zotero"]["library_id"]),
            api_key=secrets["zotero"]["api_key"],
            library_type=real_config.zotero.library_type,
            local_storage_path=real_config.zotero.local_storage_path,
        )
    except KeyError as exc:
        _skip_or_fail(f"Missing Zotero credential in config.toml: {exc}")
    except Exception as exc:
        _skip_or_fail(f"ZoteroClient init failed: {exc}")
    return zot


# ---------------------------------------------------------------------------
# Ephemeral storage fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """Temporary ChromaDB directory (auto-deleted by pytest after test)."""
    return tmp_path / "chroma"


@pytest.fixture
def tmp_state_db(tmp_path):
    """StateDB backed by a real SQLite file in a temp directory."""
    from lit_monitor.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")
