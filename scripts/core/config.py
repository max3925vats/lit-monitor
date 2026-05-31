"""
Core configuration loader.
Loads and validates config/paths.yaml and config/extraction.yaml.
Secrets (Zotero key, API keys) are read from ~/.config/lit-monitor/config.toml
which is never read by this module — only check_configured.py touches it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from scripts.core.config_schema import ExtractionConfig, PathsConfig
from scripts.core.strict_mode import strict_fallback

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "lit-monitor"
_DEFAULT_SECRETS_PATH = _DEFAULT_CONFIG_DIR / "config.toml"


def _find_project_root() -> Path:
    """Locate the project root containing pyproject.toml.

    Search order:
    1. LIT_MONITOR_ROOT env var (hard override for any install layout)
    2. Walk up from CWD — works when running lit-monitor from the project dir
    3. Walk up from __file__ — works for editable installs / venv setups
    Raises RuntimeError if no pyproject.toml is found in either walk.
    """
    if root := os.environ.get("LIT_MONITOR_ROOT"):
        return Path(root)
    # CWD walk-up: the common case — user runs from the project directory
    candidate = Path.cwd()
    for _ in range(8):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    # __file__ walk-up: editable installs / .venv inside project root
    candidate = Path(__file__).resolve().parent
    for _ in range(12):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(
        "lit-monitor: could not locate project root (no pyproject.toml found).\n"
        "Either run lit-monitor from within the project directory, or set the\n"
        "LIT_MONITOR_ROOT environment variable to the project path."
    )


_PROJECT_ROOT = _find_project_root()
_CONFIG_DIR = _PROJECT_ROOT / "config"
# ---------------------------------------------------------------------------
# Config classes (simple namespaces — not dataclasses to keep it lightweight)
# ---------------------------------------------------------------------------
class _Namespace:
    """Dot-access dict wrapper."""
    def __init__(self, data: dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else v)
    def get(self, attr: str, default: Any = None) -> Any:
        return getattr(self, attr, default)
class Config:
    """
    Top-level configuration object assembled from:
    - config/paths.yaml      — filesystem paths (vault, zotero, state db, logs)
    - config/extraction.yaml — LLM model selection per pipeline mode
    - config/topics.yaml     — recurring search queries  (optional)
    - config/researchers.yaml — tracked researchers   (optional)
    - config/concepts.yaml   — vocabulary themes      (optional)
    - config/domain_context.yaml — domain description (optional)
    - ~/.config/lit-monitor/config.toml — API keys and credentials (never read here)
    All paths are normalised (~ expanded, forward/backslash unified) at load time.
    """
    def __init__(
        self,
        paths_yaml: Path | None = None,
        extraction_yaml: Path | None = None,
    ) -> None:
        paths_yaml = paths_yaml or _CONFIG_DIR / "paths.yaml"
        extraction_yaml = extraction_yaml or _CONFIG_DIR / "extraction.yaml"
        raw_paths = _load_yaml(paths_yaml)
        # Validate structure; validated model provides Pydantic-coerced values.
        validated_paths = PathsConfig.model_validate(raw_paths)

        raw_extraction = _load_yaml(extraction_yaml)
        # Validates provider, temperature ranges, etc. before any LLM call.
        validated_extraction = ExtractionConfig.model_validate(raw_extraction)
        # --- Zotero paths (read from validated model so defaults and types are correct) ---
        z = validated_paths.zotero
        self.zotero = _Namespace({
            "library_type": z.library_type,
            "library_id": z.library_id,
            "local_storage_path": _expand(z.local_storage_path),
            "collection_name": z.collection_name,
        })
        # --- Obsidian paths ---
        o = validated_paths.obsidian
        vault_path = o.vault_path
        _validate_vault_path(vault_path)
        self.obsidian = _Namespace({
            "vault_path": Path(vault_path),
            "papers_folder": o.papers_folder,
            "books_folder": o.books_folder,
            "digests_folder": o.digests_folder,
            "connections_folder": o.connections_folder,
        })
        # --- State DB ---
        self.state_db = _Namespace({
            "path": Path(_expand(validated_paths.state_db.path)),
        })
        # --- Logs ---
        lg = validated_paths.logs
        self.logs = _Namespace({
            "path": Path(_expand(lg.path)),
            "retention_days": lg.retention_days,  # already int via Pydantic
        })
        # --- Extraction (LLM config per mode — use model_dump() so Pydantic
        #     coercions like temperature→float and default values apply) ---
        self.brain_build = _Namespace(validated_extraction.brain_build.model_dump())
        self.ingestion = _Namespace(validated_extraction.ingestion.model_dump())
        self.build_vocabulary = _Namespace(validated_extraction.build_vocabulary.model_dump())
        self.embeddings = _Namespace(validated_extraction.embeddings.model_dump())
        # Dump back to dicts so downstream code (model_compare.py) can keep
        # using ``m.get('provider', ...)`` / ``m.get('model', ...)`` patterns.
        self.comparison_models: list[dict] = [
            m.model_dump() for m in validated_extraction.comparison_models
        ]
        # --- P10: Discovery config (notify + per-paper notes control) ---
        # Read directly from raw_extraction so the block is optional; missing
        # keys default to safe values.  Nested dicts become _Namespace objects
        # so callers can use dot-access (config.discovery.notes.auto_write_per_paper).
        raw_discovery = raw_extraction.get("discovery", {}) or {}
        _raw_disc_notes = raw_discovery.get("notes", {}) or {}
        self.discovery = _Namespace({
            "notify": raw_discovery.get("notify", {}) or {},
            "notes": _Namespace({
                # Default TRUE — preserves existing inline write behaviour.
                "auto_write_per_paper": bool(
                    _raw_disc_notes.get("auto_write_per_paper", True)
                ),
            }),
        })
        # --- G9: Retrieval config (default_mode + graph_db location) ---
        raw_retrieval = raw_extraction.get("retrieval", {})
        _ret_mode = raw_retrieval.get("default_mode", "vector")
        if _ret_mode not in {"vector", "graph", "hybrid"}:
            import warnings
            warnings.warn(
                f"retrieval.default_mode {_ret_mode!r} is not one of "
                "vector/graph/hybrid — falling back to 'vector'.",
                stacklevel=2,
            )
            _ret_mode = "vector"
        _ret_graph = raw_retrieval.get("graph_db", {}) or {}
        self.retrieval = _Namespace({
            "default_mode": _ret_mode,
            "graph_db": _Namespace(_ret_graph) if isinstance(_ret_graph, dict) else _ret_graph,
        })
        # --- Optional configs (loaded lazily if files exist) ---
        self._topics: list[dict] | None = None
        self._discovery_top_k: int = 20
        self._date_window_days: int = 14
        self._researchers: list[dict] | None = None
        self._concepts: dict | None = None
        self._domain_context: str = ""
    # -- Optional config accessors --
    @property
    def topics(self) -> list[dict]:
        if self._topics is None:
            p = _CONFIG_DIR / "topics.yaml"
            if p.exists():
                raw = _load_yaml(p)
                self._topics = raw.get("searches", [])

                self._discovery_top_k = raw.get("discovery_top_k", 20)
                self._date_window_days = raw.get("date_window_days", 14)
            else:
                self._topics = []
                self._discovery_top_k = 20
                self._date_window_days = 14
        return self._topics
    @property
    def discovery_top_k(self) -> int:
        self.topics  # ensure topics.yaml is loaded and cached
        return self._discovery_top_k
    @property
    def date_window_days(self) -> int:
        self.topics  # ensure topics.yaml is loaded and cached
        return self._date_window_days
    @property
    def researchers(self) -> list[dict]:
        if self._researchers is None:
            p = _CONFIG_DIR / "researchers.yaml"
            self._researchers = (
                _load_yaml(p).get("researchers", []) if p.exists() else []
            )
        return self._researchers
    @property
    def concepts(self) -> dict:
        if self._concepts is None:
            p = _CONFIG_DIR / "concepts.yaml"
            self._concepts = _load_yaml(p) if p.exists() else {}
        return self._concepts
    @property
    def domain_context(self) -> str:
        if not self._domain_context:
            p = _CONFIG_DIR / "domain_context.yaml"
            if p.exists():
                self._domain_context = _load_yaml(p).get("domain_focus", "")
        return self._domain_context
    # -- Convenience helpers --
    def obsidian_paper_dir(self) -> Path:
        return self.obsidian.vault_path / self.obsidian.papers_folder
    def obsidian_book_dir(self) -> Path:
        return self.obsidian.vault_path / self.obsidian.books_folder
    def obsidian_digest_dir(self) -> Path:
        return self.obsidian.vault_path / self.obsidian.digests_folder
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_config_logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        # Parse failures are operator-actionable: log ERROR (not WARNING)
        # so they show up in standard log filters, then route through
        # strict_fallback so strict mode raises instead of swallowing.
        _config_logger.error("YAML parse failed in %s: %s", path, exc)
        strict_fallback(
            _config_logger,
            f"YAML parse failed in {path}: {exc} — treating as empty dict.",
            exc,
        )
        return {}
    if not isinstance(data, dict):
        strict_fallback(
            _config_logger,
            f"Config file {path} did not parse to a dict "
            f"(got {type(data).__name__!r}) — treating as empty. "
            "Check for YAML syntax errors or an empty file.",
        )
        return {}
    return data

def _expand(raw: str) -> str:
    """Expand ~ and environment variables; normalise to forward slashes."""
    expanded = os.path.expandvars(os.path.expanduser(raw))
    return str(Path(expanded))
def _validate_vault_path(vault_path: str) -> None:
    """
    Raise ValueError if vault_path starts with ~ or is empty.
    The Obsidian vault path must be a full absolute path because:
    1. ~ is not expanded on all platforms predictably.
    2. Obsidian syncs to iCloud on macOS, placing the vault outside the
       home directory on some configurations.
    """
    if not vault_path:
        raise ValueError(
            "obsidian.vault_path is not set in config/paths.yaml. "
            "Set it to the full absolute path to your Obsidian vault."
        )
    if vault_path.startswith("~"):
        raise ValueError(
            f"obsidian.vault_path must be a full absolute path, not starting "
            f"with '~'. Got: {vault_path!r}"
        )
# ---------------------------------------------------------------------------
# Singleton loader (used by scripts that need config)
# ---------------------------------------------------------------------------
_config_cache: Config | None = None
def get_config(
    paths_yaml: Path | None = None,
    extraction_yaml: Path | None = None,
    *,
    force_reload: bool = False,
) -> Config:
    """Return cached Config, loading on first call."""
    global _config_cache
    if _config_cache is None or force_reload:
        _config_cache = Config(paths_yaml=paths_yaml, extraction_yaml=extraction_yaml)
    return _config_cache
