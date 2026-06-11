"""Atomic load/save helpers for wizard-managed config files.

YAML side: load_config / save_config — for config/{name}.yaml
TOML side: load_secrets / save_secrets — for ~/.config/lit-monitor/config.toml

All saves are atomic: write to a sibling temp file in the same directory,
then ``os.replace()`` it into place.  On POSIX this is atomic provided the
temp file and the target live on the same filesystem — using the target's
parent directory for the temp file guarantees that.
"""
from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
import yaml

from lit_monitor.core.atomic_write import atomic_write_text
from lit_monitor.core.path_utils import resolve_path as _resolve_path
from lit_monitor.setup._paths import SECRETS_PATH

logger = logging.getLogger(__name__)

# Module-level so tests can monkeypatch a tmp_path / "config" directory in.
CONFIG_DIR = Path("config")


class SettingsValidationError(ValueError):
    """E3 / B8: a settings payload failed its per-section pydantic schema.

    Subclasses ``ValueError`` so existing direct callers (and the helper's
    documented contract) keep treating validation failures as ValueErrors.
    The HTTP route catches this specific type to return 422 (malformed
    payload) while a plain ValueError (e.g. unknown section) stays 400.
    """


def load_config(name: str) -> dict[str, Any]:
    """Load ``config/{name}.yaml``.

    Falls back to ``config/{name}.example.yaml`` if the real file is absent,
    via :func:`scripts.core.path_utils.resolve_path`.

    Raises:
        FileNotFoundError: If neither the real nor the example file exists
            (per the :func:`_resolve_path` contract).
        yaml.YAMLError: If the file exists but cannot be parsed as YAML.
            Wizard routes are expected to wrap this call and render the
            parse error to the user rather than letting it 500.
    """
    path = _resolve_path(CONFIG_DIR / f"{name}.yaml")
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def save_config(name: str, data: dict[str, Any]) -> Path:
    """Atomically write ``config/{name}.yaml``. Returns the final path.

    Uses ruamel.yaml round-trip mode so any comments / quote style /
    key ordering on the EXISTING file survive when the wizard re-saves
    after a single-field edit. New top-level keys in ``data`` are
    appended at the bottom in insertion order.
    """
    target = CONFIG_DIR / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    yaml_rt, existing = _round_trip_load(target)
    # Replace top-level keys from ``data`` while preserving any pre-existing
    # comments + ordering on keys that aren't being edited. ``data`` is the
    # full new document, so swap in each key explicitly.
    for k, v in data.items():
        existing[k] = v
    # Drop top-level keys the new dict no longer contains. The wizard always
    # passes the COMPLETE intended document, so absence = explicit deletion.
    for k in list(existing.keys()):
        if k not in data:
            del existing[k]
    return _atomic_write(target, _round_trip_dump(yaml_rt, existing))


def load_secrets() -> dict[str, Any]:
    """Read the secrets TOML. Returns ``{}`` when absent, empty, or malformed.

    Returns ``{}`` only for the "no usable content" cases: missing file,
    directory-where-a-file-should-be, or malformed TOML.  ``PermissionError``
    and other ``OSError`` subclasses propagate so that the wizard's setup
    route can distinguish "not configured yet" from "exists but unreadable"
    and render a specific error to the user.
    """
    if not SECRETS_PATH.exists():
        return {}
    try:
        with SECRETS_PATH.open("rb") as fh:
            return tomllib.load(fh)
    except (FileNotFoundError, IsADirectoryError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Could not parse secrets TOML at %s: %s", SECRETS_PATH, exc)
        return {}


def save_secrets(data: dict[str, Any]) -> Path:
    """Atomically write the secrets TOML with mode 0600.

    Creates parent directories if missing.  Returns the final path.
    """
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rendered = tomli_w.dumps(data)
    final = _atomic_write(SECRETS_PATH, rendered)
    os.chmod(final, 0o600)
    return final


def load_server_config() -> dict[str, Any]:
    """Return the [server] block from secrets TOML, or {} if absent.

    Caller is responsible for applying defaults — this just exposes
    whatever the user has persisted.
    """
    secrets = load_secrets()
    block = secrets.get("server", {})
    # Defensive: a malformed TOML could have [server] as a non-table.
    return block if isinstance(block, dict) else {}


def save_server_config(host: str, port: int, open_browser: bool) -> None:
    """Merge a [server] block into the secrets TOML atomically.

    Preserves all other top-level keys (zotero/pubmed/scopus/etc.) by
    round-tripping through load_secrets() + save_secrets().  Atomic write
    via os.replace; 0o600 mode preserved (save_secrets handles both).
    """
    secrets = load_secrets()
    secrets["server"] = {
        "host": host,
        "port": int(port),
        "open_browser": bool(open_browser),
    }
    save_secrets(secrets)


# P4: valid viewer identifiers for the notification-chooser preference.
_VALID_VIEWERS: frozenset[str] = frozenset({"browser", "obsidian", "none"})


def _round_trip_load(path: Path):
    """Load a YAML file with comment-preserving round-trip semantics.

    Returns (yaml_rt, data). Callers mutate ``data`` and pass it to
    :func:`_round_trip_dump` to render the modified document back to text.
    For a missing file the data is a fresh ``CommentedMap`` so any new
    nested keys land cleanly.

    All writers in this module that target user-edited YAML route through
    this helper. PyYAML's ``safe_dump`` silently strips comments + quote
    style + key ordering — the v0.9.0 Bundle H regression that prompted
    this hotfix.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    yaml_rt.width = 4096  # don't visually drift long-line values
    # ruamel's default emits None as the empty scalar (canonical YAML). We
    # round-trip many user configs that explicitly write ``null``, so install
    # a representer that keeps the explicit token to avoid visual drift
    # (e.g. ``device: null`` → ``device:`` would silently rewrite the file).
    yaml_rt.representer.add_representer(
        type(None),
        lambda r, d: r.represent_scalar("tag:yaml.org,2002:null", "null"),
    )
    if path.exists():
        data = yaml_rt.load(path.read_text(encoding="utf-8")) or CommentedMap()
    else:
        data = CommentedMap()
    return yaml_rt, data


def _round_trip_dump(yaml_rt, data) -> str:
    """Companion to :func:`_round_trip_load` — render the data back to text."""
    from io import StringIO

    buf = StringIO()
    yaml_rt.dump(data, buf)
    return buf.getvalue()


def safe_save_preference(
    viewer: str,
    *,
    enabled: bool | None = None,
    config_path: Path | None = None,
) -> None:
    """P4: atomically update ``discovery.notify`` in ``extraction.yaml``.

    Sets ``preferred_viewer`` to ``viewer`` and ``asked_user`` to ``True``.
    If ``enabled`` is not ``None``, also updates the ``enabled`` flag.

    Parameters
    ----------
    viewer:
        One of ``{"browser", "obsidian", "none"}``.  Raises ``ValueError``
        for any other value so callers get a clear error rather than silently
        persisting garbage.
    enabled:
        When provided, overrides ``discovery.notify.enabled``.  ``None``
        (default) leaves the existing value untouched.
    config_path:
        Explicit path to the YAML file.  Defaults to
        ``config/extraction.yaml`` (the live config).  Override in tests to
        point at a ``tmp_path`` copy.

    Raises
    ------
    ValueError
        If ``viewer`` is not in the valid set.
    OSError
        If the file cannot be read or written.
    """
    if viewer not in _VALID_VIEWERS:
        raise ValueError(
            f"invalid viewer: {viewer!r}; must be one of {sorted(_VALID_VIEWERS)}"
        )

    path = Path(config_path) if config_path is not None else CONFIG_DIR / "extraction.yaml"

    # v0.9.1: ruamel round-trip preserves user comments + quote style.
    yaml_rt, data = _round_trip_load(path)

    # Navigate / create the nested structure defensively.
    data.setdefault("discovery", {})
    if not isinstance(data["discovery"], dict):
        data["discovery"] = {}
    data["discovery"].setdefault("notify", {})
    if not isinstance(data["discovery"]["notify"], dict):
        data["discovery"]["notify"] = {}

    data["discovery"]["notify"]["preferred_viewer"] = viewer
    data["discovery"]["notify"]["asked_user"] = True
    if enabled is not None:
        data["discovery"]["notify"]["enabled"] = bool(enabled)

    rendered = _round_trip_dump(yaml_rt, data)
    # Delegate to the same atomic helper used by save_config / save_secrets.
    _atomic_write(path, rendered)


def safe_save_digest_auto_write(
    value: bool,
    *,
    config_path: Path | None = None,
) -> None:
    """P10b: atomically update ``discovery.digest.auto_write`` in ``extraction.yaml``.

    Parameters
    ----------
    value:
        ``True`` to enable automatic digest writes (default); ``False`` to disable.
    config_path:
        Explicit path to the YAML file.  Defaults to ``config/extraction.yaml``.
        Override in tests to point at a ``tmp_path`` copy.

    Raises
    ------
    OSError
        If the file cannot be read or written.
    """
    path = Path(config_path) if config_path is not None else CONFIG_DIR / "extraction.yaml"

    # v0.9.1: ruamel round-trip preserves user comments + quote style.
    yaml_rt, data = _round_trip_load(path)

    # Navigate / create the nested structure defensively.
    data.setdefault("discovery", {})
    if not isinstance(data["discovery"], dict):
        data["discovery"] = {}
    data["discovery"].setdefault("digest", {})
    if not isinstance(data["discovery"]["digest"], dict):
        data["discovery"]["digest"] = {}

    data["discovery"]["digest"]["auto_write"] = bool(value)

    rendered = _round_trip_dump(yaml_rt, data)
    _atomic_write(path, rendered)


def safe_save_topics(
    new_topic: dict,
    *,
    topics_path: Path | None = None,
) -> None:
    """Bundle E: atomically add a new topic entry to topics.yaml (searches list).

    Reads the existing topics.yaml (or starts with an empty searches list if the
    file does not exist), appends the new_topic dict, and writes back atomically
    via tempfile + os.replace. Never removes or modifies existing topics.

    Args:
        new_topic:   Dict with at minimum a "name" and "query" key. Appended
                     directly to the searches list without validation — caller
                     is responsible for schema correctness.
        topics_path: Override the path for testing. Defaults to
                     config/topics.yaml (relative to cwd, same convention as
                     CONFIG_DIR for other helpers).

    Raises:
        OSError: If the file cannot be read or written (e.g. permissions).
    """
    path = Path(topics_path) if topics_path is not None else CONFIG_DIR / "topics.yaml"

    # v0.9.1: ruamel round-trip preserves any user comments / formatting on
    # the existing topics.yaml (e.g. category section headers).
    yaml_rt, data = _round_trip_load(path)

    # Ensure the searches list exists.
    searches = data.get("searches")
    if not isinstance(searches, list):
        searches = []
    searches.append(new_topic)
    data["searches"] = searches

    _atomic_write(path, _round_trip_dump(yaml_rt, data))


def _validate_settings_section(section: str, data: dict[str, Any]) -> None:
    """E3 / B8: validate ``data`` against the section's pydantic schema.

    Each schema sets ``extra="forbid"``, so unknown keys or wrong types raise
    ``pydantic.ValidationError``. That is re-raised as ``ValueError`` so the HTTP
    route's existing ``except ValueError`` branch handles it uniformly.

    The section name is assumed already validated against the allowed set by
    the caller. An unmapped (but allowed) section would be a programming error,
    so we fail loud rather than silently skipping validation.
    """
    from pydantic import ValidationError

    from lit_monitor.core.config_schema import (
        ClusteringSettings,
        DiscoverySettings,
        EmbeddingSettings,
        FeedbackSettings,
        QueryExpansionSettings,
        RankingSettings,
        ResearcherGatingSettings,
        TrendingConceptsSettings,
        WebUiSettings,
    )

    section_models = {
        "ranking": RankingSettings,
        "clustering": ClusteringSettings,
        "trending_concepts": TrendingConceptsSettings,
        "query_expansion": QueryExpansionSettings,
        "researcher_gating": ResearcherGatingSettings,
        "web_ui": WebUiSettings,
        "embedding": EmbeddingSettings,
        "discovery": DiscoverySettings,
        "feedback": FeedbackSettings,
    }
    model = section_models[section]
    try:
        model.model_validate(data)
    except ValidationError as exc:
        raise SettingsValidationError(
            f"invalid payload for section {section!r}: {exc}"
        ) from exc


def safe_save_settings_section(
    section: str,
    data: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> None:
    """Bundle H (v0.9.1 hotfix): atomically update one top-level section of extraction.yaml.

    Uses ruamel.yaml round-trip mode so user comments, quote style, and key
    ordering on the unchanged sections survive the write. PyYAML's safe_dump
    silently strips all of those; that was the v0.9.0 regression that
    destroyed the carefully curated comments in ``config/extraction.yaml``
    the first time the Advanced-Settings panel toggled any flag.

    The function reads the current file via ruamel round-trip loader,
    replaces only the named section, and re-dumps the same data structure.
    All other top-level keys retain their comments and formatting.

    Args:
        section:     Top-level key in extraction.yaml to replace.  Must be
                     one of the allowed section names.
        data:        New dict to store under ``section``.
        config_path: Override for the YAML path.  Defaults to
                     ``config/extraction.yaml``.  Override in tests.

    Raises:
        ValueError: If ``section`` is not in the allowed set, if ``data`` is
            not a dict, or (E3/B8) if ``data`` fails the section's pydantic
            schema validation (unknown keys / wrong types).
        OSError:    If the file cannot be read or written.
    """
    _ALLOWED_SECTIONS: frozenset[str] = frozenset(
        {
            "ranking",
            "clustering",
            "trending_concepts",
            "query_expansion",
            "researcher_gating",
            "web_ui",
            "embedding",
            "discovery",
            "feedback",
        }
    )
    if section not in _ALLOWED_SECTIONS:
        raise ValueError(
            f"unknown section: {section!r}; allowed: {sorted(_ALLOWED_SECTIONS)}"
        )

    # P2.4 (minimal guard): a top-level config section is always a mapping.
    # Refuse to write a list/scalar/None verbatim, which would corrupt the
    # section's shape. This is defense-in-depth: the HTTP route already
    # rejects non-object bodies with 422, but safe_save_settings_section is
    # an importable helper that can be called directly, so it guards itself.
    if not isinstance(data, dict):
        raise ValueError(
            f"section {section!r} data must be a mapping, got {type(data).__name__!r}"
        )

    # E3 / B8: validate the incoming dict against the section's pydantic model
    # (extra="forbid", so unknown keys and wrong types are rejected) BEFORE we
    # touch the file. A ValidationError is re-raised as ValueError so the HTTP
    # route's existing ValueError handler maps it to 4xx — see below for the
    # 422-not-400 escalation note. Only the incoming section's data is checked;
    # the rest of the file is round-tripped untouched, preserving comments and
    # the other sections' formatting/ordering.
    #
    # We validate the raw dict but write the ORIGINAL ``data`` verbatim (not the
    # model_dump), so we never inject schema defaults into a section the user
    # only partially specified — the existing per-section "replace" semantics
    # are preserved exactly; validation is a gate, not a normalizer.
    _validate_settings_section(section, data)

    path = Path(config_path) if config_path is not None else CONFIG_DIR / "extraction.yaml"

    # v0.9.1: ruamel round-trip preserves user comments + quote style.
    yaml_rt, full = _round_trip_load(path)
    full[section] = data
    _atomic_write(path, _round_trip_dump(yaml_rt, full))


def _atomic_write(target: Path, content: str) -> Path:
    """Atomically write text to ``target`` (delegates to scripts.core.atomic_write).

    Kept as a thin wrapper so existing callers (save_config / save_secrets) keep
    working unchanged.  The canonical implementation lives in
    :func:`scripts.core.atomic_write.atomic_write_text`, which additionally
    fsyncs before the rename — closing the power-loss window that this
    function's previous local copy had.
    """
    return atomic_write_text(target, content)
