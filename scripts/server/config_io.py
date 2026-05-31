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

from scripts.core.atomic_write import atomic_write_text
from scripts.core.path_utils import resolve_path as _resolve_path
from scripts.setup._paths import SECRETS_PATH

logger = logging.getLogger(__name__)

# Module-level so tests can monkeypatch a tmp_path / "config" directory in.
CONFIG_DIR = Path("config")


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
    """Atomically write ``config/{name}.yaml``. Returns the final path."""
    target = CONFIG_DIR / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return _atomic_write(target, rendered)


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

    raw = path.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(raw) or {}

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

    rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    # Delegate to the same atomic helper used by save_config / save_secrets.
    _atomic_write(path, rendered)


def _atomic_write(target: Path, content: str) -> Path:
    """Atomically write text to ``target`` (delegates to scripts.core.atomic_write).

    Kept as a thin wrapper so existing callers (save_config / save_secrets) keep
    working unchanged.  The canonical implementation lives in
    :func:`scripts.core.atomic_write.atomic_write_text`, which additionally
    fsyncs before the rename — closing the power-loss window that this
    function's previous local copy had.
    """
    return atomic_write_text(target, content)
