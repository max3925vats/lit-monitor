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
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
import yaml

from scripts.llm.prompt_registry import _resolve_path
from scripts.setup._paths import SECRETS_PATH

logger = logging.getLogger(__name__)

# Module-level so tests can monkeypatch a tmp_path / "config" directory in.
CONFIG_DIR = Path("config")


def load_config(name: str) -> dict[str, Any]:
    """Load ``config/{name}.yaml``.

    Falls back to ``config/{name}.example.yaml`` if the real file is absent,
    mirroring :func:`scripts.llm.prompt_registry._resolve_path`.  Raises
    ``FileNotFoundError`` if neither exists.
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
    """Read the secrets TOML. Returns ``{}`` when absent or malformed.

    Never raises — wizard callers should re-validate before saving so that
    parse errors surface inline rather than as a 500.
    """
    if not SECRETS_PATH.exists():
        return {}
    try:
        with SECRETS_PATH.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
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


def _atomic_write(target: Path, content: str) -> Path:
    """Write ``content`` to ``target`` via a same-directory temp file + os.replace.

    The temp file is created in the target's parent so the eventual rename is
    a same-filesystem operation (atomic on POSIX).  If the rename fails we
    do our best to unlink the leftover temp file before re-raising.
    """
    fd, tmp = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target
