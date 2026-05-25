"""Shared filesystem paths for the setup checks.

Keeps SECRETS_PATH in one place so both check_configured and check_zotero
agree on where to look for the user's credentials TOML.
"""
from __future__ import annotations

from pathlib import Path

SECRETS_PATH: Path = Path.home() / ".config" / "lit-monitor" / "config.toml"
