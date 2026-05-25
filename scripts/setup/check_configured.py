"""
Check that the user's secrets config (~/.config/lit-monitor/config.toml) exists
and contains the minimum required keys.
Returns a dict of {check_name: (ok: bool, message: str)}.
"""
from __future__ import annotations

import tomllib

from scripts.setup._paths import SECRETS_PATH as _SECRETS_PATH

_REQUIRED_KEYS: list[tuple[str, str]] = [
    # (section, key)
    ("zotero", "api_key"),
    ("zotero", "library_id"),
    ("pubmed", "email"),   # required for Unpaywall OA PDF enrichment
]
_OPTIONAL_KEYS: list[tuple[str, str]] = [
    ("scopus", "api_key"),
]
def check_configured() -> dict[str, tuple[bool, str]]:
    """
    Return {check_name: (ok, message)} for each configuration check.
    Does not raise — always returns results.
    """
    results: dict[str, tuple[bool, str]] = {}
    # --- Secrets file exists ---
    if not _SECRETS_PATH.exists():
        results["secrets_file"] = (
            False,
            f"Missing: {_SECRETS_PATH}\n"
            "  Create it with at minimum:\n"
            "    [zotero]\n"
            "    api_key = \"YOUR_ZOTERO_KEY\"\n"
            "    library_id = \"YOUR_LIBRARY_ID\"",
        )
        # Cannot check individual keys without the file
        for section, key in _REQUIRED_KEYS + _OPTIONAL_KEYS:
            results[f"{section}.{key}"] = (False, "Secrets file not found")
        return results
    results["secrets_file"] = (True, f"Found: {_SECRETS_PATH}")
    # --- Parse the file ---
    try:
        with _SECRETS_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        results["secrets_parse"] = (False, f"Failed to parse TOML: {exc}")
        return results
    results["secrets_parse"] = (True, "Parsed OK")
    # --- Required keys ---
    for section, key in _REQUIRED_KEYS:
        val = data.get(section, {}).get(key, "")
        check_name = f"{section}.{key}"
        if val and str(val).strip():
            results[check_name] = (True, f"{section}.{key} is set")

        else:
            results[check_name] = (False, f"Missing or empty: [{section}] {key}")
    # --- Optional keys (warn but do not fail) ---
    for section, key in _OPTIONAL_KEYS:
        val = data.get(section, {}).get(key, "")
        check_name = f"{section}.{key}"
        if val and str(val).strip():
            results[check_name] = (True, f"{section}.{key} is set (optional)")
        else:
            results[check_name] = (True, f"{section}.{key} not set — database will be skipped")
    return results
