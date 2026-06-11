"""
Check that the Zotero API key and library ID are valid by making a test API call.
Returns a dict of {check_name: (ok: bool, message: str)}.
"""
from __future__ import annotations

import tomllib

import requests

from scripts.setup._paths import SECRETS_PATH as _SECRETS_PATH

_ZOTERO_BASE = "https://api.zotero.org"
_TIMEOUT = 10
def check_zotero() -> dict[str, tuple[bool, str]]:
    """
    Return {check_name: (ok, message)} for each Zotero check.
    Does not raise — always returns results.
    """
    results: dict[str, tuple[bool, str]] = {}
    # --- Load secrets ---
    if not _SECRETS_PATH.exists():
        results["zotero_credentials"] = (
            False,
            f"Secrets file not found: {_SECRETS_PATH}",
        )
        return results
    try:
        with _SECRETS_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        results["zotero_credentials"] = (False, f"Failed to parse secrets: {exc}")
        return results
    api_key = data.get("zotero", {}).get("api_key", "")
    library_id = data.get("zotero", {}).get("library_id", "")
    library_type = data.get("zotero", {}).get("library_type", "user")
    if not api_key or not library_id:
        results["zotero_credentials"] = (
            False,
            "Missing zotero.api_key or zotero.library_id in secrets file",
        )
        return results
    results["zotero_credentials"] = (True, "Credentials found in secrets file")
    # --- Test API call ---
    if library_type == "group":
        url = f"{_ZOTERO_BASE}/groups/{library_id}/items?limit=1"
    else:
        url = f"{_ZOTERO_BASE}/users/{library_id}/items?limit=1"
    try:
        resp = requests.get(
            url,
            headers={"Zotero-API-Key": api_key, "Zotero-API-Version": "3"},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            results["zotero_api"] = (

                True,
                f"Zotero API reachable — library {library_type}/{library_id} accessible",
            )
        elif resp.status_code == 403:
            results["zotero_api"] = (
                False,
                "Zotero API key rejected (HTTP 403) — check your api_key",
            )
        elif resp.status_code == 404:
            results["zotero_api"] = (
                False,
                f"Library {library_type}/{library_id} not found (HTTP 404) — "
                f"check your library_id and library_type",
            )
        else:
            results["zotero_api"] = (
                False,
                f"Zotero API returned HTTP {resp.status_code}",
            )
    except requests.exceptions.ConnectionError:
        results["zotero_api"] = (
            False,
            "Cannot reach api.zotero.org — check internet connection",
        )
    except Exception as exc:
        results["zotero_api"] = (False, f"Zotero API check failed: {exc}")
    return results
