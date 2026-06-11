"""
Check that the user's secrets config (~/.config/lit-monitor/config.toml) exists
and contains the minimum required keys.
Returns a dict of {check_name: CheckResult(ok, message, severity)}.

CheckResult is a NamedTuple, so existing call sites that do
``for name, (ok, msg) in results.items():`` keep working — tuple unpacking
takes only the first two fields. New severity-aware callers can read
``result.severity`` to render a 3-state (ok/warn/fail) UI.
"""
from __future__ import annotations

import tomllib
from typing import Literal, NamedTuple

from lit_monitor.setup._paths import SECRETS_PATH as _SECRETS_PATH


class CheckResult(NamedTuple):
    """Result of a single config check.

    ``ok`` is True for "acceptable" outcomes — that includes both genuinely-
    configured keys AND optional-key-absent (we still let the run proceed,
    just without that source). ``severity`` distinguishes the two so the UI
    can render a yellow ⚠ for the optional-absent case instead of a green ✓.

    Backward-compat: ``severity`` defaults to "auto" which means callers
    that ignore it should infer from ``ok`` (True→ok, False→fail). New
    callers (health badge aggregator, dev page) should read severity.
    """

    ok: bool
    message: str
    severity: Literal["ok", "warn", "fail", "auto"] = "auto"


_REQUIRED_KEYS: list[tuple[str, str]] = [
    # (section, key)
    ("zotero", "api_key"),
    ("zotero", "library_id"),
    ("pubmed", "email"),   # required for Unpaywall OA PDF enrichment
]
_OPTIONAL_KEYS: list[tuple[str, str]] = [
    ("scopus", "api_key"),
]


def check_configured() -> dict[str, CheckResult]:
    """
    Return {check_name: CheckResult} for each configuration check.
    Does not raise — always returns results.
    """
    results: dict[str, CheckResult] = {}
    # --- Secrets file exists ---
    if not _SECRETS_PATH.exists():
        results["secrets_file"] = CheckResult(
            False,
            f"Missing: {_SECRETS_PATH}\n"
            "  Create it with at minimum:\n"
            "    [zotero]\n"
            "    api_key = \"YOUR_ZOTERO_KEY\"\n"
            "    library_id = \"YOUR_LIBRARY_ID\"",
            severity="fail",
        )
        # Cannot check individual keys without the file
        for section, key in _REQUIRED_KEYS + _OPTIONAL_KEYS:
            results[f"{section}.{key}"] = CheckResult(
                False, "Secrets file not found", severity="fail"
            )
        return results
    results["secrets_file"] = CheckResult(
        True, f"Found: {_SECRETS_PATH}", severity="ok"
    )
    # --- Parse the file ---
    try:
        with _SECRETS_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        results["secrets_parse"] = CheckResult(
            False, f"Failed to parse TOML: {exc}", severity="fail"
        )
        return results
    results["secrets_parse"] = CheckResult(True, "Parsed OK", severity="ok")
    # --- Required keys ---
    for section, key in _REQUIRED_KEYS:
        val = data.get(section, {}).get(key, "")
        check_name = f"{section}.{key}"
        if val and str(val).strip():
            results[check_name] = CheckResult(
                True, f"{section}.{key} is set", severity="ok"
            )
        else:
            results[check_name] = CheckResult(
                False, f"Missing or empty: [{section}] {key}", severity="fail"
            )
    # --- Optional keys: absent → warn (not fail) so the rollup turns yellow,
    # not green. The boolean stays True because the run is still allowed to
    # proceed without this source.
    for section, key in _OPTIONAL_KEYS:
        val = data.get(section, {}).get(key, "")
        check_name = f"{section}.{key}"
        if val and str(val).strip():
            results[check_name] = CheckResult(
                True, f"{section}.{key} is set (optional)", severity="ok"
            )
        else:
            results[check_name] = CheckResult(
                True,
                f"{section}.{key} not set — database will be skipped",
                severity="warn",
            )
    return results
