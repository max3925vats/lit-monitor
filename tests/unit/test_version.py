"""Version assertion (closeout sentinel).

P11: confirms pyproject.toml is at the expected release version so a
stale bump doesn't silently slip through CI.
"""
from __future__ import annotations

import re
from pathlib import Path


def test_version_is_0_8_0() -> None:
    """P11 / Bundle I: pyproject.toml version is 0.9.0 (bumped from 0.8.0 in Bundle I)."""
    raw = Path("pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', raw, re.MULTILINE)
    assert match is not None, "version field not found in pyproject.toml"
    assert match.group(1) == "0.9.0", f"expected 0.9.0, got {match.group(1)}"
