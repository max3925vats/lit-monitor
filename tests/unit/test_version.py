"""Version assertion (closeout sentinel).

P11: confirms pyproject.toml is at the expected release version so a
stale bump doesn't silently slip through CI.
"""
from __future__ import annotations

import re
from pathlib import Path


def test_version_matches_release() -> None:
    """Closeout sentinel: pyproject.toml is at the expected release version so a
    stale bump doesn't silently slip through CI.

    0.10.0 marks the complete web UI (corpus/graph/insights/ask/nav) +
    active-learning observability, on top of the 0.9.0 engine.
    """
    raw = Path("pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', raw, re.MULTILINE)
    assert match is not None, "version field not found in pyproject.toml"
    assert match.group(1) == "0.10.0", f"expected 0.10.0, got {match.group(1)}"
