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

    0.11.0 marks the pip-installable PyPI packaging milestone (vendored
    findpapers, clean-room-verified wheel, trusted-publishing workflow) on top
    of the 0.10.0 web UI + active-learning observability.
    """
    raw = Path("pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', raw, re.MULTILINE)
    assert match is not None, "version field not found in pyproject.toml"
    assert match.group(1) == "0.11.0", f"expected 0.11.0, got {match.group(1)}"
