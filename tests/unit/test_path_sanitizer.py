"""Unit tests for ``sanitize_filename_component`` (path-injection hardening).

Note paths in this codebase are built from extracted/user-supplied data (paper
titles, DOIs, synthesis topics). ``sanitize_filename_component`` reduces such a
value to a benign single path segment so a crafted title can't traverse out of
the vault/papers directory (CodeQL py/path-injection).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lit_monitor.core.path_utils import sanitize_filename_component


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "../../etc/passwd",
        "foo/bar",
        "..\\..\\windows",
        "a\x00b",
        "name\nwith\ncontrol",
        "..",
        ".",
        "   ",
        "...",
        "/leading-slash",
        "trailing/",
    ],
)
def test_no_separator_nul_or_traversal_survives(raw: str) -> None:
    """Result is always a safe single segment — no separator, NUL, or dot-name."""
    result = sanitize_filename_component(raw)
    assert "/" not in result
    assert "\\" not in result
    assert "\x00" not in result
    assert result not in ("", ".", "..")
    # Joining onto a base dir must not escape it.
    base = Path("/tmp/vault/papers")
    joined = (base / f"{result}.md").resolve()
    assert str(joined).startswith(str(base.resolve()) + "/")


@pytest.mark.unit
def test_legitimate_names_preserved() -> None:
    """Ordinary titles and DOIs survive (internal dots kept; only seps replaced)."""
    assert sanitize_filename_component("Smith2024_GraphRAG") == "Smith2024_GraphRAG"
    # DOI: the '/' becomes '_', internal dots are kept.
    assert sanitize_filename_component("10.1234/jacs.5b00") == "10.1234_jacs.5b00"


@pytest.mark.unit
def test_empty_falls_back() -> None:
    assert sanitize_filename_component("", fallback="x") == "x"
    # A pure-dot input collapses to nothing usable → fallback.
    assert sanitize_filename_component("..", fallback="x") == "x"


@pytest.mark.unit
def test_separators_become_underscores_not_fallback() -> None:
    """Separators are replaced (not dropped), so "///" is a safe '___' leaf."""
    result = sanitize_filename_component("///", fallback="x")
    assert result == "___"
    assert "/" not in result


@pytest.mark.unit
def test_max_length_enforced() -> None:
    result = sanitize_filename_component("a" * 500, max_length=50)
    assert len(result) == 50


@pytest.mark.unit
def test_atomic_write_rejects_nul(tmp_path: Path) -> None:
    """atomic_write_text rejects an embedded NUL in the path (defence in depth)."""
    from lit_monitor.core.atomic_write import atomic_write_text

    with pytest.raises(ValueError, match="NUL"):
        atomic_write_text(tmp_path / "bad\x00name.md", "content")


@pytest.mark.unit
def test_atomic_write_still_writes_legitimate_absolute_path(tmp_path: Path) -> None:
    """Legitimate absolute paths keep working unchanged."""
    from lit_monitor.core.atomic_write import atomic_write_text

    target = tmp_path / "note.md"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"
