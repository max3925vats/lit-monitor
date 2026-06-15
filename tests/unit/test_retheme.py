"""
Unit tests for lit_monitor/obsidian_tools/retheme.py.

Covers:
- retheme() does not clobber an existing target page
- _rewrite_wikilinks() does not overmatch on prefix substrings
"""
from __future__ import annotations

import pytest

from lit_monitor.obsidian_tools.retheme import _rewrite_wikilinks, retheme


@pytest.mark.unit
def test_retheme_does_not_clobber_existing_target(tmp_path):
    (tmp_path / "Old.md").write_text("# Old\n", encoding="utf-8")
    (tmp_path / "New.md").write_text("# New — keep me\n", encoding="utf-8")
    retheme(str(tmp_path), "Old", "New")
    assert (tmp_path / "New.md").read_text(encoding="utf-8") == "# New — keep me\n"


@pytest.mark.unit
def test_rewrite_wikilinks_does_not_overmatch_prefix():
    out, n = _rewrite_wikilinks("[[topic2]] and [[topic]]", "topic", "subject")
    assert n == 1
    assert "[[topic2]]" in out
    assert "[[subject]]" in out
