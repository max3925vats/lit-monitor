"""Regression tests for persist-zone preservation.

Audit-29 #4: 5 of 6 malformed persist-marker cases were silently destroying
user content on rerender. These tests pin the new contract:

- Well-formed markers: zone content preserved across rerender (control).
- Malformed markers (missing endpersist, missing quote, empty name,
  unquoted name): default mode aborts rerender + logs WARNING, leaving the
  ORIGINAL file untouched. Strict mode raises RuntimeError instead.
- Nested zones are NOT flagged here; deferred (see _find_malformed_persist_markers
  docstring).
"""
from __future__ import annotations

import logging

import pytest

from lit_monitor.core.strict_mode import set_strict
from lit_monitor.output.obsidian_writer import (
    _find_malformed_persist_markers,
    update_note_preserve_persist_zones,
)


@pytest.fixture(autouse=True)
def _reset_strict():
    """Ensure each test starts in default (non-strict) mode."""
    set_strict(False)
    yield
    set_strict(False)


# ---------------------------------------------------------------------------
# _find_malformed_persist_markers — pure detection logic
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_clean_note_has_no_malformed_markers():
    text = (
        "# Note\n"
        '{% persist "rw" %}\n'
        "My links\n"
        "{% endpersist %}\n"
    )
    assert _find_malformed_persist_markers(text) == []


@pytest.mark.unit
def test_note_without_any_persist_tokens_is_fast_clean():
    text = "# Note\n\nJust prose, no Jinja tags here.\n"
    assert _find_malformed_persist_markers(text) == []


@pytest.mark.unit
def test_missing_endpersist_is_flagged():
    text = (
        "# Note\n"
        '{% persist "rw" %}\n'
        "My links\n"  # no {% endpersist %}
    )
    issues = _find_malformed_persist_markers(text)
    assert len(issues) >= 1
    assert "line 2" in issues[0]


@pytest.mark.unit
def test_missing_closing_quote_is_flagged():
    text = (
        "# Note\n"
        '{% persist "rw %}\n'  # no closing quote
        "My links\n"
        "{% endpersist %}\n"
    )
    issues = _find_malformed_persist_markers(text)
    assert len(issues) >= 1


@pytest.mark.unit
def test_empty_name_is_flagged():
    text = (
        "# Note\n"
        '{% persist "" %}\n'
        "My links\n"
        "{% endpersist %}\n"
    )
    issues = _find_malformed_persist_markers(text)
    assert len(issues) >= 1


@pytest.mark.unit
def test_unquoted_name_is_flagged():
    text = (
        "# Note\n"
        "{% persist rw %}\n"  # no quotes around the name
        "My links\n"
        "{% endpersist %}\n"
    )
    issues = _find_malformed_persist_markers(text)
    assert len(issues) >= 1


# ---------------------------------------------------------------------------
# update_note_preserve_persist_zones — default mode (abort + WARNING)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_well_formed_zone_content_preserved_across_rerender(tmp_path):
    """Control: the regular round-trip still works."""
    note = tmp_path / "n.md"
    note.write_text(
        '# Old\n{% persist "rw" %}\nUSER EDIT\n{% endpersist %}\n',
        encoding="utf-8",
    )
    new = '# New\n{% persist "rw" %}\n*(auto)*\n{% endpersist %}\n'
    update_note_preserve_persist_zones(note, new)
    result = note.read_text(encoding="utf-8")
    assert "# New" in result
    assert "USER EDIT" in result  # preserved
    assert "*(auto)*" not in result  # replaced by user content


@pytest.mark.unit
@pytest.mark.parametrize("existing", [
    '# X\n{% persist "rw" %}\nUSER EDIT\n',                       # no endpersist
    '# X\n{% persist "rw %}\nUSER EDIT\n{% endpersist %}\n',      # no closing quote
    '# X\n{% persist "" %}\nUSER EDIT\n{% endpersist %}\n',       # empty name
    '# X\n{% persist rw %}\nUSER EDIT\n{% endpersist %}\n',       # unquoted name
])
def test_malformed_marker_aborts_rerender_in_default_mode(tmp_path, caplog, existing):
    """Default mode: original file untouched + WARNING logged."""
    note = tmp_path / "n.md"
    note.write_text(existing, encoding="utf-8")
    new = '# REWRITTEN\n{% persist "rw" %}\n*(auto)*\n{% endpersist %}\n'

    with caplog.at_level(logging.WARNING, logger="lit_monitor.output.obsidian_writer"):
        update_note_preserve_persist_zones(note, new)

    # Original file untouched.
    assert note.read_text(encoding="utf-8") == existing
    assert "REWRITTEN" not in note.read_text(encoding="utf-8")
    # WARNING describes the problem.
    assert any("malformed" in r.message.lower() for r in caplog.records), \
        f"expected malformed-marker WARNING; got {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# update_note_preserve_persist_zones — strict mode (raise)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("existing", [
    '# X\n{% persist "rw" %}\nUSER\n',                            # no endpersist
    '# X\n{% persist "rw %}\nUSER\n{% endpersist %}\n',           # no closing quote
])
def test_malformed_marker_raises_in_strict_mode(tmp_path, existing):
    set_strict(True)
    note = tmp_path / "n.md"
    note.write_text(existing, encoding="utf-8")
    new = '# REWRITTEN\n{% persist "rw" %}\n*(auto)*\n{% endpersist %}\n'

    with pytest.raises(RuntimeError, match="malformed"):
        update_note_preserve_persist_zones(note, new)

    # Original file STILL untouched (strict raise happens before write).
    assert note.read_text(encoding="utf-8") == existing


# ---------------------------------------------------------------------------
# update_note_preserve_persist_zones — new-file path (no existing content)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_new_file_with_no_existing_content_writes_template_unchanged(tmp_path):
    """When the note doesn't exist yet, no malformed-marker check applies."""
    note = tmp_path / "fresh.md"
    new = '# Fresh\n{% persist "rw" %}\n*(auto)*\n{% endpersist %}\n'
    update_note_preserve_persist_zones(note, new)
    assert note.read_text(encoding="utf-8") == new
