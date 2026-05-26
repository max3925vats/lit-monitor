"""Unit tests for the helpers in scripts/server/routes/setup.py."""
import pytest

from scripts.server.routes.setup import (
    _build_step_descriptors,
    _key_tail,
    _merge_credentials,
    _validate_credentials,
)


@pytest.mark.unit
def test_key_tail_empty():
    assert _key_tail("") == ""


@pytest.mark.unit
def test_key_tail_masks_all_but_last_four():
    out = _key_tail("AbCdEfGhIjKlMnOp")
    assert out.endswith("MnOp")
    assert out.startswith("•")
    assert "AbCdEf" not in out


@pytest.mark.unit
def test_merge_credentials_empty_form_keeps_existing():
    existing = {"zotero": {"api_key": "real", "library_id": "123"}, "pubmed": {"email": "x@y"}}
    merged = _merge_credentials(existing, {
        "zotero_api_key": "",
        "zotero_library_id": "",
        "pubmed_email": "",
        "ollama_api_key": "",
    })
    assert merged["zotero"]["api_key"] == "real"
    assert merged["zotero"]["library_id"] == "123"
    assert merged["pubmed"]["email"] == "x@y"


@pytest.mark.unit
def test_merge_credentials_masked_input_keeps_existing():
    """A submitted form value starting with the masked-prefix must NOT overwrite."""
    existing = {"zotero": {"api_key": "real-key-1234"}}
    merged = _merge_credentials(existing, {
        "zotero_api_key": "••••••••1234",  # what the form pre-fills
        "zotero_library_id": "",
        "pubmed_email": "",
        "ollama_api_key": "",
    })
    assert merged["zotero"]["api_key"] == "real-key-1234"


@pytest.mark.unit
def test_merge_credentials_real_input_replaces():
    existing = {"zotero": {"api_key": "old"}}
    merged = _merge_credentials(existing, {
        "zotero_api_key": "NEW_KEY_HERE",
        "zotero_library_id": "",
        "pubmed_email": "",
        "ollama_api_key": "",
    })
    assert merged["zotero"]["api_key"] == "NEW_KEY_HERE"


@pytest.mark.unit
def test_merge_credentials_preserves_unrelated_sections():
    """A [scopus] or other section in the existing TOML must survive the merge."""
    existing = {
        "zotero": {"api_key": "z"},
        "scopus": {"api_key": "s"},
        "weird": {"x": 1},
    }
    merged = _merge_credentials(existing, {
        "zotero_api_key": "",
        "zotero_library_id": "",
        "pubmed_email": "",
        "ollama_api_key": "",
    })
    assert merged["scopus"] == {"api_key": "s"}
    assert merged["weird"] == {"x": 1}


@pytest.mark.unit
def test_validate_credentials_all_missing():
    errors = _validate_credentials({})
    assert any("Zotero API key" in e for e in errors)
    assert any("library ID" in e for e in errors)
    assert any("PubMed email" in e for e in errors)


@pytest.mark.unit
def test_validate_credentials_non_numeric_library_id():
    errors = _validate_credentials({
        "zotero": {"api_key": "k", "library_id": "abc"},
        "pubmed": {"email": "x@y"},
    })
    assert any("numeric" in e for e in errors)


@pytest.mark.unit
def test_validate_credentials_happy_path():
    errors = _validate_credentials({
        "zotero": {"api_key": "k", "library_id": "12345"},
        "pubmed": {"email": "x@y"},
    })
    assert errors == []


@pytest.mark.unit
def test_build_step_descriptors_all_ok_marks_step1_ok():
    checks = {
        "secrets_file": (True, ""),
        "secrets_parse": (True, ""),
        "zotero.api_key": (True, ""),
        "zotero.library_id": (True, ""),
        "pubmed.email": (True, ""),
    }
    steps = _build_step_descriptors(checks)
    step1 = next(s for s in steps if s["num"] == 1)
    assert step1["status"] == "ok"


@pytest.mark.unit
def test_build_step_descriptors_any_missing_marks_step1_missing():
    checks = {
        "secrets_file": (True, ""),
        "secrets_parse": (True, ""),
        "zotero.api_key": (False, "missing"),  # one failure
        "zotero.library_id": (True, ""),
        "pubmed.email": (True, ""),
    }
    steps = _build_step_descriptors(checks)
    step1 = next(s for s in steps if s["num"] == 1)
    assert step1["status"] == "missing"


@pytest.mark.unit
def test_build_step_descriptors_steps_3_through_8_are_todo():
    """Steps 3-8 haven't been wired yet — landing must show them as 'todo'.

    Step 2 is exempt: F2.5 derives its status from paths.yaml (see
    ``test_build_step_descriptors_step2_status_*`` below).
    """
    steps = _build_step_descriptors({
        "secrets_file": (True, ""),
        "secrets_parse": (True, ""),
        "zotero.api_key": (True, ""),
        "zotero.library_id": (True, ""),
        "pubmed.email": (True, ""),
    })
    for s in steps:
        if s["num"] not in (1, 2):
            assert s["status"] == "todo", f"step {s['num']} should be 'todo' until F2.{s['num'] + 3} lands"
