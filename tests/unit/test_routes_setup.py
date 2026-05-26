"""Unit tests for the helpers in scripts/server/routes/setup.py."""
from unittest.mock import patch

import pytest

from scripts.server.routes.setup import (
    _build_step_descriptors,
    _key_tail,
    _merge_credentials,
    _merge_paths,
    _validate_credentials,
    _validate_paths,
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
def test_build_step_descriptors_steps_4_through_8_are_todo():
    """Steps 4-8 haven't been wired yet — landing must show them as 'todo'.

    Steps 2 and 3 are exempt: F2.5 derives step 2's status from paths.yaml,
    and F2.6 derives step 3's status from extraction.yaml.
    """
    steps = _build_step_descriptors({
        "secrets_file": (True, ""),
        "secrets_parse": (True, ""),
        "zotero.api_key": (True, ""),
        "zotero.library_id": (True, ""),
        "pubmed.email": (True, ""),
    })
    for s in steps:
        if s["num"] not in (1, 2, 3):
            assert s["status"] == "todo", f"step {s['num']} should be 'todo' until F2.{s['num'] + 3} lands"


@pytest.mark.unit
def test_merge_paths_preserves_unrelated_top_level_keys():
    existing = {
        "zotero": {"library_id": "1"},
        "obsidian": {"vault_path": "/x"},
        "state_db": {"path": "~/state.db"},
        "logs": {"path": "./logs", "retention_days": 7},
        "weird_section": {"x": 1},
    }
    merged = _merge_paths(existing, {
        "vault_path": "/new",
        "collection_name": "C",
        "library_id": "2",
        "library_type": "user",
        "local_storage_path": "~/Zotero/storage",
    })
    assert merged["state_db"] == {"path": "~/state.db"}
    assert merged["logs"] == {"path": "./logs", "retention_days": 7}
    assert merged["weird_section"] == {"x": 1}


@pytest.mark.unit
def test_merge_paths_preserves_unrelated_obsidian_subkeys():
    existing = {
        "obsidian": {"vault_path": "/x", "papers_folder": "MyPapers", "custom_folder": "Foo"},
    }
    merged = _merge_paths(existing, {
        "vault_path": "/new",
        "collection_name": "C",
        "library_id": "1",
        "library_type": "user",
        "local_storage_path": "~/Zotero/storage",
    })
    assert merged["obsidian"]["papers_folder"] == "MyPapers"
    assert merged["obsidian"]["custom_folder"] == "Foo"
    assert merged["obsidian"]["vault_path"] == "/new"


@pytest.mark.unit
def test_merge_paths_backfills_obsidian_defaults_when_absent():
    """When existing obsidian has only vault_path, default sub-folders are added."""
    merged = _merge_paths({}, {
        "vault_path": "/new",
        "collection_name": "C",
        "library_id": "1",
        "library_type": "user",
        "local_storage_path": "~/Zotero/storage",
    })
    o = merged["obsidian"]
    assert o["papers_folder"] == "Literature/Papers"
    assert o["books_folder"] == "Literature/Books"
    assert o["digests_folder"] == "Literature/Digests"
    assert o["connections_folder"] == "Literature/Connections"


@pytest.mark.unit
def test_validate_paths_rejects_tampered_library_type():
    errors = _validate_paths({
        "obsidian": {"vault_path": "/tmp"},  # exists on every machine
        "zotero": {
            "library_id": "12345",
            "collection_name": "C",
            "library_type": "admin",   # not in {user, group}
        },
    })
    assert any("user" in e and "group" in e for e in errors)


@pytest.mark.unit
def test_validate_paths_rejects_nonexistent_vault(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    errors = _validate_paths({
        "obsidian": {"vault_path": str(nonexistent)},
        "zotero": {"library_id": "12345", "collection_name": "C", "library_type": "user"},
    })
    assert any("does not exist" in e for e in errors)


@pytest.mark.unit
def test_validate_paths_rejects_non_dir_vault(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    errors = _validate_paths({
        "obsidian": {"vault_path": str(f)},
        "zotero": {"library_id": "12345", "collection_name": "C", "library_type": "user"},
    })
    assert any("not a directory" in e for e in errors)


@pytest.mark.unit
def test_validate_paths_happy_path(tmp_path):
    errors = _validate_paths({
        "obsidian": {"vault_path": str(tmp_path)},
        "zotero": {"library_id": "12345", "collection_name": "C", "library_type": "user"},
    })
    assert errors == []


@pytest.mark.unit
def test_build_step_descriptors_step2_ok_when_vault_exists(tmp_path):
    """When paths.yaml has a vault_path pointing at a real dir, step 2 is 'ok'."""
    fake_paths = {"obsidian": {"vault_path": str(tmp_path)}}
    checks = {
        "secrets_file": (True, ""),
        "secrets_parse": (True, ""),
        "zotero.api_key": (True, ""),
        "zotero.library_id": (True, ""),
        "pubmed.email": (True, ""),
    }
    with patch("scripts.server.routes.setup.load_config", return_value=fake_paths):
        steps = _build_step_descriptors(checks)
    step2 = next(s for s in steps if s["num"] == 2)
    assert step2["status"] == "ok"


@pytest.mark.unit
def test_build_step_descriptors_step2_missing_when_paths_yaml_absent():
    """When load_config raises FileNotFoundError, step 2 is 'missing'."""
    checks = {
        "secrets_file": (True, ""),
        "secrets_parse": (True, ""),
        "zotero.api_key": (True, ""),
        "zotero.library_id": (True, ""),
        "pubmed.email": (True, ""),
    }
    with patch("scripts.server.routes.setup.load_config", side_effect=FileNotFoundError):
        steps = _build_step_descriptors(checks)
    step2 = next(s for s in steps if s["num"] == 2)
    assert step2["status"] == "missing"
