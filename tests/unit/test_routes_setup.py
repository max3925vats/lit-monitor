"""Unit tests for the helpers in scripts/server/routes/setup.py."""
from unittest.mock import patch

import pytest

from scripts.server.routes.setup import (
    _build_step_descriptors,
    _key_tail,
    _merge_credentials,
    _merge_extraction,
    _merge_paths,
    _mode_view,
    _validate_credentials,
    _validate_extraction,
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
def test_build_step_descriptors_no_todo_placeholders():
    """After F2.12 closes the wizard, every step has a disk-derived status.

    Every step's status must be one of {'ok', 'missing'} — never 'todo'.
    """
    steps = _build_step_descriptors({
        "secrets_file": (True, ""),
        "secrets_parse": (True, ""),
        "zotero.api_key": (True, ""),
        "zotero.library_id": (True, ""),
        "pubmed.email": (True, ""),
    })
    for s in steps:
        assert s["status"] in ("ok", "missing"), (
            f"step {s['num']} has placeholder status {s['status']!r}"
        )


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


@pytest.mark.unit
def test_mode_view_uses_existing_values_when_present():
    existing = {"brain_build": {"model": "qwen2.5:7b", "ollama_host": "https://ollama.com",
                                "temperature": 0.3, "think": True}}
    view = _mode_view(existing, "brain_build")
    assert view["model"] == "qwen2.5:7b"
    assert view["ollama_host"] == "https://ollama.com"
    assert view["temperature"] == 0.3
    assert view["think"] is True


@pytest.mark.unit
def test_mode_view_falls_back_to_defaults_when_absent():
    view = _mode_view({}, "brain_build")
    assert view["provider"] == "ollama"
    assert view["temperature"] == 0.1
    assert view["timeout"] == 7200
    assert view["think"] is False


@pytest.mark.unit
def test_merge_extraction_preserves_non_wizard_keys_in_mode():
    """pass_strategy, max_tokens_per_call, etc. must survive a wizard save."""
    existing = {
        "brain_build": {
            "model": "old",
            "pass_strategy": "all",
            "max_tokens_per_call": 24576,
            "num_ctx_override": 131072,
            "chunk_chars": None,
        }
    }
    form = {mode: _mode_view({}, mode) | {"model": "new-model"}
            for mode in ("brain_build", "ingestion", "build_vocabulary")}
    merged = _merge_extraction(existing, form)
    bb = merged["brain_build"]
    assert bb["model"] == "new-model"
    assert bb["pass_strategy"] == "all"
    assert bb["max_tokens_per_call"] == 24576
    assert bb["num_ctx_override"] == 131072
    assert bb["chunk_chars"] is None


@pytest.mark.unit
def test_merge_extraction_preserves_non_wizard_top_level_sections():
    """embeddings, reranker, comparison_models must round-trip untouched."""
    existing = {
        "brain_build": {"model": "old"},
        "embeddings": {"model": "mxbai-embed-large"},
        "reranker": {"enabled": True, "device": "mps"},
        "comparison_models": [{"provider": "ollama", "model": "x"}],
    }
    form = {mode: _mode_view({}, mode) | {"model": "new"}
            for mode in ("brain_build", "ingestion", "build_vocabulary")}
    merged = _merge_extraction(existing, form)
    assert merged["embeddings"] == {"model": "mxbai-embed-large"}
    assert merged["reranker"] == {"enabled": True, "device": "mps"}
    assert merged["comparison_models"] == [{"provider": "ollama", "model": "x"}]


@pytest.mark.unit
def test_merge_extraction_drops_litellm_model_on_provider_switchback():
    """If user switches litellm→ollama, the stale litellm_model is removed."""
    existing = {"brain_build": {"model": "x", "provider": "litellm",
                                "litellm_model": "anthropic/claude-haiku-4-5"}}
    form_data = _mode_view(existing, "brain_build") | {"provider": "ollama", "litellm_model": ""}
    form = {"brain_build": form_data,
            "ingestion": _mode_view({}, "ingestion") | {"model": "x"},
            "build_vocabulary": _mode_view({}, "build_vocabulary") | {"model": "x"}}
    merged = _merge_extraction(existing, form)
    assert "litellm_model" not in merged["brain_build"]


@pytest.mark.unit
def test_validate_extraction_rejects_bad_provider():
    bad = {mode: _mode_view({}, mode) | {"model": "x", "provider": "openai"}
           for mode in ("brain_build", "ingestion", "build_vocabulary")}
    errors = _validate_extraction(bad)
    assert any("ollama" in e and "litellm" in e for e in errors)


@pytest.mark.unit
def test_validate_extraction_rejects_empty_model():
    bad = {mode: _mode_view({}, mode) for mode in ("brain_build", "ingestion", "build_vocabulary")}
    errors = _validate_extraction(bad)
    # All three modes complain.
    assert sum(1 for e in errors if "model is required" in e) == 3


@pytest.mark.unit
def test_validate_extraction_rejects_temperature_outside_range():
    bad = {mode: _mode_view({}, mode) | {"model": "x", "temperature": 1.5}
           for mode in ("brain_build", "ingestion", "build_vocabulary")}
    errors = _validate_extraction(bad)
    assert any("temperature must be in" in e for e in errors)


@pytest.mark.unit
def test_validate_extraction_rejects_zero_timeout():
    bad = {mode: _mode_view({}, mode) | {"model": "x", "timeout": 0}
           for mode in ("brain_build", "ingestion", "build_vocabulary")}
    errors = _validate_extraction(bad)
    assert any("timeout must be" in e for e in errors)


@pytest.mark.unit
def test_validate_extraction_rejects_litellm_without_model():
    bad = {mode: _mode_view({}, mode) | {"model": "x", "provider": "litellm", "litellm_model": ""}
           for mode in ("brain_build", "ingestion", "build_vocabulary")}
    errors = _validate_extraction(bad)
    assert any("litellm_model is required" in e for e in errors)


@pytest.mark.unit
def test_validate_extraction_happy_path():
    good = {mode: _mode_view({}, mode) | {"model": "gemma4:31b-cloud"}
            for mode in ("brain_build", "ingestion", "build_vocabulary")}
    assert _validate_extraction(good) == []


# --- F3.4: collection switcher --------------------------------------------


@pytest.mark.unit
def test_update_collection_writes_back():
    """POST /setup/api/paths/collection rewrites only the collection_name key."""
    from fastapi.testclient import TestClient

    from scripts.server.app import create_app
    from scripts.server.runtime import reset_runtime

    reset_runtime()
    client = TestClient(create_app())

    fake_paths = {
        "zotero": {"library_id": "1", "library_type": "user", "collection_name": "OLD"},
        "obsidian": {"vault_path": "/tmp", "papers_folder": "P"},
        "state_db": {"path": "x"},
    }
    saved_with: dict = {}

    def fake_save(name, data):
        saved_with["name"] = name
        saved_with["data"] = data

    with patch("scripts.server.routes.setup.load_config", return_value=fake_paths), \
         patch("scripts.server.routes.setup.save_config", side_effect=fake_save):
        resp = client.post(
            "/setup/api/paths/collection", data={"collection_name": "NEW"}
        )

    assert resp.status_code == 200
    assert resp.headers.get("HX-Refresh") == "true"
    assert saved_with["name"] == "paths"
    assert saved_with["data"]["zotero"]["collection_name"] == "NEW"
    # Other keys must round-trip untouched.
    assert saved_with["data"]["zotero"]["library_id"] == "1"
    assert saved_with["data"]["obsidian"] == {"vault_path": "/tmp", "papers_folder": "P"}
    assert saved_with["data"]["state_db"] == {"path": "x"}


@pytest.mark.unit
def test_update_collection_rejects_empty():
    from fastapi.testclient import TestClient

    from scripts.server.app import create_app
    from scripts.server.runtime import reset_runtime

    reset_runtime()
    client = TestClient(create_app())
    resp = client.post(
        "/setup/api/paths/collection", data={"collection_name": "  "}
    )
    assert resp.status_code == 400


@pytest.mark.unit
def test_update_collection_returns_400_when_paths_missing():
    from fastapi.testclient import TestClient

    from scripts.server.app import create_app
    from scripts.server.runtime import reset_runtime

    reset_runtime()
    client = TestClient(create_app())
    with patch(
        "scripts.server.routes.setup.load_config", side_effect=FileNotFoundError
    ):
        resp = client.post(
            "/setup/api/paths/collection", data={"collection_name": "X"}
        )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# P4: safe_save_preference — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSafeSavePreference:
    """P4: atomic preference write via safe_save_preference."""

    def test_writes_preferred_viewer_and_asked_user(self, tmp_path):
        import shutil
        from pathlib import Path

        import yaml

        from scripts.server.config_io import safe_save_preference

        src = Path("config/extraction.example.yaml")
        dst = tmp_path / "extraction.yaml"
        shutil.copy(src, dst)
        safe_save_preference("browser", config_path=dst)
        data = yaml.safe_load(dst.read_text())
        assert data["discovery"]["notify"]["preferred_viewer"] == "browser"
        assert data["discovery"]["notify"]["asked_user"] is True

    def test_enabled_optional_kwarg(self, tmp_path):
        import shutil

        import yaml

        from scripts.server.config_io import safe_save_preference

        dst = tmp_path / "extraction.yaml"
        shutil.copy("config/extraction.example.yaml", dst)
        safe_save_preference("obsidian", enabled=False, config_path=dst)
        data = yaml.safe_load(dst.read_text())
        assert data["discovery"]["notify"]["enabled"] is False
        assert data["discovery"]["notify"]["preferred_viewer"] == "obsidian"

    def test_enabled_true_kwarg(self, tmp_path):
        import shutil

        import yaml

        from scripts.server.config_io import safe_save_preference

        dst = tmp_path / "extraction.yaml"
        shutil.copy("config/extraction.example.yaml", dst)
        safe_save_preference("none", enabled=True, config_path=dst)
        data = yaml.safe_load(dst.read_text())
        assert data["discovery"]["notify"]["enabled"] is True

    def test_invalid_viewer_raises(self, tmp_path):
        import shutil

        from scripts.server.config_io import safe_save_preference

        dst = tmp_path / "extraction.yaml"
        shutil.copy("config/extraction.example.yaml", dst)
        with pytest.raises(ValueError):
            safe_save_preference("bogus", config_path=dst)

    def test_atomic_write_no_corruption(self, tmp_path):
        """P4: a successful write leaves the file as valid YAML with other keys intact."""
        import shutil

        import yaml

        from scripts.server.config_io import safe_save_preference

        dst = tmp_path / "extraction.yaml"
        shutil.copy("config/extraction.example.yaml", dst)
        before = yaml.safe_load(dst.read_text())
        safe_save_preference("none", config_path=dst)
        after = yaml.safe_load(dst.read_text())
        # Non-notify top-level keys preserved
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        assert before_keys.issubset(after_keys), (
            f"missing keys after write: {before_keys - after_keys}"
        )

    def test_does_not_leave_tmp_file(self, tmp_path):
        import shutil

        from scripts.server.config_io import safe_save_preference

        dst = tmp_path / "extraction.yaml"
        shutil.copy("config/extraction.example.yaml", dst)
        safe_save_preference("browser", config_path=dst)
        # No .tmp files left behind
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == [], f"tmp files left: {leftover}"

    def test_none_viewer_valid(self, tmp_path):
        """'none' is a valid viewer value (user opts out of viewer preference)."""
        import shutil

        import yaml

        from scripts.server.config_io import safe_save_preference

        dst = tmp_path / "extraction.yaml"
        shutil.copy("config/extraction.example.yaml", dst)
        safe_save_preference("none", config_path=dst)
        data = yaml.safe_load(dst.read_text())
        assert data["discovery"]["notify"]["preferred_viewer"] == "none"


# ---------------------------------------------------------------------------
# P4: setup wizard /complete notify panel — integration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetupCompleteNotifyPanel:
    """P4: GET /setup/complete renders the notify panel; POST persists the choice."""

    def _make_client(self):
        from fastapi.testclient import TestClient

        from scripts.server.app import create_app
        from scripts.server.runtime import reset_runtime

        reset_runtime()
        return TestClient(create_app())

    def test_panel_renders(self):
        """P4: GET /setup/complete includes the notification-preferences panel."""
        client = self._make_client()
        r = client.get("/setup/complete")
        assert r.status_code == 200
        body = r.text.lower()
        assert "notification" in body or "notify" in body
        # 3 radio options present
        for opt in ("browser", "obsidian", "none"):
            assert opt in body

    def test_panel_reflects_current_config(self):
        """P4: GET /setup/complete reflects the current extraction.yaml values."""
        client = self._make_client()
        fake_extraction = {
            "discovery": {
                "notify": {
                    "enabled": True,
                    "preferred_viewer": "obsidian",
                    "asked_user": True,
                }
            }
        }
        with patch("scripts.server.routes.setup.load_config", return_value=fake_extraction):
            r = client.get("/setup/complete")
        assert r.status_code == 200
        # The selected viewer value should appear somewhere in the rendered page
        assert "obsidian" in r.text.lower()

    def test_post_persists_choice(self):
        """P4: POST /setup/complete/notify calls safe_save_preference."""
        client = self._make_client()
        with patch("scripts.server.routes.setup.safe_save_preference") as m:
            r = client.post(
                "/setup/complete/notify",
                data={"viewer": "browser", "enabled": "on"},
            )
        assert r.status_code in (200, 303, 302), r.text
        m.assert_called_once()
        call_args = m.call_args.args
        call_kwargs = m.call_args.kwargs
        # viewer must be first positional arg or explicit kwarg
        viewer_sent = call_args[0] if call_args else call_kwargs.get("viewer")
        assert viewer_sent == "browser"

    def test_post_without_enabled_flag(self):
        """P4: POST /setup/complete/notify without enabled=on sets enabled=False."""
        client = self._make_client()
        with patch("scripts.server.routes.setup.safe_save_preference") as m:
            r = client.post(
                "/setup/complete/notify",
                data={"viewer": "obsidian"},
            )
        assert r.status_code in (200, 303, 302), r.text
        m.assert_called_once()
        call_kwargs = m.call_args.kwargs
        assert call_kwargs.get("enabled") is False

    def test_post_invalid_viewer_returns_400(self):
        """P4: POST /setup/complete/notify with invalid viewer returns 400."""
        client = self._make_client()
        r = client.post(
            "/setup/complete/notify",
            data={"viewer": "bogus"},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# P10b: digest auto-write checkbox
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDigestAutoWriteCheckbox:
    """P10b: safe_save_digest_auto_write + wizard checkbox round-trip."""

    def _make_client(self):
        from fastapi.testclient import TestClient

        from scripts.server.app import create_app
        from scripts.server.runtime import reset_runtime

        reset_runtime()
        return TestClient(create_app())

    def test_safe_save_digest_auto_write_writes_yaml(self, tmp_path):
        """P10b: safe_save_digest_auto_write toggles the flag and round-trips."""
        import shutil

        import yaml

        from scripts.server.config_io import safe_save_digest_auto_write

        dst = tmp_path / "extraction.yaml"
        shutil.copy("config/extraction.example.yaml", dst)

        safe_save_digest_auto_write(False, config_path=dst)
        data = yaml.safe_load(dst.read_text())
        assert data["discovery"]["digest"]["auto_write"] is False

        # Round-trip back to True
        safe_save_digest_auto_write(True, config_path=dst)
        data = yaml.safe_load(dst.read_text())
        assert data["discovery"]["digest"]["auto_write"] is True

    def test_complete_page_renders_checkbox(self):
        """P10b: GET /setup/complete includes the digest auto-write checkbox."""
        client = self._make_client()
        r = client.get("/setup/complete")
        assert r.status_code == 200
        body = r.text.lower()
        assert "digest" in body
        # Must contain some variant of the flag name
        assert "auto-write" in body or "auto_write" in body or "auto write" in body

    def test_post_persists_checkbox_value(self):
        """P10b: POST /setup/complete/notify persists digest_auto_write=True when 'on'."""
        client = self._make_client()
        with patch("scripts.server.routes.setup.safe_save_digest_auto_write") as m, \
             patch("scripts.server.routes.setup.safe_save_preference"):
            r = client.post(
                "/setup/complete/notify",
                data={"viewer": "browser", "enabled": "on", "digest_auto_write": "on"},
            )
        assert r.status_code in (200, 303, 302), r.text
        m.assert_called_once()
        # Confirm the value passed is truthy True
        called_val = (
            m.call_args.args[0] if m.call_args.args
            else m.call_args.kwargs.get("value")
        )
        assert called_val is True

    def test_post_without_digest_flag_passes_false(self):
        """P10b: POST without digest_auto_write field passes False (unchecked checkbox)."""
        client = self._make_client()
        with patch("scripts.server.routes.setup.safe_save_digest_auto_write") as m, \
             patch("scripts.server.routes.setup.safe_save_preference"):
            r = client.post(
                "/setup/complete/notify",
                data={"viewer": "browser"},
            )
        assert r.status_code in (200, 303, 302), r.text
        m.assert_called_once()
        called_val = (
            m.call_args.args[0] if m.call_args.args
            else m.call_args.kwargs.get("value")
        )
        assert called_val is False
