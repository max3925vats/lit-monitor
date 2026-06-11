"""Bundle H: HTTP endpoint + HTML page tests for /settings and /api/settings/*.

Covers:
- GET /settings — HTML page renders
- POST /api/settings/ranking — saves section to extraction.yaml
- POST /api/settings/web_ui — saves section
- POST /api/settings/unknown_section — 400
- POST /api/settings/ranking (bad JSON body) — 422
- safe_save_settings_section: atomic write confirmed (tempfile + os.replace)
- safe_save_settings_section: unknown section raises ValueError
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from lit_monitor.server.routes.settings import router as settings_router
from lit_monitor.server.runtime import reset_runtime

TEMPLATES_DIR = Path(__file__).parents[2] / "lit_monitor" / "server" / "templates"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture(autouse=True)
def restore_app_templates():
    """Restore scripts.server.app.templates after each test.

    _make_client() reassigns app.templates to a test-local Jinja2Templates
    instance; without restoration that object leaks into sibling tests run later
    in the same session (order-dependent flakiness).
    """
    import lit_monitor.server.app as app_mod
    original = app_mod.templates
    yield
    app_mod.templates = original


def _make_client(config_path=None) -> TestClient:
    import json as _json

    import lit_monitor.server.app as app_mod

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["fromjson"] = _json.loads
    app_mod.templates = templates

    app = FastAPI()
    app.include_router(settings_router)
    app.state.dev_mode = False
    app.state.version = "test"
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /settings HTML
# ---------------------------------------------------------------------------

class TestSettingsPage:
    def test_renders_200(self):
        with patch(
            "lit_monitor.server.routes.settings._load_extraction_config",
            return_value={"ranking": {"semantic_weight": 0.5}},
        ):
            client = _make_client()
            r = client.get("/settings")
        assert r.status_code == 200
        assert "Advanced Settings" in r.text

    def test_reflects_current_config(self):
        import re

        with patch(
            "lit_monitor.server.routes.settings._load_extraction_config",
            return_value={
                "web_ui": {"show_feedback_buttons": True},
            },
        ):
            client = _make_client()
            r = client.get("/settings")
        assert r.status_code == 200
        # The SPECIFIC show_feedback_buttons checkbox must carry `checked` when the
        # config value is True — a bare `"checked" in r.text` would also match any
        # other checked control on the page, so pin the input by name.
        feedback_input = re.search(
            r'<input[^>]*name="show_feedback_buttons"[^>]*>', r.text, re.DOTALL
        )
        assert feedback_input is not None, "show_feedback_buttons checkbox not rendered"
        assert "checked" in feedback_input.group(0), (
            "show_feedback_buttons checkbox must be `checked` when config is True"
        )

    def test_checkbox_unchecked_when_config_false(self):
        """Inverse of test_reflects_current_config: False config → no `checked`."""
        import re

        with patch(
            "lit_monitor.server.routes.settings._load_extraction_config",
            return_value={
                "web_ui": {"show_feedback_buttons": False},
            },
        ):
            client = _make_client()
            r = client.get("/settings")
        assert r.status_code == 200
        feedback_input = re.search(
            r'<input[^>]*name="show_feedback_buttons"[^>]*>', r.text, re.DOTALL
        )
        assert feedback_input is not None, "show_feedback_buttons checkbox not rendered"
        assert "checked" not in feedback_input.group(0), (
            "show_feedback_buttons checkbox must NOT be `checked` when config is False"
        )


# ---------------------------------------------------------------------------
# POST /api/settings/{section}
# ---------------------------------------------------------------------------

class TestSettingsPost:
    def test_saves_ranking_section(self, tmp_path):
        config_path = tmp_path / "extraction.yaml"
        config_path.write_text("ranking:\n  semantic_weight: 0.4\n", encoding="utf-8")

        with patch(
            "lit_monitor.server.routes.settings.safe_save_settings_section"
        ) as mock_save:
            client = _make_client()
            r = client.post(
                "/api/settings/ranking",
                json={"semantic_weight": 0.6},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["section"] == "ranking"
        mock_save.assert_called_once_with("ranking", {"semantic_weight": 0.6})

    def test_unknown_section_400(self):
        with patch(
            "lit_monitor.server.routes.settings.safe_save_settings_section",
            side_effect=ValueError("unknown section: 'bad_section'"),
        ):
            client = _make_client()
            r = client.post("/api/settings/bad_section", json={"foo": "bar"})
        assert r.status_code == 400

    def test_garbage_key_returns_422(self, tmp_path):
        """B8 (route): an unknown key in a known section → HTTP 422.

        Not mocked — the real safe_save_settings_section validates and raises
        SettingsValidationError before any file write, which the route maps to
        422 (distinct from the 400 used for an unknown section name).

        E8: routed through _post_to_real_file so the (rejected) write target is a
        tmp extraction.yaml, never the repo's live config — a future validation
        regression must not be able to mutate the user's real file.
        """
        r, _ = _post_to_real_file(tmp_path, "ranking", {"randomgarbage": True})
        assert r.status_code == 422

    def test_feedback_section_saves_and_rejects_unknown_key(self, tmp_path):
        """P5/K-a revived ``feedback`` as a REAL, editable settings section
        (atrophy ``minimum_cluster_floor`` + ``exploration_budget_pct``), so the
        E8 "dead section → 400" contract no longer holds. A valid payload now
        saves (200, round-trips); an unknown key in this KNOWN section → 422
        (malformed payload), distinct from the 400 used for a genuinely unknown
        section name.
        """
        r, loaded = _post_to_real_file(
            tmp_path, "feedback", {"minimum_cluster_floor": 0.2}
        )
        assert r.status_code == 200, r.text
        assert loaded["feedback"]["minimum_cluster_floor"] == 0.2

        # Unknown key in a KNOWN section → 422 (not the 400 reserved for an
        # unknown section name). FeedbackSettings is extra="forbid".
        r2, _ = _post_to_real_file(tmp_path, "feedback", {"anything": "value"})
        assert r2.status_code == 422

    def test_oserror_returns_generic_500_no_leak(self, caplog):
        """A3-5: an OSError from the writer → 500 whose body does NOT contain
        the exception text/path, and the failure is logged server-side.
        """
        import logging

        secret = "/Users/secret/config/extraction.yaml"
        with caplog.at_level(
            logging.ERROR, logger="lit_monitor.server.routes.settings"
        ):
            with patch(
                "lit_monitor.server.routes.settings.safe_save_settings_section",
                side_effect=OSError(f"[Errno 13] Permission denied: '{secret}'"),
            ):
                client = _make_client()
                r = client.post(
                    "/api/settings/ranking", json={"semantic_weight": 0.6}
                )
        assert r.status_code == 500
        body = r.json()
        assert body["detail"] == "File I/O error"
        # Info-leak guard: the path must not reach the client.
        assert secret not in str(body)
        assert "Permission denied" not in str(body)
        # But it must be logged server-side.
        assert any(
            secret in rec.getMessage() or rec.exc_info for rec in caplog.records
        )

    def test_non_object_body_422(self):
        client = _make_client()
        # Send a JSON array instead of an object.
        r = client.post(
            "/api/settings/ranking",
            content=b"[1, 2, 3]",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# E3/B8 review fix: round-trip the EXACT nested body the reworked template emits
#
# The Advanced Settings UI was posting fictional flat keys (semantic_weight,
# graph_weight, k, algorithm) that the per-section pydantic models reject with
# 422. The template now posts the REAL nested config keys. These tests pin the
# corrected shape (200 + written) and prove the old flat shape is rejected (422).
# ---------------------------------------------------------------------------


def _post_to_real_file(tmp_path, section, payload):
    """POST through the route while pointing config_io at a tmp extraction.yaml.

    Patches CONFIG_DIR so safe_save_settings_section reads/writes the tmp file
    instead of the repo's real config. Returns (response, loaded_yaml_dict).
    """
    import lit_monitor.server.config_io as cfg_io

    config_path = tmp_path / "extraction.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    with patch.object(cfg_io, "CONFIG_DIR", tmp_path):
        client = _make_client()
        r = client.post(f"/api/settings/{section}", json=payload)

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return r, loaded


class TestSettingsUIRoundTrip:
    """Round-trip the exact nested payloads the reworked template emits."""

    # The complete ranking section the form serializes (all fields, nested).
    RANKING_BODY = {
        "weights": {
            "domain_context": 0.3,
            "cluster_centroid": 0.2,
            "graph_entity_overlap": 0.15,
            "graph_citation": 0.1,
            "graph_shared_authors": 0.05,
        },
        "domain_filter": {
            "enabled": True,
            "threshold": 0.4,
            "minimum_off_domain_slots_pct": 0.1,
        },
        "s2_supplement_cap": {
            "enabled": True,
            "min_relevance": 0.5,
        },
    }

    # The complete clustering section the form serializes (all fields, nested).
    CLUSTERING_BODY = {
        "enabled": True,
        "min_papers_threshold": 120,
        "k_min": 6,
        "k_max": 18,
        "recompute_frequency": "nightly",
        "use_existing_collections_as_priors": {
            "enabled": True,
            "collection_names": ["Chromatography", "Filtration"],
        },
        "write_back": {
            "tags": {"enabled": True, "namespace": "lm"},
            "collections": {"enabled": False, "parent_collection": "lit-monitor"},
        },
    }

    def test_ranking_nested_payload_round_trips(self, tmp_path):
        r, loaded = _post_to_real_file(tmp_path, "ranking", self.RANKING_BODY)
        assert r.status_code == 200, r.text
        assert loaded["ranking"]["weights"]["domain_context"] == 0.3
        assert loaded["ranking"]["weights"]["graph_shared_authors"] == 0.05
        assert loaded["ranking"]["domain_filter"]["enabled"] is True
        assert loaded["ranking"]["domain_filter"]["threshold"] == 0.4
        assert loaded["ranking"]["s2_supplement_cap"]["min_relevance"] == 0.5

    def test_clustering_nested_payload_round_trips(self, tmp_path):
        r, loaded = _post_to_real_file(tmp_path, "clustering", self.CLUSTERING_BODY)
        assert r.status_code == 200, r.text
        assert loaded["clustering"]["enabled"] is True
        assert loaded["clustering"]["k_min"] == 6
        assert loaded["clustering"]["k_max"] == 18
        assert loaded["clustering"]["use_existing_collections_as_priors"][
            "collection_names"
        ] == ["Chromatography", "Filtration"]
        assert loaded["clustering"]["write_back"]["tags"]["namespace"] == "lm"

    def test_old_flat_ranking_payload_rejected_422(self, tmp_path):
        """Migration proof: the pre-fix flat ranking shape now fails validation.

        E8: routed through _post_to_real_file so the (rejected) write target is a
        tmp file, not the repo's live config/extraction.yaml.
        """
        r, _ = _post_to_real_file(
            tmp_path,
            "ranking",
            {"semantic_weight": 0.4, "graph_weight": 0.3, "domain_weight": 0.3},
        )
        assert r.status_code == 422

    def test_old_flat_clustering_payload_rejected_422(self, tmp_path):
        """Migration proof: the pre-fix flat clustering shape now fails validation.

        E8: routed through _post_to_real_file so the (rejected) write target is a
        tmp file, not the repo's live config/extraction.yaml.
        """
        r, _ = _post_to_real_file(tmp_path, "clustering", {"k": 8, "algorithm": "kmeans"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# safe_save_settings_section unit tests
# ---------------------------------------------------------------------------

class TestSafeSaveSettingsSection:
    def test_atomic_write(self, tmp_path):
        """Verify the function writes and the file is readable YAML after save."""
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        config_path.write_text(
            "ranking:\n  weights:\n    domain_context: 0.4\n"
            "clustering:\n  k_min: 8\n",
            encoding="utf-8",
        )

        safe_save_settings_section(
            "ranking",
            {"weights": {"domain_context": 0.9, "cluster_centroid": 0.1}},
            config_path=config_path,
        )

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["ranking"]["weights"]["domain_context"] == 0.9
        # Other sections untouched.
        assert result["clustering"]["k_min"] == 8

    def test_creates_file_if_absent(self, tmp_path):
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        assert not config_path.exists()

        safe_save_settings_section("web_ui", {"show_feedback_buttons": False}, config_path=config_path)

        assert config_path.exists()
        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["web_ui"]["show_feedback_buttons"] is False

    def test_unknown_section_raises_value_error(self, tmp_path):
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        config_path.write_text("{}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unknown section"):
            safe_save_settings_section("nonexistent_section", {}, config_path=config_path)

    @pytest.mark.parametrize("bad", [[1, 2, 3], "a string", 42, True, None])
    def test_non_dict_data_raises_value_error(self, tmp_path, bad):
        """P2.4 minimal guard: a section value must be a mapping.

        A list/scalar/None for ``data`` would corrupt the section's shape, so
        the helper rejects it with ValueError (the route maps ValueError to a
        4xx) and the file on disk is left UNCHANGED.
        """
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        original = "ranking:\n  semantic_weight: 0.4\n"
        config_path.write_text(original, encoding="utf-8")

        with pytest.raises(ValueError, match="must be a mapping"):
            safe_save_settings_section("ranking", bad, config_path=config_path)

        # File must be untouched by the rejected write.
        assert config_path.read_text(encoding="utf-8") == original

    def test_garbage_key_rejected_file_unchanged(self, tmp_path):
        """B8: an unknown key in a known section → ValueError, file untouched.

        ``{"randomgarbage": true}`` is a dict (so it survives the P2 non-dict
        guard) but has no place in the ranking schema. The per-section model
        uses extra="forbid", so this must raise and NOT corrupt the file.
        """
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        original = "ranking:\n  weights:\n    domain_context: 0.2\n"
        config_path.write_text(original, encoding="utf-8")

        with pytest.raises(ValueError):
            safe_save_settings_section(
                "ranking", {"randomgarbage": True}, config_path=config_path
            )

        assert config_path.read_text(encoding="utf-8") == original

    def test_wrong_type_for_known_key_rejected(self, tmp_path):
        """B8: a known key with the wrong type → ValueError, file untouched.

        ``clustering.enabled`` is a bool; a non-coercible string must be
        rejected rather than written verbatim.
        """
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        original = "clustering:\n  enabled: true\n"
        config_path.write_text(original, encoding="utf-8")

        with pytest.raises(ValueError):
            safe_save_settings_section(
                "clustering", {"enabled": "not-a-bool"}, config_path=config_path
            )

        assert config_path.read_text(encoding="utf-8") == original

    @pytest.mark.parametrize(
        ("section", "payload"),
        [
            ("ranking", {"weights": {"domain_context": 0.3}}),
            ("clustering", {"enabled": True, "k_min": 6}),
            ("trending_concepts", {"enabled": True, "min_recent_mentions": 4}),
            ("query_expansion", {"enabled": True, "top_k_co_entities": 5}),
            ("researcher_gating", {"enabled": True, "min_graph_overlap": 2}),
            ("embedding", {"provider": "ollama", "ollama": {"dim": 768}}),
            ("discovery", {"notify": {"enabled": False}}),
            ("web_ui", {"show_feedback_buttons": True}),
        ],
    )
    def test_valid_payload_written(self, tmp_path, section, payload):
        """B8: a valid payload for each section is accepted and persisted."""
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        config_path.write_text("{}\n", encoding="utf-8")

        safe_save_settings_section(section, payload, config_path=config_path)

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert section in result

    def test_feedback_section_is_known_and_validates(self, tmp_path):
        """P5/K-a revived ``feedback`` as a real config section (atrophy
        ``minimum_cluster_floor`` + ``exploration_budget_pct`` are read by
        config.py). It is therefore NO LONGER an 'unknown section':
        safe_save_settings_section accepts valid keys and raises
        SettingsValidationError (NOT a plain 'unknown section' ValueError) on an
        unknown key, since FeedbackSettings is extra="forbid".
        """
        from lit_monitor.server.config_io import (
            SettingsValidationError,
            safe_save_settings_section,
        )

        config_path = tmp_path / "extraction.yaml"
        config_path.write_text("{}\n", encoding="utf-8")

        # Valid key saves without error and round-trips to disk.
        safe_save_settings_section(
            "feedback", {"exploration_budget_pct": 0.25}, config_path=config_path
        )
        saved = yaml.safe_load(config_path.read_text())
        assert saved["feedback"]["exploration_budget_pct"] == 0.25

        # Unknown key → SettingsValidationError (known section, malformed body),
        # NOT the plain "unknown section" ValueError reserved for bad section names.
        with pytest.raises(SettingsValidationError):
            safe_save_settings_section(
                "feedback", {"anything": "value"}, config_path=config_path
            )

    def test_preserves_comments_on_unchanged_sections(self, tmp_path):
        """v0.9.1 regression: comments + quoting on untouched sections survive the write.

        Bundle H originally used yaml.safe_dump which silently stripped every
        comment in extraction.yaml the first time a user toggled an Advanced
        Settings flag. The fix uses ruamel.yaml round-trip mode.
        """
        from lit_monitor.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        original = (
            "# top-of-file comment\n"
            "brain_build:\n"
            '  model: "phi4-mini"  # quality ceiling\n'
            "  # nested comment about chunking\n"
            "  chunk_chars: null\n"
            "\n"
            "ranking:\n"
            "  weights:\n"
            "    domain_context: 0.4\n"
        )
        config_path.write_text(original, encoding="utf-8")

        # Toggle a setting via the production code path.
        safe_save_settings_section(
            "ranking",
            {"weights": {"domain_context": 0.9, "cluster_centroid": 0.1}},
            config_path=config_path,
        )

        written = config_path.read_text(encoding="utf-8")
        # All comments preserved.
        assert "# top-of-file comment" in written, (
            "top-of-file comment was stripped — ruamel round-trip is not active"
        )
        assert "# quality ceiling" in written, "inline comment was stripped"
        assert "# nested comment about chunking" in written, "nested comment was stripped"
        # Quoted string stays quoted.
        assert '"phi4-mini"' in written, "quoted scalar lost its quotes"
        # The actual edit landed.
        result = yaml.safe_load(written)
        assert result["ranking"]["weights"]["domain_context"] == 0.9
        assert result["ranking"]["weights"]["cluster_centroid"] == 0.1
        # Untouched section semantically intact.
        assert result["brain_build"]["model"] == "phi4-mini"
        assert result["brain_build"]["chunk_chars"] is None
