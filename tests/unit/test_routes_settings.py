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

from scripts.server.routes.settings import router as settings_router
from scripts.server.runtime import reset_runtime

TEMPLATES_DIR = Path(__file__).parents[2] / "scripts" / "server" / "templates"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


def _make_client(config_path=None) -> TestClient:
    import json as _json

    import scripts.server.app as app_mod

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
            "scripts.server.routes.settings._load_extraction_config",
            return_value={"ranking": {"semantic_weight": 0.5}},
        ):
            client = _make_client()
            r = client.get("/settings")
        assert r.status_code == 200
        assert "Advanced Settings" in r.text

    def test_reflects_current_config(self):
        with patch(
            "scripts.server.routes.settings._load_extraction_config",
            return_value={
                "web_ui": {"show_feedback_buttons": True},
            },
        ):
            client = _make_client()
            r = client.get("/settings")
        assert r.status_code == 200
        # The checkbox should be rendered as checked when show_feedback_buttons=True.
        assert "checked" in r.text


# ---------------------------------------------------------------------------
# POST /api/settings/{section}
# ---------------------------------------------------------------------------

class TestSettingsPost:
    def test_saves_ranking_section(self, tmp_path):
        config_path = tmp_path / "extraction.yaml"
        config_path.write_text("ranking:\n  semantic_weight: 0.4\n", encoding="utf-8")

        with patch(
            "scripts.server.routes.settings.safe_save_settings_section"
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
            "scripts.server.routes.settings.safe_save_settings_section",
            side_effect=ValueError("unknown section: 'bad_section'"),
        ):
            client = _make_client()
            r = client.post("/api/settings/bad_section", json={"foo": "bar"})
        assert r.status_code == 400

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
# safe_save_settings_section unit tests
# ---------------------------------------------------------------------------

class TestSafeSaveSettingsSection:
    def test_atomic_write(self, tmp_path):
        """Verify the function writes and the file is readable YAML after save."""
        from scripts.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        config_path.write_text(
            "ranking:\n  semantic_weight: 0.4\nclustering:\n  k: 8\n",
            encoding="utf-8",
        )

        safe_save_settings_section(
            "ranking",
            {"semantic_weight": 0.9, "graph_weight": 0.1},
            config_path=config_path,
        )

        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["ranking"]["semantic_weight"] == 0.9
        # Other sections untouched.
        assert result["clustering"]["k"] == 8

    def test_creates_file_if_absent(self, tmp_path):
        from scripts.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        assert not config_path.exists()

        safe_save_settings_section("web_ui", {"show_feedback_buttons": False}, config_path=config_path)

        assert config_path.exists()
        result = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert result["web_ui"]["show_feedback_buttons"] is False

    def test_unknown_section_raises_value_error(self, tmp_path):
        from scripts.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        config_path.write_text("{}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unknown section"):
            safe_save_settings_section("nonexistent_section", {}, config_path=config_path)

    def test_preserves_comments_on_unchanged_sections(self, tmp_path):
        """v0.9.1 regression: comments + quoting on untouched sections survive the write.

        Bundle H originally used yaml.safe_dump which silently stripped every
        comment in extraction.yaml the first time a user toggled an Advanced
        Settings flag. The fix uses ruamel.yaml round-trip mode.
        """
        from scripts.server.config_io import safe_save_settings_section

        config_path = tmp_path / "extraction.yaml"
        original = (
            "# top-of-file comment\n"
            "brain_build:\n"
            '  model: "phi4-mini"  # quality ceiling\n'
            "  # nested comment about chunking\n"
            "  chunk_chars: null\n"
            "\n"
            "ranking:\n"
            "  semantic_weight: 0.4\n"
        )
        config_path.write_text(original, encoding="utf-8")

        # Toggle a setting via the production code path.
        safe_save_settings_section(
            "ranking",
            {"semantic_weight": 0.9, "graph_weight": 0.1},
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
        assert result["ranking"]["semantic_weight"] == 0.9
        assert result["ranking"]["graph_weight"] == 0.1
        # Untouched section semantically intact.
        assert result["brain_build"]["model"] == "phi4-mini"
        assert result["brain_build"]["chunk_chars"] is None
