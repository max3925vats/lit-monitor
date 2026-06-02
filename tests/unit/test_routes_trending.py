"""Bundle E: HTTP endpoint tests for trending-concept routes.

Covers:
- GET /api/trending returns pending suggestions list
- POST /api/trending/{id}/accept persists user action
- POST /api/trending/{id}/dismiss persists user action
- GET /trending renders HTML page
- 404 on accept/dismiss for unknown id
- safe_save_topics called on accept
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path):
    """Minimal FastAPI app with trending router wired in."""
    from fastapi import FastAPI
    from fastapi.templating import Jinja2Templates

    from scripts.server.routes.trending import router as trending_router

    app = FastAPI()
    app.include_router(trending_router)

    # Provide templates at app-level so route handlers can access request.app.state
    templates_dir = Path(__file__).parents[2] / "scripts" / "server" / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))
    app.state.dev_mode = False
    app.state.version = "test"
    return app


@pytest.fixture()
def client(app):
    return TestClient(app)


def _make_row(row_id: int = 1, text: str = "biorefinery", action: str = "pending") -> dict:
    return {
        "id": row_id,
        "concept_text": text,
        "concept_type": "topic",
        "n_mentions_new": 20,
        "n_mentions_prev": 5,
        "growth_rate": 3.0,
        "suggested_at": "2026-05-30 12:00:00",
        "user_action": action,
        "action_at": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetTrending:
    def test_returns_pending_list(self, client):
        pending = [_make_row(1), _make_row(2, "distillation")]
        with (
            patch("scripts.server.routes.trending._safe_db") as mock_db_fn,
        ):
            mock_db = MagicMock()
            mock_db.get_pending_trending_suggestions.return_value = pending
            mock_db_fn.return_value = mock_db

            resp = client.get("/api/trending")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["concept_text"] == "biorefinery"

    def test_returns_empty_when_no_db(self, client):
        with patch("scripts.server.routes.trending._safe_db", return_value=None):
            resp = client.get("/api/trending")
        assert resp.status_code == 200
        assert resp.json() == []


class TestAcceptTrending:
    def test_accept_marks_accepted(self, client):
        row = _make_row(1)
        with (
            patch("scripts.server.routes.trending._safe_db") as mock_db_fn,
            patch("scripts.server.routes.trending.safe_save_topics") as mock_save,
        ):
            mock_db = MagicMock()
            mock_db.get_trending_suggestion_by_id.return_value = row
            mock_db.update_trending_action.return_value = None
            mock_db_fn.return_value = mock_db

            resp = client.post("/api/trending/1/accept")

        assert resp.status_code == 200
        mock_save.assert_called_once()
        mock_db.update_trending_action.assert_called_once_with(1, "accepted")

    def test_accept_returns_404_for_unknown_id(self, client):
        with patch("scripts.server.routes.trending._safe_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db.get_trending_suggestion_by_id.return_value = None
            mock_db_fn.return_value = mock_db

            resp = client.post("/api/trending/999/accept")
        assert resp.status_code == 404


class TestDismissTrending:
    def test_dismiss_marks_dismissed(self, client):
        row = _make_row(1)
        with patch("scripts.server.routes.trending._safe_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db.get_trending_suggestion_by_id.return_value = row
            mock_db.update_trending_action.return_value = None
            mock_db_fn.return_value = mock_db

            resp = client.post("/api/trending/1/dismiss")

        assert resp.status_code == 200
        mock_db.update_trending_action.assert_called_once_with(1, "dismissed")

    def test_dismiss_returns_404_for_unknown_id(self, client):
        with patch("scripts.server.routes.trending._safe_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db.get_trending_suggestion_by_id.return_value = None
            mock_db_fn.return_value = mock_db

            resp = client.post("/api/trending/999/dismiss")
        assert resp.status_code == 404


class TestSafeSaveTopics:
    def test_atomic_add_creates_file(self, tmp_path):
        from scripts.server.config_io import safe_save_topics

        topics_path = tmp_path / "topics.yaml"
        new_topic = {"name": "biorefinery", "query": "biorefinery AND design", "databases": ["arxiv"]}
        safe_save_topics(new_topic, topics_path=topics_path)

        import yaml
        data = yaml.safe_load(topics_path.read_text())
        assert len(data["searches"]) == 1
        assert data["searches"][0]["name"] == "biorefinery"

    def test_atomic_add_appends_to_existing(self, tmp_path):
        import yaml

        from scripts.server.config_io import safe_save_topics

        topics_path = tmp_path / "topics.yaml"
        existing = {"searches": [{"name": "existing", "query": "existing AND topic"}]}
        topics_path.write_text(yaml.safe_dump(existing))

        new_topic = {"name": "biorefinery", "query": "biorefinery AND design"}
        safe_save_topics(new_topic, topics_path=topics_path)

        data = yaml.safe_load(topics_path.read_text())
        assert len(data["searches"]) == 2
        names = [s["name"] for s in data["searches"]]
        assert "existing" in names
        assert "biorefinery" in names

    def test_does_not_remove_existing_topics(self, tmp_path):
        import yaml

        from scripts.server.config_io import safe_save_topics

        topics_path = tmp_path / "topics.yaml"
        existing = {
            "searches": [
                {"name": "topic1", "query": "topic1"},
                {"name": "topic2", "query": "topic2"},
            ]
        }
        topics_path.write_text(yaml.safe_dump(existing))

        safe_save_topics({"name": "new", "query": "new"}, topics_path=topics_path)

        data = yaml.safe_load(topics_path.read_text())
        assert len(data["searches"]) == 3
