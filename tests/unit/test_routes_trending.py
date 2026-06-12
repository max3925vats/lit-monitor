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

    import lit_monitor.server.app as app_mod
    from lit_monitor.server.routes.trending import router as trending_router

    app = FastAPI()
    app.include_router(trending_router)

    # Provide templates at app-level so route handlers can access request.app.state
    templates_dir = Path(__file__).parents[2] / "lit_monitor" / "server" / "templates"
    app.state.templates = app_mod.configure_templates(
        Jinja2Templates(directory=str(templates_dir))
    )
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
            patch("lit_monitor.server.routes.trending._safe_db") as mock_db_fn,
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
        with patch("lit_monitor.server.routes.trending._safe_db", return_value=None):
            resp = client.get("/api/trending")
        assert resp.status_code == 200
        assert resp.json() == []


class TestAcceptTrending:
    def test_accept_marks_accepted(self, client):
        row = _make_row(1)
        with (
            patch("lit_monitor.server.routes.trending._safe_db") as mock_db_fn,
            patch("lit_monitor.server.routes.trending.safe_save_topics") as mock_save,
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
        with patch("lit_monitor.server.routes.trending._safe_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db.get_trending_suggestion_by_id.return_value = None
            mock_db_fn.return_value = mock_db

            resp = client.post("/api/trending/999/accept")
        assert resp.status_code == 404


class TestDismissTrending:
    def test_dismiss_marks_dismissed(self, client):
        row = _make_row(1)
        with patch("lit_monitor.server.routes.trending._safe_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db.get_trending_suggestion_by_id.return_value = row
            mock_db.update_trending_action.return_value = None
            mock_db_fn.return_value = mock_db

            resp = client.post("/api/trending/1/dismiss")

        assert resp.status_code == 200
        mock_db.update_trending_action.assert_called_once_with(1, "dismissed")

    def test_dismiss_returns_404_for_unknown_id(self, client):
        with patch("lit_monitor.server.routes.trending._safe_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db.get_trending_suggestion_by_id.return_value = None
            mock_db_fn.return_value = mock_db

            resp = client.post("/api/trending/999/dismiss")
        assert resp.status_code == 404


class TestTrendingPageRender:
    """TF-2 (Audit-3 A3-2): GET /trending must render the full HTML page (200).

    This is the test that would have caught A3-2: the route used the legacy
    2-arg ``TemplateResponse("name", {"request": ...})`` form, which Starlette
    1.1+ rejects → 500. We exercise the real production handler (which renders
    via ``scripts.server.app.templates``) through the test client, mocking only
    the data layer (_safe_db) as the other route tests do.
    """

    def test_get_trending_renders_page_200(self, client):
        pending = [_make_row(1), _make_row(2, "distillation")]
        with patch("lit_monitor.server.routes.trending._safe_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db.get_pending_trending_suggestions.return_value = pending
            mock_db_fn.return_value = mock_db

            resp = client.get("/trending")

        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        # The suggestions must render into the page body.
        assert "biorefinery" in resp.text

    def test_get_trending_renders_with_no_db(self, client):
        """Even with no DB (empty suggestions), the page must render 200, not 500."""
        with patch("lit_monitor.server.routes.trending._safe_db", return_value=None):
            resp = client.get("/trending")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestTrendingPageCopy:
    """C2 (Audit-2): the /trending page must honestly frame the feature as
    library-activity / ingest-recency, not field-level publication trends.

    Renders the production template body block directly (the {% block content %}
    is self-contained and needs no app.state context) and asserts the copy.
    """

    def _render_content(self) -> str:
        import jinja2

        templates_dir = Path(__file__).parents[2] / "lit_monitor" / "server" / "templates"
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(templates_dir)))
        # Isolate the content block so we don't need base.html's app.state context.
        template = env.get_template("trending/index.html")
        ctx = template.new_context({"suggestions": []})
        return "".join(template.blocks["content"](ctx))

    def test_page_uses_honest_library_activity_framing(self):
        body = self._render_content().lower()
        # Honest framing present: the page makes clear this is about the user's
        # own library and ingest time, not field-level publication trends.
        assert "your library" in body
        assert "ingest" in body or "added" in body
        # The page must explicitly disclaim publication-recency / field trends so
        # the user can't be misled (the word "published" may appear, but only in
        # a disclaiming "not when they were published" sense).
        assert "not" in body and "published" in body
        assert "not a field-level publication trend" in body


class TestSafeSaveTopics:
    def test_atomic_add_creates_file(self, tmp_path):
        from lit_monitor.server.config_io import safe_save_topics

        topics_path = tmp_path / "topics.yaml"
        new_topic = {"name": "biorefinery", "query": "biorefinery AND design", "databases": ["arxiv"]}
        safe_save_topics(new_topic, topics_path=topics_path)

        import yaml
        data = yaml.safe_load(topics_path.read_text())
        assert len(data["searches"]) == 1
        assert data["searches"][0]["name"] == "biorefinery"

    def test_atomic_add_appends_to_existing(self, tmp_path):
        import yaml

        from lit_monitor.server.config_io import safe_save_topics

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

        from lit_monitor.server.config_io import safe_save_topics

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
