"""Bundle H: HTTP endpoint + HTML page tests for /themes and /api/themes/*.

Covers:
- GET /api/themes — empty list when no clusters
- GET /api/themes — populated list from seeded clusters
- GET /api/themes/{id} — 200 with cluster + papers
- GET /api/themes/{id} — 404 for unknown id
- GET /api/themes/{id} — 404 for archived cluster
- POST /api/themes/{id}/write-back/tags — dry_run default, response shape
- POST /api/themes/{id}/write-back/tags?confirm=true — calls live write
- POST /api/themes/{id}/write-back/{bad_mode} — 400
- GET /themes — HTML index 200
- GET /themes/{cluster_id} — HTML detail 200
- GET /themes/{cluster_id} — HTML detail 404
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from lit_monitor.server.routes.themes import router as themes_router
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


@pytest.fixture()
def empty_db(tmp_path):
    from lit_monitor.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")


@pytest.fixture()
def seeded_db(tmp_path):
    """StateDB with one active cluster and two papers in it."""
    from lit_monitor.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    with db._connect() as conn:
        conn.execute(
            "INSERT INTO clusters (display_name, n_papers, cohesion_score) VALUES (?, ?, ?)",
            ("Topic A", 2, 0.85),
        )
        cluster_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO papers (doi, title) VALUES (?, ?)",
            ("10.1/test1", "Paper One"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO papers (doi, title) VALUES (?, ?)",
            ("10.1/test2", "Paper Two"),
        )
        conn.execute(
            "INSERT INTO cluster_assignments (doi, cluster_id, distance_to_centroid) VALUES (?, ?, ?)",
            ("10.1/test1", cluster_id, 0.1),
        )
        conn.execute(
            "INSERT INTO cluster_assignments (doi, cluster_id, distance_to_centroid) VALUES (?, ?, ?)",
            ("10.1/test2", cluster_id, 0.3),
        )
    return db, cluster_id


def _make_api_client(db, monkeypatch) -> TestClient:
    """Minimal app with themes router, runtime patched to use db."""
    rt = MagicMock()
    rt.state_db = db
    monkeypatch.setattr("lit_monitor.server.routes.themes.get_runtime", lambda: rt)
    app = FastAPI()
    app.include_router(themes_router)
    app.state.dev_mode = False
    app.state.version = "test"
    return TestClient(app), rt


def _make_html_client(db, monkeypatch) -> TestClient:
    """App with themes router + real templates for HTML tests."""
    import lit_monitor.server.app as app_mod

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app_mod.configure_templates(templates)
    # monkeypatch auto-restores app.templates after the test so this test-local
    # templates object does not leak into sibling tests (order-dependent flakiness).
    monkeypatch.setattr(app_mod, "templates", templates)

    rt = MagicMock()
    rt.state_db = db
    monkeypatch.setattr("lit_monitor.server.routes.themes.get_runtime", lambda: rt)
    monkeypatch.setattr("lit_monitor.server.config_io.load_config", lambda name: {})

    app = FastAPI()
    app.include_router(themes_router)
    app.state.dev_mode = False
    app.state.version = "test"
    return TestClient(app), rt


# ---------------------------------------------------------------------------
# API tests — GET /api/themes
# ---------------------------------------------------------------------------

class TestListThemesEmpty:
    def test_empty_db_returns_empty_list(self, empty_db, monkeypatch):
        client, _ = _make_api_client(empty_db, monkeypatch)
        r = client.get("/api/themes")
        assert r.status_code == 200
        assert r.json() == []


class TestListThemesSeeded:
    def test_returns_cluster_rows(self, seeded_db, monkeypatch):
        db, cluster_id = seeded_db
        client, _ = _make_api_client(db, monkeypatch)
        r = client.get("/api/themes")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["display_name"] == "Topic A"
        assert data[0]["n_papers"] == 2


# ---------------------------------------------------------------------------
# API tests — GET /api/themes/{id}
# ---------------------------------------------------------------------------

class TestGetThemeDetail:
    def test_200_with_papers(self, seeded_db, monkeypatch):
        db, cluster_id = seeded_db
        client, _ = _make_api_client(db, monkeypatch)
        r = client.get(f"/api/themes/{cluster_id}")
        assert r.status_code == 200
        data = r.json()
        assert "cluster" in data
        assert "papers" in data
        assert data["cluster"]["display_name"] == "Topic A"
        assert len(data["papers"]) == 2

    def test_404_for_unknown(self, empty_db, monkeypatch):
        client, _ = _make_api_client(empty_db, monkeypatch)
        r = client.get("/api/themes/9999")
        assert r.status_code == 404

    def test_404_for_archived_cluster(self, tmp_path, monkeypatch):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO clusters (display_name, n_papers, archived) VALUES (?, ?, ?)",
                ("Archived", 0, 1),
            )
            cluster_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        client, _ = _make_api_client(db, monkeypatch)
        r = client.get(f"/api/themes/{cluster_id}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# API tests — POST /api/themes/{id}/write-back/{mode}
# ---------------------------------------------------------------------------

class TestWriteBack:
    def test_dry_run_default_tags(self, seeded_db, monkeypatch):
        db, cluster_id = seeded_db
        rt = MagicMock()
        rt.state_db = db
        rt.zotero_client = MagicMock()
        monkeypatch.setattr("lit_monitor.server.routes.themes.get_runtime", lambda: rt)

        fake_report = [{"doi": "10.1/test1", "tag": "lm/topic-a"}]
        with patch(
            "lit_monitor.clustering.write_back.push_tags_to_zotero",
            return_value=fake_report,
        ):
            app = FastAPI()
            app.include_router(themes_router)
            client = TestClient(app)
            r = client.post(f"/api/themes/{cluster_id}/write-back/tags")

        assert r.status_code == 200
        data = r.json()
        assert data["dry_run"] is True

    def test_confirm_true_calls_live_write(self, seeded_db, monkeypatch):
        db, cluster_id = seeded_db
        rt = MagicMock()
        rt.state_db = db
        rt.zotero_client = MagicMock()
        monkeypatch.setattr("lit_monitor.server.routes.themes.get_runtime", lambda: rt)

        fake_report = [{"doi": "10.1/test1", "tag": "lm/topic-a"}]
        with patch(
            "lit_monitor.clustering.write_back.push_tags_to_zotero",
            return_value=fake_report,
        ) as mock_push:
            app = FastAPI()
            app.include_router(themes_router)
            client = TestClient(app)
            r = client.post(
                f"/api/themes/{cluster_id}/write-back/tags?confirm=true"
            )

        assert r.status_code == 200
        data = r.json()
        assert data["dry_run"] is False
        mock_push.assert_called_once()
        _, kwargs = mock_push.call_args
        assert kwargs["dry_run"] is False

    def test_bad_mode_returns_400(self, seeded_db, monkeypatch):
        db, cluster_id = seeded_db
        client, _ = _make_api_client(db, monkeypatch)
        r = client.post(f"/api/themes/{cluster_id}/write-back/invalid")
        assert r.status_code == 400

    def test_write_back_error_returns_generic_500_no_leak(
        self, seeded_db, monkeypatch, caplog
    ):
        """A3-5 / TF-3: when the write-back tool raises, the 500 body must NOT
        contain the exception text/path — only a generic detail — and the full
        exception must be logged server-side.
        """
        import logging

        db, cluster_id = seeded_db
        rt = MagicMock()
        rt.state_db = db
        rt.zotero_client = MagicMock()
        monkeypatch.setattr("lit_monitor.server.routes.themes.get_runtime", lambda: rt)

        # A path-shaped secret the raw str(exc) would leak to the client.
        secret = "/Users/secret/zotero/storage/ABCD1234/leak.sqlite"
        with caplog.at_level(logging.ERROR, logger="lit_monitor.server.routes.themes"):
            with patch(
                "lit_monitor.clustering.write_back.push_tags_to_zotero",
                side_effect=FileNotFoundError(secret),
            ):
                app = FastAPI()
                app.include_router(themes_router)
                client = TestClient(app, raise_server_exceptions=False)
                r = client.post(f"/api/themes/{cluster_id}/write-back/tags")

        assert r.status_code == 500
        body = r.json()
        assert body["detail"] == "Internal error"
        # Info-leak guard: no path / exception type / traceback in the body.
        serialized = str(body)
        assert secret not in serialized
        assert "FileNotFoundError" not in serialized
        # The real error must still be logged server-side (with the path).
        assert any(
            secret in rec.getMessage() or rec.exc_info for rec in caplog.records
        ), "write-back failure must be logged server-side with the full exception"


# ---------------------------------------------------------------------------
# HTML page tests
# ---------------------------------------------------------------------------

class TestThemesHTMLPages:
    def test_themes_index_200(self, seeded_db, monkeypatch):
        db, _ = seeded_db
        client, _ = _make_html_client(db, monkeypatch)
        r = client.get("/themes")
        assert r.status_code == 200
        assert "Topic A" in r.text

    def test_themes_detail_200(self, seeded_db, monkeypatch):
        db, cluster_id = seeded_db
        client, _ = _make_html_client(db, monkeypatch)
        r = client.get(f"/themes/{cluster_id}")
        assert r.status_code == 200
        assert "Topic A" in r.text

    def test_themes_detail_404(self, empty_db, monkeypatch):
        client, _ = _make_html_client(empty_db, monkeypatch)
        r = client.get("/themes/9999")
        assert r.status_code == 404
