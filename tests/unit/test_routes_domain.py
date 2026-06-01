"""Bundle G (v0.9): /api/domain HTTP endpoint tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient whose route handlers see a runtime backed by a shared tmp StateDB.

    Patches ``scripts.server.routes.domain.get_runtime`` to return a MagicMock
    runtime whose ``.state_db`` IS the same instance the test seeds — so
    writes via the fixture and reads via the route hit the same connection.
    """
    from scripts.core.state_db import StateDB
    from scripts.server.app import create_app

    db_path = tmp_path / "state.db"
    db = StateDB(db_path)

    fake_cfg = MagicMock()
    fake_cfg.state_db.path = str(db_path)
    monkeypatch.setattr("scripts.core.config.get_config", lambda: fake_cfg)

    fake_runtime = MagicMock()
    fake_runtime.state_db = db
    monkeypatch.setattr(
        "scripts.server.routes.domain.get_runtime",
        lambda: fake_runtime,
    )

    app = create_app()
    client = TestClient(app)
    client.state_db = db  # type: ignore[attr-defined]
    return client


class TestGetDomain:
    def test_empty_returns_stable_shape(self, client_with_runtime: TestClient) -> None:
        r = client_with_runtime.get("/api/domain")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "topics", "methods", "materials", "adjacent_fields", "exclusions",
        }
        for v in body.values():
            assert v == []

    def test_returns_seeded_extraction(self, client_with_runtime: TestClient) -> None:
        db = client_with_runtime.state_db  # type: ignore[attr-defined]
        db.save_domain_extraction(
            {
                "topics": ["alpha"],
                "methods": ["beta"],
                "materials": [],
                "adjacent_fields": [],
                "exclusions": [],
            }
        )
        body = client_with_runtime.get("/api/domain").json()
        assert body["topics"][0]["value"] == "alpha"
        assert body["methods"][0]["value"] == "beta"


class TestPostAnalyze:
    def test_missing_yaml_returns_404(
        self,
        client_with_runtime: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)  # ensure config/domain_context.yaml is absent
        r = client_with_runtime.post("/api/domain/analyze")
        assert r.status_code == 404

    def test_empty_yaml_returns_422(
        self,
        client_with_runtime: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "domain_context.yaml").write_text("domain_focus: ''\n")
        r = client_with_runtime.post("/api/domain/analyze")
        assert r.status_code == 422

    def test_llm_failure_returns_502(
        self,
        client_with_runtime: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "domain_context.yaml").write_text(
            "domain_focus: 'I work on X.'\n"
        )
        # Patch the symbol AS IMPORTED IN routes/domain.py
        monkeypatch.setattr(
            "scripts.server.routes.domain.analyze_domain",
            lambda text: None,
        )
        r = client_with_runtime.post("/api/domain/analyze")
        assert r.status_code == 502

    def test_happy_path_persists(
        self,
        client_with_runtime: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "domain_context.yaml").write_text(
            "domain_focus: 'I work on bioprocessing.'\n"
        )

        fake_extraction = {
            "topics": ["bioprocessing"],
            "methods": ["chromatography"],
            "materials": [],
            "adjacent_fields": [],
            "exclusions": [],
        }
        monkeypatch.setattr(
            "scripts.server.routes.domain.analyze_domain",
            lambda text: fake_extraction,
        )
        r = client_with_runtime.post("/api/domain/analyze")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["extraction"] == fake_extraction

        # And persisted
        get_body = client_with_runtime.get("/api/domain").json()
        assert get_body["topics"][0]["value"] == "bioprocessing"


class TestConfirmReject:
    def test_confirm_toggles_on(self, client_with_runtime: TestClient) -> None:
        db = client_with_runtime.state_db  # type: ignore[attr-defined]
        db.save_domain_extraction(
            {
                "topics": ["x"],
                "methods": [], "materials": [],
                "adjacent_fields": [], "exclusions": [],
            }
        )
        row_id = db.list_domain_extraction()["topics"][0]["id"]
        r = client_with_runtime.post(f"/api/domain/{row_id}/confirm")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "row_id": row_id, "confirmed": True}
        assert db.list_domain_extraction()["topics"][0]["user_confirmed"] is True

    def test_reject_toggles_off(self, client_with_runtime: TestClient) -> None:
        db = client_with_runtime.state_db  # type: ignore[attr-defined]
        db.save_domain_extraction(
            {
                "topics": ["x"],
                "methods": [], "materials": [],
                "adjacent_fields": [], "exclusions": [],
            }
        )
        row_id = db.list_domain_extraction()["topics"][0]["id"]
        db.set_domain_extraction_confirmed(row_id, True)
        r = client_with_runtime.post(f"/api/domain/{row_id}/reject")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "row_id": row_id, "confirmed": False}
        assert db.list_domain_extraction()["topics"][0]["user_confirmed"] is False
