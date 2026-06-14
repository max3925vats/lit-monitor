"""WA1: GET /ask page + POST /ask/answer fragment route tests.

All graph / LLM dependencies are mocked — no live KuzuDB, Ollama, or network.
Mirrors the `client` fixture from tests/unit/test_routes_query.py.

NOTE (plan drift): scripts.graph.ask.AskResult requires a `rendered` positional
field (cypher, rows, rendered, prose) — the plan's example AskResult(...) calls
omit it, which would raise TypeError. We pass `rendered=` here to match the real
dataclass.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    from lit_monitor.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


class TestAskPage:
    def test_get_ask_renders_form_when_graph_available(self, client):
        with patch("lit_monitor.server.routes.ask.safe_graph_db", return_value=object()):
            r = client.get("/ask")
        assert r.status_code == 200
        # The page form should use sl-textarea (Shoelace), not the native textarea.
        # Assert sl-button variant="primary" with id="ask-spin" — both unique to
        # the page form after migration (drawer uses ask-drawer-spin, not ask-spin).
        assert 'name="question"' in r.text  # question field present
        assert '<textarea name="question"' not in r.text  # native replaced
        assert 'id="ask-spin"' in r.text  # page-form spinner present
        assert "/ask/answer" in r.text  # hx-post target

    def test_get_ask_shows_no_graph_notice_when_absent(self, client):
        with patch("lit_monitor.server.routes.ask.safe_graph_db", return_value=None):
            r = client.get("/ask")
        assert r.status_code == 200
        assert "graph backfill" in r.text  # the build-graph-first hint
        # The page's own Ask form is gated on has_graph. The global Ask drawer in
        # the app-shell always renders an <sl-textarea name="question">, so we
        # cannot assert <sl-textarea absent page-wide. Instead assert the PAGE
        # form's HTMX target (#ask-result) is absent — that target is unique to
        # the page form (the drawer uses #ask-drawer-result).
        assert 'hx-target="#ask-result"' not in r.text
        assert 'id="ask-result"' not in r.text

    def test_nav_has_ask_link(self, client):
        r = client.get("/")
        assert 'href="/ask"' in r.text


class TestAskHistory:
    """WA4: recent-questions history container + site.js wiring.

    The history itself lives in localStorage (client-side); here we only pin
    the server-rendered wiring: the container exists when the graph is
    available, the page loads site.js (the module that drives it), and the
    container is NOT rendered on the no-graph page.
    """

    def test_ask_page_includes_history_container(self, client):
        with patch("lit_monitor.server.routes.ask.safe_graph_db", return_value=object()):
            r = client.get("/ask")
        assert r.status_code == 200
        assert 'id="ask-history"' in r.text  # history mount point
        assert "Recent questions" in r.text  # WA4 heading (the new element)
        assert "/static/site.js" in r.text  # the module that drives it (base.html)

    def test_history_container_absent_without_graph(self, client):
        with patch("lit_monitor.server.routes.ask.safe_graph_db", return_value=None):
            r = client.get("/ask")
        assert r.status_code == 200
        assert 'id="ask-history"' not in r.text  # only shown when form is shown
        assert "Recent questions" not in r.text  # heading gated with the form too


class TestAskFullWidthAndEmptyState:
    """Task D: D1 full-width class + D2 empty-state default text."""

    def test_ask_question_field_full_width_class(self, client):
        # The question textarea carries the full-width class when graph is available.
        with patch("lit_monitor.server.routes.ask.safe_graph_db", return_value=object()):
            r = client.get("/ask")
        assert "ask-question" in r.text

    def test_ask_history_empty_state_default(self, client):
        # The #ask-history div ships with an empty-state paragraph as default content.
        with patch("lit_monitor.server.routes.ask.safe_graph_db", return_value=object()):
            r = client.get("/ask")
        assert "No questions asked yet" in r.text


class TestAskAnswer:
    def _result(self):
        from lit_monitor.graph.ask import AskResult  # noqa: PLC0415

        return AskResult(
            cypher="MATCH (p:Paper) RETURN p LIMIT 5",
            rows=[{"title": "Smith 2021", "year": 2021}],
            rendered="| title | year |\n| --- | --- |\n| Smith 2021 | 2021 |",
            prose="Smith 2021 is the key paper.",
        )

    def test_answer_renders_prose_table_and_cypher(self, client):
        with patch(
            "lit_monitor.server.routes.ask.run_pipeline", return_value=self._result()
        ):
            r = client.post("/ask/answer", data={"question": "key paper?"})
        assert r.status_code == 200
        assert "Smith 2021 is the key paper." in r.text  # prose
        assert "Smith 2021" in r.text and "2021" in r.text  # table cell
        assert "MATCH (p:Paper)" in r.text  # cypher block

    def test_answer_empty_question_is_rejected(self, client):
        r = client.post("/ask/answer", data={"question": "   "})
        assert r.status_code == 200
        assert "enter a question" in r.text.lower()

    def test_answer_empty_rows_says_no_matches(self, client):
        from lit_monitor.graph.ask import AskResult  # noqa: PLC0415

        res = AskResult(
            cypher="MATCH (p) RETURN p",
            rows=[],
            rendered="_(no results)_",
            prose="No data.",
        )
        with patch("lit_monitor.server.routes.ask.run_pipeline", return_value=res):
            r = client.post("/ask/answer", data={"question": "x"})
        assert "no matches" in r.text.lower()

    def test_answer_pipeline_error_is_generic_no_leak(self, client, caplog):
        secret = "kuzu://secret/path/db"
        with caplog.at_level(logging.ERROR, logger="lit_monitor.server.routes.ask"):
            with patch(
                "lit_monitor.server.routes.ask.run_pipeline",
                side_effect=RuntimeError(secret),
            ):
                r = client.post("/ask/answer", data={"question": "x"})
        assert r.status_code == 200  # fragment, not a 500 page
        assert secret not in r.text  # no leak to browser
        assert any(secret in rec.getMessage() for rec in caplog.records)  # logged


class TestAskCypher:
    """WA2: POST /ask/cypher — B3-guarded, read-only Cypher re-run.

    The guard MUST run before any execution: a mutation is rejected
    pre-exec, so the executor spy never fires.
    """

    def test_cypher_rerun_renders_table(self, client):
        rows = [{"name": "Smith"}]
        with patch(
            "lit_monitor.server.routes.ask._execute_guarded_cypher", return_value=rows
        ):
            r = client.post("/ask/cypher", data={"cypher": "MATCH (p) RETURN p"})
        assert r.status_code == 200
        assert "Smith" in r.text

    def test_cypher_mutation_blocked_not_executed(self, client):
        called = {"n": 0}

        def _spy(*a, **k):
            called["n"] += 1
            return []

        with patch(
            "lit_monitor.server.routes.ask._execute_guarded_cypher", side_effect=_spy
        ):
            r = client.post(
                "/ask/cypher", data={"cypher": "MATCH (p) DETACH DELETE p"}
            )
        # guard runs BEFORE execution; a mutation must be rejected pre-exec.
        assert "read-only" in r.text.lower()
        assert called["n"] == 0


class TestAskSave:
    """WA3: POST /ask/save — parse the hidden payload, write the note.

    Config is monkeypatched to a tmp-vault SimpleNamespace so no real vault is
    touched. Bad JSON → a generic error fragment (no traceback in the body).
    """

    def _cfg(self, tmp_path):
        from types import SimpleNamespace  # noqa: PLC0415

        return SimpleNamespace(
            obsidian=SimpleNamespace(
                vault_path=str(tmp_path / "vault"),
                connections_folder="Literature/Connections",
            )
        )

    def test_ask_save_writes_note(self, client, tmp_path, monkeypatch):
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        monkeypatch.setattr(
            "lit_monitor.server.routes.ask.get_config",
            lambda: self._cfg(tmp_path),
        )
        payload = json.dumps(
            {
                "question": "Which papers cite Smith 2021?",
                "prose": "Three papers cite it.",
                "rows": [{"title": "Jones 2022"}],
                "cypher": "MATCH (p)-[:CITES]->(q) RETURN p",
            }
        )
        r = client.post("/ask/save", data={"payload": payload})
        assert r.status_code == 200
        assert "Saved" in r.text
        connections = tmp_path / "vault" / "Literature" / "Connections"
        written = list(connections.glob("Ask_*.md"))
        assert written, "expected an Ask_*.md note under the tmp vault"
        text = Path(written[0]).read_text(encoding="utf-8")
        assert "Jones 2022" in text

    def test_ask_save_bad_payload_is_generic_error(self, client, monkeypatch):
        # Even with a valid config, a malformed payload must not 500-leak.
        monkeypatch.setattr(
            "lit_monitor.server.routes.ask.get_config",
            lambda: None,
        )
        r = client.post("/ask/save", data={"payload": "{not json"})
        assert r.status_code == 200  # generic fragment, not a 500 page
        assert "traceback" not in r.text.lower()
        assert "{not json" not in r.text
