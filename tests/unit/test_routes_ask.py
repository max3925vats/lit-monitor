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
    from scripts.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


class TestAskPage:
    def test_get_ask_renders_form_when_graph_available(self, client):
        with patch("scripts.server.routes.ask.safe_graph_db", return_value=object()):
            r = client.get("/ask")
        assert r.status_code == 200
        assert 'name="question"' in r.text  # the form is present
        assert "/ask/answer" in r.text  # hx-post target

    def test_get_ask_shows_no_graph_notice_when_absent(self, client):
        with patch("scripts.server.routes.ask.safe_graph_db", return_value=None):
            r = client.get("/ask")
        assert r.status_code == 200
        assert "graph backfill" in r.text  # the build-graph-first hint
        assert 'name="question"' not in r.text  # form NOT shown

    def test_nav_has_ask_link(self, client):
        r = client.get("/")
        assert 'href="/ask"' in r.text


class TestAskAnswer:
    def _result(self):
        from scripts.graph.ask import AskResult  # noqa: PLC0415

        return AskResult(
            cypher="MATCH (p:Paper) RETURN p LIMIT 5",
            rows=[{"title": "Smith 2021", "year": 2021}],
            rendered="| title | year |\n| --- | --- |\n| Smith 2021 | 2021 |",
            prose="Smith 2021 is the key paper.",
        )

    def test_answer_renders_prose_table_and_cypher(self, client):
        with patch(
            "scripts.server.routes.ask.run_pipeline", return_value=self._result()
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
        from scripts.graph.ask import AskResult  # noqa: PLC0415

        res = AskResult(
            cypher="MATCH (p) RETURN p",
            rows=[],
            rendered="_(no results)_",
            prose="No data.",
        )
        with patch("scripts.server.routes.ask.run_pipeline", return_value=res):
            r = client.post("/ask/answer", data={"question": "x"})
        assert "no matches" in r.text.lower()

    def test_answer_pipeline_error_is_generic_no_leak(self, client, caplog):
        secret = "kuzu://secret/path/db"
        with caplog.at_level(logging.ERROR, logger="scripts.server.routes.ask"):
            with patch(
                "scripts.server.routes.ask.run_pipeline",
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
            "scripts.server.routes.ask._execute_guarded_cypher", return_value=rows
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
            "scripts.server.routes.ask._execute_guarded_cypher", side_effect=_spy
        ):
            r = client.post(
                "/ask/cypher", data={"cypher": "MATCH (p) DETACH DELETE p"}
            )
        # guard runs BEFORE execution; a mutation must be rejected pre-exec.
        assert "read-only" in r.text.lower()
        assert called["n"] == 0
