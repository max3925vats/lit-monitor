"""H8: POST /api/ask + POST /api/cypher route tests.

Coverage:
  ask:    happy path, empty-question 422, whitespace-only 422,
          model kwarg pass-through.
  cypher: valid read query 200, mutation 400 (keyword and mixed-case),
          LIMIT appended when absent, LIMIT preserved when present,
          empty-query 422, mutation detail does NOT echo the query string.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Minimal test client — state DB pointed at tmp so no real FS side-effects."""
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    from scripts.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


# ---------------------------------------------------------------------------
# /api/ask
# ---------------------------------------------------------------------------


class TestAsk:
    def test_ask_happy_path(self, client, monkeypatch):
        """200 with expected prose/rows/cypher from a mocked pipeline."""
        from unittest.mock import patch

        from scripts.graph.ask import AskResult  # noqa: PLC0415

        fake = AskResult(
            cypher="MATCH (p) RETURN p LIMIT 100",
            rows=[{"doi": "10.1/a"}],
            rendered="| doi |\n| --- |\n| 10.1/a |",
            prose="One paper found.",
        )
        with patch("scripts.graph.ask.run_pipeline", return_value=fake):
            r = client.post("/api/ask", json={"question": "what is X?"})

        assert r.status_code == 200
        body = r.json()
        assert body["prose"] == "One paper found."
        assert body["rows"] == [{"doi": "10.1/a"}]
        assert body["cypher"] is not None and body["cypher"].startswith("MATCH")

    def test_ask_empty_question_422(self, client):
        """Empty string should be rejected by Pydantic with 422."""
        r = client.post("/api/ask", json={"question": ""})
        assert r.status_code == 422

    def test_ask_whitespace_only_422(self, client):
        """Whitespace-only question must also be rejected."""
        r = client.post("/api/ask", json={"question": "   "})
        assert r.status_code == 422

    def test_ask_with_model_passes_through(self, client):
        """model kwarg from the request body reaches run_pipeline."""
        from unittest.mock import patch

        from scripts.graph.ask import AskResult  # noqa: PLC0415

        captured: dict = {}

        def fake_run(question: str, **kw: object) -> AskResult:
            captured.update(kw)
            return AskResult(cypher="MATCH", rows=[], rendered="_(no results)_", prose=None)

        with patch("scripts.graph.ask.run_pipeline", side_effect=fake_run):
            client.post("/api/ask", json={"question": "q", "model": "gemma2:9b"})

        assert captured.get("model") == "gemma2:9b"

    def test_ask_partial_state_still_200(self, client):
        """Pipeline failure (None cypher, empty rows) still returns 200 — best-effort."""
        from unittest.mock import patch

        from scripts.graph.ask import AskResult  # noqa: PLC0415

        fake = AskResult(
            cypher=None,
            rows=[],
            rendered="_(no results)_",
            prose="I couldn't translate your question.",
        )
        with patch("scripts.graph.ask.run_pipeline", return_value=fake):
            r = client.post("/api/ask", json={"question": "some question"})

        assert r.status_code == 200
        body = r.json()
        assert body["cypher"] is None
        assert body["rows"] == []
        assert "couldn't" in body["prose"]


# ---------------------------------------------------------------------------
# /api/cypher
# ---------------------------------------------------------------------------


class TestCypher:
    def test_cypher_valid_query_200(self, client):
        """A clean read-only query returns 200 with rows."""
        from unittest.mock import patch

        with patch(
            "scripts.server.routes.query._execute_cypher", return_value=[{"n": 1}]
        ):
            r = client.post(
                "/api/cypher", json={"query": "MATCH (p) RETURN count(p) AS n"}
            )

        assert r.status_code == 200
        assert r.json()["rows"] == [{"n": 1}]

    def test_cypher_mutation_400(self, client):
        """CREATE keyword must be rejected with 400 and fixed detail string."""
        r = client.post("/api/cypher", json={"query": "CREATE (p:Paper)"})
        assert r.status_code == 400
        assert r.json()["detail"] == "mutation rejected"

    def test_cypher_mixed_case_mutation_blocked(self, client):
        """Mixed-case DELETE must still be caught by the guard."""
        r = client.post("/api/cypher", json={"query": "match (p) DeLeTe p"})
        assert r.status_code == 400

    def test_cypher_limit_enforced_when_absent(self, client):
        """LIMIT 100 is appended when the query has no LIMIT clause."""
        from unittest.mock import patch

        captured: dict = {}

        def cap(q: str) -> list:
            captured["q"] = q
            return []

        with patch("scripts.server.routes.query._execute_cypher", side_effect=cap):
            r = client.post(
                "/api/cypher", json={"query": "MATCH (p) RETURN p"}
            )

        assert r.status_code == 200
        assert "LIMIT 100" in captured["q"]

    def test_cypher_existing_limit_preserved(self, client):
        """An existing LIMIT clause must NOT be doubled by the guard."""
        from unittest.mock import patch

        captured: dict = {}

        def cap(q: str) -> list:
            captured["q"] = q
            return []

        with patch("scripts.server.routes.query._execute_cypher", side_effect=cap):
            r = client.post(
                "/api/cypher", json={"query": "MATCH (p) RETURN p LIMIT 5"}
            )

        assert r.status_code == 200
        assert "LIMIT 5" in captured["q"]
        # Guard must not have appended a second LIMIT.
        assert "LIMIT 100" not in captured["q"]

    def test_cypher_empty_query_422(self, client):
        """Empty query string must be rejected by Pydantic with 422."""
        r = client.post("/api/cypher", json={"query": ""})
        assert r.status_code == 422

    def test_cypher_error_detail_does_not_echo_query(self, client):
        """H8 Opus check: mutation rejection detail MUST NOT echo the user's query.

        The detail field must be exactly the fixed string 'mutation rejected'
        — no query content, no exception message, no partial echo.
        """
        r = client.post("/api/cypher", json={"query": "DROP TABLE Paper"})
        assert r.status_code == 400
        body = r.json()
        # Opus reviewer requirement: no query text in the response.
        assert "DROP TABLE" not in body["detail"]
        assert body["detail"] == "mutation rejected"

    def test_cypher_merge_blocked(self, client):
        """MERGE is a mutation keyword and must be rejected."""
        r = client.post("/api/cypher", json={"query": "MERGE (p:Paper {doi: 'x'})"})
        assert r.status_code == 400
        assert r.json()["detail"] == "mutation rejected"

    def test_cypher_set_blocked(self, client):
        """SET is a mutation keyword and must be rejected."""
        r = client.post(
            "/api/cypher", json={"query": "MATCH (p) WHERE p.doi='x' SET p.title='y'"}
        )
        assert r.status_code == 400
