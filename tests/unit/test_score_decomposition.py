"""Bundle B: score decomposition tests.

Covers:
- rank_papers() populates score_breakdown on every output paper
- score_breakdown values are floats rounded to 3 decimals
- breakdown keys: vector (always), domain_context (0.0 when disabled, > 0 when enabled)
- sum(breakdown.values()) ≈ total similarity_score
- digest markdown table opt-in (show_in_md flag)
- digest markdown table format (Signal | Score columns + total row)
- state_db additive migration (score_breakdown_json column)
- add_discovery_paper accepts and persists score_breakdown_json
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(score: float = 0.5) -> MagicMock:
    db = MagicMock()
    db.find_similar_to_text.return_value = [
        {"score": score, "id": "y", "document": "", "metadata": {}}
    ]
    return db


def _make_llm(doi: str = "10.1/a") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = json.dumps({doi: "relevant paper"})
    return llm


def _make_candidate(
    doi: str = "10.1/a",
    score: float = 0.5,
    has_embedding: bool = True,
) -> dict[str, Any]:
    cand: dict[str, Any] = {
        "doi": doi,
        "title": f"Paper {doi}",
        "abstract": "Abstract text.",
    }
    if has_embedding:
        cand["_embedding"] = np.array([score, 0.0, 0.0], dtype=np.float32)
    return cand


# ===========================================================================
# TestScoreBreakdownInRanker
# ===========================================================================


@pytest.mark.unit
class TestScoreBreakdownInRanker:
    """rank_papers() must attach score_breakdown to every output paper."""

    def test_breakdown_present_on_every_paper(self):
        """rank_papers() adds score_breakdown dict to every output paper."""
        from scripts.llm.ranker import rank_papers

        candidates = [
            _make_candidate("10.1/a", 0.7),
            _make_candidate("10.1/b", 0.3),
        ]
        db = MagicMock()
        db.find_similar_to_text.side_effect = [
            [{"score": 0.7, "id": "x", "document": "", "metadata": {}}],
            [{"score": 0.3, "id": "y", "document": "", "metadata": {}}],
        ]
        llm = MagicMock()
        llm.complete.return_value = '{"10.1/a": "r", "10.1/b": "r"}'

        ranked = rank_papers(candidates, db, llm)

        for paper in ranked:
            assert "score_breakdown" in paper, (
                f"score_breakdown missing from paper {paper.get('doi')}"
            )
            assert isinstance(paper["score_breakdown"], dict)

    def test_breakdown_values_rounded_to_3_decimals(self):
        """All values in score_breakdown are floats rounded to 3 decimal places."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.123456)]
        db = _make_db(0.123456)
        llm = _make_llm("10.1/a")

        ranked = rank_papers(candidates, db, llm)
        breakdown = ranked[0]["score_breakdown"]

        for key, val in breakdown.items():
            assert isinstance(val, float), f"breakdown[{key!r}] is not float: {val!r}"
            # Check <= 3 decimal places by round-trip
            assert round(val, 3) == val, (
                f"breakdown[{key!r}] = {val} not rounded to 3 decimals"
            )

    def test_breakdown_sums_approximately_to_total_score(self):
        """sum(breakdown.values()) ≈ similarity_score (within rounding tolerance)."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.6)]
        db = _make_db(0.6)
        llm = _make_llm("10.1/a")

        ranked = rank_papers(candidates, db, llm)
        paper = ranked[0]
        total = paper["similarity_score"]
        decomposed_sum = sum(paper["score_breakdown"].values())

        # Allow 0.01 tolerance due to 3-decimal rounding of individual signals
        assert abs(decomposed_sum - total) <= 0.01, (
            f"breakdown sum {decomposed_sum} too far from total {total}"
        )

    def test_breakdown_includes_vector_signal(self):
        """score_breakdown always has a 'vector' key."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.8)]
        db = _make_db(0.8)
        llm = _make_llm("10.1/a")

        ranked = rank_papers(candidates, db, llm)
        assert "vector" in ranked[0]["score_breakdown"]

    def test_breakdown_vector_value_matches_similarity_score_when_no_domain(self):
        """Without domain context, breakdown['vector'] == similarity_score."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a")]
        db = _make_db(0.75)
        llm = _make_llm("10.1/a")

        ranked = rank_papers(candidates, db, llm)
        paper = ranked[0]

        # When no domain_context_emb, vector is the full score
        assert paper["score_breakdown"]["vector"] == round(0.75, 3)

    def test_breakdown_includes_domain_context_signal_when_enabled(self):
        """domain_context key is present and > 0.0 when domain_context_emb + weight provided."""
        from scripts.llm.ranker import rank_papers

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [_make_candidate("10.1/a", 1.0)]  # embedding aligned with domain
        db = _make_db(0.5)
        llm = _make_llm("10.1/a")

        ranked = rank_papers(
            candidates,
            db,
            llm,
            domain_context_emb=domain_emb,
            domain_context_weight=0.3,
        )
        breakdown = ranked[0]["score_breakdown"]
        assert "domain_context" in breakdown
        assert breakdown["domain_context"] > 0.0

    def test_breakdown_domain_zero_when_disabled(self):
        """domain_context is 0.0 when weight=0.0 (disabled)."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_db(0.5)
        llm = _make_llm("10.1/a")

        ranked = rank_papers(
            candidates,
            db,
            llm,
            domain_context_emb=None,
            domain_context_weight=0.0,
        )
        breakdown = ranked[0]["score_breakdown"]
        assert "domain_context" in breakdown
        assert breakdown["domain_context"] == 0.0

    def test_breakdown_domain_zero_when_no_embedding_on_candidate(self):
        """domain_context = 0.0 when candidate has no _embedding (no crash)."""
        from scripts.llm.ranker import rank_papers

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [_make_candidate("10.1/noemb", has_embedding=False)]
        db = _make_db(0.6)
        llm = _make_llm("10.1/noemb")

        ranked = rank_papers(
            candidates,
            db,
            llm,
            domain_context_emb=domain_emb,
            domain_context_weight=0.3,
        )
        breakdown = ranked[0]["score_breakdown"]
        assert breakdown["domain_context"] == 0.0

    def test_breakdown_json_serializable(self):
        """score_breakdown can be serialized to JSON without error."""
        from scripts.llm.ranker import rank_papers

        candidates = [_make_candidate("10.1/a", 0.5)]
        db = _make_db(0.5)
        llm = _make_llm("10.1/a")

        ranked = rank_papers(candidates, db, llm)
        # Must not raise
        serialized = json.dumps(ranked[0]["score_breakdown"])
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

    def test_empty_candidates_returns_empty(self):
        """rank_papers([]) returns [] even with Bundle B present."""
        from scripts.llm.ranker import rank_papers

        result = rank_papers([], MagicMock(), MagicMock())
        assert result == []


# ===========================================================================
# TestDigestMarkdownDecomposition
# ===========================================================================


@pytest.mark.unit
class TestDigestMarkdownDecomposition:
    """render_digest() markdown table behaviour controlled by show_in_md flag."""

    def _make_paper(self, doi: str = "10.1/a", score: float = 0.7) -> dict[str, Any]:
        return {
            "doi": doi,
            "title": "Test Paper",
            "authors": ["A. Author"],
            "year": 2024,
            "similarity_score": score,
            "llm_rationale": "Relevant.",
            "score_breakdown": {
                "vector": round(score * 0.8, 3),
                "domain_context": round(score * 0.2, 3),
            },
        }

    def test_show_in_md_false_no_table(self):
        """show_in_md=False → no breakdown table in the rendered Markdown."""
        from scripts.output.digest_renderer import render_digest

        run = {"run_date": "2026-05-31"}
        papers = [self._make_paper()]

        md = render_digest(
            run,
            papers,
            show_score_decomposition=False,
            _today="2026-05-31",
        )

        # No breakdown table heading should appear
        assert "| Signal |" not in md
        assert "| vector |" not in md

    def test_show_in_md_default_is_false(self):
        """Without show_score_decomposition kwarg, no table appears (default off)."""
        from scripts.output.digest_renderer import render_digest

        run = {"run_date": "2026-05-31"}
        papers = [self._make_paper()]

        md = render_digest(run, papers, _today="2026-05-31")

        assert "| Signal |" not in md

    def test_show_in_md_true_renders_table(self):
        """show_in_md=True → breakdown table present per paper."""
        from scripts.output.digest_renderer import render_digest

        run = {"run_date": "2026-05-31"}
        papers = [self._make_paper()]

        md = render_digest(
            run,
            papers,
            show_score_decomposition=True,
            _today="2026-05-31",
        )

        assert "| Signal |" in md
        assert "| vector |" in md

    def test_table_has_score_column(self):
        """Breakdown table has 'Score' column header."""
        from scripts.output.digest_renderer import render_digest

        run = {"run_date": "2026-05-31"}
        papers = [self._make_paper()]

        md = render_digest(
            run,
            papers,
            show_score_decomposition=True,
            _today="2026-05-31",
        )

        assert "| Score |" in md or "| Score" in md

    def test_table_has_total_row(self):
        """Breakdown table includes a bold **total** row."""
        from scripts.output.digest_renderer import render_digest

        run = {"run_date": "2026-05-31"}
        papers = [self._make_paper()]

        md = render_digest(
            run,
            papers,
            show_score_decomposition=True,
            _today="2026-05-31",
        )

        assert "**total**" in md

    def test_table_domain_context_signal_present(self):
        """domain_context signal row appears in the table when breakdown has it."""
        from scripts.output.digest_renderer import render_digest

        run = {"run_date": "2026-05-31"}
        papers = [self._make_paper()]

        md = render_digest(
            run,
            papers,
            show_score_decomposition=True,
            _today="2026-05-31",
        )

        assert "domain_context" in md

    def test_no_breakdown_key_no_crash(self):
        """Papers without score_breakdown key: show_in_md=True renders gracefully (no table)."""
        from scripts.output.digest_renderer import render_digest

        run = {"run_date": "2026-05-31"}
        papers = [{"doi": "10.1/a", "title": "T", "similarity_score": 0.5}]

        # Must not raise even with show_score_decomposition=True
        md = render_digest(
            run,
            papers,
            show_score_decomposition=True,
            _today="2026-05-31",
        )
        assert isinstance(md, str)


# ===========================================================================
# TestStateDBAugmentedMigration
# ===========================================================================


@pytest.mark.unit
class TestStateDBAugmentedMigration:
    """Additive migration: score_breakdown_json column in discovery_paper_results."""

    def test_column_added_on_fresh_db(self, tmp_path):
        """Fresh StateDB has score_breakdown_json column on discovery_paper_results."""
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            rows = conn.execute(
                "PRAGMA table_info(discovery_paper_results)"
            ).fetchall()
        col_names = [r[1] for r in rows]
        assert "score_breakdown_json" in col_names

    def test_add_discovery_paper_persists_breakdown(self, tmp_path):
        """add_discovery_paper with score_breakdown persists JSON to the column."""
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run(run_params=None)
        breakdown = {"vector": 0.7, "domain_context": 0.2}

        db.add_discovery_paper(
            run_id=run_id,
            doi="10.1/test",
            title="Test Paper",
            score=0.9,
            rationale="Relevant.",
            ingested=False,
            score_breakdown=breakdown,
        )

        with db._connect() as conn:
            row = conn.execute(
                "SELECT score_breakdown_json FROM discovery_paper_results "
                "WHERE doi = ? LIMIT 1",
                ("10.1/test",),
            ).fetchone()

        assert row is not None
        stored = json.loads(row[0])
        assert stored["vector"] == 0.7
        assert stored["domain_context"] == 0.2

    def test_add_discovery_paper_no_breakdown_stores_null(self, tmp_path):
        """add_discovery_paper without score_breakdown stores NULL (backward compat)."""
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run(run_params=None)

        # No score_breakdown kwarg — backward-compatible call
        db.add_discovery_paper(
            run_id=run_id,
            doi="10.1/nobreakdown",
            title="Old Paper",
            score=0.5,
            rationale="",
            ingested=False,
        )

        with db._connect() as conn:
            row = conn.execute(
                "SELECT score_breakdown_json FROM discovery_paper_results "
                "WHERE doi = ? LIMIT 1",
                ("10.1/nobreakdown",),
            ).fetchone()

        assert row is not None
        assert row[0] is None  # NULL stored when no breakdown provided


# ===========================================================================
# TestWebUIPanel
# ===========================================================================


@pytest.mark.unit
class TestWebUIPanel:
    """Bundle B: /discovery/{run_id} page renders the 'Why this paper?' panel."""

    @pytest.fixture()
    def _seeded_db_with_breakdown(self, tmp_path):
        """StateDB with one run + one paper that has score_breakdown_json."""
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state_webui.db")
        run_id = db.start_discovery_run(run_params=None)
        db.add_discovery_paper(
            run_id=run_id,
            doi="10.0/panel",
            title="Panel Paper",
            score=0.88,
            rationale="Test rationale.",
            ingested=False,
            score_breakdown={"vector": 0.7, "domain_context": 0.18},
        )
        db.finish_discovery_run(run_id, status="success", total_found=1, total_ingested=0)
        return db, run_id

    def test_why_panel_present_when_breakdown_stored(self, _seeded_db_with_breakdown):
        """GET /discovery/{run_id} includes 'Why this paper?' when breakdown stored."""
        from fastapi.testclient import TestClient

        from scripts.server.app import create_app
        from scripts.server.runtime import reset_runtime

        reset_runtime()
        db, run_id = _seeded_db_with_breakdown
        rt = MagicMock()
        rt.state_db = db

        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            client = TestClient(create_app())
            r = client.get(f"/discovery/{run_id}")

        assert r.status_code == 200
        assert "Why this paper?" in r.text

    def test_why_panel_uses_details_element(self, _seeded_db_with_breakdown):
        """Panel uses a native <details> element (no JS framework)."""
        from fastapi.testclient import TestClient

        from scripts.server.app import create_app
        from scripts.server.runtime import reset_runtime

        reset_runtime()
        db, run_id = _seeded_db_with_breakdown
        rt = MagicMock()
        rt.state_db = db

        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            client = TestClient(create_app())
            r = client.get(f"/discovery/{run_id}")

        assert r.status_code == 200
        html = r.text.lower()
        assert "<details" in html

    def test_no_panel_when_no_breakdown(self, tmp_path):
        """Paper without score_breakdown_json: no 'Why this paper?' panel shown."""
        from fastapi.testclient import TestClient

        from scripts.core.state_db import StateDB
        from scripts.server.app import create_app
        from scripts.server.runtime import reset_runtime

        reset_runtime()
        db = StateDB(tmp_path / "state_nobreakdown.db")
        run_id = db.start_discovery_run(run_params=None)
        # Old-style insert without score_breakdown
        db.add_discovery_paper(
            run_id=run_id,
            doi="10.0/nobreak",
            title="No Breakdown Paper",
            score=0.5,
            rationale="",
            ingested=False,
        )
        db.finish_discovery_run(run_id, status="success", total_found=1, total_ingested=0)

        rt = MagicMock()
        rt.state_db = db

        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            client = TestClient(create_app())
            r = client.get(f"/discovery/{run_id}")

        assert r.status_code == 200
        assert "Why this paper?" not in r.text
