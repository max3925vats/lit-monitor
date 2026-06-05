"""
Unit tests for Bundle J2: `lit-monitor learning` CLI commands.

Tests cover:
- learning recompute: synthetic feedback + fake embeddings → store_interest_vector
  called with the right n_events.
- learning recompute: 0 events → inert message, no crash, no store.
- learning recompute: 0 library embeddings → graceful message, no store.
- learning view: with and without a stored vector.

All embeddings / DB access is faked — NO live Ollama or ChromaDB.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config():
    cfg = MagicMock()
    cfg.state_db.path = "/tmp/test_state.db"
    return cfg


def _feedback_event(doi: str, signal_type: str, weight: float) -> dict:
    """A minimal feedback_events row dict (matches list_feedback_events shape)."""
    return {
        "id": 1,
        "doi": doi,
        "signal_type": signal_type,
        "weight": weight,
        "rating": None,
        "source": None,
        "created_at": "2026-06-01 12:00:00",
    }


def _patches(state_db: MagicMock, embeddings_db: MagicMock):
    """Patch the three CLI factory helpers used by the learning commands."""
    return (
        patch("scripts.cli._make_config", return_value=_mock_config()),
        patch("scripts.cli._make_state_db", return_value=state_db),
        patch("scripts.cli._make_embeddings_db", return_value=embeddings_db),
    )


# ---------------------------------------------------------------------------
# learning recompute
# ---------------------------------------------------------------------------


class TestLearningRecomputeCmd:
    def test_recompute_stores_vector_with_right_n_events(self):
        """≥10 saved events with embeddings → store_interest_vector called with
        n_events == number of contributing events and a finite vector."""
        from scripts.cli import main

        # 12 saved (positive) events on distinct DOIs, all with stored embeddings.
        n = 12
        events = [
            _feedback_event(f"10.1/p{i}", "saved", 1.0) for i in range(n)
        ]
        # A small synthetic embedding library: each event DOI has a 3-d vector
        # pointing roughly toward [1,0,0]; plus a couple of extra library papers.
        embeddings = {
            f"10.1/p{i}": np.array([1.0, 0.05 * i, 0.0], dtype=np.float32)
            for i in range(n)
        }
        embeddings["10.1/extra1"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        state_db = MagicMock()
        state_db.list_feedback_events.return_value = events
        embeddings_db = MagicMock()
        embeddings_db.get_paper_embeddings.return_value = embeddings

        p_cfg, p_sdb, p_edb = _patches(state_db, embeddings_db)
        with p_cfg, p_sdb, p_edb:
            result = CliRunner().invoke(main, ["learning", "recompute"])

        assert result.exit_code == 0, result.output
        state_db.store_interest_vector.assert_called_once()
        args, _ = state_db.store_interest_vector.call_args
        scope, vector, n_events = args
        assert scope == "global"
        assert n_events == n, f"expected {n} contributing events, got {n_events}"
        # Vector must be finite (no NaN reached the store).
        assert np.isfinite(np.asarray(vector)).all()
        # 12 ≥ cold-start floor → not inert.
        assert "INERT" not in result.output
        assert "soft_gate" in result.output

    def test_recompute_zero_events_is_inert_no_store(self):
        """No feedback events → inert message, store_interest_vector NOT called."""
        from scripts.cli import main

        state_db = MagicMock()
        state_db.list_feedback_events.return_value = []
        embeddings_db = MagicMock()

        p_cfg, p_sdb, p_edb = _patches(state_db, embeddings_db)
        with p_cfg, p_sdb, p_edb:
            result = CliRunner().invoke(main, ["learning", "recompute"])

        assert result.exit_code == 0, result.output
        state_db.store_interest_vector.assert_not_called()
        assert "No feedback events" in result.output

    def test_recompute_below_floor_marks_inert(self):
        """<10 contributing events → vector stored but flagged INERT."""
        from scripts.cli import main

        events = [_feedback_event(f"10.1/p{i}", "saved", 1.0) for i in range(4)]
        embeddings = {
            f"10.1/p{i}": np.array([1.0, 0.0, 0.0], dtype=np.float32)
            for i in range(4)
        }
        state_db = MagicMock()
        state_db.list_feedback_events.return_value = events
        embeddings_db = MagicMock()
        embeddings_db.get_paper_embeddings.return_value = embeddings

        p_cfg, p_sdb, p_edb = _patches(state_db, embeddings_db)
        with p_cfg, p_sdb, p_edb:
            result = CliRunner().invoke(main, ["learning", "recompute"])

        assert result.exit_code == 0, result.output
        # Vector IS stored (J1 contract: compute+store always), but inert.
        state_db.store_interest_vector.assert_called_once()
        _, _, n_events = state_db.store_interest_vector.call_args[0]
        assert n_events == 4
        assert "INERT" in result.output

    def test_recompute_no_embeddings_graceful(self):
        """Events present but no library embeddings → graceful message, no store."""
        from scripts.cli import main

        events = [_feedback_event("10.1/p0", "saved", 1.0)]
        state_db = MagicMock()
        state_db.list_feedback_events.return_value = events
        embeddings_db = MagicMock()
        embeddings_db.get_paper_embeddings.return_value = {}

        p_cfg, p_sdb, p_edb = _patches(state_db, embeddings_db)
        with p_cfg, p_sdb, p_edb:
            result = CliRunner().invoke(main, ["learning", "recompute"])

        assert result.exit_code == 0, result.output
        state_db.store_interest_vector.assert_not_called()
        assert "No paper embeddings" in result.output


# ---------------------------------------------------------------------------
# learning view
# ---------------------------------------------------------------------------


class TestLearningViewCmd:
    def test_view_with_stored_vector(self):
        """view prints n_events, soft_gate, dim, and a top-K alignment list."""
        from scripts.cli import main

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state_db = MagicMock()
        state_db.get_interest_vector.return_value = (vec, 15)
        # _connect() context-manager returns a fake conn with computed_at row.
        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchone.return_value = {
            "computed_at": "2026-06-04 09:00:00"
        }
        state_db._connect.return_value.__enter__.return_value = fake_conn
        state_db.get_paper.return_value = {"title": "Aligned paper"}

        embeddings_db = MagicMock()
        embeddings_db.get_paper_embeddings.return_value = {
            "10.1/aligned": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "10.1/orth": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        p_cfg, p_sdb, p_edb = _patches(state_db, embeddings_db)
        with p_cfg, p_sdb, p_edb:
            result = CliRunner().invoke(main, ["learning", "view"])

        assert result.exit_code == 0, result.output
        assert "15" in result.output            # n_events
        assert "soft_gate" in result.output
        assert "dimensionality" in result.output
        # The aligned paper (cos≈1) should rank first in the alignment list.
        assert "Aligned paper" in result.output

    def test_view_without_stored_vector(self):
        """view with nothing stored → recompute hint, exit 0."""
        from scripts.cli import main

        state_db = MagicMock()
        state_db.get_interest_vector.return_value = None
        embeddings_db = MagicMock()

        p_cfg, p_sdb, p_edb = _patches(state_db, embeddings_db)
        with p_cfg, p_sdb, p_edb:
            result = CliRunner().invoke(main, ["learning", "view"])

        assert result.exit_code == 0, result.output
        assert "No interest vector stored" in result.output
