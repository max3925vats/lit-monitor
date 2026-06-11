"""P10: per-paper note decoupling + obsidian sync tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestNotesSyncedColumn:
    def test_column_exists(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()]
        assert "notes_synced" in cols

    def test_defaults_to_zero(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute("INSERT INTO papers (doi) VALUES ('10.0/a')")
            conn.commit()
            row = conn.execute("SELECT notes_synced FROM papers WHERE doi='10.0/a'").fetchone()
        assert row[0] == 0


class TestSetNotesSynced:
    def test_updates_to_one(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute("INSERT INTO papers (doi) VALUES ('10.0/x')")
            conn.commit()
        db.set_notes_synced("10.0/x", 1)
        with db._connect() as conn:
            row = conn.execute("SELECT notes_synced FROM papers WHERE doi='10.0/x'").fetchone()
        assert row[0] == 1

    def test_missing_doi_is_silent(self, tmp_path):
        """UPDATE on missing DOI affects 0 rows — should not raise."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        db.set_notes_synced("10.0/missing", 1)  # must not raise

    def test_reset_to_zero(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute("INSERT INTO papers (doi, notes_synced) VALUES ('10.0/r', 1)")
            conn.commit()
        db.set_notes_synced("10.0/r", 0)
        with db._connect() as conn:
            row = conn.execute("SELECT notes_synced FROM papers WHERE doi='10.0/r'").fetchone()
        assert row[0] == 0


class TestGetNotesPending:
    def test_returns_only_unsynced_and_indexed(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            # a: embeddings_indexed=1 + unsynced → PENDING
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/a', 1, 0)"
            )
            # b: embeddings_indexed=1 + synced → DONE (skip)
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/b', 1, 1)"
            )
            # c: embeddings_indexed=0 + unsynced → NOT READY (skip)
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/c', 0, 0)"
            )
            conn.commit()
        pending = db.get_notes_pending()
        assert "10/a" in pending
        assert "10/b" not in pending
        assert "10/c" not in pending

    def test_limit_respected(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES (?, 1, 0)",
                    (f"10/{i}",),
                )
            conn.commit()
        assert len(db.get_notes_pending(limit=3)) == 3

    def test_no_limit_returns_all(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            for i in range(4):
                conn.execute(
                    "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES (?, 1, 0)",
                    (f"10/e{i}",),
                )
            conn.commit()
        assert len(db.get_notes_pending()) == 4

    def test_empty_db_returns_empty(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        assert db.get_notes_pending() == []


class TestSyncNotes:
    def test_calls_rerender_and_marks_synced(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/z', 1, 0)"
            )
            conn.commit()

        with patch("lit_monitor.obsidian_tools.sync._rerender_one") as mock:
            mock.return_value = None
            from lit_monitor.obsidian_tools.sync import sync_notes
            result = sync_notes(db, MagicMock())

        assert result["processed"] == 1
        assert result["succeeded"] == 1
        assert result["failed"] == 0
        with db._connect() as conn:
            row = conn.execute(
                "SELECT notes_synced FROM papers WHERE doi='10/z'"
            ).fetchone()
        assert row[0] == 1

    def test_per_paper_failure_continues(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/fail', 1, 0)"
            )
            conn.commit()

        with patch(
            "lit_monitor.obsidian_tools.sync._rerender_one",
            side_effect=RuntimeError("boom"),
        ):
            from lit_monitor.obsidian_tools.sync import sync_notes
            result = sync_notes(db, MagicMock())

        assert result["failed"] == 1
        assert result["succeeded"] == 0
        # notes_synced stays 0 on failure
        with db._connect() as conn:
            row = conn.execute(
                "SELECT notes_synced FROM papers WHERE doi='10/fail'"
            ).fetchone()
        assert row[0] == 0

    def test_per_paper_failure_raises_under_strict(self, tmp_path):
        """P4.4: in --strict, a per-paper sync failure escalates to a raise."""
        import lit_monitor.core.strict_mode as _sm
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/fail', 1, 0)"
            )
            conn.commit()

        _sm.set_strict(True)
        with patch(
            "lit_monitor.obsidian_tools.sync._rerender_one",
            side_effect=RuntimeError("boom"),
        ):
            from lit_monitor.obsidian_tools.sync import sync_notes
            with pytest.raises(RuntimeError, match="P10 sync: failed"):
                sync_notes(db, MagicMock())

    def test_doi_filter(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/a', 1, 0)"
            )
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/b', 1, 0)"
            )
            conn.commit()

        with patch("lit_monitor.obsidian_tools.sync._rerender_one"):
            from lit_monitor.obsidian_tools.sync import sync_notes
            result = sync_notes(db, MagicMock(), doi="10/a")

        assert result["processed"] == 1  # only the filtered DOI

    def test_idempotent_rerun_on_synced(self, tmp_path):
        """Re-running sync on already-synced papers is a no-op."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES ('10/done', 1, 1)"
            )
            conn.commit()

        with patch("lit_monitor.obsidian_tools.sync._rerender_one") as mock:
            from lit_monitor.obsidian_tools.sync import sync_notes
            # --all path uses get_notes_pending which filters out notes_synced=1
            result = sync_notes(db, MagicMock())

        assert result["processed"] == 0
        mock.assert_not_called()

    def test_limit_applied_in_bulk_sync(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES (?, 1, 0)",
                    (f"10/lim{i}",),
                )
            conn.commit()

        with patch("lit_monitor.obsidian_tools.sync._rerender_one"):
            from lit_monitor.obsidian_tools.sync import sync_notes
            result = sync_notes(db, MagicMock(), limit=2)

        assert result["processed"] == 2

    def test_multiple_failures_all_counted(self, tmp_path):
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            for i in range(3):
                conn.execute(
                    "INSERT INTO papers (doi, embeddings_indexed, notes_synced) VALUES (?, 1, 0)",
                    (f"10/mf{i}",),
                )
            conn.commit()

        with patch(
            "lit_monitor.obsidian_tools.sync._rerender_one",
            side_effect=RuntimeError("oops"),
        ):
            from lit_monitor.obsidian_tools.sync import sync_notes
            result = sync_notes(db, MagicMock())

        assert result["failed"] == 3
        assert result["succeeded"] == 0
        assert result["processed"] == 3


class TestPipelineGate:
    """P10 critical: flag default TRUE preserves existing behavior."""

    def test_flag_default_true_in_example_yaml(self):
        """Confirm extraction.example.yaml default is true."""
        from importlib.resources import files
        example = files("lit_monitor._data.config_examples") / "extraction.example.yaml"
        raw = example.read_text()
        assert "auto_write_per_paper:" in raw
        for line in raw.splitlines():
            if "auto_write_per_paper" in line:
                assert "true" in line.lower(), (
                    f"default must be true, got: {line!r}"
                )
                break


class TestCliObsidianSync:
    def test_obsidian_sync_subcommand_registered(self):
        from click.testing import CliRunner

        from lit_monitor.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["obsidian", "sync", "--help"])
        assert result.exit_code == 0
        assert "sync" in result.output.lower()

    def test_obsidian_sync_no_args_errors(self):
        """Without --all/--doi/--since, should exit non-zero."""
        from click.testing import CliRunner

        from lit_monitor.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["obsidian", "sync"])
        assert result.exit_code != 0
