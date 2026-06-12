def test_record_corpus_snapshot_writes_one_row(tmp_path):
    from lit_monitor.api._stats_snapshot import record_corpus_snapshot
    from lit_monitor.core.state_db import StateDB
    db = StateDB(tmp_path / "state.db")
    record_corpus_snapshot(db, run_type="discovery", run_id="r1")  # graph absent → 0, no raise
    assert len(db.get_recent_snapshots(limit=5)) == 1

def test_record_corpus_snapshot_never_raises(tmp_path):
    from lit_monitor.api._stats_snapshot import record_corpus_snapshot
    class Boom:
        def count_with_extraction(self): raise RuntimeError("boom")
    # must NOT raise even when the state_db is broken
    record_corpus_snapshot(Boom(), run_type="discovery", run_id="x")
