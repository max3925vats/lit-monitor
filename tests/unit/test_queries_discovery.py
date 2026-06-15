"""Task 6: unit tests for get_discovery_run_conversion query.

Exercises the recommendation→ingestion conversion counts that back the
run-detail headline (M recommendations · X since ingested).
"""
from __future__ import annotations


def test_get_discovery_run_conversion_counts(tmp_path):
    """Three recommendations, two later ingested → {recommendations:3, ingested:2}."""
    from lit_monitor.api.queries import get_discovery_run_conversion
    from lit_monitor.core.state_db import StateDB

    db = StateDB(db_path=tmp_path / "s.db")
    run_id = db.start_discovery_run({}, trigger="manual")
    for i in range(3):
        db.add_discovery_paper(
            run_id=run_id,
            doi=f"10.1/{i}",
            title="t",
            score=0.5,
            rationale="",
            ingested=False,
        )
    db.mark_discovery_recommendations_ingested("10.1/0")
    db.mark_discovery_recommendations_ingested("10.1/1")
    assert get_discovery_run_conversion(db, run_id) == {"recommendations": 3, "ingested": 2}


def test_get_discovery_run_conversion_zero(tmp_path):
    """No papers → {recommendations:0, ingested:0}."""
    from lit_monitor.api.queries import get_discovery_run_conversion
    from lit_monitor.core.state_db import StateDB

    db = StateDB(db_path=tmp_path / "s.db")
    run_id = db.start_discovery_run({})
    assert get_discovery_run_conversion(db, run_id) == {"recommendations": 0, "ingested": 0}


def test_get_discovery_run_conversion_all_ingested(tmp_path):
    """All papers ingested at add time → recommendations==ingested."""
    from lit_monitor.api.queries import get_discovery_run_conversion
    from lit_monitor.core.state_db import StateDB

    db = StateDB(db_path=tmp_path / "s.db")
    run_id = db.start_discovery_run({})
    for i in range(2):
        db.add_discovery_paper(
            run_id=run_id,
            doi=f"10.2/{i}",
            title="t",
            score=0.7,
            rationale="",
            ingested=True,  # ingested at write time
        )
    result = get_discovery_run_conversion(db, run_id)
    assert result == {"recommendations": 2, "ingested": 2}


def test_get_discovery_run_conversion_wrong_run(tmp_path):
    """Papers from run A don't bleed into counts for run B."""
    from lit_monitor.api.queries import get_discovery_run_conversion
    from lit_monitor.core.state_db import StateDB

    db = StateDB(db_path=tmp_path / "s.db")
    run_a = db.start_discovery_run({})
    run_b = db.start_discovery_run({})
    db.add_discovery_paper(run_a, doi="10.3/a", title="t", score=0.5, rationale="", ingested=True)
    # run_b has no papers
    assert get_discovery_run_conversion(db, run_b) == {"recommendations": 0, "ingested": 0}
