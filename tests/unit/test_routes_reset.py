"""Unit tests for RR4: reset routes (status + per-component reset POSTs).

TDD: tests are written first. Stash-and-rerun verification is documented in
the bundle definition-of-done section.

Config-isolated by conftest.py autouse fixtures (config absent by default).
Routes that need config/runtime inject fake runtimes via patch.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app
from lit_monitor.server.runtime import reset_runtime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_runtime():
    """Reset the runtime singleton before and after each test."""
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# 1. First-run-safe status
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_status_first_run_safe(client):
    """GET /api/reset/status returns 200 with configured=False when config absent."""
    resp = client.get("/api/reset/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is False
    # All component metrics should be empty/zeroed (new shape: count/unit/size_bytes)
    for comp in ("vectors", "graph", "notes"):
        assert comp in data["components"]
        assert data["components"][comp]["present"] is False
        assert data["components"][comp]["count"] == 0
        assert data["components"][comp]["size_bytes"] == 0
    # enrichment and busy must always be present
    assert "enrichment" in data
    assert "busy" in data


# ---------------------------------------------------------------------------
# 2. Page endpoint first-run safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_page_first_run_safe(client):
    """GET /setup/reset returns 200 even when no config is present."""
    resp = client.get("/setup/reset")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. "everything" reset rejects bad confirmation phrase
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_everything_reset_rejects_bad_phrase(client):
    """POST /api/reset/everything with wrong phrase → 400; perform_state_reset not called."""
    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        resp = client.post("/api/reset/everything", data={"confirm": "wrong phrase"})
    assert resp.status_code == 400
    mock_reset.assert_not_called()


@pytest.mark.unit
def test_everything_reset_rejects_missing_phrase(client):
    """POST /api/reset/everything with no confirm field → 400; nothing deleted."""
    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        resp = client.post("/api/reset/everything", data={})
    assert resp.status_code == 400
    mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Busy-guard blocks reset when pipeline running
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_blocked_when_pipeline_running(client):
    """POST /api/reset/vectors → 409 when a pipeline slot is running."""
    # Build a fake runtime where brain_build is running
    fake_runtime = MagicMock()
    brain_slot = MagicMock()
    brain_slot.is_running.return_value = True
    # Other slots idle
    idle_slot = MagicMock()
    idle_slot.is_running.return_value = False
    idle_slot.sequence_active = False
    fake_runtime.processes = {
        "brain_build": brain_slot,
        "discovery": idle_slot,
        "vocabulary": idle_slot,
        "rebuild_vectors": idle_slot,
        "rebuild_graph": idle_slot,
        "rebuild_notes": idle_slot,
    }

    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
            resp = client.post("/api/reset/vectors")

    assert resp.status_code == 409
    data = resp.json()
    assert "busy" in data
    mock_reset.assert_not_called()


@pytest.mark.unit
def test_reset_blocked_when_rebuild_running(client):
    """POST /api/reset/graph → 409 when a rebuild slot has sequence_active."""
    fake_runtime = MagicMock()
    idle_proc = MagicMock()
    idle_proc.is_running.return_value = False
    busy_rebuild = MagicMock()
    busy_rebuild.is_running.return_value = False
    busy_rebuild.sequence_active = True  # between commands in a sequence

    fake_runtime.processes = {
        "brain_build": idle_proc,
        "discovery": idle_proc,
        "vocabulary": idle_proc,
        "rebuild_vectors": idle_proc,
        "rebuild_graph": busy_rebuild,
        "rebuild_notes": idle_proc,
    }

    with patch("lit_monitor.server.routes.reset.perform_state_reset") as mock_reset:
        with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
            resp = client.post("/api/reset/graph")

    assert resp.status_code == 409
    mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# 5. vectors reset happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vectors_reset_happy_path(tmp_path):
    """POST /api/reset/vectors deletes chroma dir, calls reset_embeddings_indexed,
    and calls reset_runtime.
    """
    # Create a fake chroma directory to simulate an existing target
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "dummy.bin").write_bytes(b"data")

    # Build a fake config pointing at this temp chroma dir
    fake_config = MagicMock()
    state_db_path = tmp_path / "state.db"
    fake_config.state_db.path = str(state_db_path)
    # obsidian — needed for vault_targets not to blow up but not used in vectors reset
    fake_config.obsidian.vault_path = str(tmp_path / "vault")

    # Build a fake state_db with reset_embeddings_indexed spy
    fake_state_db = MagicMock()

    # Build a fake runtime with all slots idle
    idle_slot = MagicMock()
    idle_slot.is_running.return_value = False
    idle_slot.sequence_active = False

    fake_runtime = MagicMock()
    fake_runtime.config = fake_config
    fake_runtime.state_db = fake_state_db
    fake_runtime.processes = {
        "brain_build": idle_slot,
        "discovery": idle_slot,
        "vocabulary": idle_slot,
        "rebuild_vectors": idle_slot,
        "rebuild_graph": idle_slot,
        "rebuild_notes": idle_slot,
    }

    client = TestClient(create_app())

    with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
        with patch("lit_monitor.server.routes.reset.reset_runtime") as mock_reset_runtime:
            resp = client.post("/api/reset/vectors")

    assert resp.status_code == 200, resp.text

    # Chroma dir should be gone
    assert not chroma_dir.exists()

    # reset_embeddings_indexed should have been called
    fake_state_db.reset_embeddings_indexed.assert_called_once()

    # reset_runtime must be called to reconnect fresh DBs
    mock_reset_runtime.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Unknown component → 404
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_unknown_component_returns_404(client):
    """POST /api/reset/bogus → 404."""
    resp = client.post("/api/reset/bogus")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# RR5 — Rebuild endpoints
# ---------------------------------------------------------------------------

def _make_idle_runtime():
    """Return a fake runtime with all slots idle (no rebuild/pipeline running)."""
    idle = MagicMock()
    idle.is_running.return_value = False
    idle.sequence_active = False
    runtime = MagicMock()
    runtime.processes = {
        "brain_build": idle,
        "discovery": idle,
        "vocabulary": idle,
        "rebuild_vectors": idle,
        "rebuild_graph": idle,
        "rebuild_notes": idle,
    }
    return runtime


@pytest.mark.unit
def test_rebuild_graph_enrich_blocked_without_capability(client):
    """POST /api/rebuild/graph with enrich=true → 400 when capability is missing.

    _schedule_rebuild must NOT be called.
    """
    fake_runtime = _make_idle_runtime()
    schedule_mock = MagicMock()

    with (
        patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime),
        patch(
            "lit_monitor.server.routes.reset.enrichment_capability",
            return_value={"nlp": False, "ollama_key": False},
        ),
        patch("lit_monitor.server.routes.reset._schedule_rebuild", schedule_mock),
    ):
        resp = client.post("/api/rebuild/graph", data={"enrich": "true"})

    assert resp.status_code == 400
    data = resp.json()
    assert "enrichment unavailable" in data["error"]
    assert "capability" in data
    schedule_mock.assert_not_called()


@pytest.mark.unit
def test_rebuild_blocked_when_busy(client):
    """POST /api/rebuild/vectors → 409 when any pipeline slot is running.

    _schedule_rebuild must NOT be called.
    """
    # brain_build is running
    busy_slot = MagicMock()
    busy_slot.is_running.return_value = True
    idle = MagicMock()
    idle.is_running.return_value = False
    idle.sequence_active = False

    fake_runtime = MagicMock()
    fake_runtime.processes = {
        "brain_build": busy_slot,
        "discovery": idle,
        "vocabulary": idle,
        "rebuild_vectors": idle,
        "rebuild_graph": idle,
        "rebuild_notes": idle,
    }

    schedule_mock = MagicMock()

    with (
        patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime),
        patch("lit_monitor.server.routes.reset._schedule_rebuild", schedule_mock),
    ):
        resp = client.post("/api/rebuild/vectors")

    assert resp.status_code == 409
    schedule_mock.assert_not_called()


@pytest.mark.unit
def test_rebuild_idle_schedules(client):
    """POST /api/rebuild/notes → 200 and _schedule_rebuild called once with correct argvs."""
    fake_runtime = _make_idle_runtime()
    # Give the notes slot a real output list so SSE could follow it
    fake_runtime.processes["rebuild_notes"].output = []

    schedule_mock = MagicMock()

    with (
        patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime),
        patch("lit_monitor.server.routes.reset._schedule_rebuild", schedule_mock),
    ):
        resp = client.post("/api/rebuild/notes")

    assert resp.status_code == 200
    data = resp.json()
    assert data["started"] is True
    assert data["component"] == "notes"

    schedule_mock.assert_called_once()
    # Verify the slot and argvs passed to _schedule_rebuild
    call_args = schedule_mock.call_args
    slot_arg, argvs_arg = call_args[0]
    assert argvs_arg == [["lit-monitor", "obsidian", "rerender"]]


@pytest.mark.unit
def test_rebuild_graph_base_schedules_without_capability(client):
    """POST /api/rebuild/graph (no enrich) → 200 even if capability is absent.

    The base graph rebuild is NOT capability-gated.
    """
    fake_runtime = _make_idle_runtime()
    schedule_mock = MagicMock()

    with (
        patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime),
        patch(
            "lit_monitor.server.routes.reset.enrichment_capability",
            return_value={"nlp": False, "ollama_key": False},
        ),
        patch("lit_monitor.server.routes.reset._schedule_rebuild", schedule_mock),
    ):
        resp = client.post("/api/rebuild/graph")  # no enrich field

    assert resp.status_code == 200
    schedule_mock.assert_called_once()
    _, argvs_arg = schedule_mock.call_args[0]
    assert argvs_arg == [["lit-monitor", "graph", "backfill", "--all"]]


@pytest.mark.unit
def test_rebuild_unknown_component_404(client):
    """POST /api/rebuild/bogus → 404."""
    resp = client.post("/api/rebuild/bogus")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# RR5-review — regression tests for the monotonic-counter / front-drop fix
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_process_slot_total_appended_monotonic():
    """total_appended keeps climbing even when the bounded buffer drops front lines.

    Specifically: after cap+150 appends the counter must equal cap+150 while the
    buffer holds at most ``output_cap`` lines.  The first line ("L0") must have
    been evicted from the front of the buffer.
    """
    from lit_monitor.server.runtime import ProcessSlot

    slot = ProcessSlot()
    cap = slot.output_cap  # 1000
    n = cap + 150

    for i in range(n):
        slot.append_line(f"L{i}")

    assert slot.total_appended == n, (
        f"total_appended should be {n}, got {slot.total_appended}"
    )
    assert len(slot.output) <= cap, (
        f"buffer should be capped at {cap}, got {len(slot.output)}"
    )
    # The front was dropped — "L0" must no longer be in the buffer.
    assert slot.output[0] != "L0", "L0 should have been evicted from the buffer front"
    # The last appended line must still be present.
    assert slot.output[-1] == f"L{n - 1}", "last line must be in the buffer"


@pytest.mark.unit
def test_new_lines_helper_no_gap_after_front_drop():
    """_new_lines emits ALL buffered lines with no gap or IndexError after a front-drop.

    Simulates: total_appended=1100, buffer holds 901 lines (indices 199..1099).
    A follower that last emitted idx=199 must get all 901 lines in one pass,
    ending at idx=1100, with no gap.
    """
    from lit_monitor.server.routes.reset import _new_lines

    total = 1100
    buf_len = 901  # total - buf_len = 199 = base (first buffered line has absolute index 199)
    # Simulate slot.output: lines for absolute indices 199..1099
    out = [f"L{i}" for i in range(199, 1100)]
    assert len(out) == buf_len

    # Follower arrives claiming it has emitted up to idx=199 (base exactly).
    lines, new_idx = _new_lines(out, total, 199)

    assert len(lines) == buf_len, (
        f"expected {buf_len} lines, got {len(lines)}"
    )
    assert new_idx == total, f"new_idx should be {total}, got {new_idx}"
    assert lines[0] == "L199", f"first emitted line should be 'L199', got {lines[0]!r}"
    assert lines[-1] == "L1099", f"last emitted line should be 'L1099', got {lines[-1]!r}"


@pytest.mark.unit
def test_new_lines_helper_clamps_idx_below_base():
    """_new_lines clamps idx forward to base when the follower fell behind a front-drop.

    Simulates a follower stuck at idx=50 while the buffer has already dropped
    the first 199 lines (base=199).  The follower must NOT skip lines 199..1099.
    """
    from lit_monitor.server.routes.reset import _new_lines

    total = 1100
    out = [f"L{i}" for i in range(199, 1100)]  # base=199

    # Follower is stuck at idx=50 (below base).
    lines, new_idx = _new_lines(out, total, 50)

    # It must emit all 901 buffered lines (no lines were skipped).
    assert len(lines) == 901
    assert new_idx == total
    assert lines[0] == "L199"


@pytest.mark.unit
def test_new_lines_old_logic_would_skip_lines():
    """Demonstrate that the OLD min()-clamp logic silently skips lines.

    This test encodes the OLD buggy logic and asserts it produces fewer lines
    than expected — proving the regression would have been caught.
    """
    # base=199 (absolute index of out[0] = total_appended - len(out) = 1100 - 901)
    out = [f"L{i}" for i in range(199, 1100)]  # len=901

    # OLD logic: idx = min(idx, len(out))  then emit out[idx:]
    old_idx = 50
    old_idx = min(old_idx, len(out))  # stays 50 — does NOT clamp to base
    # out[50] is actually "L249", not "L199" → lines 199..248 are silently skipped
    lines_old = out[old_idx:]

    assert len(lines_old) < 901, (
        "old logic should produce fewer than 901 lines (it skips the first 50 buffer entries)"
    )
    assert lines_old[0] != "L199", (
        f"old logic starts at {lines_old[0]!r}, not 'L199' — proving lines were skipped"
    )


# ---------------------------------------------------------------------------
# RR7 — Page structure (four component cards)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_page_has_component_cards():
    """GET /setup/reset renders the four required component sections.

    First-run-safe: page must return 200 even with no config present (the
    JS handles the 'not configured' state client-side; the template must not
    require config to render).
    """
    # Re-use the module-level imports (TestClient + create_app are already at
    # the top of this file); creating a fresh client here avoids shared state.
    html = TestClient(create_app()).get("/setup/reset").text
    for token in ("Vectors", "Knowledge graph", "Obsidian notes", "Reset Everything"):
        assert token in html, f"Expected card token {token!r} not found in /setup/reset"


@pytest.mark.unit
def test_rebuild_stream_emits_buffered_lines(client):
    """GET /api/rebuild/vectors/stream returns SSE with buffered lines and a done event."""
    fake_runtime = _make_idle_runtime()

    # Set up the vectors slot with pre-buffered output and sequence inactive.
    vectors_slot = MagicMock()
    vectors_slot.output = ["a", "b"]
    vectors_slot.total_appended = 2  # required by the monotonic-counter follower
    vectors_slot.sequence_active = False
    vectors_slot.is_running.return_value = False
    fake_runtime.processes["rebuild_vectors"] = vectors_slot

    with (
        patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime),
        patch(
            "lit_monitor.server.routes.reset.is_rebuild_busy",
            return_value=False,
        ),
    ):
        resp = client.get(
            "/api/rebuild/vectors/stream",
            headers={"accept": "text/event-stream"},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    body = resp.text
    # Both buffered lines must appear in the SSE body
    assert "a" in body
    assert "b" in body
    # A done event must close the stream
    assert "done" in body
    # Named event contract: the server MUST emit "event: progress" lines so the
    # JS addEventListener("progress", …) listener in site.js can receive them.
    # An unnamed/default event would only fire the "message" handler, which we
    # intentionally removed. This assertion locks the server↔client contract.
    assert "event: progress" in body, (
        "SSE body must contain 'event: progress' lines; "
        "the JS rebuild log listener subscribes to the 'progress' named event"
    )


# ---------------------------------------------------------------------------
# Reviewer bugs — FIX 2: configured=True contract
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_status_configured_true_with_valid_runtime(tmp_path):
    """FIX 2: GET /api/reset/status returns configured=True when runtime has a valid config.

    Locks the contract so configured can't silently flip to False when config is present.
    """
    from lit_monitor.core.state_db import StateDB

    # Create a real temp StateDB so count_embeddings_indexed / count_graph_indexed work.
    db_path = tmp_path / "state.db"
    real_state_db = StateDB(str(db_path))

    # Temp dirs that "exist" — so targets are present.
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    graph_file = tmp_path / "graph.kuzu"
    graph_file.write_bytes(b"")
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    fake_config = MagicMock()
    fake_config.state_db.path = str(db_path)
    fake_config.obsidian.vault_path = str(vault_dir)
    # retrieval.graph_db.persist_dir is consulted by _graph_persist_dir
    fake_config.retrieval.graph_db.persist_dir = str(graph_file)

    idle_slot = MagicMock()
    idle_slot.is_running.return_value = False
    idle_slot.sequence_active = False

    fake_runtime = MagicMock()
    fake_runtime.config = fake_config
    fake_runtime.state_db = real_state_db
    fake_runtime.processes = {
        "brain_build": idle_slot,
        "discovery": idle_slot,
        "vocabulary": idle_slot,
        "rebuild_vectors": idle_slot,
        "rebuild_graph": idle_slot,
        "rebuild_notes": idle_slot,
    }

    client = TestClient(create_app())
    with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
        resp = client.get("/api/reset/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True, (
        f"Expected configured=True with valid runtime, got {data['configured']!r}"
    )


# ---------------------------------------------------------------------------
# Reviewer bugs — FIX 3: status shows paper/note counts, not raw file counts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_status_counts_are_papers_not_files(tmp_path):
    """FIX 3: /api/reset/status components.vectors.count is paper count, not file count.

    Seeds a real StateDB with N papers where M have embeddings_indexed=1 and K have
    graph_indexed=1. Asserts that the JSON response reflects M and K, not filesystem
    file counts.
    """
    from lit_monitor.core.state_db import StateDB

    db_path = tmp_path / "state.db"
    real_state_db = StateDB(str(db_path))

    # Upsert 5 papers; mark 3 embeddings_indexed=1 and 2 graph_indexed=1 directly.
    for i in range(5):
        real_state_db.upsert_paper({
            "doi": f"10.0/p{i}",
            "title": f"Paper {i}",
            "year": 2024,
            "embeddings_indexed": 1 if i < 3 else 0,
        })
    # Set graph_indexed via the dedicated setter (R28 invariant).
    for i in range(2):
        real_state_db.set_graph_indexed(f"10.0/p{i}", 1)

    # Temp chroma dir with some internal ChromaDB files (should NOT be the count).
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    for j in range(9):
        (chroma_dir / f"file{j}.bin").write_bytes(b"x")  # 9 internal files

    # Graph file: a single file (should yield 0 from old logic).
    graph_file = tmp_path / "graph.kuzu"
    graph_file.write_bytes(b"")

    # Vault dir with some .md notes.
    vault_dir = tmp_path / "vault" / "Literature" / "Papers"
    vault_dir.mkdir(parents=True)
    for j in range(4):
        (vault_dir / f"note{j}.md").write_bytes(b"")

    fake_config = MagicMock()
    fake_config.state_db.path = str(db_path)
    fake_config.obsidian.vault_path = str(tmp_path / "vault")
    fake_config.obsidian.papers_folder = "Literature/Papers"
    fake_config.obsidian.digests_folder = "Literature/Digests"
    fake_config.obsidian.connections_folder = "Literature/Connections"
    fake_config.retrieval.graph_db.persist_dir = str(graph_file)

    idle_slot = MagicMock()
    idle_slot.is_running.return_value = False
    idle_slot.sequence_active = False

    fake_runtime = MagicMock()
    fake_runtime.config = fake_config
    fake_runtime.state_db = real_state_db
    fake_runtime.processes = {
        "brain_build": idle_slot,
        "discovery": idle_slot,
        "vocabulary": idle_slot,
        "rebuild_vectors": idle_slot,
        "rebuild_graph": idle_slot,
        "rebuild_notes": idle_slot,
    }

    client = TestClient(create_app())
    with patch("lit_monitor.server.routes.reset.get_runtime", return_value=fake_runtime):
        resp = client.get("/api/reset/status")

    assert resp.status_code == 200
    data = resp.json()
    comps = data["components"]

    # vectors: count must be 3 (embeddings_indexed=1), NOT 9 (chroma internal files).
    assert comps["vectors"]["count"] == 3, (
        f"vectors.count should be 3 (embedded papers), got {comps['vectors']['count']}"
    )
    assert comps["vectors"]["unit"] == "paper"

    # graph: count must be 2 (graph_indexed=1), NOT 0 (single-file kuzu).
    assert comps["graph"]["count"] == 2, (
        f"graph.count should be 2 (graph-indexed papers), got {comps['graph']['count']}"
    )
    assert comps["graph"]["unit"] == "paper"

    # notes: 4 .md files in Papers folder.
    assert comps["notes"]["count"] == 4, (
        f"notes.count should be 4, got {comps['notes']['count']}"
    )
    assert comps["notes"]["unit"] == "note"

    # Old 'files' key must be gone.
    assert "files" not in comps["vectors"], "Legacy 'files' key must be removed"
    assert "files" not in comps["graph"], "Legacy 'files' key must be removed"


@pytest.mark.unit
def test_status_first_run_safe_new_shape(client):
    """FIX 3: config-absent status still returns count/unit/size_bytes shape (not files)."""
    resp = client.get("/api/reset/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is False
    for comp in ("vectors", "graph", "notes"):
        assert "count" in data["components"][comp], (
            f"{comp} missing 'count' key"
        )
        assert "unit" in data["components"][comp], (
            f"{comp} missing 'unit' key"
        )
        assert data["components"][comp]["count"] == 0
        assert data["components"][comp]["size_bytes"] == 0
