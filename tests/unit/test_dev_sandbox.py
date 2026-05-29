"""Unit tests for scripts/server/dev_sandbox.py (Task #66 D2).

Covers the four invariants that make the sandbox safe to use from /dev:

1. ``sandbox_status`` returns zeroed counts when the sandbox is empty / not
   yet materialised on disk.
2. Writing through ``sandbox_state_db()`` never touches the production
   ``state.db`` file.
3. ``clear_sandbox()`` without ``confirm=True`` is a no-op.
4. ``clear_sandbox(confirm=True)`` removes the sandbox state DB file.

ChromaDB and the obsidian vault are stubbed where needed via monkeypatch so
the tests stay hermetic and never reach the user's real config.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.server import dev_sandbox


@pytest.fixture(autouse=True)
def _reset_sandbox_embeddings_cache() -> None:
    """Clear the sandbox_embeddings_db and sandbox_graph_db singletons between tests.

    The singletons leak across tests if not reset — e.g. a tmp_path-bound client
    from test A would survive into test B's fresh tmp_path. Force-reset before AND
    after each test so monkeypatched SANDBOX_CHROMA_DIR / SANDBOX_GRAPH_DIR are
    honoured.
    """
    dev_sandbox.sandbox_embeddings_db.cache_clear()
    if hasattr(dev_sandbox, "sandbox_graph_db"):
        dev_sandbox.sandbox_graph_db.cache_clear()
    yield
    dev_sandbox.sandbox_embeddings_db.cache_clear()
    if hasattr(dev_sandbox, "sandbox_graph_db"):
        dev_sandbox.sandbox_graph_db.cache_clear()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _stub_get_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vault_path: Path,
    state_db_path: Path,
) -> None:
    """Replace ``dev_sandbox.get_config`` with a stub matching the real shape.

    The sandbox only reads ``cfg.state_db.path`` and ``cfg.obsidian.vault_path``,
    so a SimpleNamespace covers it without dragging in the full Pydantic schema.
    """
    fake_cfg = SimpleNamespace(
        state_db=SimpleNamespace(path=str(state_db_path)),
        obsidian=SimpleNamespace(vault_path=vault_path),
    )
    monkeypatch.setattr(dev_sandbox, "get_config", lambda: fake_cfg)


# ---------------------------------------------------------------------------
# 1) Empty sandbox → zeroed counts
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_status_empty_when_nothing_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no sandbox state on disk, status should report all-zero counts."""
    # Redirect the sandbox sqlite + vault into tmp_path; chroma uses tmp too.
    sandbox_db = tmp_path / "state_dev.db"
    prod_db = tmp_path / "state.db"  # never gets written
    sandbox_chroma = tmp_path / "chroma_dev"
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", sandbox_chroma)
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=prod_db,
    )

    status = dev_sandbox.sandbox_status()

    assert status["state_db_rows"] == 0
    assert status["papers_collection_count"] == 0
    assert status["chunks_collection_count"] == 0
    assert status["vault_file_count"] == 0
    assert status["last_modified"] is None


# ---------------------------------------------------------------------------
# 2) Sandbox state DB is fully separate from production state.db
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_state_db_separate_from_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write into the sandbox must NOT create/modify production state.db."""
    sandbox_db = tmp_path / "state_dev.db"
    prod_db = tmp_path / "state.db"
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", tmp_path / "chroma_dev")
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=prod_db,
    )

    assert not sandbox_db.exists()
    assert not prod_db.exists()

    db = dev_sandbox.sandbox_state_db()
    db.upsert_paper(
        {
            "doi": "10.0/dev-sandbox-test",
            "title": "Dev Sandbox Probe",
            "authors": ["Tester"],
            "year": 2025,
            "source_type": "paper",
            "status": "pending",
        }
    )

    # Sandbox file came into existence; production file did NOT.
    assert sandbox_db.exists(), "Sandbox sqlite file should be created on write."
    assert not prod_db.exists(), "Production state.db must remain untouched."

    # And the row landed in the sandbox where we expect.
    status = dev_sandbox.sandbox_status()
    assert status["state_db_rows"] == 1


# ---------------------------------------------------------------------------
# 3) clear_sandbox() without confirm is a no-op
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_clear_sandbox_without_confirm_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``confirm=True``, ``clear_sandbox`` must leave files in place."""
    sandbox_db = tmp_path / "state_dev.db"
    sandbox_db.write_bytes(b"dummy-sqlite-bytes-not-a-real-db")
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", tmp_path / "chroma_dev")
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=tmp_path / "state.db",
    )

    result = dev_sandbox.clear_sandbox()

    assert result == {"status": "skipped", "reason": "confirm=False"}
    assert sandbox_db.exists(), "Sandbox DB must NOT be deleted without confirm."


# ---------------------------------------------------------------------------
# 4) clear_sandbox(confirm=True) removes the sandbox state DB file
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_clear_sandbox_with_confirm_removes_state_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clear_sandbox(confirm=True)`` deletes the sandbox sqlite file."""
    sandbox_db = tmp_path / "state_dev.db"
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", tmp_path / "chroma_dev")
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=tmp_path / "state.db",
    )

    # Materialise a real sandbox sqlite file so the unlink branch executes.
    dev_sandbox.sandbox_state_db().upsert_paper(
        {"doi": "10.0/x", "title": "x", "year": 2025, "source_type": "paper"}
    )
    assert sandbox_db.exists()

    result = dev_sandbox.clear_sandbox(confirm=True)

    assert result["status"] == "cleared"
    assert not sandbox_db.exists(), "Sandbox DB file should be deleted."
    # Action log must mention the unlink so partial-wipe failures are visible.
    assert "unlinked" in result["actions"]


# ---------------------------------------------------------------------------
# 5) Partial failure surfaces as status="partial"
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_clear_sandbox_returns_partial_on_failure(monkeypatch, tmp_path):
    """If any cleanup step raises, status should be 'partial', not 'cleared'."""
    from scripts.server import dev_sandbox
    sandbox_db = tmp_path / "state_dev.db"
    sandbox_db.touch()
    sandbox_chroma = tmp_path / "chroma_dev"
    sandbox_chroma.mkdir()
    (sandbox_chroma / "dummy").write_text("x")
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", sandbox_chroma)
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=tmp_path / "state.db",
    )
    # Force shutil.rmtree on the chroma dir to raise to exercise the partial path.
    import shutil

    def _bad_rmtree(path, ignore_errors=False):
        raise RuntimeError("chroma rmtree boom")

    monkeypatch.setattr(dev_sandbox.shutil, "rmtree", _bad_rmtree)
    # Also need to preserve shutil for vault rmtree branch — but the vault dir
    # doesn't exist here, so that branch is skipped. Restore shutil after.
    _ = shutil  # silence unused-import
    result = dev_sandbox.clear_sandbox(confirm=True)
    assert result["status"] == "partial", f"expected partial, got {result}"
    assert "FAILED" in result["actions"]


# ---------------------------------------------------------------------------
# 6) Sandbox chroma persist dir must differ from production chroma dir
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_chroma_dir_is_separate_from_production() -> None:
    """The sandbox chroma persist dir must NOT collide with production.

    Production resolves to ``Path(cfg.state_db.path).parent / "chroma"`` (see
    ``scripts/server/runtime.py::ServerRuntime.embeddings_db``). The sandbox
    constant must use a different directory name so chromadb's per-path
    singleton cache can hold both clients simultaneously without conflict.

    Hermetic check: assert the sandbox dir's basename is NOT ``"chroma"`` —
    that's the production sibling. (Avoids loading real config so this test
    runs cleanly in CI where config/paths.yaml is unseeded.)
    """
    assert dev_sandbox.SANDBOX_CHROMA_DIR.name != "chroma", (
        f"sandbox dir basename collides with production: "
        f"{dev_sandbox.SANDBOX_CHROMA_DIR}"
    )
    # And it should match the documented `chroma_dev` convention.
    assert dev_sandbox.SANDBOX_CHROMA_DIR.name == "chroma_dev"


# ---------------------------------------------------------------------------
# 7) sandbox_embeddings_db is a singleton (dodges chromadb's per-path lock)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_embeddings_db_is_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two back-to-back calls must return the same instance.

    Without caching, chromadb refuses the second ``PersistentClient`` on the
    same persist dir with "An instance of Chroma already exists ... with
    different settings". The cache is the fix; this test pins it.
    """
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", tmp_path / "chroma_dev")
    # sandbox_embeddings_db touches cfg.embeddings/cfg.brain_build for host/model
    # config; the bare _stub_get_config doesn't include those, so build the
    # SimpleNamespace inline with the extra fields.
    fake_cfg = SimpleNamespace(
        state_db=SimpleNamespace(path=str(tmp_path / "state.db")),
        obsidian=SimpleNamespace(vault_path=tmp_path / "vault"),
        embeddings=SimpleNamespace(ollama_host="http://localhost:11434", model="m"),
        brain_build=SimpleNamespace(ollama_host="http://localhost:11434"),
    )
    monkeypatch.setattr(dev_sandbox, "get_config", lambda: fake_cfg)

    # Stub EmbeddingsDB so we don't have to spin up real chromadb. A unique
    # object per call would prove the cache is broken; the cache should fold
    # repeated calls to a single invocation of the constructor.
    construction_count = {"n": 0}

    class _FakeEmbDB:
        def __init__(self, **kwargs):
            construction_count["n"] += 1

    monkeypatch.setattr(
        "scripts.output.embeddings.EmbeddingsDB", _FakeEmbDB
    )

    first = dev_sandbox.sandbox_embeddings_db()
    second = dev_sandbox.sandbox_embeddings_db()

    assert first is second, "sandbox_embeddings_db must be cached as a singleton"
    assert construction_count["n"] == 1, (
        f"EmbeddingsDB constructor ran {construction_count['n']} times; "
        "cache is not folding repeated calls"
    )


@pytest.mark.unit
def test_clear_sandbox_invalidates_embeddings_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clear_sandbox(confirm=True)`` must drop the cached EmbeddingsDB.

    Otherwise the next caller gets a stale client pointing at the rmtree'd
    persist dir and writes silently fail (or worse, succeed against a
    recreated dir with no schema).
    """
    sandbox_chroma = tmp_path / "chroma_dev"
    sandbox_chroma.mkdir()
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", tmp_path / "state_dev.db")
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", sandbox_chroma)
    fake_cfg = SimpleNamespace(
        state_db=SimpleNamespace(path=str(tmp_path / "state.db")),
        obsidian=SimpleNamespace(vault_path=tmp_path / "vault"),
        embeddings=SimpleNamespace(ollama_host="http://localhost:11434", model="m"),
        brain_build=SimpleNamespace(ollama_host="http://localhost:11434"),
    )
    monkeypatch.setattr(dev_sandbox, "get_config", lambda: fake_cfg)

    class _FakeEmbDB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "scripts.output.embeddings.EmbeddingsDB", _FakeEmbDB
    )

    first = dev_sandbox.sandbox_embeddings_db()
    # Clear invalidates the cache (and rmtree's the dir along the way).
    dev_sandbox.clear_sandbox(confirm=True)
    second = dev_sandbox.sandbox_embeddings_db()

    assert id(first) != id(second), (
        "Post-clear sandbox_embeddings_db() returned the cached pre-clear "
        "instance — clear_sandbox didn't invalidate the cache"
    )


# ---------------------------------------------------------------------------
# 9) sandbox_collections returns real chromadb collections from EmbeddingsDB
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_collections_returns_real_chroma_collections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sandbox_collections() must return objects with a callable .count().

    This catches a recurrent class of bug: someone renames EmbeddingsDB's
    private collection attribute (today: _collection and _chunks_collection)
    and the getattr lookups in sandbox_collections silently fall back to
    None, surfacing later as a vague "EmbeddingsDB does not expose ..."
    warning from sandbox_status. With the real EmbeddingsDB constructed
    against a real-but-empty chromadb dir, this asserts the contract end
    to end — no mocks at the EmbeddingsDB level.
    """
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", tmp_path / "chroma_dev")
    fake_cfg = SimpleNamespace(
        state_db=SimpleNamespace(path=str(tmp_path / "state.db")),
        obsidian=SimpleNamespace(vault_path=tmp_path / "vault"),
        embeddings=SimpleNamespace(
            ollama_host="http://localhost:11434", model="mxbai-embed-large"
        ),
        brain_build=SimpleNamespace(ollama_host="http://localhost:11434"),
    )
    monkeypatch.setattr(dev_sandbox, "get_config", lambda: fake_cfg)

    papers, chunks = dev_sandbox.sandbox_collections()

    # The two returned objects must be real chromadb collections — they
    # expose a `.count()` method and live behind the same chromadb client
    # the cached EmbeddingsDB holds (i.e. no parallel client construction).
    assert callable(getattr(papers, "count", None)), (
        f"papers collection has no callable .count(); got {type(papers)!r}"
    )
    assert callable(getattr(chunks, "count", None)), (
        f"chunks collection has no callable .count(); got {type(chunks)!r}"
    )
    # Counts should be 0 on a freshly-created persist dir — proves the
    # collections are queryable, not just placeholder objects.
    assert papers.count() == 0
    assert chunks.count() == 0


# ---------------------------------------------------------------------------
# G12: sandbox graph DB tests
# ---------------------------------------------------------------------------

def _patch_all_sandbox_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    graph_path: Path | None = None,
) -> None:
    """Redirect ALL sandbox module-level path constants into tmp_path.

    Matches the pattern used by the existing sandbox tests: redirect every
    path constant so no test ever touches the real ~/.config/lit-monitor/.
    """
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", tmp_path / "state_dev.db")
    monkeypatch.setattr(dev_sandbox, "SANDBOX_CHROMA_DIR", tmp_path / "chroma_dev")
    monkeypatch.setattr(
        dev_sandbox, "SANDBOX_GRAPH_DIR", str(graph_path or tmp_path / "graph_dev.kuzu")
    )
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=tmp_path / "state.db",
    )


@pytest.mark.unit
def test_sandbox_graph_db_returns_graphdb_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: sandbox_graph_db() returns a GraphDB bound to SANDBOX_GRAPH_DIR."""
    sandbox_graph_path = tmp_path / "graph_dev.kuzu"
    _patch_all_sandbox_paths(monkeypatch, tmp_path, graph_path=sandbox_graph_path)
    dev_sandbox.sandbox_graph_db.cache_clear()

    # Stub GraphDB so the test is hermetic (no kuzu required at unit-test time).
    class _FakeGraphDB:
        def __init__(self, persist_dir: str) -> None:
            self.persist_dir = persist_dir

    monkeypatch.setattr("scripts.graph.db.GraphDB", _FakeGraphDB)
    # Also patch the import inside dev_sandbox so the module-level import resolves
    import scripts.graph as graph_mod
    monkeypatch.setattr(graph_mod, "GraphDB", _FakeGraphDB)

    graph = dev_sandbox.sandbox_graph_db()
    assert isinstance(graph, _FakeGraphDB)
    # The persist_dir handed to the constructor must match SANDBOX_GRAPH_DIR.
    assert graph.persist_dir == str(sandbox_graph_path)
    dev_sandbox.sandbox_graph_db.cache_clear()


@pytest.mark.unit
def test_sandbox_graph_db_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: repeated calls return the same instance (lru_cache)."""
    _patch_all_sandbox_paths(monkeypatch, tmp_path)
    dev_sandbox.sandbox_graph_db.cache_clear()

    construction_count = {"n": 0}

    class _FakeGraphDB:
        def __init__(self, persist_dir: str) -> None:
            construction_count["n"] += 1

    import scripts.graph as graph_mod
    monkeypatch.setattr(graph_mod, "GraphDB", _FakeGraphDB)

    g1 = dev_sandbox.sandbox_graph_db()
    g2 = dev_sandbox.sandbox_graph_db()

    assert g1 is g2, "sandbox_graph_db must be cached as a singleton"
    assert construction_count["n"] == 1, (
        f"GraphDB constructor ran {construction_count['n']} times; cache is broken"
    )
    dev_sandbox.sandbox_graph_db.cache_clear()


@pytest.mark.unit
def test_clear_sandbox_removes_graph_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: clear_sandbox(confirm=True) removes the SANDBOX_GRAPH_DIR file."""
    sandbox_graph_path = tmp_path / "graph_dev.kuzu"
    # Materialise as a file (KuzuDB stores as a file, not directory).
    sandbox_graph_path.write_bytes(b"kuzu-placeholder")
    _patch_all_sandbox_paths(monkeypatch, tmp_path, graph_path=sandbox_graph_path)
    dev_sandbox.sandbox_graph_db.cache_clear()

    result = dev_sandbox.clear_sandbox(confirm=True)

    assert result["status"] in ("cleared", "partial"), f"unexpected status: {result}"
    assert not sandbox_graph_path.exists(), "SANDBOX_GRAPH_DIR file must be removed"


@pytest.mark.unit
def test_clear_sandbox_removes_graph_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: clear_sandbox handles SANDBOX_GRAPH_DIR that is a directory."""
    sandbox_graph_path = tmp_path / "graph_dev.kuzu"
    # Materialise as a directory (handle both shapes per G12 brief).
    sandbox_graph_path.mkdir()
    (sandbox_graph_path / "data").write_bytes(b"x")
    _patch_all_sandbox_paths(monkeypatch, tmp_path, graph_path=sandbox_graph_path)
    dev_sandbox.sandbox_graph_db.cache_clear()

    result = dev_sandbox.clear_sandbox(confirm=True)

    assert result["status"] in ("cleared", "partial"), f"unexpected status: {result}"
    assert not sandbox_graph_path.exists(), "SANDBOX_GRAPH_DIR directory must be removed"


@pytest.mark.unit
def test_clear_sandbox_removes_graph_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: clear_sandbox also removes the .schema_version sentinel if present."""
    sandbox_graph_path = tmp_path / "graph_dev.kuzu"
    sentinel = Path(str(sandbox_graph_path) + ".schema_version")
    sentinel.write_text("3")
    sandbox_graph_path.write_bytes(b"kuzu-placeholder")
    _patch_all_sandbox_paths(monkeypatch, tmp_path, graph_path=sandbox_graph_path)
    dev_sandbox.sandbox_graph_db.cache_clear()

    dev_sandbox.clear_sandbox(confirm=True)

    assert not sentinel.exists(), "graph .schema_version sentinel must be removed"


@pytest.mark.unit
def test_clear_sandbox_clears_graph_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: clear_sandbox invalidates the sandbox_graph_db lru_cache."""
    _patch_all_sandbox_paths(monkeypatch, tmp_path)
    dev_sandbox.sandbox_graph_db.cache_clear()

    construction_count = {"n": 0}

    class _FakeGraphDB:
        def __init__(self, persist_dir: str) -> None:
            construction_count["n"] += 1

    import scripts.graph as graph_mod
    monkeypatch.setattr(graph_mod, "GraphDB", _FakeGraphDB)

    first = dev_sandbox.sandbox_graph_db()
    assert construction_count["n"] == 1

    dev_sandbox.clear_sandbox(confirm=True)

    # After clear, the next call must construct a new instance.
    second = dev_sandbox.sandbox_graph_db()
    assert construction_count["n"] == 2, (
        "clear_sandbox did not invalidate sandbox_graph_db cache; "
        f"constructor ran {construction_count['n']} times total"
    )
    assert first is not second
    dev_sandbox.sandbox_graph_db.cache_clear()


@pytest.mark.unit
def test_sandbox_status_includes_graph_node_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: sandbox_status returns a graph_node_count key (int)."""
    _patch_all_sandbox_paths(monkeypatch, tmp_path)

    # Mock sandbox_graph_db to return a fake with a queryable _conn.
    class _FakeResult:
        def has_next(self) -> bool:
            return True

        def get_next(self) -> list:
            return [7]  # 7 nodes

    class _FakeConn:
        def execute(self, query: str) -> _FakeResult:
            return _FakeResult()

    class _FakeGraphDB:
        def __init__(self, persist_dir: str) -> None:
            self._conn = _FakeConn()

    import scripts.graph as graph_mod
    monkeypatch.setattr(graph_mod, "GraphDB", _FakeGraphDB)
    # Reset the cache so the next sandbox_graph_db() uses _FakeGraphDB.
    dev_sandbox.sandbox_graph_db.cache_clear()

    status = dev_sandbox.sandbox_status()

    assert "graph_node_count" in status, (
        f"sandbox_status missing 'graph_node_count' key; got keys: {list(status)}"
    )
    assert isinstance(status["graph_node_count"], int)
    assert status["graph_node_count"] == 7
    dev_sandbox.sandbox_graph_db.cache_clear()


@pytest.mark.unit
def test_sandbox_status_graph_node_count_zero_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G12: graph query errors in sandbox_status must not raise; return 0."""
    _patch_all_sandbox_paths(monkeypatch, tmp_path)

    class _BrokenConn:
        def execute(self, query: str) -> None:
            raise RuntimeError("kuzu not available in test env")

    class _FakeGraphDB:
        def __init__(self, persist_dir: str) -> None:
            self._conn = _BrokenConn()

    import scripts.graph as graph_mod
    monkeypatch.setattr(graph_mod, "GraphDB", _FakeGraphDB)
    dev_sandbox.sandbox_graph_db.cache_clear()

    status = dev_sandbox.sandbox_status()

    assert status["graph_node_count"] == 0
    dev_sandbox.sandbox_graph_db.cache_clear()


@pytest.mark.unit
def test_sandbox_graph_dir_is_separate_from_production() -> None:
    """G12: SANDBOX_GRAPH_DIR must not collide with production graph.kuzu."""
    # Production is ~/.config/lit-monitor/graph.kuzu
    import os
    prod = os.path.expanduser("~/.config/lit-monitor/graph.kuzu")
    assert dev_sandbox.SANDBOX_GRAPH_DIR != prod, (
        f"SANDBOX_GRAPH_DIR collides with production: {dev_sandbox.SANDBOX_GRAPH_DIR}"
    )
    # Must use the 'graph_dev.kuzu' naming convention.
    assert "graph_dev" in dev_sandbox.SANDBOX_GRAPH_DIR
