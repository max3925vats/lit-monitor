"""H9: prove HTTP /api/papers/{doi} and MCP get_paper_details share the same shape.

This is the cross-surface parity guarantee: any future refactor that
changes the shape of one MUST update the other (and this test catches it).

Both surfaces delegate to scripts.api.queries.get_paper_snapshot, so the
shapes should always be identical. If they diverge, this test fails and
forces an explicit resolution.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
def fixture_graph(tmp_path):
    """Small KuzuDB with one paper for parity testing.

    Uses tmp_path so each test run gets a fresh DB (no cross-test pollution).
    add_paper expects EntityTuple objects (not plain dicts) for the entities list.
    """
    from scripts.graph import GraphDB
    from scripts.graph.entity_extractor import EntityTuple

    db = GraphDB(persist_dir=str(tmp_path / "parity.kuzu"))
    db.add_paper(
        doi="10.9999/parity",
        paper_metadata={"title": "Parity Test Paper", "year": 2024, "journal": "J"},
        entities=[
            EntityTuple(
                canonical_id="method-a",
                type="method",
                surface="Method A",
                field="methods",
                span_start=None,
                span_end=None,
            ),
        ],
        relationships=[],
    )
    return db


def test_http_mcp_get_paper_shape_parity(fixture_graph, monkeypatch):
    """H9: GET /api/papers/{doi} and MCP get_paper_details return identical keys.

    Validates that:
    1. Both surfaces return HTTP 200 / a non-error dict for the same fixture DOI.
    2. Top-level keys are identical.
    3. Nested ``metadata`` keys are identical (when present on both sides).

    If this test fails after a refactor, update BOTH surfaces before merging.
    """
    # Patch BOTH surfaces to return the SAME shared ``fixture_graph`` handle.
    # papers.py binds safe_graph_db at module level; tools.py exposes
    # _get_graph_db as a module-level callable. Patch both bindings directly.
    import scripts.server.routes.papers as _papers_module
    from scripts.mcp import tools as _tools_module

    monkeypatch.setattr(_papers_module, "safe_graph_db", lambda *a, **kw: fixture_graph)
    monkeypatch.setattr(_tools_module, "_get_graph_db", lambda: fixture_graph)

    from scripts.server.app import create_app

    # --- MCP surface FIRST ---
    # Order matters: AR-2 made the HTTP route close() its graph handle in a
    # try/finally after use (the correct production behaviour — each real
    # request gets its own safe_graph_db() handle and releases it). Here both
    # surfaces share ONE handle (fixture_graph), so if HTTP ran first it would
    # close fixture_graph and the MCP call would then hit a closed connection
    # ('NoneType' has no attribute 'execute'). The MCP surface does NOT close
    # its handle, so calling it first leaves fixture_graph open for HTTP, which
    # closes it last. Nothing reads the graph after the HTTP call — the
    # assertions below only consume the returned dicts — so this ordering is safe.
    mcp_result = _tools_module.get_paper_details("10.9999/parity")

    # --- HTTP surface SECOND (closes fixture_graph via AR-2's finally) ---
    client = TestClient(create_app())
    http_response = client.get("/api/papers/10.9999/parity")
    assert http_response.status_code == 200, (
        f"HTTP GET returned {http_response.status_code}: {http_response.text}"
    )
    http_result = http_response.json()

    # Both results must be non-empty dicts (not error stubs).
    assert isinstance(http_result, dict) and http_result, "HTTP result is empty"
    assert isinstance(mcp_result, dict) and mcp_result, "MCP result is empty"

    # Top-level keys must match exactly.
    assert set(http_result.keys()) == set(mcp_result.keys()), (
        f"Top-level key mismatch:\n"
        f"  HTTP keys: {sorted(http_result.keys())}\n"
        f"  MCP keys:  {sorted(mcp_result.keys())}"
    )

    # Drill one level: metadata sub-keys must also match.
    if "metadata" in http_result and "metadata" in mcp_result:
        http_meta_keys = set(http_result["metadata"].keys())
        mcp_meta_keys = set(mcp_result["metadata"].keys())
        assert http_meta_keys == mcp_meta_keys, (
            f"metadata key mismatch:\n"
            f"  HTTP metadata keys: {sorted(http_meta_keys)}\n"
            f"  MCP metadata keys:  {sorted(mcp_meta_keys)}"
        )
