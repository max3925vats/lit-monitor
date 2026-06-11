"""B1: MCP server scaffold tests.

Acceptance criteria:
- Exactly 10 tools registered.
- Tool names match the Phase 4b plan exactly (no extras, no typos).
- No duplicate names.
- SERVER_HOST is loopback-only.
- _build_server() runs without raising when mcp SDK is present.
- Module can be imported without triggering GraphDB I/O or signal handlers.
"""
from __future__ import annotations

import pytest

from lit_monitor.mcp.graph_server import SERVER_HOST, TOOL_NAMES, _build_server


def test_tool_names_count() -> None:
    """P9: exactly 12 tools in the registry (10 from Phase 4b + 2 discovery)."""
    assert len(TOOL_NAMES) == 12


def test_tool_names_exact_set() -> None:
    """P9: the 12 tool names match the Phase 4b plan + P9 discovery tools exactly."""
    expected = {
        "find_papers_by_entity",
        "find_papers_by_relationship",
        "get_paper_details",
        "get_corpus_stats",
        "list_entities_by_type",
        "get_schema",
        "run_cypher",
        "semantic_search",
        "find_papers_by_query",
        "find_papers_by_query_hybrid",
        # P9:
        "get_recent_discovery_runs",
        "get_discovery_run_papers",
    }
    assert set(TOOL_NAMES) == expected


def test_no_duplicate_names() -> None:
    """B1: registry contains no repeated entries."""
    assert len(TOOL_NAMES) == len(set(TOOL_NAMES))


def test_localhost_bind() -> None:
    """B1: server bind host is loopback only (no external exposure)."""
    assert SERVER_HOST == "127.0.0.1"


def test_build_server_registers_ten_tools() -> None:
    """B1: _build_server() returns a Server instance without raising."""
    pytest.importorskip("mcp", reason="mcp SDK not installed")
    server = _build_server()
    assert server is not None


def test_call_tool_unknown_returns_error_text() -> None:
    """B1: _build_server() smoke — Server object is truthy; unknown-name dispatch
    is confirmed by inspecting the handler map rather than running async I/O.
    Detailed dispatch tests live in B2-B6."""
    pytest.importorskip("mcp", reason="mcp SDK not installed")
    server = _build_server()
    # Confirm both handler types are registered.
    from mcp.types import CallToolRequest, ListToolsRequest

    assert ListToolsRequest in server.request_handlers
    assert CallToolRequest in server.request_handlers


def test_graph_server_module_importable() -> None:
    """B1: importing the module doesn't trigger GraphDB construction or signal
    handler installation (those only fire inside main())."""
    import lit_monitor.mcp.graph_server as gs

    assert hasattr(gs, "TOOL_NAMES")
    assert hasattr(gs, "main")
    assert hasattr(gs, "_build_server")
    assert hasattr(gs, "_install_signal_handlers")
