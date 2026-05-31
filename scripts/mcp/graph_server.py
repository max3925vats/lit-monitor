"""MCP server exposing lit-monitor graph + vector RAG as 10 tools.

Phase 4b B1: scaffold.  B2 fills in the 8 high-level tools.
B3 fills in run_cypher (safety-guarded Cypher escape hatch).
B6 fills in semantic_search.

Architecture:
- stdio transport (matches Claude Desktop's MCP client expectation).
- Single Server instance with a 10-name registry.
- Signal handler closes GraphDB on SIGTERM/SIGINT.
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)

# === Tool registry =========================================================
# Order matters for human readability; the list_tools handler returns them
# in this order.
TOOL_NAMES: tuple[str, ...] = (
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
)

# Server bind host — localhost only by design (matches lit-monitor serve)
SERVER_HOST: str = "127.0.0.1"

# Module-level handle to the GraphDB; set in main(), closed in shutdown()
_graph_db: Any = None


def _build_server():
    """Build the MCP Server with all 10 tools registered.

    Returns the Server instance (not yet running). Caller invokes
    .run() via stdio_server depending on transport.
    """
    # Lazy import so test_mcp_server can import this module without
    # the mcp SDK installed (the package import succeeds; the function
    # call requires it).
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    # B2 tool implementations; imported lazily so the module can be
    # imported without triggering GraphDB I/O.
    from scripts.mcp import tools as _tools  # noqa: PLC0415

    # Dispatch table: maps MCP tool name → callable.
    _DISPATCH = {
        "find_papers_by_entity": _tools.find_papers_by_entity,
        "find_papers_by_relationship": _tools.find_papers_by_relationship,
        "get_paper_details": _tools.get_paper_details,
        "get_corpus_stats": _tools.get_corpus_stats,
        "list_entities_by_type": _tools.list_entities_by_type,
        "get_schema": _tools.get_schema,
        "find_papers_by_query": _tools.find_papers_by_query,
        "find_papers_by_query_hybrid": _tools.find_papers_by_query_hybrid,
        # B3: read-only Cypher escape hatch with safety guard.
        "run_cypher": _tools.run_cypher,
        # B6: ChromaDB vector retrieval (paper + chunk granularity).
        "semantic_search": _tools.semantic_search,
    }

    server = Server("lit-monitor-graph")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=f"lit-monitor graph tool: {name}",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            )
            for name in TOOL_NAMES
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name in _DISPATCH:
            try:
                result = _DISPATCH[name](**arguments)
            except ValueError as exc:
                # Validation errors are user-actionable; return as error text.
                return [TextContent(type="text", text=f"Error: {exc}")]
            except Exception as exc:  # noqa: BLE001
                logger.warning("MCP tool %s failed: %s", name, exc)
                return [TextContent(type="text", text=f"Tool error: {exc}")]
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        # Truly unknown tool name (should not happen if client uses list_tools).
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


def _install_signal_handlers() -> None:
    """Register SIGTERM/SIGINT handlers that close GraphDB before exit."""

    def _shutdown(signum: int, frame: Any) -> None:
        logger.info("MCP server received signal %d; closing GraphDB", signum)
        global _graph_db
        if _graph_db is not None:
            try:
                if hasattr(_graph_db, "close"):
                    _graph_db.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("GraphDB close failed: %s", exc)
            _graph_db = None
        # B6: drop the EmbeddingsDB handle so ChromaDB cleans itself up.
        try:
            from scripts.mcp.tools import close_embeddings_db  # noqa: PLC0415
            close_embeddings_db()
        except Exception as exc:  # noqa: BLE001
            logger.warning("EmbeddingsDB close failed: %s", exc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)


async def _run_stdio() -> None:
    """Run the server over stdio (Claude Desktop's expected transport)."""
    from mcp.server.stdio import stdio_server

    server = _build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point: ``python -m scripts.mcp.graph_server``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [mcp] %(message)s",
    )
    _install_signal_handlers()

    # Lazy-construct the GraphDB so import-only tests don't trigger I/O.
    global _graph_db
    try:
        from scripts.core.config import get_config
        from scripts.graph import GraphDB, safe_graph_db  # noqa: F401

        cfg = get_config()
        _graph_db = safe_graph_db(cfg)
        if _graph_db is None:
            logger.warning(
                "Graph backend unavailable; MCP server starting in degraded mode"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "GraphDB initialization failed: %s; starting without graph", exc
        )

    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
