"""MCP server exposing lit-monitor graph + vector RAG as 10 tools.

Phase 4b B1: scaffold only. Tool bodies stub NotImplementedError;
B2/B3/B6 fill them in.

Architecture:
- stdio transport (matches Claude Desktop's MCP client expectation).
- Single Server instance with a 10-name registry.
- Signal handler closes GraphDB on SIGTERM/SIGINT.
"""
from __future__ import annotations

import asyncio
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

    server = Server("lit-monitor-graph")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        # Skeleton: minimal inputSchema; B2/B3/B6 will refine descriptions
        # and property schemas per tool.
        return [
            Tool(
                name=name,
                description=f"(B1 scaffold — implementation pending) {name}",
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
        if name not in TOOL_NAMES:
            # Return error text rather than raising so the client gets a
            # well-formed CallToolResult with isError semantics.
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        # B2/B3/B6 will dispatch to actual implementations here.
        raise NotImplementedError(f"Tool {name} not yet implemented (B1 scaffold)")

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
        from scripts.core.config import load_config
        from scripts.graph import GraphDB, safe_graph_db  # noqa: F401

        cfg = load_config()
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
