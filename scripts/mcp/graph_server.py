"""MCP server exposing lit-monitor graph + vector RAG as 12 tools.

Phase 4b B1: scaffold.  B2 fills in the 8 high-level tools.
B3 fills in run_cypher (safety-guarded Cypher escape hatch).
B6 fills in semantic_search.
P9 adds the two discovery-run inspection tools (10 → 12).

Architecture:
- stdio transport (matches Claude Desktop's MCP client expectation).
- Single Server instance with a 12-name registry.
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
    # P9: discovery run inspection tools.
    "get_recent_discovery_runs",
    "get_discovery_run_papers",
)

# === Per-tool input schemas (B11) =========================================
# Each entry is a JSON Schema object derived directly from the matching
# handler signature in scripts/mcp/tools.py (the source of truth for params,
# types, defaults, and required-ness).  These replace the previous permissive
# empty schema so MCP clients get accurate type hints.
#
# Conventions:
# - additionalProperties is False everywhere: every tool has a closed,
#   fully-enumerated parameter set (none take arbitrary **kwargs), so clients
#   can rely on the advertised properties being exhaustive.
# - enum values for the closed-vocabulary params are pulled from the same
#   constants the handlers validate against (no duplicated literals), so the
#   schema can never drift from the runtime check.
# - This is metadata only: call_tool still dispatches **arguments unchanged.


def _build_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return {tool_name: {description, inputSchema}} for all 12 tools.

    Built lazily inside a function (not at import time) so importing the
    closed-vocabulary constants from scripts.mcp.tools / the validator does
    not run at module import — keeps import-only tests cheap and avoids
    pulling heavy deps before they are needed.
    """
    # Closed vocabularies — imported from their single source of truth so the
    # advertised enums stay in lock-step with the handlers' runtime validation.
    from scripts.graph.relationship_validator import VALID_PREDICATES  # noqa: PLC0415
    from scripts.mcp.tools import _ENTITY_TYPES  # noqa: PLC0415

    predicate_enum = sorted(VALID_PREDICATES)
    entity_type_enum = sorted(_ENTITY_TYPES)

    return {
        "find_papers_by_entity": {
            "description": "Find papers mentioning an entity (alias-resolved).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Surface form or canonical ID of the entity.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max papers to return.",
                    },
                },
                "required": ["entity"],
                "additionalProperties": False,
            },
        },
        "find_papers_by_relationship": {
            "description": "Find papers participating in a typed relationship.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "predicate": {
                        "type": "string",
                        "enum": predicate_enum,
                        "description": "One of the closed-vocabulary predicates.",
                    },
                    "target": {
                        "type": "string",
                        "description": (
                            "Optional target entity canonical_id or paper DOI "
                            "to filter by."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max papers to return.",
                    },
                },
                "required": ["predicate"],
                "additionalProperties": False,
            },
        },
        "get_paper_details": {
            "description": "Full snapshot of one paper (metadata, entities, edges).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "Paper DOI (matches ^10.[0-9]+/[^ ]+$).",
                    },
                },
                "required": ["doi"],
                "additionalProperties": False,
            },
        },
        "get_corpus_stats": {
            "description": "Aggregate paper/entity/edge counts for the corpus.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        "list_entities_by_type": {
            "description": "List entities of a given type, ranked by mention count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": entity_type_enum,
                        "description": "One of the 6 closed entity types.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 50,
                        "description": "Max entities to return.",
                    },
                },
                "required": ["entity_type"],
                "additionalProperties": False,
            },
        },
        "get_schema": {
            "description": "Markdown-formatted graph schema description.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        "run_cypher": {
            "description": "Execute a read-only, safety-guarded Cypher query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Raw read-only Cypher (mutation keywords rejected)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "default": 100,
                        "description": "Max rows (appended as LIMIT if absent).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        "semantic_search": {
            "description": "Vector retrieval over the ChromaDB index.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text query (embedded on the fly).",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Number of results (1-100).",
                    },
                    "granularity": {
                        "type": "string",
                        "enum": sorted(("paper", "chunk")),
                        "default": "paper",
                        "description": "paper-level or chunk-level hits.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        "find_papers_by_query": {
            "description": "Free-text query via the graph entity alias chain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search string.",
                    },
                    "k": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max results.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        "find_papers_by_query_hybrid": {
            "description": "Free-text query via RRF-fused graph + vector retrieval.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search string.",
                    },
                    "k": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max results.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        "get_recent_discovery_runs": {
            "description": "Most recent discovery runs (newest first).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Max runs to return (1-100).",
                    },
                },
                "additionalProperties": False,
            },
        },
        "get_discovery_run_papers": {
            "description": "Paper results for a discovery run, sorted by score DESC.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Primary key of the discovery_runs row.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Max papers to return (1-100).",
                    },
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    }


# Documentation-only loopback marker.  The MCP server runs over stdio (see
# main()/stdio_server) and never binds a network socket, so this constant is
# not consumed by the transport.  It is retained as an explicit "loopback by
# design" declaration and is asserted by tests/unit/test_mcp_server.py.
SERVER_HOST: str = "127.0.0.1"

# Module-level handle to the GraphDB; set in main(), closed in shutdown()
_graph_db: Any = None


def _build_server():
    """Build the MCP Server with all 12 tools registered.

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
        # P9: discovery run inspection.
        "get_recent_discovery_runs": _tools.get_recent_discovery_runs,
        "get_discovery_run_papers": _tools.get_discovery_run_papers,
    }

    server = Server("lit-monitor-graph")

    # B11: accurate per-tool input schemas, derived from the handler
    # signatures in scripts/mcp/tools.py.  Built once per server instance.
    _schemas = _build_tool_schemas()

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        tools_out: list[Tool] = []
        for name in TOOL_NAMES:
            spec = _schemas[name]
            tools_out.append(
                Tool(
                    name=name,
                    description=spec["description"],
                    inputSchema=spec["inputSchema"],
                )
            )
        return tools_out

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
