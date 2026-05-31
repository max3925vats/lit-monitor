"""MCP server exposing the lit-monitor graph + vector RAG as tools.

Phase 4b: graph_server.py registers 10 tools that AI agents (Claude
Desktop, Cursor, Continue) can call to query the literature graph.
"""

__all__ = ["graph_server"]
