"""
scripts.graph — KuzuDB-backed knowledge graph for lit-monitor (Phase 1).

Public API
----------
GraphDB : the main wrapper class; import is always safe. Instantiation
          raises ImportError if kuzu is not installed (install with
          ``uv sync --extra graph``).
"""
from __future__ import annotations

# Re-export GraphDB so callers can write ``from scripts.graph import GraphDB``
# without knowing the internal module layout.  The import is always available
# regardless of whether kuzu is installed; the lazy kuzu import inside GraphDB
# ensures that non-graph users see no errors at import time.
from scripts.graph.db import GraphDB
from scripts.graph.import_citations import safe_graph_db

__all__ = ["GraphDB", "safe_graph_db"]
