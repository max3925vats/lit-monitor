"""A1: KuzuDB schema describer for the ask command.

Generates a markdown description of the current schema by calling
``CALL show_tables()`` + ``CALL table_info(<name>)`` +
``CALL show_connection(<name>)``.  Used by Phase 4a A2 to inject the schema
into the NL→Cypher prompt, and by Phase 4b B2 MCP get_schema tool.

Row shapes in Kuzu 0.11.x (verified against installed version):
  show_tables()      → [int id, str name, str type, str database, str comment]
  table_info(name)   → [int id, str prop_name, str prop_type, str default, bool|str is_pk]
                       • NODE: is_pk is bool (True/False)
                       • REL:  is_pk is a string direction ("both"/etc.)
  show_connection(n) → [str from_table, str to_table, str from_pk, str to_pk]

Cached per-process keyed by id(graph_db). Invalidated by GraphDB.add_paper
so describes reflect current state without re-querying on every ask call.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level cache. Key: id(graph_db). Value: rendered markdown.
# Cleared by invalidate_schema_cache when the graph is mutated.
_CACHE: dict[int, str] = {}


def invalidate_schema_cache(graph_db: Any) -> None:
    """Drop the cached schema description for this graph_db instance."""
    _CACHE.pop(id(graph_db), None)


def describe_schema(graph_db: Any) -> str:
    """A1: render the current KuzuDB schema as markdown.

    Cached per-process. Call invalidate_schema_cache(graph_db) after any
    mutation that changes schema or data (handled automatically by
    GraphDB.add_paper).

    Args:
        graph_db: GraphDB instance (or any object exposing ._conn).

    Returns:
        Markdown string grouping Node tables (alphabetical) then REL tables
        (alphabetical).  REL headings include FROM → TO labels.
    """
    key = id(graph_db)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        result = _build_schema_text(graph_db)
    except Exception as exc:
        logger.warning("describe_schema: build failed: %s", exc)
        return "(schema introspection failed — check KuzuDB connection)"

    _CACHE[key] = result
    return result


def _build_schema_text(graph_db: Any) -> str:
    """Internal: query the live schema and render markdown.

    Enumerates tables via show_tables(), then calls table_info() and
    (for REL tables) show_connection() per table.
    """
    conn = graph_db._conn

    # --- Enumerate tables via CALL show_tables() RETURN * ---
    # Row shape: [int id, str name, str type, str database, str comment]
    # row[1] = table name, row[2] = "NODE" or "REL"
    res = conn.execute("CALL show_tables() RETURN *")
    nodes: list[str] = []
    rels: list[str] = []
    while res.has_next():
        row = res.get_next()
        name: str = row[1]
        kind: str = row[2]
        if kind == "NODE":
            nodes.append(name)
        elif kind == "REL":
            rels.append(name)

    out: list[str] = []

    # Node tables alphabetically
    for name in sorted(nodes):
        out.append(_describe_node(conn, name))
        out.append("")  # blank line separator

    # REL tables alphabetically
    for name in sorted(rels):
        out.append(_describe_rel(conn, name))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def _describe_node(conn: Any, name: str) -> str:
    """Render one NODE table's markdown block.

    Row shape from table_info(): [int id, str prop_name, str prop_type,
    str default, bool is_pk]
    """
    lines = [f"## Node: {name}"]
    try:
        res = conn.execute(f"CALL table_info('{name}') RETURN *")
        while res.has_next():
            row = res.get_next()
            # row[1] = property name, row[2] = type string, row[4] = is_pk (bool)
            prop_name: str = row[1]
            prop_type: str = row[2]
            is_pk: bool = row[4] is True  # True only when the bool is literally True
            suffix = " (PRIMARY KEY)" if is_pk else ""
            lines.append(f"- {prop_name}: {prop_type}{suffix}")
    except Exception as exc:
        logger.warning("_describe_node: table_info(%s) failed: %s", name, exc)
        lines.append("- (introspection failed)")
    return "\n".join(lines)


def _describe_rel(conn: Any, name: str) -> str:
    """Render one REL table's markdown block including FROM → TO labels.

    Uses show_connection() to retrieve FROM/TO node labels.
    Row shape: [str from_table, str to_table, str from_pk, str to_pk]

    Falls back to "Node → Node" if show_connection is unavailable (Kuzu
    version variance).

    table_info() row shape for REL: [int id, str prop_name, str prop_type,
    str default, str direction]  — is_pk is a string ("both"), not bool.
    """
    from_label = "Node"
    to_label = "Node"
    try:
        res = conn.execute(f"CALL show_connection('{name}') RETURN *")
        if res.has_next():
            row = res.get_next()
            # row[0] = from_table, row[1] = to_table
            from_label = row[0]
            to_label = row[1]
    except Exception:
        # Kuzu version variance — fall through with default labels
        pass

    lines = [f"## REL: {name} ({from_label} → {to_label})"]
    try:
        res = conn.execute(f"CALL table_info('{name}') RETURN *")
        while res.has_next():
            row = res.get_next()
            prop_name: str = row[1]
            prop_type: str = row[2]
            lines.append(f"- {prop_name}: {prop_type}")
    except Exception as exc:
        logger.warning("_describe_rel: table_info(%s) failed: %s", name, exc)
        lines.append("- (introspection failed)")
    return "\n".join(lines)
