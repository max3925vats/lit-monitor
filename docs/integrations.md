# Integrations

[← Back to README](../README.md)

lit-monitor exposes the same query layer over two programmatic surfaces: an MCP
server for AI clients and an HTTP API.

## MCP server

For Claude Desktop, Cursor, Continue, and any other MCP-capable agent:

```bash
lit-monitor mcp serve
```

stdio transport. Register in your MCP client config:

```json
"lit-monitor-graph": {
  "command": "lit-monitor",
  "args": ["mcp", "serve"]
}
```

Twelve tools — `find_papers_by_entity`, `find_papers_by_relationship`,
`get_paper_details`, `find_papers_by_query_hybrid` (RRF-fused), `run_cypher`
(read-only with safety guard), `semantic_search`, `get_recent_discovery_runs`,
and five more. Run `lit-monitor mcp serve` to see the registry.

`get_paper_details` returns each paper's `metadata` with a `zotero_key` and a
derived `zotero_deeplink` (or `null` when the paper has no linked Zotero item),
so a paper found through lit-monitor can be handed to a Zotero MCP client without
a title round-trip.

## HTTP API

`lit-monitor serve` exposes the same query layer over HTTP, with FastAPI
auto-docs at `http://127.0.0.1:8765/docs`. Endpoints include:

- `POST /api/ingest`
- `GET /api/papers/{doi}`
- `POST /api/ask`
- `POST /api/cypher`
- `POST /api/search`
- `GET /api/discovery/runs`
- trigger endpoints for relink and re-extract

The server binds to `127.0.0.1` by default and has no authentication of its own;
see the [Web UI guide](web-ui.md) for host/binding notes.
