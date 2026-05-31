# External API Surfaces

lit-monitor exposes three external surfaces over the same underlying knowledge
graph (KuzuDB) and vector store (ChromaDB). All three share the query layer at
`scripts/api/queries.py` — any shape change in one must propagate to the others.

| Surface | Entry point | Audience |
|---|---|---|
| CLI ask | `lit-monitor ask "<question>"` | Terminal / scripting |
| MCP server | `lit-monitor mcp serve` (stdio transport) | Claude Desktop / Claude Code / Cursor / Continue |
| HTTP API | `lit-monitor serve` (FastAPI on 127.0.0.1) | Web UI, future Zotero plugin, browser automation |

## Shared implementation

All three surfaces ultimately call functions in `scripts/api/queries.py`:

| Function | HTTP endpoint | MCP tool | CLI |
|---|---|---|---|
| `get_paper_snapshot(doi, graph_db)` | `GET /api/papers/{doi}` (H4) | `get_paper_details` (B2) | — |
| `get_related_papers(doi, graph_db, mode, k, cfg)` | `GET /api/papers/{doi}/related` (H5) | — | — |
| `get_entity_neighborhood(canonical_id, graph_db)` | `GET /api/entities/{id}` (H6) | — | — |
| `list_entities(type, top_k, graph_db)` | `GET /api/entities?type=...` (H6) | `list_entities_by_type` (B2) | — |
| `get_corpus_stats(graph_db)` | — | `get_corpus_stats` (B2) | — |
| `get_schema_text(graph_db)` | — | `get_schema` (B2, via A1's describe_schema) | — |
| `get_papers_by_query(query, mode, k, cfg, ...)` | `POST /api/search` (H10) | `find_papers_by_query` / `find_papers_by_query_hybrid` (B2) | `lit-monitor ask` (A-series) |

### Tools that call GraphDB directly (intentional)

Two MCP tools in `scripts/mcp/tools.py` use GraphDB methods directly rather
than going through `queries.py`:

- **`find_papers_by_entity`** — calls `db.resolve_query_entity` and
  `db.find_papers_by_entities`. No HTTP equivalent exists; no `queries.py`
  shim was added to avoid dead code. If an HTTP endpoint for entity search is
  introduced, a shared helper should be extracted then.
- **`find_papers_by_relationship`** — uses inline Cypher against `db._conn`.
  No equivalent in `queries.py` and no HTTP endpoint exposes this predicate
  filter. The escape hatch `run_cypher` MCP tool covers ad-hoc Cypher queries
  in the meantime.

## Parity guarantee

The shape contract between HTTP and MCP for `get_paper_details` /
`GET /api/papers/{doi}` is pinned by:

```
tests/integration/test_api_mcp_parity.py
```

If you refactor one surface and forget the other, that test fails. The test
builds a fixture KuzuDB, calls both surfaces against the same DOI, and asserts
that top-level keys (and nested `metadata` keys) are identical.

## When to use which surface

- **Building a Zotero plugin or browser UI?** Use the HTTP API.
- **Building a Claude / Cursor agent?** Use the MCP server.
- **One-off terminal query?** Use `lit-monitor ask`.

For HTTP endpoint reference, see the FastAPI auto-generated OpenAPI docs at
`http://127.0.0.1:8000/docs` when running `lit-monitor serve`.

For MCP tool reference, see [docs/MCP_TOOLS.md](MCP_TOOLS.md).

## Phase 5 surfaces (v0.8.0)

Phase 5 (discovery pipeline + notifications) added the following surfaces.
MCP tool count increases from 10 to **12**.

### HTTP endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/discovery/runs` | List all discovery runs (id, timestamp, paper count, status) |
| `GET` | `/api/discovery/runs/{id}` | Single run metadata + summary stats |
| `GET` | `/api/discovery/runs/{id}/papers` | Paginated per-paper results for a run |
| `GET` | `/discovery/notify-handler` | Notification chooser page (opens on notification click) |
| `POST` | `/discovery/notify-handler/save-preference` | Persist viewer preference (browser / obsidian / none) |

### MCP tools (additions — total now 12)

| Tool | Description |
|---|---|
| `get_recent_discovery_runs` | Returns the N most recent discovery runs with metadata |
| `get_discovery_run_papers` | Returns per-paper scored results for a given run id |

### CLI commands (additions)

| Command | Description |
|---|---|
| `lit-monitor discovery view` | Rich-formatted table of results for a run (`--run latest` or run id) |
| `lit-monitor discovery export-md` | Export a run's results as a Markdown digest (`--run`, `--to`) |
| `lit-monitor obsidian sync` | Sync deferred per-paper Obsidian notes (`--all` or `--run`) |
