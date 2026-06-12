# MCP server

[← Home](index.md)

lit-monitor ships a [Model Context Protocol](https://modelcontextprotocol.io)
server so any MCP-capable agent — Claude Desktop, Claude Code, Cursor, Continue —
can query your knowledge graph and vector index directly. It's the same query
layer the [HTTP API](http-api.md) exposes, surfaced as MCP tools.

## Starting the server

```bash
lit-monitor mcp serve
```

stdio transport. Register it in your MCP client config:

```json
"lit-monitor-graph": {
  "command": "lit-monitor",
  "args": ["mcp", "serve"]
}
```

The server reads the same local state DB, ChromaDB store, and KuzuDB graph as the
rest of lit-monitor — nothing leaves your machine, and it's read-only by design.

## The twelve tools

| Tool | What it does |
|---|---|
| `find_papers_by_entity` | Find papers mentioning an entity (alias-resolved). |
| `find_papers_by_relationship` | Find papers participating in a typed relationship (`CITES`, `EXTENDS`, `CONTRADICTS`, …). |
| `get_paper_details` | Full snapshot of one paper — metadata, entities, and graph edges. |
| `get_corpus_stats` | Aggregate paper / entity / edge counts for the whole corpus. |
| `list_entities_by_type` | List entities of a given type, ranked by mention count. |
| `get_schema` | Markdown-formatted description of the graph schema (for prompt grounding). |
| `find_papers_by_query` | Free-text query resolved through the graph entity alias chain. |
| `find_papers_by_query_hybrid` | Free-text query via RRF-fused graph + vector retrieval (the strongest default). |
| `semantic_search` | Pure vector retrieval over the ChromaDB index — paper-level or chunk-level hits. |
| `run_cypher` | Execute a **read-only, safety-guarded** Cypher query against the graph. |
| `get_recent_discovery_runs` | The most recent discovery runs, newest first. |
| `get_discovery_run_papers` | The ranked paper results for one discovery run, score-descending. |

### The Cypher escape hatch

`run_cypher` lets an agent run arbitrary graph queries, but every statement passes
through a safety guard that rejects anything mutating — only read paths
(`MATCH` / `RETURN` and friends) are allowed, write clauses (`CREATE`, `DELETE`,
`SET`, `MERGE`, …) are blocked, and a `LIMIT` is appended when absent. Pair it
with `get_schema` so the model knows the node labels, properties, and the
closed-vocabulary predicates before it composes a query.

### Zotero hand-off

`get_paper_details` returns each paper's `metadata` with a `zotero_key` and a
derived `zotero_deeplink` (or `null` when the paper has no linked Zotero item), so
a paper found through lit-monitor can be handed straight to a Zotero MCP client
without a title round-trip.

To see the live tool registry with full input schemas, run `lit-monitor mcp serve`
and inspect the `tools/list` response from your client.
