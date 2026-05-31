# lit-monitor MCP Tools Reference

The `lit-monitor mcp serve` command starts an MCP server over stdio that
exposes 10 tools for querying the knowledge graph and vector index.
AI clients (Claude Desktop, Continue, Cursor) can call these tools to
retrieve papers, entities, relationships, and semantic matches from your
Zotero-backed corpus.

**Prerequisites:** `uv sync --extra mcp --extra graph`

---

## Tool index

| # | Name | Backend |
|---|------|---------|
| 1 | `find_papers_by_entity` | KuzuDB graph |
| 2 | `find_papers_by_relationship` | KuzuDB graph |
| 3 | `get_paper_details` | KuzuDB graph |
| 4 | `get_corpus_stats` | KuzuDB graph |
| 5 | `list_entities_by_type` | KuzuDB graph |
| 6 | `get_schema` | KuzuDB graph |
| 7 | `find_papers_by_query` | KuzuDB graph (entity alias chain) |
| 8 | `find_papers_by_query_hybrid` | KuzuDB + ChromaDB (RRF fusion) |
| 9 | `run_cypher` | KuzuDB (read-only, safety-guarded) |
| 10 | `semantic_search` | ChromaDB vector index |

---

## 1. find_papers_by_entity

Find papers that mention a given entity. The query string is resolved
through the Phase 2 normalizer + alias chain before querying.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity` | string | required | Surface form or canonical ID of the entity |
| `top_k` | int | 20 | Maximum papers to return |

**Returns** `list[{doi, title, year, journal}]`

**Example call**
```json
{ "entity": "monoclonal antibody", "top_k": 5 }
```

**Example response**
```json
[
  {"doi": "10.1016/j.foo.2024.01", "title": "mAb purification review", "year": 2024, "journal": "J Chromatogr A"},
  {"doi": "10.1002/btpr.3456",     "title": "CEX screening of mAbs",   "year": 2023, "journal": "Biotechnol Prog"}
]
```

**Raises** `ValueError` when `entity` is empty or not a string.

---

## 2. find_papers_by_relationship

Find papers participating in a given typed relationship.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `predicate` | string | required | One of the 10 closed-vocabulary predicates |
| `target` | string | `null` | Optional target entity canonical_id or paper DOI |
| `top_k` | int | 20 | Maximum papers to return |

**Closed-vocabulary predicates** (10 total):
`MENTIONS`, `CITES`, `COMPARES_TO`, `DEPENDS_ON`, `PROPOSES`,
`LIMITED_BY`, `INTRODUCES`, `RAISES_QUESTION`, `EXTENDS`, `CONTRADICTS`

**Returns** `list[{doi, title, year, journal}]`

**Example call**
```json
{ "predicate": "EXTENDS", "top_k": 10 }
```

**Example response**
```json
[
  {"doi": "10.1002/btpr.3500", "title": "Extended CEX model", "year": 2024, "journal": "Biotechnol Prog"}
]
```

**Raises** `ValueError` when `predicate` is not in the closed vocabulary.

---

## 3. get_paper_details

Return a full snapshot of one paper: metadata, entities grouped by type,
and all typed relationship edges in and out.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `doi` | string | required | DOI of the paper (must match `^10.[0-9]+/[^ ]+$`) |

**Returns**
```
{
  "metadata": {doi, title, year, journal},
  "entities_by_type": {
    "topic":    [{canonical, surface, mention_count}, ...],
    "method":   [...],
    ...
  },
  "relationships_in":  [{predicate, source_doi, evidence}, ...],
  "relationships_out": [{predicate, target_id,  evidence}, ...]
}
```

**Example call**
```json
{ "doi": "10.1016/j.foo.2024.01" }
```

**Raises** `ValueError` when `doi` fails DOI validation.

---

## 4. get_corpus_stats

Return aggregate counts for the entire graph corpus.

**Arguments** None.

**Returns**
```
{
  "paper_count":  <int>,
  "entity_count": <int>,
  "edge_counts_by_predicate": {
    "MENTIONS": <int>,
    "CITES":    <int>,
    ...
  }
}
```

**Example call**
```json
{}
```

**Example response**
```json
{
  "paper_count": 847,
  "entity_count": 3412,
  "edge_counts_by_predicate": {"MENTIONS": 21043, "CITES": 204, "EXTENDS": 37}
}
```

---

## 5. list_entities_by_type

List entities of a given type, ranked by mention count descending.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity_type` | string | required | One of: `topic`, `method`, `material`, `author`, `journal`, `keyword` |
| `top_k` | int | 50 | Maximum entities to return |

**Returns** `list[{canonical, type, mention_count, surface_forms}]`

**Example call**
```json
{ "entity_type": "method", "top_k": 10 }
```

**Example response**
```json
[
  {"canonical": "cation exchange chromatography", "type": "method", "mention_count": 312, "surface_forms": ["CEX", "cation exchange"]},
  {"canonical": "size exclusion chromatography",  "type": "method", "mention_count": 278, "surface_forms": ["SEC"]}
]
```

**Raises** `ValueError` when `entity_type` is not in the closed vocabulary.

---

## 6. get_schema

Return a Markdown-formatted schema description for prompt injection.
Useful for priming the LLM before authoring `run_cypher` queries.

**Arguments** None.

**Returns** `string` — Markdown describing node types, edge types, and properties.

**Example call**
```json
{}
```

**Example response** (truncated)
```
## Node types

### Paper
- doi STRING (primary key)
- title STRING
- year INT64
- journal STRING

### Entity
- canonical_id STRING (primary key)
- type STRING  — one of: topic, method, material, author, journal, keyword
- surface STRING

## Relationship types
...
```

---

## 7. find_papers_by_query

Find papers via free-text query using the graph entity alias chain.
Resolves the query to a canonical entity ID, then returns papers ranked
by MENTIONS overlap. Returns `[]` if the query cannot be resolved.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | string | required | Free-text search string |
| `k` | int | 20 | Maximum results |

**Returns** `list[{doi, title, year, journal}]`

**Example call**
```json
{ "query": "protein A chromatography", "k": 10 }
```

---

## 8. find_papers_by_query_hybrid

Find papers via free-text query using RRF-fused graph + vector retrieval.

Graph leg: entity alias-resolve → MENTIONS overlap ranking.
Vector leg: ChromaDB semantic search (paper-level embeddings). When the
vector backend is unavailable, degrades gracefully to graph-only results.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | string | required | Free-text search string |
| `k` | int | 20 | Maximum results |

**Returns** `list[{doi, title, year, journal}]`, RRF-fused when both
legs are available, otherwise graph-only.

**Example call**
```json
{ "query": "antibody aggregate removal", "k": 15 }
```

---

## 9. run_cypher

Execute a read-only Cypher query against the live KuzuDB graph.
Power-user escape hatch — call `get_schema` first to understand the
node/edge types available.

Every query passes through a safety guard before reaching Kuzu:
- Comments stripped (`//` and `/* */`).
- Mutation keywords blocked (`CREATE`, `MERGE`, `DELETE`, `SET`, `DROP`,
  `ALTER`, `REMOVE`, `LOAD CSV`) — raises `ValueError`.
- `LIMIT {limit}` appended when the query has none.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | string | required | Read-only Cypher query string |
| `limit` | int | 100 | Maximum rows returned; ignored when query already has LIMIT |

**Returns** `list[{col: value, ...}]` — one dict per result row, column
names as keys. Values are JSON-serializable (Kuzu node objects are
flattened to their property dicts).

**Example call**
```json
{ "query": "MATCH (p:Paper)-[:EXTENDS]->(q:Paper) RETURN p.doi AS src, q.doi AS tgt", "limit": 50 }
```

**Example response**
```json
[
  {"src": "10.1002/btpr.3500", "tgt": "10.1016/j.foo.2024.01"}
]
```

**Raises** `ValueError` (CypherSafetyError) for mutation queries.
`RuntimeError` when the graph backend is unavailable.

---

## 10. semantic_search

Vector retrieval over the production ChromaDB persist dir.
Uses `mxbai-embed-large` (via Ollama) to embed the query on the fly.

Returns `[]` without raising when the persist dir is missing or the
collection is empty.

**Arguments**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | string | required | Free-text query string |
| `top_k` | int | 10 | Results to return; must be in [1, 100] |
| `granularity` | string | `"paper"` | `"paper"` or `"chunk"` |

**Returns (paper granularity)**
```
list[{doi, title, year, score}]
```

**Returns (chunk granularity)**
```
list[{doi, chunk_id, snippet, score}]
```

`score` is the cosine similarity in [0, 1].
`snippet` is truncated to 500 characters.
`chunk_id` format: `"<doi>#<chunk_index>"`.

**Example call — paper**
```json
{ "query": "HIC purification of monoclonal antibodies", "top_k": 5, "granularity": "paper" }
```

**Example response — paper**
```json
[
  {"doi": "10.1016/j.foo.2024.01", "title": "mAb purification review", "year": 2024, "score": 0.91},
  {"doi": "10.1002/btpr.3456",     "title": "CEX screening of mAbs",   "year": 2023, "score": 0.84}
]
```

**Example call — chunk**
```json
{ "query": "aggregate clearance polishing step", "top_k": 3, "granularity": "chunk" }
```

**Example response — chunk**
```json
[
  {
    "doi":      "10.1016/j.foo.2024.01",
    "chunk_id": "10.1016/j.foo.2024.01#2",
    "snippet":  "The polishing step on a mixed-mode resin reduced HMW species from 3.2% to 0.4%...",
    "score":    0.88
  }
]
```

**Raises** `ValueError` when `granularity` is not `"paper"` or `"chunk"`,
or when `top_k` is not an int in [1, 100].
