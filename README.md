# lit-monitor

> **Status: Alpha.** Working tool, used personally by the author. APIs, configs,
> and on-disk state may still change without notice. No backwards-compat
> guarantees pre-1.0. Issues + feedback welcome.

A personal literature tracker for any research field. It searches
PubMed, arXiv, and Scopus on a schedule, ranks new papers by **semantic
similarity to your existing Zotero library**, extracts structured fields with
an LLM, builds a knowledge graph of entities and relationships, and writes
everything into your Obsidian vault. Ask natural-language questions of the
corpus from the terminal, a browser, or any MCP-compatible AI client.

Nothing in the pipeline is domain-specific: your Zotero library, search
topics, and a free-text domain paragraph are the only signals it needs, so
it adapts to whatever you read. It was built for — and is dogfooded daily on —
downstream biopharmaceutical process development, which is why the defaults
and some examples lean that way. Ready-made starting configs for other fields
(ML research, climate science, bioprocessing) live in
[`config/examples/`](config/examples/) — see [Install](#install).

Drive it from a **localhost web UI** (`lit-monitor serve`) for setup and
day-to-day operation, or from the **CLI** (`lit-monitor --help`) for scripted
work. Same pipeline either way.

## What you get

- **Topic search + relevance ranking.** Recurring searches across PubMed,
  arXiv, and Scopus. Each candidate ranked by cosine similarity to embeddings
  of your existing Zotero library. Embeddings run locally via
  `mxbai-embed-large` (or any LiteLLM-compatible provider) against a
  per-machine ChromaDB store.
- **Obsidian-native output.** Every paper becomes a structured Markdown note
  with persist zones for your own annotations, two-phase LLM extraction
  (simple regurgitative fields + complex synthesis fields), and a citation
  graph rebuild path.
- **Knowledge graph + ask interface.** A KuzuDB graph stores entities (topics,
  methods, materials, authors, journals, keywords) and ten typed
  relationships across the corpus. Ask questions in plain English from the
  CLI (`lit-monitor ask "what methods extend Carta 2009?"`) or via HTTP /
  MCP. The pipeline writes both the vector and the graph backends atomically
  per paper. `ask` is cluster-aware: when your library has been theme-clustered,
  responses are automatically contextualised to the most relevant theme.
- **Three retrieval modes** — vector (semantic), graph (entity-typed), and
  hybrid (reciprocal-rank fusion). Switch per-command with
  `--rag-mode {vector,graph,hybrid}`.
- **MCP server for AI clients.** Twelve tools that Claude Desktop, Cursor,
  Continue, and any other MCP-capable agent can call to query the graph and
  vector index. Includes a read-only Cypher escape hatch with a regex safety
  guard.
- **Notifications + flexible delivery.** OS notification when a discovery run
  finishes. Click it to land in your preferred viewer (browser, Obsidian, or
  dismiss). Weekly digest as `.md`, on-demand Markdown export, or rich
  table in the terminal — your choice.

## How it works

### Library-as-signal

Every paper in your Zotero collection is embedded and stored in a per-machine
ChromaDB instance.  When a discovery run finds a candidate, its abstract is
embedded and compared against your library — the cosine similarity is the base
relevance score.  Papers that are already close to what you've read rank
higher; papers in a completely different domain rank lower.

### Score decomposition

The final ranking score is a weighted sum of six signals:

| Signal | What it measures |
|---|---|
| `vector` | Cosine similarity of the candidate to your Zotero library centroid |
| `domain_context` | Cosine similarity to the optional free-text domain focus paragraph |
| `cluster_centroid` | Similarity to the nearest theme cluster in your library |
| `graph_entity_overlap` | How many named entities the candidate shares with your graph |
| `graph_citation` | Citation edges (CITES / EXTENDS) to papers already in the graph |
| `graph_shared_authors` | Authors appearing in both the candidate and your graph |

Every paper in a discovery result carries a `score_breakdown` dict with all
six values.  The web UI's paper card shows the decomposition as a stacked bar.

### Theme clustering

Once your library reaches ~100 papers (configurable), `lit-monitor` runs
K-means over the embedding space and asks the LLM to name each cluster.  The
named clusters (themes) are written back to Zotero as tags or collections on
request.  Clusters update automatically on each subsequent brain-build.

### Domain extraction

`domain_context.yaml` accepts a free-text paragraph describing your focus
areas.  The `domain analyze` command uses a single LLM call to extract
structured concepts (techniques, targets, assay types, keywords) and stores
them in `state.db`.  These feed the `domain_context` ranking signal without
any manual tagging.

### Trending concepts and query expansion

The graph tracks which entity types are mentioned most in recently accepted
papers.  `lit-monitor trending suggest` surfaces new concepts that are rising
in frequency.  Accept a suggestion to add it to your active topics; dismiss
it to suppress it until the next review cycle.

### Embedding providers

Embeddings default to local Ollama (`mxbai-embed-large`).  Switch to any
LiteLLM-compatible provider (OpenAI `text-embedding-3-small`, Anthropic, etc.)
without changing the rest of the pipeline:

```bash
uv sync --extra litellm
lit-monitor embeddings switch --provider litellm --model text-embedding-3-small
lit-monitor embeddings rebuild   # re-embed your whole library under the new model
```

The `embeddings status` command shows current provider, model, and how many
papers are already embedded.

## Setup the way you work

Three common configurations, from zero to fully customised.

### (a) Just run it — minimal config

Enough to get a weekly discovery feed and Obsidian notes.

1. `cp config/*.example.yaml config/` (if you didn't run `install.sh`).
2. Add Zotero credentials to `~/.config/lit-monitor/config.toml`.
3. Set `obsidian_vault_path` and `zotero_library_id` in `config/paths.yaml`.
4. Add 2-3 search topics in `config/topics.yaml`.
5. `lit-monitor check` — verify connectivity.
6. `lit-monitor brain-build` — index your existing library (one-time).
7. `lit-monitor run` — first discovery run.
8. `lit-monitor serve` to browse results at `http://127.0.0.1:8765`.

No domain context, no clustering, no graph signals yet — all optional.
The vector similarity signal works on its own.

### (b) Clustering on — after 100 papers

Enable after `brain-build` has indexed at least 100 papers.

1. `lit-monitor cluster recompute` — run K-means and name clusters.
2. `lit-monitor cluster view` — inspect the resulting themes.
3. Optionally: `lit-monitor cluster write-back tags` to tag Zotero items.
4. Set `cluster_centroid_weight: 0.2` under `ranking:` in
   `config/extraction.yaml` to include the cluster signal in scoring.

Clustering re-runs automatically on subsequent `brain-build` calls once
`clustering.enabled: true` is set.  Adjust `clustering.n_clusters` for the
right granularity (8-15 is a good starting range for a 500-paper library).

```bash
lit-monitor cluster recompute        # run K-means + LLM naming
lit-monitor cluster view             # show cluster themes + paper counts
lit-monitor cluster assign           # assign every paper to nearest cluster
lit-monitor cluster write-back tags  # tag Zotero items with theme names
```

### (c) Graph signals — tuning the ranking mix

Enable once the knowledge graph has been populated (`lit-monitor graph backfill
--all`) and you want the citation and entity-overlap signals.

1. Run `lit-monitor graph backfill --all` (first time only; incremental
   thereafter).
2. Set nonzero weights in `config/extraction.yaml`:

```yaml
ranking:
  graph_entity_overlap_weight: 0.15
  graph_citation_weight:       0.10
  graph_shared_authors_weight: 0.05
```

3. (Optional) add a `domain_context.yaml` paragraph and run
   `lit-monitor domain analyze` to activate the `domain_context` signal:

```yaml
ranking:
  domain_context_weight: 0.20
```

4. Check the score decomposition in the discovery web UI or with
   `lit-monitor discovery view --run latest --breakdown` to confirm all
   signals are contributing as expected.

Trending-concept suggestions and researcher gating are independent features —
see `lit-monitor trending suggest` and `config/researchers.yaml` respectively.
- **Run on any schedule.** One-click installer for launchd (macOS) or
  systemd user timers (Linux). Weekly works. Run ad-hoc from the dashboard
  whenever you want.

LLM calls default to local **Ollama**. Switch to Ollama Cloud or any LiteLLM-
compatible provider (Anthropic, OpenAI, Vertex AI) from the setup wizard or
by editing `config/extraction.yaml`. Embeddings always run locally.

## Requirements

- macOS or Linux (developed on an M2 MacBook Air; ARM Linux including
  Raspberry Pi 4/5 works for cloud-Ollama configurations)
- Python 3.11+
- [Ollama](https://ollama.com) installed locally for embeddings
  (`ollama pull mxbai-embed-large`)
- A Zotero library (Better BibTeX optional)
- An Obsidian vault (full absolute path required)

## Install

```bash
git clone https://github.com/max3925vats/lit-monitor.git
cd lit-monitor
./install.sh
```

The script installs [`uv`](https://docs.astral.sh/uv/) if needed, creates a
project-local `.venv`, resolves all dependencies, and seeds personal configs
from `config/*.example.yaml`.

**Starting from a non-biopharma field?** `config/examples/` ships filled-in,
synthetic config sets for `bioprocessing/`, `ml-research/`, and
`climate-science/`. Pick the closest one and copy its `topics.yaml`,
`domain_context.yaml`, `concepts.yaml`, and `researchers.yaml` into `config/`
as a head start, then edit:

```bash
cp config/examples/ml-research/*.yaml config/   # or bioprocessing/ climate-science/
```

The non-domain configs (`paths.yaml`, `extraction.yaml`) still come from
`config/*.example.yaml`. See [`config/examples/README.md`](config/examples/README.md).

If you prefer to drive `uv` yourself:

```bash
uv venv && source .venv/bin/activate
uv sync                                  # web UI, graph, MCP, notifications all included
for f in config/*.example.yaml; do cp -n "$f" "${f%.example.yaml}.yaml"; done
```

Optional extras:

- `--extra nlp` — BiobertNER for entity extraction (~3 GB; transformers +
  torch). Without it, the LLM fallback handles entities.
- `--extra litellm` — multi-provider LLM routing (Anthropic, OpenAI, Vertex
  AI, etc.). Without it, you're on Ollama only.
- `--extra dev` — contributor tooling (ruff, pytest, mypy).

## Quickstart — web UI

```bash
lit-monitor serve
```

Open **`http://127.0.0.1:8765/setup`** in any browser. An 8-step wizard
covers credentials, paths, extraction config, topics, domain context, theme
vocabulary, tracked researchers, and item routing. Live credential checks at
each step.

After setup, the dashboards take over:

| URL | What |
|---|---|
| `/brain-build` | Extract your existing Zotero library into the index. Progress bar, per-paper table, recent runs. Start / Stop / Resume. Live JSONL log stream. |
| `/discovery` | Latest discovery run summary, run history, per-paper cards with one-click relink / re-extract actions. Run-now / Dry-run / Stop buttons. |
| `/corpus` | A read lens on every processed paper in the state DB: searchable / filterable list, plus a per-paper detail view (extraction, score, knowledge-graph, related work, Zotero / Obsidian links, relink / re-extract). Not a Zotero replacement — it reads what the pipeline already processed. |
| `/ask` | Ask natural-language questions of your corpus (the `lit-monitor ask` pipeline in the browser): prose answer + results table, with show/edit Cypher, recent-questions history, and save-to-vault. Requires the knowledge graph (`lit-monitor graph backfill --all`). |
| `/schedule` | Install or remove a recurring schedule (launchd / systemd). |

The server binds to `127.0.0.1` by default. Pass `--host 0.0.0.0` if you
want LAN access; the server has no authentication of its own.

## Quickstart — CLI

If you'd rather edit YAML by hand:

**1. Create credentials** at `~/.config/lit-monitor/config.toml`:

```toml
[zotero]
api_key    = "YOUR_ZOTERO_API_KEY"
library_id = "YOUR_NUMERIC_LIBRARY_ID"

[pubmed]
email = "you@example.com"

[ollama]
api_key = "YOUR_OLLAMA_CLOUD_KEY"   # optional, only for cloud Ollama
```

**2. Edit the seeded YAMLs in `./config/`**: `paths.yaml`, `topics.yaml`,
`domain_context.yaml`, `concepts.yaml`, `researchers.yaml`. The example
files in git document every field; the real `.yaml` files are gitignored.

**3. Verify** — `lit-monitor check`.

## Day-to-day CLI

```bash
# One-time setup
lit-monitor build-vocabulary             # cluster Zotero tags into themes
lit-monitor brain-build --resume         # extract every paper in the collection

# Discovery
lit-monitor run                          # discover new papers + ingest new Zotero items
lit-monitor run --dry-run                # preview without writes
lit-monitor discovery view --run latest  # rich-table view of the latest results
lit-monitor discovery export-md --to ~/digest.md   # on-demand Markdown export

# Ask (cluster-aware since v0.9)
lit-monitor ask "what methods extend cation exchange?" --rag-mode hybrid
# Responses are automatically contextualised to your dominant theme when clusters exist.

# Domain focus extraction (v0.9)
lit-monitor domain analyze               # LLM-extract structured concepts from domain_context.yaml
lit-monitor domain view                  # show extracted focus areas
lit-monitor domain clear                 # reset extracted domain focus

# Theme clustering (v0.9, requires ~100 papers)
lit-monitor cluster recompute            # run K-means + LLM naming
lit-monitor cluster view                 # show cluster themes + paper counts
lit-monitor cluster assign               # assign papers to nearest cluster
lit-monitor cluster write-back tags      # tag Zotero items with theme names
lit-monitor cluster write-back collections  # move Zotero items into per-theme collections

# Trending concepts + query expansion (v0.9)
lit-monitor trending suggest             # surface rising entity types
lit-monitor trending view                # list pending / accepted / dismissed suggestions
lit-monitor trending accept <id>         # add concept to active topics
lit-monitor trending dismiss <id>        # suppress concept until next cycle

# Embeddings (v0.9)
lit-monitor embeddings status            # show provider, model, coverage
lit-monitor embeddings switch --provider litellm --model text-embedding-3-small
lit-monitor embeddings rebuild           # re-embed library under new model

# Knowledge graph
lit-monitor graph status                 # node + edge counts
lit-monitor graph backfill --all         # index existing papers into the graph
lit-monitor graph propose-aliases        # suggest entity normalization rules

# Obsidian helpers
lit-monitor obsidian relink              # update Related Work sections
lit-monitor obsidian rerender            # regenerate notes from stored extractions
lit-monitor obsidian sync --all          # write deferred per-paper notes
lit-monitor obsidian synthesize --topic "..."   # chunk-level RAG with reranking

# Health
lit-monitor check                        # config + Ollama + Zotero reachability
lit-monitor diagnose                     # strict-mode config audit
lit-monitor status                       # extraction + embedding + graph counts
```

Top-level `-v` / `--verbose` enables DEBUG console output; a full DEBUG log
is always written to `logs/{date}_{command}.jsonl`. See
`lit-monitor --help` for the full command surface.

## MCP server (for Claude Desktop / Cursor / Continue)

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
`get_paper_details`, `find_papers_by_query_hybrid` (RRF-fused),
`run_cypher` (read-only with safety guard), `semantic_search`,
`get_recent_discovery_runs`, and 5 more. Run `lit-monitor mcp serve` to see
the registry.

## HTTP API

`lit-monitor serve` exposes the same query layer over HTTP. FastAPI
auto-docs at `http://127.0.0.1:8765/docs`. Endpoints include
`POST /api/ingest`, `GET /api/papers/{doi}`, `POST /api/ask`,
`POST /api/cypher`, `POST /api/search`, `GET /api/discovery/runs`, plus
trigger endpoints for relink and re-extract.

## Notifications + delivery

OS notification when a discovery run completes. Clicking the notification
lands at the chooser page on first use, then remembers your preferred
surface (browser / Obsidian / dismiss). Four delivery flags under
`discovery:` in `config/extraction.yaml`:

| Key | Default | Effect |
|---|---|---|
| `notify.enabled` | `true` | Fire OS notification at run end |
| `notify.preferred_viewer` | `""` | Skip the chooser when set |
| `notes.auto_write_per_paper` | `true` | `false` → defer per-paper notes to `obsidian sync` |
| `digest.auto_write` | `true` | `false` → no inline digest; use `discovery export-md` |

## Strict mode + diagnose

```bash
lit-monitor --strict run --dry-run        # CLI flag
LIT_MONITOR_STRICT=1 lit-monitor run      # env var
lit-monitor diagnose --config-only        # validate every tracked config
lit-monitor diagnose                      # full health (config + Ollama + Zotero)
```

Strict mode turns every silent fallback (corrupt config, unreadable
attachment, unexpected API response) into a hard error.

## LLM providers — Ollama (default) or LiteLLM

Local Ollama is the default for both LLM inference and embeddings.

### LLM routing

To route any extraction mode through LiteLLM:

```bash
uv sync --extra litellm
```

Then per-mode in `config/extraction.yaml`:

```yaml
modes:
  simple:
    provider: litellm
    litellm_model: claude-3-5-sonnet-20241022
    # ... other keys unchanged ...
  complex:
    provider: litellm
    litellm_model: claude-opus-4-5
```

Mix per-mode — local Ollama for `simple`, cloud Claude for `complex`. API
keys come from your environment per
[LiteLLM's provider docs](https://docs.litellm.ai/docs/providers).

### Embedding routing (v0.9)

Embeddings also support LiteLLM providers, letting you use OpenAI, Cohere,
Vertex AI, or any other provider that exposes an embeddings endpoint:

```bash
lit-monitor embeddings switch --provider litellm --model text-embedding-3-small
# Or switch back to local Ollama:
lit-monitor embeddings switch --provider ollama --model mxbai-embed-large
```

After switching, run `lit-monitor embeddings rebuild` to re-embed your
library under the new model.  The `embeddings status` command shows current
provider, model, embedding dimensionality, and how many papers are indexed.

## Running tests

```bash
.venv/bin/python -m pytest tests/unit/ tests/llm/ -q

# Integration tests — SKIP if Ollama / Zotero / vault are unavailable
lit-monitor check && .venv/bin/python -m pytest tests/integration/ -m integration -v

# Integration tests in STRICT mode — missing services FAIL the run
LIT_MONITOR_LIVE=1 .venv/bin/python -m pytest tests/integration/ -m integration -v
```

## Deployment

Runs on a recurring schedule on a single workstation. Install via
`/schedule` (launchd on macOS, systemd user timer on Linux). Weekly is a
sensible default; any cadence works. Once the local state DB and ChromaDB
store have been populated — either by brain-build in one shot, or
organically by repeated `lit-monitor run` calls — the schedule can be handed
off to a low-power node (Raspberry Pi 4/5) provided extraction is routed
through Ollama Cloud or LiteLLM.

## License

[MIT](LICENSE)
