# lit-monitor

> **Status: Alpha.** Working tool, used personally by the author. APIs, configs,
> and on-disk state may still change without notice. No backwards-compat
> guarantees pre-1.0. Issues + feedback welcome.

A personal literature tracker for biopharmaceutical research. It searches
PubMed, arXiv, and Scopus on a schedule, ranks new papers by **semantic
similarity to your existing Zotero library**, extracts structured fields with
an LLM, builds a knowledge graph of entities and relationships, and writes
everything into your Obsidian vault. Ask natural-language questions of the
corpus from the terminal, a browser, or any MCP-compatible AI client.

Drive it from a **localhost web UI** (`lit-monitor serve`) for setup and
day-to-day operation, or from the **CLI** (`lit-monitor --help`) for scripted
work. Same pipeline either way.

## What you get

- **Topic search + relevance ranking.** Recurring searches across PubMed,
  arXiv, and Scopus. Each candidate ranked by cosine similarity to embeddings
  of your existing Zotero library. Embeddings run locally via
  `mxbai-embed-large` against a per-machine ChromaDB store.
- **Obsidian-native output.** Every paper becomes a structured Markdown note
  with persist zones for your own annotations, two-phase LLM extraction
  (simple regurgitative fields + complex synthesis fields), and a citation
  graph rebuild path.
- **Knowledge graph + ask interface.** A KuzuDB graph stores entities (topics,
  methods, materials, authors, journals, keywords) and ten typed
  relationships across the corpus. Ask questions in plain English from the
  CLI (`lit-monitor ask "what methods extend Carta 2009?"`) or via HTTP /
  MCP. The pipeline writes both the vector and the graph backends atomically
  per paper.
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

# Ask
lit-monitor ask "what methods extend cation exchange?" --rag-mode hybrid

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

Local Ollama is the default. To route any mode through LiteLLM:

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
