# lit-monitor

> **Status: Alpha (v0.2.0).** Working MVP, used personally by the author. The
> web UI shipped in v0.2.0 makes setup + day-to-day operation point-and-click;
> APIs, configs, and on-disk state layout may still change without notice. No
> backwards-compatibility guarantees pre-1.0. Feedback and issues welcome.

A personal literature tracker that ranks new papers by **semantic similarity
to your existing Zotero library** — find what matters, written into Obsidian.
Automate the discovery on a schedule (weekly works) to stay current.

Drive it from a **localhost web UI** (`lit-monitor serve`) or — for scripted /
power-user work — from the **CLI** (`lit-monitor --help`). Same pipeline either
way; the web UI is just a thin layer on top of the same Click commands.

## What it does

- **Topic search + relevance ranking** — searches PubMed, arXiv, and Scopus
  for papers matching configured topics, then **ranks results by cosine
  similarity to embeddings of your existing Zotero library**. The relevance
  signal that makes the digest worth reading. Embeddings run locally via
  `mxbai-embed-large` against a per-machine ChromaDB store.
- **Obsidian-native output** — every paper becomes a structured Markdown note
  with persist zones for your own annotations, two-phase LLM extraction (simple
  regurgitative fields, then complex synthesis fields), chunk-level RAG for
  cross-paper synthesis (`obsidian synthesize`), cross-encoder reranking for
  "show me what I should read next" (`obsidian relink`), and a citation-graph
  rebuilder (`rebuild-citations`).
- **Run on any schedule** — once a week is a sensible default; `lit-monitor
  serve` exposes a one-click schedule installer (launchd on macOS, systemd
  user timer on Linux). Run ad-hoc from the dashboard's "Run now" button
  whenever you want.
- **Optional accelerators** (cost real LLM tokens) — `brain-build` batches
  your existing Zotero library into the embedding index in one pass;
  `build-vocabulary` clusters Zotero tags into theme labels. Both are
  **skippable** — the embedding index also grows organically from week 1 if you
  just configure topics and `lit-monitor run`. Use these to front-load
  relevance ranking if you already have a populated Zotero library.

LLM calls default to **Ollama Cloud** (`gemma4:31b-cloud`). Switch to local
Ollama or any LiteLLM-compatible provider (Anthropic, OpenAI, Vertex AI) from
the web UI's setup wizard, or by editing `config/extraction.yaml` directly.
Embeddings always run locally via Ollama.

## Requirements

- macOS or Linux (developed on an M2 MacBook Air; ARM Linux including Raspberry
  Pi 4/5 works for cloud-Ollama configurations)
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
project-local `.venv`, resolves all dependencies (`uv` routes around a known
`findpapers`/`chromadb` metadata conflict via a `[tool.uv]` override in
`pyproject.toml`), and seeds the personal configs from `config/*.example.yaml`.

If you prefer to drive `uv` yourself:

```bash
uv venv && source .venv/bin/activate
uv sync --extra dev --extra litellm --extra server
for f in config/*.example.yaml; do cp -n "$f" "${f%.example.yaml}.yaml"; done
```

Drop `--extra litellm` if you do not need LiteLLM routing (`--extra cloud` also
works as a deprecated alias). The `--extra server` brings in FastAPI + uvicorn
for `lit-monitor serve`; drop it only if you plan to run CLI-only and never
touch the web UI.

## Quickstart — web UI

The fastest path to a working install. Walks you through every config file,
runs live credential checks, and ships you to the dashboard when you're done.

```bash
lit-monitor serve
```

Then open **`http://127.0.0.1:8765/setup`** in any browser. The 8-step wizard
covers:

1. **Credentials** — Zotero API key + library ID, PubMed email, optional
   Ollama Cloud key. Stored at `~/.config/lit-monitor/config.toml` (mode 0600).
   Live Zotero ping after save.
2. **Paths** — Obsidian vault picker (server-side folder browser modal) +
   Zotero collection dropdown (live-loaded from your library).
3. **Extraction** — LLM provider/model/temperature/timeout per mode, with a
   live "Test Ollama" button per host.
4. **Topics** — recurring search queries (add/remove rows).
5. **Domain context** — short paragraph telling the LLM what your field is.
6. **Concepts** — view the theme vocabulary; one-click "Regenerate" runs
   `lit-monitor build-vocabulary` with live log streaming.
7. **Researchers** — optional tracked-author list.
8. **Item routing** — read-only view of how Zotero item types map to
   pipelines; advanced YAML editor underneath.

After the wizard, the dashboards take over:

| URL | What |
|---|---|
| `/brain-build` | Progress bar, per-paper table, recent runs, failed-papers expander. Start / Stop / Resume buttons. Live JSONL log streaming. Collection switcher. |
| `/discovery` | Last-run summary, history, today's digest. Run-now / Dry-run / Stop buttons. |
| `/schedule` | Install or remove a recurring schedule (launchd on macOS, systemd user timer on Linux). Weekly is a sensible default. |

The server binds to `127.0.0.1` by default. Pass `--host 0.0.0.0` if you want
LAN access (e.g. running on a Pi, browsing from a laptop) — that is the only
intended remote-access mode; the server has no authentication of its own.

## Quickstart — CLI

If you'd rather skip the web UI and edit the YAML configs by hand:

**1. Create the credentials file** at `~/.config/lit-monitor/config.toml`:

```toml
[zotero]
api_key    = "YOUR_ZOTERO_API_KEY"
library_id = "YOUR_NUMERIC_LIBRARY_ID"

[pubmed]
email = "you@example.com"

[ollama]
api_key = "YOUR_OLLAMA_CLOUD_KEY"   # optional, only for cloud Ollama
```

**2. Edit the seeded configs in `./config/`**: `paths.yaml`, `topics.yaml`,
`domain_context.yaml`, `concepts.yaml`, `researchers.yaml`. The seeded
`*.example.yaml` files in git document every field; the real `*.yaml` files
are gitignored so `git pull` never touches them.

**3. Verify** — `lit-monitor check`.

## Power-user CLI

The web UI is a thin wrapper; every action is also a Click command.

```bash
# One-time setup
lit-monitor build-vocabulary             # cluster Zotero tags into themes
lit-monitor brain-build --resume         # extract every paper in the configured collection

# Discovery (the recurring run — call it weekly if you schedule it weekly)
lit-monitor run                          # discover new papers + ingest new Zotero items
lit-monitor run --dry-run                # preview discovery without writes

# Obsidian helpers
lit-monitor obsidian relink                       # update Related Work sections
lit-monitor obsidian rerender                     # regenerate notes from stored extractions
lit-monitor obsidian synthesize --topic "..."     # chunk-level RAG with reranking

# Verification + diagnostics
lit-monitor check                        # reachability check (config + Ollama + Zotero)
lit-monitor diagnose                     # strict-mode config audit
lit-monitor status                       # extraction + embedding counts from state DB

# Web UI
lit-monitor serve [--port 8765] [--host 127.0.0.1] [--reload]
```

A top-level `-v` / `--verbose` flag enables DEBUG-level console output. A full
DEBUG log is always written to `logs/{date}_{command}.jsonl`. For the full
command surface (model comparison, citation graph rebuild, targeted field
re-extraction, `--all-library` mode, etc.) see `lit-monitor --help`.

### Graph RAG (optional — v0.4.0+)

A KuzuDB knowledge graph runs alongside the ChromaDB vector store. Install the
optional extra to enable it:

```bash
uv sync --extra graph           # install kuzu

lit-monitor graph backfill --all            # index all papers into the graph
lit-monitor graph status                    # show node + edge counts
lit-monitor graph propose-aliases           # suggest entity normalization rules

# Use graph or hybrid retrieval (default is "vector"):
lit-monitor ask "what methods compare X to Y" --rag-mode hybrid
lit-monitor synthesize --rag-mode graph
```

When the `[graph]` extra is not installed, all graph-aware commands print a
friendly install message and exit 0 — the rest of the pipeline is unaffected.
`lit-monitor diagnose --config-only` includes a `graph` row showing extra
availability and persist-dir reachability. `lit-monitor status` appends a
`Graph: indexed=N / total=M  entities=K` line when the graph is active.

### Graph RAG with NER (v0.5.0+, optional)

Phase 2 layers domain-aware NER on top of Phase 1. Install:

```bash
uv sync --extra graph --extra nlp
```

Optional cloud-Ollama long-tail validation:

```bash
export OLLAMA_API_KEY=your_key
# Edit config/extraction.yaml: set graph.ner.cloud_long_tail_enabled=true
```

New flags:

- `lit-monitor graph backfill --ner` — process existing papers via BioBERT.
- `lit-monitor graph backfill --ner-with-llm` — also use cloud-Ollama
  long-tail (caps off the schema → BioBERT → LLM merge pipeline).
- `lit-monitor graph propose-aliases --with-llm` — LLM-validated alias
  consensus.
- `lit-monitor obsidian re-extract --rag-mode graph` — re-extract with
  corpus-aware graph context.
- `lit-monitor graph status --by-source` — MENTIONS counts per source.

### MCP server — graph + vector RAG over Model Context Protocol (v0.7.0+)

The MCP server exposes 10 tools that AI clients (Claude Desktop, Continue,
Cursor) can call to query the lit-monitor knowledge graph and vector index.

```bash
uv sync --extra mcp --extra graph
lit-monitor mcp serve
```

The server uses stdio transport. Register it in your MCP client's config:

```json
"lit-monitor-graph": {
  "command": "lit-monitor",
  "args": ["mcp", "serve"]
}
```

Tools include `find_papers_by_entity`, `get_paper_details`,
`find_papers_by_query_hybrid` (RRF-fused graph + vector),
`run_cypher` (read-only with safety guard), `semantic_search`, and 5 more.
See [`docs/MCP_TOOLS.md`](docs/MCP_TOOLS.md) for the full reference.

### Discovery + notifications (v0.8.0)

The discovery pipeline writes structured results (runs + per-paper scores) into
`state.db`. Multiple surfaces render them on demand.

```bash
# Rich-formatted table for the most recent run
lit-monitor discovery view --run latest

# On-demand Markdown export
lit-monitor discovery export-md --run latest --to ~/discovery.md

# Per-paper Obsidian notes deferred from the discovery pipeline can be synced later
lit-monitor obsidian sync --all
```

**Optional OS notifications** — install with `uv sync --extra notify`. On run
completion an OS notification fires (macOS Notification Center, Linux
`notify-send`, Windows toast). Clicking the notification opens
`http://localhost:8765/discovery/notify-handler?run_id=N` — a chooser page on
first use, or a direct redirect to your preferred surface (browser / Obsidian /
dismiss) after you save a preference.

**Config flags** (under `discovery:` in `config/extraction.yaml`):

| Key | Default | Effect |
|---|---|---|
| `notify.enabled` | `true` | Fire OS notification at run end |
| `notify.preferred_viewer` | `""` | Skip the chooser when set to `browser`, `obsidian`, or `none` |
| `notes.auto_write_per_paper` | `true` | `false` → defer per-paper notes to `obsidian sync` |
| `digest.auto_write` | `true` | `false` → no inline digest .md; use `discovery export-md` |

### Phase 3 LLM relationships (v0.6.0+, optional)

Phase 3 adds `EXTENDS` / `CONTRADICTS` edges + LLM-augmented schema
relationships. Enable via:

```yaml
# config/extraction.yaml
graph:
  relationships:
    llm_enabled: true
```

Plus `OLLAMA_API_KEY` env var.

New CLI:
- `lit-monitor graph backfill --relationships` — schema-only relationship
  extraction over existing corpus.
- `lit-monitor graph backfill --relationships-with-llm` — also runs the
  cloud-Ollama LLM extractor (EXTENDS + CONTRADICTS + LLM-augmented predicates).
- `lit-monitor graph status` — now includes EXTENDS + CONTRADICTS edge counts.
- `lit-monitor graph status --by-source` — typed-predicate counts broken
  down by `prompt_version` (schema vs LLM split).

### Strict mode

`lit-monitor` has a strict mode that turns every silent fallback (corrupt
config, unreadable attachment, unexpected API response) into a hard error
instead of a logged warning:

```bash
# Activate via CLI flag (any subcommand):
lit-monitor --strict run --dry-run

# Activate via environment variable (useful in CI and scripts):
LIT_MONITOR_STRICT=1 lit-monitor run --dry-run

# Health check — validates all tracked config files without needing services:
lit-monitor diagnose --config-only

# Full health check including Ollama + Zotero reachability:
lit-monitor diagnose
```

`diagnose` activates strict mode internally and reports each config file as
`OK` or `FAIL`. Use it when `lit-monitor check` returns OK but something
feels wrong (e.g. a corrupt `domain_context.yaml` silently becomes `""` and
the LLM gets no domain context — `diagnose` catches it).

### LLM providers — Ollama (default) or LiteLLM

`lit-monitor` defaults to a local Ollama instance for all LLM extraction
calls. To use a cloud provider (Anthropic, OpenAI, Vertex AI, etc.) via
LiteLLM, install the extra and configure `extraction.yaml`:

```bash
uv sync --extra litellm
```

Then in `config/extraction.yaml`, set `provider` and `litellm_model` for each
mode you want to route through the cloud:

```yaml
modes:
  simple:
    provider: litellm
    litellm_model: claude-3-5-sonnet-20241022   # any LiteLLM-compatible model string
    # ... existing keys unchanged ...
  complex:
    provider: litellm
    litellm_model: claude-opus-4-5
```

Ollama and LiteLLM can be mixed per-mode — for example, local Ollama for
`simple` and cloud Claude for `complex`. API keys are read from your
environment per [LiteLLM's provider docs](https://docs.litellm.ai/docs/providers).

## Running tests

```bash
# Unit + LLM tests — no live services needed
.venv/bin/python -m pytest tests/unit/ tests/llm/ -q

# Integration tests — silently SKIP if Ollama / Zotero / vault are unavailable
lit-monitor check && .venv/bin/python -m pytest tests/integration/ -m integration -v

# Integration tests in STRICT mode — missing services FAIL the run
# (use this before tagging a release; the release-gate script invokes it for you)
LIT_MONITOR_LIVE=1 .venv/bin/python -m pytest tests/integration/ -m integration -v
```

## Deployment

Designed to run on a recurring schedule on a single workstation. The web UI
installs the schedule directly from the `/schedule` page (launchd on macOS,
systemd user timer on Linux). Weekly is a sensible default; any cadence works.
Once the local state DB and ChromaDB store have been populated — either by
brain-build in one shot, or organically by repeated `lit-monitor run` calls —
the schedule can be handed off to a low-power node (Raspberry Pi 4/5, etc.)
via the same mechanism, provided extraction is routed through cloud Ollama
or LiteLLM.

## License

[MIT](LICENSE)
