# lit-monitor

> **Status: Alpha.** Working MVP, used personally by the author. APIs, configs,
> and on-disk state layout may change without notice. No backwards-compatibility
> guarantees yet. Feedback and issues welcome.

A personal weekly literature monitor for scientific research.
`lit-monitor` searches academic databases for new papers, ranks them by similarity
to your existing Zotero library, and writes structured Markdown notes into an
Obsidian vault via LLM extraction.

## What it does

- **Weekly discovery** — searches PubMed, arXiv, and Scopus for new papers
  matching configured topics, then ranks results by cosine similarity to your
  library embeddings.
- **Brain build** — batch-extracts every paper and review in your Zotero
  collection into structured Obsidian notes using a two-phase LLM pipeline
  (simple regurgitative fields, then complex synthesis fields).
- **Obsidian tools** — relink related notes, regenerate notes from stored
  extractions, and synthesize across topics using chunk-level RAG with
  cross-encoder reranking.

LLM calls default to **Ollama Cloud** (`gemma4:31b-cloud`). Switch to local
Ollama by editing `config/extraction.yaml`, or set `provider: litellm` for any
LiteLLM-compatible provider (Anthropic, OpenAI, Vertex AI, etc.). Embeddings
always run locally via `mxbai-embed-large`.

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
uv sync --extra dev --extra cloud
for f in config/*.example.yaml; do cp -n "$f" "${f%.example.yaml}.yaml"; done
```

Drop `--extra cloud` if you do not need LiteLLM (Anthropic / OpenAI / Vertex)
routing.

## First-time setup

**1. Create a credentials file** at `~/.config/lit-monitor/config.toml`:

```toml
[zotero]
api_key    = "YOUR_ZOTERO_API_KEY"
library_id = "YOUR_NUMERIC_LIBRARY_ID"

[pubmed]
email = "you@example.com"   # CrossRef polite-pool routing

[ollama]
api_key = "YOUR_OLLAMA_CLOUD_KEY"   # optional, only for cloud Ollama
```

The shell env var `OLLAMA_API_KEY` overrides the TOML value if both are set.

**2. Edit the seeded configs in `./config/`:**

| File | What to edit |
|---|---|
| `paths.yaml` | Zotero `library_id`, Obsidian `vault_path` (full absolute path) |
| `topics.yaml` | Your weekly search queries |
| `domain_context.yaml` | One paragraph describing your field |
| `concepts.yaml` | Theme vocabulary — or generate via `lit-monitor build-vocabulary` |
| `researchers.yaml` | Authors you want to track (optional) |

These files are gitignored, so `git pull` will never touch them. The
`*.example.yaml` versions in git serve as documented templates.

**3. Verify** — `lit-monitor check`.

## Usage

```bash
# One-time setup
lit-monitor build-vocabulary       # cluster Zotero tags into themes
lit-monitor brain-build --resume   # extract every paper in the configured collection

# Weekly
lit-monitor run                    # discover new papers + ingest new Zotero items

# Obsidian helpers
lit-monitor obsidian relink                       # update Related Work sections
lit-monitor obsidian rerender                     # regenerate notes from stored extractions
lit-monitor obsidian synthesize --topic "..."     # chunk-level RAG with reranking
```

A top-level `-v` / `--verbose` flag enables DEBUG-level console output. A full
DEBUG log is always written to `logs/{date}_{command}.jsonl`.

For the full command surface (model comparison, citation graph rebuild,
targeted field re-extraction, `--all-library` mode, etc.) see
`lit-monitor --help`.

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
the LLM gets no domain context — `diagnose` catches that).

## Deployment

Designed to run weekly on a single workstation. Once the local state DB and
ChromaDB store have been populated by a brain-build, the weekly schedule can
be handed off to a low-power node (Raspberry Pi 4/5, etc.) via a systemd
timer, provided extraction is routed through cloud Ollama or LiteLLM.

## License

[MIT](LICENSE)
