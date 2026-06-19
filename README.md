# lit-monitor for Zotero

**Semantic literature discovery for researchers who live in Zotero.**

lit-monitor tracks PubMed, arXiv, and Scopus on a schedule, ranks every new
paper against your existing Zotero library, extracts structured fields with an
LLM, and writes everything into your Obsidian vault — queryable from a browser,
the terminal, or any AI client that speaks the Model Context Protocol (MCP).

[![CI](https://github.com/max3925vats/lit-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/max3925vats/lit-monitor/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/lit-monitor)](https://pypi.org/project/lit-monitor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Platform: macOS | Linux](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](#requirements)
[![MCP compatible](https://img.shields.io/badge/MCP-compatible-blueviolet.svg)](docs/mcp.md)

Your library is the signal. Each candidate paper is scored by semantic
similarity to embeddings of what you already keep in Zotero, so the feed adapts
to whatever you read — papers close to your interests rank higher, papers in a
different domain rank lower. Nothing in the pipeline is domain-specific: your
Zotero library, a handful of search topics, and an optional free-text domain
paragraph are the only inputs it needs.

The default configuration and some examples lean toward downstream
biopharmaceutical process development — the domain the tool was originally
developed against — but ready-made starting configs for other fields (ML
research, climate science, and more) ship with the package in
[`lit_monitor/_data/config_examples/examples/`](lit_monitor/_data/config_examples/examples/).
See [Installation](docs/installation.md).

Drive it from a **localhost web UI** (`lit-monitor serve`) for setup and
day-to-day operation, or from the **CLI** (`lit-monitor --help`) for scripted
work. Same pipeline either way.

> **Local-first, free by default.** On the default configuration everything runs
> on your machine: your library is embedded and stored locally (Ollama +
> ChromaDB), with no per-call API costs. Only outbound paper searches reach
> PubMed, arXiv, and Scopus. Routing extraction or embeddings to a cloud provider
> (Anthropic, OpenAI, Vertex AI, Ollama Cloud) is opt-in.

> **Beta.** lit-monitor is feature-complete and in active daily use, but still
> maturing toward 1.0 — interfaces and on-disk formats may change between
> releases, and you may hit rough edges. Bug reports and feedback are very
> welcome via [GitHub Issues](https://github.com/max3925vats/lit-monitor/issues).
> Provided under the [MIT License](#license).

## Who it's for

lit-monitor is for researchers whose Zotero library has outrun them — the papers
you saved meaning to read, the ones you read once and half-forgot, the threads
you'd follow if the day job left the time. Rather than re-reading everything to
decide what matters now, it treats your library as a statement of interest: it
indexes what you've collected, then watches PubMed, arXiv, and Scopus and surfaces
new work ranked by closeness to *your* corpus — including the auxiliary interests
you want to keep a pulse on but don't touch day to day. New papers arrive already
searched, ranked, extracted into structured notes, and filed in your Obsidian
vault, on a schedule — so staying current becomes a digest you skim, not a Google
Scholar tab and an inbox of journal alerts you'll never open.

## Features

- **Library-relative ranking, augmentable with the knowledge graph.** Recurring
  searches across PubMed, arXiv, and Scopus, powered by a bundled copy of
  [findpapers](https://github.com/jonatasgrosman/findpapers) (see
  [Acknowledgements](#acknowledgements)). Every candidate is scored against your
  Zotero library: semantic cosine similarity to local embeddings
  (`mxbai-embed-large`, or any LiteLLM-compatible provider, in a per-machine
  ChromaDB store) is the baseline — which you can layer **graph-derived signals**
  on top of (shared entities, citation overlap, shared authors) along with a
  free-text domain-focus paragraph, each weighted in config (opt-in). Vector
  search is table stakes; the graph is where the ranking gets opinionated about
  *your* corpus. *Explicit search coverage for individual journal publishers is
  planned for a future release.*
- **Knowledge graph with an ask interface.** A LadybugDB graph stores entities
  (topics, methods, materials, authors, journals, keywords) and ten typed
  relationships across the corpus. Ask questions in plain English from the CLI,
  HTTP, or MCP — for example, `lit-monitor ask "what methods extend Carta
  2009?"`. Answers are theme-aware when the library has been clustered.
- **Three retrieval modes** — vector (semantic), graph (entity-typed), and
  hybrid (reciprocal-rank fusion), selectable per command with
  `--rag-mode {vector,graph,hybrid}`.
- **Obsidian-native output.** Every paper becomes a structured Markdown note
  with persist zones for your own annotations, two-phase LLM extraction, and a
  citation-graph rebuild path.
- **MCP server for AI clients.** Twelve tools that Claude Desktop, Cursor,
  Continue, and any other MCP-capable agent can call to query the graph and
  vector index, including a read-only Cypher escape hatch with a safety guard.
- **Notifications and flexible delivery.** An OS notification when a discovery
  run finishes, a weekly Markdown digest, on-demand Markdown export, or a rich
  terminal table — configurable.
- **Runs on any schedule.** One-command install for launchd (macOS) or systemd
  user timers (Linux), plus ad-hoc runs from the dashboard.

For the scoring model and the design behind each signal, see
[How it works](docs/how-it-works.md).

## Requirements

- macOS or Linux (ARM Linux, including Raspberry Pi 4/5, works for
  cloud-Ollama configurations)
- Python 3.11+
- [Ollama](https://ollama.com) installed locally for embeddings
  (`ollama pull mxbai-embed-large`)
- A Zotero library (Better BibTeX optional). lit-monitor ingests **Markdown
  attachments**, not PDFs — convert your PDFs to Markdown and attach them with
  [zotero-docling](https://github.com/max3925vats/zotero-docling).
- An Obsidian vault (full absolute path required)

## Install

```bash
pip install lit-monitor        # or: uvx lit-monitor / pipx install lit-monitor
lit-monitor first-run
```

That's the whole install. `lit-monitor first-run` walks you through interactive
setup and then launches the web UI. [Ollama](https://ollama.com) is a separate
prerequisite for local embeddings (`ollama pull mxbai-embed-large`) — see
[Requirements](#requirements).

For optional extras — `[nlp]` (BioBERT entity extraction) and `[litellm]`
(multi-provider cloud LLM routing — *under testing; feedback appreciated*) — and
the from-source / development install, see the
[Installation guide](docs/installation.md).

### From source (development)

```bash
git clone https://github.com/max3925vats/lit-monitor.git
cd lit-monitor
./install.sh
```

The script installs [`uv`](https://docs.astral.sh/uv/) if needed, creates a
project-local `.venv`, and resolves all dependencies, then offers to run
`lit-monitor first-run`, which seeds your config files (from the packaged
examples) and launches the web UI.

## Quickstart

### Web UI

```bash
lit-monitor first-run   # interactive first-time setup, then launches the server
# or, once credentials are configured:
lit-monitor serve
```

Open **`http://127.0.0.1:8765/setup`** in any browser. An 8-step wizard covers
credentials, paths, extraction config, topics, domain context, theme
vocabulary, tracked researchers, and item routing, with live credential checks
at each step. After setup, the dashboards take over. See the
[Web UI guide](docs/web-ui.md) for every page.

### CLI

```bash
lit-monitor check          # verify config + Ollama + Zotero connectivity
lit-monitor brain-build    # index your existing Zotero library (one-time)
lit-monitor run            # first discovery run
lit-monitor serve          # browse results at http://127.0.0.1:8765
```

Full command surface in the [CLI reference](docs/cli-reference.md). To configure
credentials and YAML by hand instead of using the wizard, see
[Configuration](docs/configuration.md).

## Documentation

| Guide | Covers |
|---|---|
| [Installation](docs/installation.md) | Install paths, optional extras, field-specific starter configs |
| [How it works](docs/how-it-works.md) | Library-as-signal, score decomposition, clustering, domain extraction, trending, embeddings |
| [Configuration](docs/configuration.md) | The config files, three setup recipes, LLM and embedding providers, notifications, strict mode, scheduled deployment |
| [Web UI](docs/web-ui.md) | Dashboard pages and the setup wizard |
| [CLI reference](docs/cli-reference.md) | Every day-to-day command |
| [MCP server](docs/mcp.md) | The twelve MCP tools for AI clients |
| [HTTP API](docs/http-api.md) | The HTTP query and ingestion surface |
| [Development](docs/development.md) | Running tests and contributing |

## Glossary

A few terms used throughout the docs:

- **Zotero** — reference manager that holds your library of papers; lit-monitor
  reads it as the relevance signal.
- **Obsidian** — Markdown-based knowledge base; lit-monitor writes one note per
  paper into a vault (a folder of Markdown files).
- **Embedding** — a numeric vector representing a paper's text, so similarity can
  be measured by distance. Papers near your library's embeddings rank higher.
- **Ollama** — runs language and embedding models locally on your machine (no
  cloud account needed for the default setup).
- **ChromaDB** — the local vector database that stores paper embeddings.
- **LadybugDB** — the local graph database that stores entities (methods, authors,
  …) and their typed relationships.
- **LiteLLM** — an optional adapter to route LLM or embedding calls to cloud
  providers (OpenAI, Anthropic, Vertex AI) instead of local Ollama.
- **MCP (Model Context Protocol)** — an open standard that lets AI clients (Claude
  Desktop, Cursor, …) call external tools; lit-monitor ships an MCP server.
- **Cypher** — the query language for the knowledge graph; the `ask` and MCP
  surfaces translate plain English into read-only Cypher under the hood.
- **brain-build** — the one-time step that indexes your existing Zotero library
  into the embedding store and graph.
- **RRF (reciprocal-rank fusion)** — the method behind `--rag-mode hybrid` that
  blends vector and graph rankings into one ordered list.

## Acknowledgements

Multi-source literature search is powered by
[**findpapers**](https://github.com/jonatasgrosman/findpapers) by Jonatas Grosman
(MIT License, © 2020). A copy is bundled under
[`lit_monitor/_vendor/findpapers`](lit_monitor/_vendor/findpapers) — with its
license retained — so that `pip install lit-monitor` resolves cleanly without an
upstream dependency conflict. The original project is gratefully acknowledged.

Explicit search coverage for individual journal publishers (beyond the sources
findpapers provides) is planned for a future release.

## License

[MIT](LICENSE)

This project bundles a copy of [findpapers](https://github.com/jonatasgrosman/findpapers)
(MIT License) — see [Acknowledgements](#acknowledgements) and
[`lit_monitor/_vendor/findpapers/LICENSE`](lit_monitor/_vendor/findpapers/LICENSE).
