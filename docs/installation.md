# Installation

[← Home](index.md)

## Requirements

- macOS or Linux (ARM Linux, including Raspberry Pi 4/5, works for cloud-Ollama
  configurations)
- Python 3.11+
- [Ollama](https://ollama.com) installed locally for embeddings
  (`ollama pull mxbai-embed-large`)
- A Zotero library (Better BibTeX optional)
- An Obsidian vault (full absolute path required)

## Install with pip (recommended)

```bash
pip install lit-monitor        # or: uvx lit-monitor / pipx install lit-monitor
lit-monitor first-run
```

`lit-monitor first-run` walks you through interactive setup and then launches the
web UI. The base install is cloud-free and runs entirely on local Ollama.
Ollama itself is a separate prerequisite — see [Requirements](#requirements).

## From source (development)

```bash
git clone https://github.com/max3925vats/lit-monitor.git
cd lit-monitor
./install.sh
```

The script installs [`uv`](https://docs.astral.sh/uv/) if needed, creates a
project-local `.venv`, and resolves all dependencies, then offers to run
`lit-monitor first-run`, which seeds your config files (from the packaged
examples) and launches the web UI.

!!! tip "Not in biopharma?"
    lit-monitor ranks against *your* library, whatever the field — and ships
    synthetic starter configs for `ml-research`, `climate-science`, and
    `bioprocessing`. See
    [Starting from a non-biopharma field](configuration.md#starting-from-a-non-biopharma-field).

## Manual install

To drive `uv` yourself instead of running `install.sh`:

```bash
uv venv && source .venv/bin/activate
uv sync                                  # web UI, graph, MCP, notifications all included
lit-monitor first-run                    # seeds config files, then launches the UI
```

## Optional extras

| Extra | Adds | Without it |
|---|---|---|
| `--extra nlp` | BiobertNER for entity extraction (~3 GB; transformers + torch) | The LLM fallback handles entities |
| `--extra litellm` | Multi-provider LLM routing (Anthropic, OpenAI, Vertex AI, etc.) | Ollama only |
| `--extra dev` | Contributor tooling (ruff, pytest, mypy) | — |

Install one or more with pip (square-bracket extras):

```bash
pip install "lit-monitor[nlp]"           # BioBERT entity extraction
pip install "lit-monitor[litellm]"       # multi-provider cloud LLM routing
pip install "lit-monitor[nlp,litellm]"   # both at once
```

Or, from a source checkout driving `uv` yourself:

```bash
uv sync --extra nlp --extra litellm
```

## Next steps

- [Configuration](configuration.md) — credentials and the three setup recipes
- [Quickstart](index.md#quickstart) — first run, web or CLI
