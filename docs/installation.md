# Installation

[← Back to README](../README.md)

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
project-local `.venv`, resolves all dependencies, and seeds working configs from
`config/*.example.yaml`.

## Starting from a non-biopharma field

`config/examples/` ships filled-in, synthetic config sets for `bioprocessing/`,
`ml-research/`, and `climate-science/`. Pick the closest one and copy its
`topics.yaml`, `domain_context.yaml`, `concepts.yaml`, and `researchers.yaml`
into `config/` as a head start, then edit:

```bash
cp config/examples/ml-research/*.yaml config/   # or bioprocessing/ climate-science/
```

The non-domain configs (`paths.yaml`, `extraction.yaml`) still come from
`config/*.example.yaml`. See
[`config/examples/README.md`](../config/examples/README.md).

## Manual install

To drive `uv` yourself instead of running `install.sh`:

```bash
uv venv && source .venv/bin/activate
uv sync                                  # web UI, graph, MCP, notifications all included
for f in config/*.example.yaml; do cp -n "$f" "${f%.example.yaml}.yaml"; done
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
- [Quickstart in the README](../README.md#quickstart) — first run, web or CLI
