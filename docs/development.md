# Development

[← Back to README](../README.md)

## Running tests

```bash
.venv/bin/python -m pytest tests/unit/ tests/llm/ -q

# Integration tests — SKIP if Ollama / Zotero / vault are unavailable
lit-monitor check && .venv/bin/python -m pytest tests/integration/ -m integration -v

# Integration tests in STRICT mode — missing services FAIL the run
LIT_MONITOR_LIVE=1 .venv/bin/python -m pytest tests/integration/ -m integration -v
```

Contributor tooling (ruff, pytest, mypy) installs with `uv sync --extra dev`.

## Deployment

lit-monitor runs on a recurring schedule on a single workstation. Install the
schedule from the `/schedule` page (launchd on macOS, systemd user timer on
Linux). Weekly is a sensible default; any cadence works.

Once the local state DB and ChromaDB store have been populated — either by
brain-build in one shot, or organically by repeated `lit-monitor run` calls — the
schedule can be handed off to a low-power node (Raspberry Pi 4/5), provided
extraction is routed through Ollama Cloud or LiteLLM. See
[Configuration](configuration.md#llm-and-embedding-providers) for provider
routing.
