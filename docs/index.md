---
hide:
  - navigation
  - toc
---

<div class="lm-hero" markdown>
<p class="lm-eyebrow">for Zotero · into Obsidian</p>

# Your library is the signal.

<p class="lm-sub">
lit-monitor tracks PubMed, arXiv & Scopus, ranks every new paper by similarity
to your <span class="lm-zotero">Zotero</span> library, and files structured notes
into <span class="lm-obsidian">Obsidian</span> — queryable from the terminal, a
browser, or any MCP client.
</p>

<div class="lm-cta">
  <a class="lm-btn lm-btn--primary" href="installation/">Get started →</a>
  <a class="lm-btn lm-btn--ghost" href="https://pypi.org/project/lit-monitor/">View on PyPI</a>
  <a class="lm-btn lm-btn--ghost" href="https://github.com/max3925vats/lit-monitor">GitHub</a>
</div>

<div class="lm-install"><b>$</b> pip install lit-monitor</div>

<div class="lm-pills">
  <span>PubMed</span><span>arXiv</span><span>Scopus</span>
  <span>Obsidian</span><span>Knowledge graph</span><span>MCP server</span>
</div>
</div>

## What it does

<div class="grid cards" markdown>

-   :material-target:{ .lg } **Library-relative ranking**

    Every candidate is scored by cosine similarity to embeddings of your Zotero
    library. Papers close to what you already keep rank higher — embeddings run
    locally via Ollama against a per-machine ChromaDB store.

-   :material-notebook-edit:{ .lg } **Obsidian-native output**

    Each paper becomes a structured Markdown note with persist zones for your own
    annotations, two-phase LLM extraction, and a citation-graph rebuild path.

-   :material-graph:{ .lg } **Knowledge graph + ask**

    A KuzuDB graph stores entities and ten typed relationships. Ask in plain
    English from the CLI, HTTP, or MCP — *"what methods extend Carta 2009?"* —
    with vector, graph, or hybrid retrieval.

-   :material-robot-happy:{ .lg } **MCP server for AI clients**

    Twelve tools that Claude Desktop, Cursor, and any MCP-capable agent can call
    to query the graph and vector index, including a guarded read-only Cypher
    escape hatch.

-   :material-shield-lock:{ .lg } **Local-first & free by default**

    Your library is embedded and stored on your machine — no per-call API costs.
    Only outbound paper searches go out. Cloud providers are strictly opt-in.

-   :material-bell-ring:{ .lg } **Runs on your schedule**

    One-command install for launchd (macOS) or systemd user timers (Linux), an OS
    notification when a run finishes, and on-demand runs from the dashboard.

</div>

## Quickstart

Ollama is the one external prerequisite (for local embeddings).

```bash
pip install lit-monitor        # or: uvx lit-monitor / pipx install lit-monitor
lit-monitor first-run          # interactive setup: credentials + seeds your config
lit-monitor serve              # browse at http://127.0.0.1:8765
```

New here? Start with the **[Installation guide](installation.md)**, then
**[How it works](how-it-works.md)** for the scoring model.

## Explore

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — install paths, extras, field-specific starter configs
- :material-cog: **[Configuration](configuration.md)** — setup recipes, providers, notifications, strict mode
- :material-console: **[CLI reference](cli-reference.md)** — every day-to-day command
- :material-monitor-dashboard: **[Web UI](web-ui.md)** — dashboards and the setup wizard
- :material-connection: **[Integrations](integrations.md)** — MCP server and HTTP API
- :material-flask: **[How it works](how-it-works.md)** — library-as-signal, score decomposition, clustering

</div>

!!! note "Beta"
    lit-monitor is feature-complete and in active daily use, but still maturing
    toward 1.0 — interfaces and on-disk formats may change between releases.
    Bug reports and feedback are very welcome via
    [GitHub Issues](https://github.com/max3925vats/lit-monitor/issues).
    Provided under the [MIT License](https://github.com/max3925vats/lit-monitor/blob/main/LICENSE).
