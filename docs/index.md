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

<div class="grid cards lm-linked" markdown>

-   :material-bell-ring:{ .lg } **[A monitor that runs itself](how-it-works.md#scheduling-the-monitor-loop)**

    Schedule it (launchd / systemd) and new papers arrive already searched,
    ranked, extracted, and filed — with a dated digest and an OS notification when
    each run finishes. Open your vault and the work is already done.

-   :material-target:{ .lg } **[Library-relative ranking](how-it-works.md#score-decomposition)**

    Your library *is* the relevance signal. Each candidate is scored by a
    transparent six-signal mix — semantic similarity to your Zotero library plus
    graph signals (shared entities, citation edges, shared authors) — and every
    paper shows the decomposition, so you see exactly why it ranked where it did.

-   :material-graph:{ .lg } **[Knowledge graph you can talk to](how-it-works.md#knowledge-graph-and-query-surfaces)**

    A real KuzuDB graph of entities and typed relationships — not just embeddings.
    Hold a conversation grounded in *your* corpus from the CLI (`ask`), an AI
    client (MCP), or HTTP, with vector, graph, or hybrid retrieval.

-   :material-notebook-edit:{ .lg } **[Obsidian-native output](how-it-works.md#obsidian-output)**

    Every kept paper becomes a structured Markdown note in your vault — plain
    `.md` files you own, with Dataview front matter and persist zones that protect
    your own annotations across rebuilds. An Obsidian companion plugin is planned.

-   :material-robot-happy:{ .lg } **[MCP server for AI clients](mcp.md)**

    Twelve tools that Claude Desktop, Cursor, and any MCP-capable agent can call
    to query the graph and vector index, including a guarded read-only Cypher
    escape hatch.

-   :material-heart:{ .lg } **[Local-first, open, and free](how-it-works.md#local-first-and-open-source)**

    MIT-licensed and built for researchers: your library is embedded and stored
    on your machine with no per-call costs and no account — open formats, no
    lock-in. Cloud providers are strictly opt-in. Contributions welcome.

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

<div class="grid cards lm-linked" markdown>

-   :material-download:{ .lg } **[Installation](installation.md)**

    Install paths, optional extras, and the prerequisites.

-   :material-cog:{ .lg } **[Configuration](configuration.md)**

    Config files, setup recipes, providers, notifications, strict mode, deployment.

-   :material-flask:{ .lg } **[How it works](how-it-works.md)**

    Library-as-signal, the knowledge graph, scheduling, the six ranking signals,
    and the local-first philosophy.

-   :material-monitor-dashboard:{ .lg } **[Web UI](web-ui.md)**

    The dashboards and the 8-step setup wizard.

-   :material-robot-happy:{ .lg } **[MCP server](mcp.md)**

    Twelve MCP tools for AI clients, each described.

-   :material-api:{ .lg } **[HTTP API](http-api.md)**

    The HTTP query and ingestion surface.

</div>

!!! note "Beta"
    lit-monitor is feature-complete and in active daily use, but still maturing
    toward 1.0 — interfaces and on-disk formats may change between releases.
    Bug reports and feedback are very welcome via
    [GitHub Issues](https://github.com/max3925vats/lit-monitor/issues).
    Provided under the [MIT License](https://github.com/max3925vats/lit-monitor/blob/main/LICENSE).
