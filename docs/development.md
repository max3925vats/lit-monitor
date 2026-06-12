# Development &amp; contributing

[← Home](index.md)

lit-monitor is open source under the [MIT License](https://github.com/max3925vats/lit-monitor/blob/main/LICENSE),
and contributions are genuinely welcome — whether that's a bug report, a docs
fix, a new field-specific starter config, or a feature. The project grows from
people using it on their own libraries, so feedback from real use is the most
valuable thing you can send.

The code lives at
[**github.com/max3925vats/lit-monitor**](https://github.com/max3925vats/lit-monitor).

## Contributing code

```bash
git clone https://github.com/max3925vats/lit-monitor.git
cd lit-monitor
./install.sh
```

The script installs [`uv`](https://docs.astral.sh/uv/) if needed, creates a
project-local `.venv`, resolves all dependencies, and seeds working configs from
`config/*.example.yaml`. From there the usual loop applies: branch, make a small
focused change, open a pull request. Issues and PRs are the right place to discuss
anything non-trivial before investing a lot of work — open one early and we can
shape the approach together.

## The developer page

Start the server with the `--dev` flag to mount an extra page at `/dev`:

```bash
lit-monitor serve --dev
```

It's a hands-on diagnostic and sandbox surface, useful whether you're hacking on
lit-monitor or just want to watch the pipeline work end-to-end on a throwaway
copy:

- **Trace a paper end-to-end** — paste a Markdown paper (or pull one from Zotero)
  and watch each stage run: chunking, two-phase extraction, embedding, graph, and
  the Obsidian note. Every step is inspectable.
- **Health and service checks** — the same `diagnose` / `check` probes the CLI
  runs, in the browser.
- **Discovery dry-run** — exercise the search-and-rank path without ingesting.
- **Graph backfill** — populate the knowledge graph with a live log stream.

Everything on `/dev` runs against an **isolated sandbox** (separate state DB,
ChromaDB collections, and a `Literature/_Dev/` vault subfolder) with a one-click
"clear sandbox" — so it never touches your real library.

## Reporting bugs &amp; requesting features

Found a bug or have an idea? Open an issue:

- **[Report a bug →](https://github.com/max3925vats/lit-monitor/issues/new?template=bug_report.yml)**
- **[Request a feature →](https://github.com/max3925vats/lit-monitor/issues/new?template=feature_request.yml)**

Or browse [all issues](https://github.com/max3925vats/lit-monitor/issues). The bug
form asks for your version, OS, install method, and steps to reproduce — and for
setup or connectivity problems, the output of `lit-monitor diagnose` is the single
most helpful thing to include.

!!! warning "Never paste secrets"
    Scrub API keys, tokens, and library credentials out of any logs or config you
    attach to an issue.
