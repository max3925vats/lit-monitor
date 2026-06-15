# Web UI

[← Home](index.md)

```bash
lit-monitor first-run   # interactive first-time setup, then launches the server
# or, once credentials are configured:
lit-monitor serve
```

`first-run` prompts for write-once credentials and your server host/port, then
starts the UI. On subsequent launches use `lit-monitor serve`.

The server binds to `127.0.0.1` by default. Pass `--host 0.0.0.0` for LAN
access; the server has no authentication of its own. A persistent health badge
in the nav polls `/api/health` and shows pipeline status at a glance. An optional
developer page is mounted at `/dev` when you start the server with
`lit-monitor serve --dev`.

!!! tip "Set a schedule — that's what makes it a *monitor*"
    The single highest-value thing to do after setup is install a recurring
    schedule from the **`/schedule`** page (launchd / systemd). With it, new
    papers arrive already searched, ranked, extracted, and filed between visits —
    see [Scheduling — the monitor loop](how-it-works.md#scheduling-the-monitor-loop).

## Setup wizard

Open **`http://127.0.0.1:8765/setup`** in any browser. An 8-step wizard covers
credentials, paths, extraction config, topics, domain context, theme vocabulary,
tracked researchers, and item routing, with live credential checks at each step.

To configure the same settings by hand instead, see
[Configuration](configuration.md).

## Dashboards

After setup, the dashboards take over. The top nav groups the pages by what
you're doing — **Monitor**, **Semantics**, **Explore**, **Tune**, **Setup**.

| URL | Group | What |
|---|---|---|
| `/discovery` | Monitor | Latest discovery run summary, run history, per-paper cards with one-click relink / re-extract actions. Run-now / Dry-run / Stop buttons. |
| `/brain-build` | Monitor | Extract your existing Zotero library into the index. Progress bar, per-paper table, recent runs. Start / Stop / Resume. Live JSONL log stream. |
| `/schedule` | Monitor | Install or remove a recurring schedule (launchd / systemd). |
| `/ask` | Semantics | Ask natural-language questions of your corpus (the `lit-monitor ask` pipeline in the browser): prose answer + results table, with show/edit Cypher, recent-questions history, and save-to-vault. Requires the knowledge graph (`lit-monitor graph backfill --all`). |
| `/corpus` | Explore | A read lens on every processed paper in the state DB: searchable list with a theme filter, plus a per-paper detail view (extraction, score, knowledge-graph, related work, Zotero / Obsidian links, relink / re-extract). Related work works without a knowledge graph (vector similarity, each result labelled by source) and the Obsidian link is a clickable deep-link into your vault. Not a Zotero replacement — it reads what the pipeline already processed. |
| `/graph` | Explore | A read-only, corpus-wide overview of the knowledge graph: entity counts by type, edges by predicate, entity counts by NER source, and graph-indexed coverage. Read lens only — running a graph backfill is a CLI action. |
| `/themes` | Explore | Review the theme clusters your library has been grouped into: per-theme paper counts and a drill-in to the papers in each theme. |
| `/trending` | Explore | The trending-concept suggestion queue — concepts rising in your recent library activity, with accept / dismiss actions. |
| `/domain` | Tune | Edit the free-text domain-focus paragraph and run the LLM extraction that turns it into the structured `domain_context` ranking signal. |
| `/insights` | Tune | A read-only window into the active-learning engine: the current interest-vector state with top-aligned papers, per-cluster atrophy weights, the feedback mix by signal and source plus its timeline, and recent feedback events. Read lens only — feedback is still captured through the discovery and themes buttons. |
| `/setup` → step 9 (Tuning) | Tune | Advanced settings — ranking weights, clustering, embeddings, and discovery / delivery config sections. Folded into the setup wizard's optional final step; there is no standalone `/settings` page (the `POST /api/settings/{section}` endpoint still backs it). |
| `/setup` | Setup | The 8-step onboarding wizard (see above). |
| `/setup/reset` | Setup | Reset & Rebuild console — reset and rebuild individual components (vectors / graph / notes) or wipe everything, with confirmation guards. |
