# CLI reference

[← Home](index.md)

Run `lit-monitor --help` for the full command surface. Top-level `-v` /
`--verbose` enables DEBUG console output; a full DEBUG log is always written to
`logs/{date}_{command}.jsonl`.

## One-time setup

```bash
lit-monitor first-run                    # interactive setup, then launch the server
lit-monitor build-vocabulary             # cluster Zotero tags into themes
lit-monitor brain-build --resume         # extract every paper in the collection
```

## Discovery

```bash
lit-monitor run                          # discover new papers + ingest new Zotero items
lit-monitor run --dry-run                # preview without writes
lit-monitor run --since-days 30          # search the last 30 days (override the adaptive window)
lit-monitor run --since 2026-01-01       # search from a date to today
lit-monitor run --from 2025-08-01 --to 2025-09-01   # search an explicit date range (catch-up/backfill)
lit-monitor discovery view --run latest  # rich-table view of the latest results
lit-monitor discovery export-md --to ~/digest.md   # on-demand Markdown export
lit-monitor discovery backfill-ingested  # mark past recommendations that are now in your library as ingested
```

`backfill-ingested` is a one-time catch-up: for every paper ever surfaced in a
discovery digest that is now in your library, it records that the recommendation
was ingested, so older run views show their true conversion. It is forward-only
and safe to re-run.

**Search window.** With no window flag, each run covers from the latest date any
prior run already searched (the *coverage frontier*) up to today — so runs march
forward without gaps or re-fetching. `--since-days` / `--since` / `--from`+`--to`
are mutually exclusive overrides for a one-off wider look-back or a historical
catch-up; an explicit range is honored as-is (no cap) and does **not** disturb the
frontier unless it extends past it.

## Ask

```bash
lit-monitor ask "what methods extend cation exchange?" --rag-mode hybrid
```

Responses are automatically contextualised to your dominant theme when clusters
exist. Requires the knowledge graph (`lit-monitor graph backfill --all`).

## Domain focus extraction

```bash
lit-monitor domain analyze               # LLM-extract structured concepts from domain_context.yaml
lit-monitor domain view                  # show extracted focus areas
lit-monitor domain clear                 # reset extracted domain focus
```

## Theme clustering

Requires ~100 indexed papers.

```bash
lit-monitor cluster recompute            # run K-means + LLM naming
lit-monitor cluster view                 # show cluster themes + paper counts
lit-monitor cluster assign               # assign papers to nearest cluster
lit-monitor cluster write-back tags      # tag Zotero items with theme names
lit-monitor cluster write-back collections  # move Zotero items into per-theme collections
```

## Trending concepts and query expansion

```bash
lit-monitor trending suggest             # surface rising entity types
lit-monitor trending view                # list pending / accepted / dismissed suggestions
lit-monitor trending accept <id>         # add concept to active topics
lit-monitor trending dismiss <id>        # suppress concept until next cycle
lit-monitor topics expansions suggest "ion exchange"   # co-occurring entities to broaden a topic query
```

## Embeddings

```bash
lit-monitor embeddings status            # show provider, model, coverage
lit-monitor embeddings switch --provider litellm --model text-embedding-3-small
lit-monitor embeddings rebuild           # re-embed library under new model
```

## Knowledge graph

```bash
lit-monitor graph status                 # node + edge counts
lit-monitor graph backfill --all         # index existing papers into the graph
lit-monitor graph rebuild --all          # drop + rebuild the whole graph from state.db
lit-monitor graph propose-aliases        # suggest entity normalization rules
```

Only papers that have been extracted are indexed into the graph (and the vector
store) — discovery candidates and not-yet-processed items are never added.

> **Upgrading from a version before this?** Earlier builds could index discovery
> candidates into the knowledge graph. Run `lit-monitor graph rebuild --all` once
> after upgrading to purge those nodes; the rebuild re-indexes only your extracted
> papers.

## Obsidian helpers

```bash
lit-monitor obsidian relink              # update Related Work sections
lit-monitor obsidian rerender            # regenerate notes from stored extractions
lit-monitor obsidian re-extract --doi <doi>     # re-run LLM extraction on one paper
lit-monitor obsidian sync --all          # write deferred per-paper notes
lit-monitor obsidian synthesize --topic "..."   # chunk-level RAG with reranking
```

## Active learning — interest vector + feedback

```bash
lit-monitor feedback <doi> --saved       # record a feedback event (also --dismissed/--thumbs-up/...)
lit-monitor learning recompute           # recompute the interest vector from feedback
lit-monitor learning view                # show the interest vector (--per-cluster for atrophy weights)
```

## Health

```bash
lit-monitor check                        # config + Ollama + Zotero reachability
lit-monitor diagnose                     # strict-mode config audit
lit-monitor status                       # extraction + embedding + graph counts
lit-monitor reset state                  # wipe state DB + all regenerable views (also: reset vault / reset all)
lit-monitor reset vectors                # wipe just the ChromaDB vector store
lit-monitor reset graph                  # wipe just the KuzuDB knowledge graph
```

Each `reset` subcommand prompts for a typed confirmation. The web UI offers the
same per-component reset **and** rebuild from the **Reset & Rebuild** console
(`/setup/reset`).

See also [Configuration](configuration.md) for strict mode and provider routing.
