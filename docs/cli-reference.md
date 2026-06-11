# CLI reference

[← Back to README](../README.md)

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
lit-monitor discovery view --run latest  # rich-table view of the latest results
lit-monitor discovery export-md --to ~/digest.md   # on-demand Markdown export
```

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
lit-monitor graph propose-aliases        # suggest entity normalization rules
```

## Obsidian helpers

```bash
lit-monitor obsidian relink              # update Related Work sections
lit-monitor obsidian rerender            # regenerate notes from stored extractions
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
lit-monitor reset state                  # reset pipeline state (also reset vault / reset all)
```

See also [Configuration](configuration.md) for strict mode and provider routing.
