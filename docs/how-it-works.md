# How it works

[← Home](index.md)

lit-monitor is a *monitor*: it watches the literature for you, on a schedule, and
keeps a running, ranked, enriched view of what's new and relevant — without you
having to run a search and triage results by hand each week.

Each cycle does five things. It **searches** PubMed, arXiv, and Scopus for your
topics; **ranks** every candidate by how close it sits to the library you already
keep in Zotero; **extracts** structured fields from the papers worth keeping with
a local LLM; **files** each one as a Markdown note in your Obsidian vault; and
**indexes** everything into a local knowledge graph and vector store you can then
query from a browser, the terminal, or any AI client. The sections below walk the
machinery behind each of those, starting with the idea the whole thing rests on:
your own library is the relevance signal.

## Library as signal

Most discovery tools rank papers by keyword match or by a global popularity
signal. lit-monitor ranks them by similarity to *your* library — so the feed is
shaped by what you actually read, and it follows you as your interests drift.

To do that, your Zotero collection is turned into **two complementary
representations**, both stored locally:

1. **A knowledge graph (KuzuDB).** Entities — methods, materials, targets,
   authors, journals, topics — and the typed relationships between them
   (`CITES`, `EXTENDS`, `CONTRADICTS`, and more) are extracted from each paper and
   stored as a real graph. This is the part that sets lit-monitor apart from
   embedding-only tools: it doesn't just know a candidate "feels similar" to your
   library, it can say *how* it connects — shares an entity, extends a method you
   already track, cites a paper you keep.
2. **A vector index (ChromaDB).** Each paper's text is embedded into a numeric
   vector (locally, via Ollama) so semantic similarity can be measured by
   distance. When a discovery run finds a candidate, it's embedded the same way
   and compared against your library; the cosine similarity is the base relevance
   score.

A third store, a small SQLite **state DB**, is the canonical record for every
paper and tracks what's been processed, indexed, and scored. The Obsidian notes
and the two indexes above are all regenerable views over it.

Papers close to what you've read rank higher; papers in a different domain rank
lower — and the graph signals can push a paper up even when raw text similarity is
modest, because it connects to your library structurally.

!!! note "Zotero companion plugin — planned"
    Today, papers enter the pipeline when `brain-build` or a scheduled discovery
    run reads your Zotero collection. A planned Zotero companion plugin (Zotero 7+)
    closes the loop from the other side: the moment you save a paper to a watched
    collection — via the browser connector, drag-and-drop, or sync — it's sent
    straight to lit-monitor for ingestion, with no terminal step. Auto-ingest on
    add, plus a right-click *"Send to lit-monitor"*. It's one-way by design: your
    Obsidian vault stays the destination — see the companion
    [Obsidian plugin](#obsidian-output) planned on the output side.

## Knowledge graph and query surfaces

The graph isn't just a ranking input — it's something you can talk to. Once your
library has been indexed (`lit-monitor graph backfill --all`), the same KuzuDB
graph and ChromaDB index sit behind three interchangeable interfaces:

- **`lit-monitor ask`** — ask in plain English (*"what methods extend Carta
  2009?"*). The question is turned into a read-only Cypher query, run against the
  graph, and the rows are summarized back into prose. Answers are theme-aware when
  your library has been clustered.
- **The [MCP server](mcp.md)** — twelve tools that Claude Desktop, Cursor, and any
  MCP-capable agent can call, so an AI assistant can hold a conversation grounded
  in *your* corpus, including a guarded read-only Cypher escape hatch.
- **The [HTTP API](http-api.md)** — the same query layer over HTTP for scripts and
  services.

All three share one query layer, so an answer is the same whichever surface you
ask through. Retrieval runs in one of three modes — **vector** (semantic),
**graph** (entity-typed), or **hybrid** (reciprocal-rank fusion of the two) —
selectable per query.

## Score decomposition

The final ranking score is a weighted sum of six signals — three from the vector
side, three from the graph:

| Signal | What it measures |
|---|---|
| `vector` | Cosine similarity of the candidate to your Zotero library centroid |
| `domain_context` | Cosine similarity to the optional free-text domain focus paragraph |
| `cluster_centroid` | Similarity to the nearest theme cluster in your library |
| `graph_entity_overlap` | How many named entities the candidate shares with your graph |
| `graph_citation` | Citation edges (CITES / EXTENDS) to papers already in the graph |
| `graph_shared_authors` | Authors appearing in both the candidate and your graph |

Every paper in a discovery result carries a `score_breakdown` dict with all six
values, and the web UI's paper card renders the decomposition as a stacked bar so
you can see *why* a paper ranked where it did. Signal weights live under
`ranking:` in `config/extraction.yaml`; see
[Configuration](configuration.md#setup-recipes) for the recommended progression
from vector-only to a full graph-aware mix.

## Entity and relationship extraction

The graph signals depend on pulling named entities and typed relationships out of
each paper. Two extractors run and their outputs are merged:

- A **BioBERT genetic NER** model (`alvaroalon2/biobert_genetic_ner`, the
  `--extra nlp` path) tags biomedical entities.
- An **LLM pass** extracts longer-tail entities and the typed relationships
  (`CITES`, `EXTENDS`, `CONTRADICTS`, …). The BioBERT layer is optional — without
  it, the LLM handles entities alone.

!!! info "NER model selection"
    The NER model is fixed to the BioBERT default in the current build. The
    underlying `BiobertNER` class already accepts a `model_id` for any Hugging
    Face token-classification model, but there is **no config field to choose it
    yet** — config-level NER model selection is planned for a future release. If
    you need a different NER model today, it's a one-line code change at the
    extractor's construction site, not a setting.

## Theme clustering

Once your library reaches ~100 papers (configurable), lit-monitor runs K-means
over the embedding space and asks the LLM to name each cluster. The named
clusters (themes) can be written back to Zotero as tags or collections on
request, and they feed the `cluster_centroid` ranking signal. Clusters update
automatically on each subsequent brain-build.

## Domain extraction

`domain_context.yaml` accepts a free-text paragraph describing your focus areas.
The `domain analyze` command uses a single LLM call to extract structured
concepts (techniques, targets, assay types, keywords) and stores them in the
state DB. These feed the `domain_context` ranking signal without any manual
tagging — a quick way to bias the feed toward where you're heading, not just
where you've been.

## Trending concepts and query expansion

The graph tracks which entity types are mentioned most in papers you've recently
added. `lit-monitor trending suggest` surfaces concepts rising in frequency in
*your* library activity (note: this reflects what you've been ingesting, not
field-wide publication trends). Accept a suggestion to add it to your active
topics; dismiss it to suppress it until the next review cycle.

## Obsidian output

Every paper worth keeping becomes a structured Markdown note written straight into
your Obsidian vault — plain `.md` files you own, readable and editable with or
without lit-monitor running. Each note carries Dataview-friendly YAML front matter
(year, journal, tags, themes, citation counts, an extraction-confidence score) so
you can query your literature from inside Obsidian.

Notes include **persist zones** — sections such as your own Related Work and
Synthesis that lit-monitor will never overwrite when it re-renders a note, so your
annotations are safe across re-extractions and graph rebuilds.

!!! info "Not locked to Markdown or Obsidian"
    The current pipeline goes Zotero → Obsidian, but the Markdown note is only one
    *rendered view*. Every paper's extracted fields and metadata are stored
    centrally and independently of how they're displayed, so the same data can be
    rendered to other targets — non-Markdown formats, or knowledge tools other than
    Obsidian — with some adaptation. Additional output workflows will be added as
    need and demand arise; if you want one, [open an issue](https://github.com/max3925vats/lit-monitor/issues).

!!! note "Obsidian companion plugin — planned"
    An Obsidian companion plugin that works directly with lit-monitor is on the
    roadmap. Today the integration is the vault itself: lit-monitor writes the
    notes, Obsidian reads them.

## Scheduling — the monitor loop

The word *monitor* is the point. lit-monitor is built to run **unattended on a
recurring schedule** so that the literature comes to you, already searched,
ranked, extracted, and filed — rather than you carving out time to go looking.

Install a schedule in one command from the [`/schedule`](web-ui.md#dashboards)
page (launchd on macOS, systemd user timers on Linux). Weekly is a sensible
default; any cadence works. On each scheduled run lit-monitor:

- searches your topics and ranks the new candidates against your library;
- ingests the papers worth keeping — extraction, embeddings, graph, and an
  Obsidian note per paper — so they're *already in your library and queryable* by
  the time you look;
- writes a dated **digest** (`Discovery_<date>.md`) summarizing the run; and
- fires an **OS notification** when it finishes, which lands you at the run's
  results on click.

The result is a literature feed that maintains itself: you open your vault (or the
dashboard) when it suits you and find new, relevant, fully-processed papers
waiting — no manual search-and-triage step. Because the heavy state lives in local
files, a populated install can even be handed to a low-power always-on node (e.g.
a Raspberry Pi) with cloud-routed extraction; see
[Deployment](configuration.md#deployment-running-on-a-schedule).

## Local-first and open source

lit-monitor is **MIT-licensed, free, and local-first by design** — built to help
researchers put modern LLM tooling to work on their own literature without cost
barriers or lock-in.

On the default configuration, everything runs on your machine: your library is
embedded and stored locally (Ollama + ChromaDB + KuzuDB), with **no per-call API
costs** and **no account required**. The only thing that leaves your machine is
the outbound paper search to PubMed, arXiv, and Scopus. Your library, your notes,
your graph, and your reading history stay yours, on disk, in open formats
(Markdown, SQLite). Routing extraction or embeddings to a cloud provider
(Anthropic, OpenAI, Vertex AI, Ollama Cloud) is strictly opt-in, for when you want
more horsepower — never the default. The code is on
[GitHub](https://github.com/max3925vats/lit-monitor); contributions and feedback
are welcome.

## What each config file feeds

The ranking and enrichment above are driven by the YAML config files. This is the
signal-level view of what each file actually does; for *where they live* and how
to edit them, see [Configuration](configuration.md#the-config-files-at-a-glance).

| File | Feeds |
|---|---|
| `paths.yaml` | The library itself — which Zotero collection gets embedded, and where notes/DBs are written. Nothing ranks without it. |
| `extraction.yaml` | The `ranking:` weights that combine the six signals above, plus model selection, `clustering:`, `embeddings:`, and the discovery delivery flags. |
| `topics.yaml` | The candidate stream — the recurring searches that produce the papers to be ranked. |
| `domain_context.yaml` | The `domain_context` signal (after `domain analyze` extracts structured concepts). |
| `concepts.yaml` | Theme assignment and the `cluster_centroid` signal's named themes. |
| `researchers.yaml` | The researcher-gating nudge for tracked authors. |
| `predicates.yaml`, `entity_types.yaml`, `entity_aliases.yaml` | The knowledge-graph vocabulary that the entity/relationship extractors map onto — which in turn feeds the three `graph_*` signals. |

## Embedding providers

Embeddings default to local Ollama (`mxbai-embed-large`). Switch to any
LiteLLM-compatible provider (OpenAI `text-embedding-3-small`, Cohere, Vertex AI,
etc.) without changing the rest of the pipeline:

```bash
uv sync --extra litellm
lit-monitor embeddings switch --provider litellm --model text-embedding-3-small
lit-monitor embeddings rebuild   # re-embed your whole library under the new model
```

The `embeddings status` command shows the current provider, model, embedding
dimensionality, and how many papers are already embedded. Provider routing for
both embeddings and LLM inference is covered in
[Configuration](configuration.md#llm-and-embedding-providers).
