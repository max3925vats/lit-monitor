# How it works

[← Back to README](../README.md)

lit-monitor ranks new papers by how close they are to the literature you already
keep, then enriches and files each one. This page explains the signals behind
the ranking and the features that build on them.

## Library-as-signal

Every paper in your Zotero collection is embedded and stored in a per-machine
ChromaDB instance. When a discovery run finds a candidate, its abstract is
embedded and compared against your library — the cosine similarity is the base
relevance score. Papers already close to what you've read rank higher; papers in
a completely different domain rank lower.

## Score decomposition

The final ranking score is a weighted sum of six signals:

| Signal | What it measures |
|---|---|
| `vector` | Cosine similarity of the candidate to your Zotero library centroid |
| `domain_context` | Cosine similarity to the optional free-text domain focus paragraph |
| `cluster_centroid` | Similarity to the nearest theme cluster in your library |
| `graph_entity_overlap` | How many named entities the candidate shares with your graph |
| `graph_citation` | Citation edges (CITES / EXTENDS) to papers already in the graph |
| `graph_shared_authors` | Authors appearing in both the candidate and your graph |

Every paper in a discovery result carries a `score_breakdown` dict with all six
values. The web UI's paper card shows the decomposition as a stacked bar.

Signal weights live under `ranking:` in `config/extraction.yaml`. See
[Configuration](configuration.md) for the recommended progression from
vector-only to a full graph-aware mix.

## Theme clustering

Once your library reaches ~100 papers (configurable), lit-monitor runs K-means
over the embedding space and asks the LLM to name each cluster. The named
clusters (themes) can be written back to Zotero as tags or collections on
request. Clusters update automatically on each subsequent brain-build.

## Domain extraction

`domain_context.yaml` accepts a free-text paragraph describing your focus areas.
The `domain analyze` command uses a single LLM call to extract structured
concepts (techniques, targets, assay types, keywords) and stores them in
`state.db`. These feed the `domain_context` ranking signal without any manual
tagging.

## Trending concepts and query expansion

The graph tracks which entity types are mentioned most in recently accepted
papers. `lit-monitor trending suggest` surfaces new concepts that are rising in
frequency. Accept a suggestion to add it to your active topics; dismiss it to
suppress it until the next review cycle.

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
