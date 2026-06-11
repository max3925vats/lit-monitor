# Configuration

[← Back to README](../README.md)

This guide covers configuring lit-monitor by hand, the three common setup
recipes, provider routing, notifications, and strict mode. To configure through
the browser instead, use the [setup wizard](web-ui.md#setup-wizard).

## Credentials and config files

**1. Create credentials** at `~/.config/lit-monitor/config.toml`:

```toml
[zotero]
api_key    = "YOUR_ZOTERO_API_KEY"
library_id = "YOUR_NUMERIC_LIBRARY_ID"

[pubmed]
email = "you@example.com"

[ollama]
api_key = "YOUR_OLLAMA_CLOUD_KEY"   # optional, only for cloud Ollama
```

**2. Edit the seeded YAMLs in `./config/`**: `paths.yaml`, `topics.yaml`,
`domain_context.yaml`, `concepts.yaml`, `researchers.yaml`. The example files in
git document every field; the real `.yaml` files are gitignored.

**3. Verify** — `lit-monitor check`.

## Setup recipes

Three configurations, from zero to fully customised.

### (a) Just run it — minimal config

Enough to get a recurring discovery feed and Obsidian notes.

1. `cp config/*.example.yaml config/` (if you didn't run `install.sh`).
2. Add Zotero credentials to `~/.config/lit-monitor/config.toml`.
3. Set `obsidian_vault_path` and `zotero_library_id` in `config/paths.yaml`.
4. Add 2–3 search topics in `config/topics.yaml`.
5. `lit-monitor check` — verify connectivity.
6. `lit-monitor brain-build` — index your existing library (one-time).
7. `lit-monitor run` — first discovery run.
8. `lit-monitor serve` to browse results at `http://127.0.0.1:8765`.

No domain context, no clustering, no graph signals yet — all optional. The
vector similarity signal works on its own.

### (b) Clustering on — after 100 papers

Enable after `brain-build` has indexed at least 100 papers.

1. `lit-monitor cluster recompute` — run K-means and name clusters.
2. `lit-monitor cluster view` — inspect the resulting themes.
3. Optionally `lit-monitor cluster write-back tags` to tag Zotero items.
4. Set `cluster_centroid_weight: 0.2` under `ranking:` in
   `config/extraction.yaml` to include the cluster signal in scoring.

Clustering re-runs automatically on subsequent `brain-build` calls once
`clustering.enabled: true` is set. Adjust `clustering.n_clusters` for the right
granularity (8–15 is a good starting range for a 500-paper library).

```bash
lit-monitor cluster recompute        # run K-means + LLM naming
lit-monitor cluster view             # show cluster themes + paper counts
lit-monitor cluster assign           # assign every paper to nearest cluster
lit-monitor cluster write-back tags  # tag Zotero items with theme names
```

### (c) Graph signals — tuning the ranking mix

Enable once the knowledge graph has been populated and you want the citation and
entity-overlap signals.

1. Run `lit-monitor graph backfill --all` (first time only; incremental
   thereafter).
2. Set nonzero weights in `config/extraction.yaml`:

```yaml
ranking:
  graph_entity_overlap_weight: 0.15
  graph_citation_weight:       0.10
  graph_shared_authors_weight: 0.05
```

3. Optionally add a `domain_context.yaml` paragraph and run
   `lit-monitor domain analyze` to activate the `domain_context` signal:

```yaml
ranking:
  domain_context_weight: 0.20
```

4. Check the score decomposition in the discovery web UI or with
   `lit-monitor discovery view --run latest --breakdown` to confirm all signals
   are contributing as expected.

Trending-concept suggestions and researcher gating are independent features —
see `lit-monitor trending suggest` and `config/researchers.yaml` respectively.

## LLM and embedding providers

Local Ollama is the default for both LLM inference and embeddings. Embeddings
always run locally unless explicitly switched to a provider.

### LLM routing

To route any extraction mode through LiteLLM:

```bash
uv sync --extra litellm
```

Then per-mode in `config/extraction.yaml`:

```yaml
modes:
  simple:
    provider: litellm
    litellm_model: claude-3-5-sonnet-20241022
    # ... other keys unchanged ...
  complex:
    provider: litellm
    litellm_model: claude-opus-4-5
```

Mix per-mode — local Ollama for `simple`, cloud Claude for `complex`. API keys
come from your environment per
[LiteLLM's provider docs](https://docs.litellm.ai/docs/providers).

### Embedding routing

Embeddings also support LiteLLM providers (OpenAI, Cohere, Vertex AI, or any
provider that exposes an embeddings endpoint):

```bash
lit-monitor embeddings switch --provider litellm --model text-embedding-3-small
# Or switch back to local Ollama:
lit-monitor embeddings switch --provider ollama --model mxbai-embed-large
```

After switching, run `lit-monitor embeddings rebuild` to re-embed your library
under the new model. The `embeddings status` command shows the current provider,
model, embedding dimensionality, and how many papers are indexed.

## Notifications and delivery

An OS notification fires when a discovery run completes. Clicking it lands at the
chooser page on first use, then remembers your preferred surface (browser,
Obsidian, or dismiss). Four delivery flags under `discovery:` in
`config/extraction.yaml`:

| Key | Default | Effect |
|---|---|---|
| `notify.enabled` | `true` | Fire OS notification at run end |
| `notify.preferred_viewer` | `""` | Skip the chooser when set |
| `notes.auto_write_per_paper` | `true` | `false` → defer per-paper notes to `obsidian sync` |
| `digest.auto_write` | `true` | `false` → no inline digest; use `discovery export-md` |

## Strict mode and diagnose

```bash
lit-monitor --strict run --dry-run        # CLI flag
LIT_MONITOR_STRICT=1 lit-monitor run      # env var
lit-monitor diagnose --config-only        # validate every tracked config
lit-monitor diagnose                      # full health (config + Ollama + Zotero)
```

Strict mode turns every silent fallback (corrupt config, unreadable attachment,
unexpected API response) into a hard error.
