# Example configs

`lit-monitor` works for any research field, not just its home turf of
biopharmaceutical process development. The directories here are ready-made,
**synthetic** starting points for three different domains:

| Directory | Flavor |
|---|---|
| `bioprocessing/` | Downstream bioprocess — chromatography, filtration, product quality |
| `ml-research/` | Machine learning / deep learning (NeurIPS / ICML / ICLR style) |
| `climate-science/` | Climate and atmospheric science |

Each directory contains the four domain-flavored, user-facing config files:

- `topics.yaml` — recurring search queries run on every `lit-monitor run`
- `domain_context.yaml` — a short prose paragraph describing your focus area
- `concepts.yaml` — controlled vocabulary mapping Zotero keywords to themes
- `researchers.yaml` — tracked authors whose new papers always surface

## How to use

Pick the domain closest to your work and copy its files into `config/`:

```bash
cp config/examples/ml-research/*.yaml config/
```

Then edit them for your actual topics, authors, and vocabulary. You still
need to seed the non-domain configs (`paths.yaml`, `extraction.yaml`, etc.)
from `config/*.example.yaml` — those are the same regardless of field. See
the main [README](../../README.md) setup section.

## Notes

- The content is **representative, not real**. Author names and ORCIDs are
  placeholders — replace them with people you actually follow. Either `orcid`
  or `scopus_id` must be set for researcher tracking to work.
- These files conform to the same schema the setup wizard reads and writes,
  and are covered by `tests/unit/test_example_configs.py` to guard against
  drift.
- For the full field-by-field documentation of each file, see the
  `config/*.example.yaml` files one level up.
