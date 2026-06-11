# Vendored third-party code

## findpapers 0.6.7 (MIT)

`findpapers/` is a vendored copy of [findpapers](https://pypi.org/project/findpapers/)
version **0.6.7**, MIT-licensed (see `findpapers/LICENSE`).

### Why it is vendored

findpapers 0.6.7 is unmaintained and its package metadata hard-pins
`typer>=0.3.2,<0.4.0`. That pin transitively forces `click<7.2`, which conflicts
with lit-monitor's `click>=8.0` (and with `chromadb`'s `typer>=0.9`). A
`[tool.uv]` override hid the conflict for `uv`, but **pip ignores it**, so a
vanilla `pip install lit-monitor` failed the resolver with `ResolutionImpossible`.

lit-monitor only ever used findpapers as a **library** — `findpapers.search`,
`findpapers.utils.persistence_util.load`, and its paper/search models. It never
used findpapers' typer CLI. Vendoring the library and dropping the PyPI
dependency removes the typer pin entirely, so `pip install lit-monitor` now
resolves cleanly.

### What was changed from upstream

- **`cli.py` removed** — the only module that imported `typer`. Nothing in the
  library path imports it.
- **Internal absolute imports rewritten** from the `findpapers.*` namespace to
  `lit_monitor._vendor.findpapers.*` so the package works at its vendored
  location.
- **`__version__`** now falls back to the hard-coded vendored version
  (`"0.6.7"`) instead of `importlib.metadata.version(__name__)`, because the
  vendored package has no installed distribution metadata to look up.

Everything else (`searchers/`, `models/`, `utils/`, `tools/`) is unmodified
upstream 0.6.7 source.

### Runtime dependencies absorbed by lit-monitor

The vendored (non-cli) code imports: `lxml`, `xmltodict`, `edlib`, `colorama`,
`inquirer`, and `requests` (`requests` was already a lit-monitor dependency).
These are declared in lit-monitor's `pyproject.toml` with loose lower bounds —
findpapers' restrictive upper caps were intentionally NOT copied, to avoid
introducing new resolver conflicts.
