"""Shared path utilities used across the codebase.

Kept under :mod:`lit_monitor.core` (not under any domain package) so that
lightweight consumers — e.g. the web server's config loader — can resolve config
files without dragging in heavier LLM-domain imports.

Public API:
  resolve_path(path) -> Path
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def _candidate_names(path: Path) -> list[Path]:
    """Return the path plus its ``*.example.yaml`` sibling (when applicable).

    The example fallback keeps the package working on a fresh clone / install
    before the user has customised a given config file (some ship-defaults are
    tracked only as ``*.example.yaml``).
    """
    candidates = [path]
    if path.suffix == ".yaml" and not path.name.endswith(".example.yaml"):
        candidates.append(path.with_suffix(".example.yaml"))
    return candidates


def _packaged_data_path(path: Path) -> Path | None:
    """Resolve ``path`` against the packaged ship-defaults under ``_data/``.

    Relocated runtime data (extraction/review schemas + prompts) ships inside
    the wheel mirrored as ``lit_monitor/_data/<path>`` — e.g.
    ``config/prompts/rationale.yaml`` → ``_data/config/prompts/rationale.yaml``.

    pip installs are unzipped, so ``files("lit_monitor._data").joinpath(...)``
    is a real on-disk location. We materialise it to a plain ``Path`` and
    return it only when it is an existing file (trying the requested name and
    its ``.example.yaml`` sibling). Returns ``None`` when nothing matches.
    """
    try:
        data_root = files("lit_monitor._data")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    for cand in _candidate_names(path):
        resource = data_root.joinpath(*cand.parts)
        # Unzipped install: str(resource) is a real filesystem path.
        fs = Path(str(resource))
        if fs.is_file():
            return fs

    # Top-level user configs (``config/<name>.yaml``) ship their ship-default
    # only as ``_data/config_examples/<name>.example.yaml`` (the same templates
    # first-run seeds into ``~/.config/lit-monitor/``). Map to that location as
    # a last resort so an UNSEEDED wheel still resolves e.g.
    # ``config/domain_context.yaml`` → the packaged example. Only applies to a
    # bare ``config/<file>`` (depth 2), never to ``config/prompts/...`` which
    # has its own relocated copies above.
    if len(path.parts) == 2 and path.parts[0] == "config" and path.suffix == ".yaml":
        stem = path.stem
        example = data_root.joinpath("config_examples", f"{stem}.example.yaml")
        fs = Path(str(example))
        if fs.is_file():
            return fs
    return None


def resolve_path(path: Path) -> Path:
    """Resolve a runtime config path across user / dev / packaged locations.

    Resolution order (first existing wins):

    1. ``path.exists()`` — an absolute path or one relative to CWD.
    2. **User config dir** — when ``path`` is rooted at ``config/`` (e.g.
       ``config/prompts/rationale.yaml``), the part after ``config/`` is tried
       under :func:`lit_monitor.core.config.config_dir` so user overrides /
       first-run-seeded configs in ``~/.config/lit-monitor/`` win.
    3. **CWD + ``__file__`` ancestor walk** — the original dev/editable
       behaviour: finds the repo ``config/`` for files still living there.
    4. **Packaged ship-defaults** — relocated runtime data under
       ``lit_monitor/_data/config/`` inside the installed wheel.

    Each step also tries a sibling ``*.example.yaml`` fallback.

    Args:
        path: Relative config path (e.g. ``Path("config/prompts/rationale.yaml")``).

    Returns:
        An existing :class:`~pathlib.Path` — the requested file or its
        ``.example.yaml`` fallback, from whichever tier resolves first.

    Raises:
        FileNotFoundError: When the file cannot be located in any tier.
    """
    candidates = _candidate_names(path)

    # 1. Absolute / CWD-relative.
    if path.exists():
        return path

    # 2. User config dir — only for paths rooted at ``config/``. Import lazily
    #    to avoid an import cycle (config.py is heavier than this module). Any
    #    failure here (e.g. config_dir() raising) must not break dev fallback.
    if path.parts and path.parts[0] == "config":
        try:
            from lit_monitor.core.config import config_dir

            user_root = config_dir()
            rest = Path(*path.parts[1:])
            for cand in _candidate_names(rest):
                full = user_root / cand
                if full.exists():
                    return full
        except Exception:  # noqa: BLE001 — never let user-dir lookup break resolution.
            pass

    # 3. CWD + __file__ ancestor walk (original dev behaviour, unchanged).
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in candidates:
            full = parent / cand
            if full.exists():
                return full

    # 4. Packaged ship-defaults inside the installed wheel.
    packaged = _packaged_data_path(path)
    if packaged is not None:
        return packaged

    raise FileNotFoundError(
        f"Cannot find {path} — searched CWD, the user config dir, "
        f"ancestors of {here}, and packaged lit_monitor._data."
    )
