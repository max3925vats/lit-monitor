"""Reusable diagnose entry point for the lit-monitor CLI + web UI.

``run_diagnose()`` returns the same per-file/per-check status that the
CLI's ``diagnose`` command prints, but suitable for direct consumption
by the web server (no Click side-effects, no exit codes).

Section helpers (``check_core_configs``, ``check_schemas``,
``check_optional_configs``, ``check_prompts``) are also exported so the
CLI wrapper can reproduce its grouped output byte-for-byte while sharing
the validation logic with the web UI.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml as _yaml

# Call ``config_dir()`` at read time (never bind the path at import) so the
# three-tier resolver honours HOME / LIT_MONITOR_ROOT / CWD changes — including
# those a test makes via monkeypatch — on every section helper below.
from lit_monitor.core import config as _config_mod
from lit_monitor.core.config import config_dir
from lit_monitor.core.path_utils import resolve_path as _resolve_path
from lit_monitor.core.strict_mode import get_strict_override, set_strict
from lit_monitor.setup.health_check import run_health_check

logger = logging.getLogger(__name__)

# Sentinel used by check_optional_configs() to flag a file that is
# missing-but-acceptable. The CLI wrapper renders this as a yellow "--"
# row instead of a green OK / red FAIL.
ABSENT_OPTIONAL_MSG = "absent (optional)"


def _validate_yaml_file(path: Path) -> tuple[bool, str]:
    """Load a YAML file and return (ok, message) without printing."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"not found: {path}")
        with path.open(encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        if data is None:
            raise ValueError("file is empty or contains only comments")
        return (True, str(path))
    except FileNotFoundError:
        # "not found" carries no tainted/exception text beyond the config path
        # we constructed ourselves — safe and useful to surface.
        return (False, f"not found: {path}")
    except Exception:
        # Info-leak guard: a YAML parse error can echo file content / offsets.
        # Log the detail; surface a constant message so no exception text
        # reaches the dev diagnostics panel (py/stack-trace-exposure).
        logger.warning("diagnose: failed to validate %s", path, exc_info=True)
        return (False, "invalid or unreadable — see server logs")


def check_core_configs() -> dict[str, tuple[bool, str]]:
    """Validate paths.yaml and extraction.yaml."""
    results: dict[str, tuple[bool, str]] = {}
    results["paths.yaml"] = _validate_yaml_file(config_dir() / "paths.yaml")
    results["extraction.yaml"] = _validate_yaml_file(config_dir() / "extraction.yaml")
    return results


def check_schemas() -> dict[str, tuple[bool, str]]:
    """Validate extraction_schema.yaml and review_schema.yaml.

    Each label includes a "(using .example.yaml fallback)" suffix when
    the resolver falls back to the example variant, matching the CLI's
    original output exactly.
    """
    results: dict[str, tuple[bool, str]] = {}
    for schema_name in ("extraction_schema.yaml", "review_schema.yaml"):
        try:
            resolved = _resolve_path(Path(f"config/{schema_name}"))
            label = schema_name
            if resolved.name.endswith(".example.yaml"):
                label += " (using .example.yaml fallback)"
            results[label] = _validate_yaml_file(resolved)
        except Exception:
            # Info-leak guard: log the detail; surface a constant message so
            # no exception text reaches the response (py/stack-trace-exposure).
            logger.warning("diagnose: failed to resolve %s", schema_name, exc_info=True)
            results[schema_name] = (False, "could not resolve — see server logs")
    return results


def check_optional_configs() -> dict[str, tuple[bool, str]]:
    """Validate optional config files (topics, domain_context, etc.).

    Missing files are reported as ``(True, ABSENT_OPTIONAL_MSG)`` —
    a missing optional is *not* a failure, but the CLI wrapper still
    needs to render it as a distinct yellow "--" row.
    """
    results: dict[str, tuple[bool, str]] = {}
    for optional_name in (
        "topics.yaml",
        "domain_context.yaml",
        "researchers.yaml",
        "concepts.yaml",
    ):
        p = config_dir() / optional_name
        if p.exists():
            results[optional_name] = _validate_yaml_file(p)
        else:
            results[optional_name] = (True, ABSENT_OPTIONAL_MSG)
    return results


def check_prompts() -> dict[str, tuple[bool, str]]:
    """Validate the four prompt YAMLs, with .example.yaml fallbacks."""
    results: dict[str, tuple[bool, str]] = {}
    prompts_dir = config_dir() / "prompts"
    for prompt_name in ("clustering.yaml", "extraction.yaml", "rationale.yaml", "synthesis.yaml"):
        p = prompts_dir / prompt_name
        example_p = prompts_dir / prompt_name.replace(".yaml", ".example.yaml")
        if p.exists():
            results[f"prompts/{prompt_name}"] = _validate_yaml_file(p)
        elif example_p.exists():
            label = f"prompts/{prompt_name} (using .example.yaml fallback)"
            results[label] = _validate_yaml_file(example_p)
        else:
            results[f"prompts/{prompt_name}"] = (False, "absent")
    return results


def _check_graph_extra() -> tuple[bool, str]:
    """G13: check whether the [graph] extra is installed and the persist dir is reachable.

    Returns:
        ``(ok, message)`` — ``ok`` is always True unless the persist directory
        parent exists but is not writable.  A missing [graph] extra is not a
        failure — it's an optional feature.
    """
    try:
        import kuzu  # noqa: F401, PLC0415
    except ImportError:
        return (True, "[graph] extra not installed (optional)")

    # Extra is present — check that the persist dir parent is reachable
    try:
        cfg = _config_mod.get_config()
        # Navigate the config hierarchy defensively (retrieval.graph_db.persist_dir)
        _retrieval = getattr(cfg, "retrieval", None)
        _graph_section = getattr(_retrieval, "graph_db", None) if _retrieval else None
        _persist_dir = getattr(_graph_section, "persist_dir", None) if _graph_section else None
    except Exception:  # noqa: BLE001
        _persist_dir = None

    persist_path = Path(
        _persist_dir or "~/.config/lit-monitor/graph.kuzu"
    ).expanduser()

    # This is a read-only diagnostic: probe writability WITHOUT creating any
    # directories. Walk up to the nearest existing ancestor and check it is a
    # writable directory; the missing leaves under it will be created by the
    # actual GraphDB open, not by diagnose.
    parent = persist_path.parent
    probe = parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    if not probe.is_dir():
        return (False, f"persist dir parent not reachable: {probe}")
    if not os.access(probe, os.W_OK):
        return (False, f"persist dir parent not writable: {probe}")

    return (True, f"ok ({persist_path})")


def run_diagnose(*, config_only: bool = False) -> dict[str, tuple[bool, str]]:
    """Read-only validation of all config files (and optionally services).

    Returns a flat ``{file_or_check_name: (ok, message)}`` dict suitable
    for CI checks and the web UI's diagnostics panel. ``config_only=True``
    skips Ollama/Zotero probes — used by CI (no live services) and by the
    web UI when it already has its own service probes via
    ``run_health_check()``.

    Activates strict mode internally so silent config-fallbacks surface
    as hard errors, mirroring the CLI's original behaviour.

    Note:
        Optional-config files that are absent appear as
        ``(True, ABSENT_OPTIONAL_MSG)`` rather than being omitted — the
        CLI wrapper relies on this to render the yellow "--" row.
    """
    # Temporarily force strict mode, restoring the caller's prior override in a
    # finally so this diagnostic never leaves a process-wide side effect behind
    # (it used to set strict on and never reset it).
    _prev_strict = get_strict_override()
    set_strict(True)
    try:
        results: dict[str, tuple[bool, str]] = {}
        results.update(check_core_configs())
        results.update(check_schemas())
        results.update(check_optional_configs())
        results.update(check_prompts())

        # G13: always include a 'graph' row — optional extra, so absence is OK (not a failure)
        results["graph"] = _check_graph_extra()

        if not config_only:
            # The "config" sub-dict contains CheckResult NamedTuples (3-tuples
            # of ok/message/severity), while the ollama/zotero/vault sub-dicts
            # contain plain (ok, msg) 2-tuples. Use indexed access so both
            # shapes pass through without ValueError. See Audit_28_May_2026.md
            # bundle H2 for the regression that motivated this.
            hc: dict[str, dict[str, Sequence[Any]]] = run_health_check()
            for section, sub in hc.items():
                for check_name, value in sub.items():
                    ok, msg = value[0], value[1]
                    # Namespace service-check keys so they don't collide with
                    # config-file keys in the flat dict.
                    results[f"{section}.{check_name}"] = (ok, msg)

        return results
    finally:
        set_strict(_prev_strict)


__all__: list[Any] = [
    "ABSENT_OPTIONAL_MSG",
    "check_core_configs",
    "check_schemas",
    "check_optional_configs",
    "check_prompts",
    "run_diagnose",
]
