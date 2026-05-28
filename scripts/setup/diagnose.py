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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml as _yaml

# Import the source module rather than the attribute so test-suite
# patches like ``patch("scripts.core.config._config_mod._CONFIG_DIR", tmp_path)``
# affect every read inside the section helpers below. Binding the name
# at import time would freeze the path before the patch applies.
from scripts.core import config as _config_mod
from scripts.core.strict_mode import set_strict
from scripts.llm.extraction_schema import _resolve_path
from scripts.setup.health_check import run_health_check

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
    except Exception as exc:
        return (False, str(exc))


def check_core_configs() -> dict[str, tuple[bool, str]]:
    """Validate paths.yaml and extraction.yaml."""
    results: dict[str, tuple[bool, str]] = {}
    results["paths.yaml"] = _validate_yaml_file(_config_mod._CONFIG_DIR / "paths.yaml")
    results["extraction.yaml"] = _validate_yaml_file(_config_mod._CONFIG_DIR / "extraction.yaml")
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
        except Exception as exc:
            results[schema_name] = (False, str(exc))
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
        p = _config_mod._CONFIG_DIR / optional_name
        if p.exists():
            results[optional_name] = _validate_yaml_file(p)
        else:
            results[optional_name] = (True, ABSENT_OPTIONAL_MSG)
    return results


def check_prompts() -> dict[str, tuple[bool, str]]:
    """Validate the four prompt YAMLs, with .example.yaml fallbacks."""
    results: dict[str, tuple[bool, str]] = {}
    prompts_dir = _config_mod._CONFIG_DIR / "prompts"
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
    set_strict(True)

    results: dict[str, tuple[bool, str]] = {}
    results.update(check_core_configs())
    results.update(check_schemas())
    results.update(check_optional_configs())
    results.update(check_prompts())

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


__all__: list[Any] = [
    "ABSENT_OPTIONAL_MSG",
    "check_core_configs",
    "check_schemas",
    "check_optional_configs",
    "check_prompts",
    "run_diagnose",
]
