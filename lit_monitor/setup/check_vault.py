"""Validate the Obsidian vault path is reachable and writable.

Used by ``run_health_check()`` to expose a vault probe section to the web
UI's health badge. The CLI does not yet call this directly — Task #65
introduces it for the badge's per-check breakdown.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def check_vault(*, cfg: Any = None) -> dict[str, tuple[bool, str]]:
    """Probe the configured Obsidian vault path.

    Args:
        cfg: optional pre-loaded config (test injection). When ``None``,
            ``get_config()`` is called at runtime. Injecting a ``MagicMock``
            bypasses the filesystem lookup so unit tests never touch paths.yaml.

    Returns:
        ``{check_name: (ok, message)}`` describing whether the vault is
        configured, exists, is a directory, and is writable. Probes
        short-circuit on the first failing precondition.
    """
    # Lazy-import so the module is importable even if config is broken;
    # the health-check route catches the result and falls back to "unconfigured".
    try:
        if cfg is None:
            from lit_monitor.core.config import get_config
            cfg = get_config()
        vault_path = Path(cfg.obsidian.vault_path).expanduser()
    except FileNotFoundError:
        return {"paths_yaml": (False, "paths.yaml not found")}
    except Exception:  # noqa: BLE001 — defensive against any config-load failure
        # Info-leak guard: the exception can embed config paths. Log the detail
        # server-side; surface only a constant message so nothing tainted
        # reaches the health detail panel (py/stack-trace-exposure).
        logger.warning("check_vault: failed to load paths.yaml", exc_info=True)
        return {"paths_yaml": (False, "Failed to load paths.yaml — see server logs.")}

    results: dict[str, tuple[bool, str]] = {}
    is_set = bool(str(vault_path).strip())
    results["vault_path_set"] = (
        is_set, f"vault_path={vault_path}" if is_set else "vault_path empty"
    )
    if not is_set:
        return results

    results["vault_exists"] = (vault_path.exists(), str(vault_path))
    if not vault_path.exists():
        return results

    results["vault_is_dir"] = (vault_path.is_dir(), str(vault_path))
    if not vault_path.is_dir():
        return results

    # Writable check — touch a sentinel file and remove it.
    probe = vault_path / ".lit-monitor-probe"
    try:
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        results["vault_writable"] = (True, "writable")
    except Exception:  # noqa: BLE001 — surface any OS-level failure as a probe result
        # Info-leak guard: log the OS error detail; surface a constant message
        # so no exception text reaches the response (py/stack-trace-exposure).
        logger.warning("check_vault: vault not writable", exc_info=True)
        results["vault_writable"] = (False, "not writable — see server logs")
    return results


__all__ = ["check_vault"]
