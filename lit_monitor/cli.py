"""
lit-monitor CLI — entry point for all pipeline commands.
Usage:
    lit-monitor build-vocabulary
    lit-monitor brain-build    [--batch-size N] [--model PROVIDER:MODEL] [--resume/--no-resume]
    lit-monitor run            [--dry-run] [--screen-all]
    lit-monitor obsidian relink
    lit-monitor obsidian retheme --old OLD --new NEW
    lit-monitor obsidian rerender [--source-type TYPE]
    lit-monitor obsidian synthesize --topic TOPIC
    lit-monitor obsidian re-extract --doi DOI [--field FIELD] [--scope doi|failed]
    lit-monitor obsidian rebuild-citations --doi DOI [--scope doi|all|failed]
    lit-monitor status
    lit-monitor check
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tomllib
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from lit_monitor.core.strict_mode import set_strict, strict_fallback
from lit_monitor.setup.reset import (
    ResetResult,
    ResetTarget,
    _format_size_bytes,
    graph_targets,
    perform_state_reset,
    perform_vault_reset,
    state_targets,
    vault_targets,
    vectors_targets,
)

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Canonical log location. Logs live under the user config dir
# (~/.config/lit-monitor/logs/), co-located with state.db / chroma / graph.kuzu —
# the same location reset.py treats as canonical (and intentionally preserves).
# The old repo-relative ``Path(__file__).parent.parent / "logs"`` only worked
# when running from a source checkout. ``_resolve_log_dir`` reads the configured
# state-db parent so a user override of ``state_db.path`` co-locates logs too.
_DEFAULT_LOG_DIR = Path("~/.config/lit-monitor/logs").expanduser()


def _resolve_log_dir() -> Path:
    """Resolve the canonical log dir: ``<state_db parent>/logs``.

    Mirrors how reset.py co-locates runtime state under the config dir. Falls
    back to ``~/.config/lit-monitor/logs`` when config is unavailable so logging
    setup never fails just because config can't be loaded.
    """
    try:
        cfg = _make_config()
        state_path = getattr(getattr(cfg, "state_db", None), "path", None)
        if state_path:
            return Path(state_path).expanduser().parent / "logs"
    except Exception:  # noqa: BLE001 — logging must work even without config
        pass
    return _DEFAULT_LOG_DIR


def _setup_logging(mode: str, log_dir: Path | None = None, verbose: bool = False) -> None:
    """Configure root logger: console + JSONL file.

    Root logger is always set to DEBUG so the JSONL file captures everything
    that passes per-logger filters. Visibility is controlled per-handler:

    - Console (stderr): WARNING+ by default; DEBUG+ when ``--verbose`` is set.
    - JSONL file:       always DEBUG+ — full diagnostic record on disk.

    Noisy third-party loggers (httpx, chromadb, urllib3, httpcore) are filtered
    at the logger level to WARNING regardless of verbose flag.  Their DEBUG/INFO
    chatter never reaches any handler — too noisy to keep on disk.
    """
    # Silence noisy third-party loggers — filtered at logger level, never propagates.
    for _noisy in ("chromadb.telemetry", "chromadb", "httpx", "urllib3", "httpcore"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    effective_dir = log_dir or _resolve_log_dir()
    effective_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = effective_dir / f"{date_str}_{mode}.jsonl"
    root = logging.getLogger()
    # Root at DEBUG always — each handler picks its own visibility threshold.
    # Setting root to WARNING here would filter records BEFORE handlers see them,
    # silently dropping INFO/DEBUG from the JSONL backup file too.
    root.setLevel(logging.DEBUG)
    # Console handler — user-facing; quiet by default.
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
               for h in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.DEBUG if verbose else logging.WARNING)
        console.setFormatter(logging.Formatter("%(levelname)s  %(name)s: %(message)s"))
        root.addHandler(console)
    # JSONL file handler — structured; always captures DEBUG+. Only attached once
    # per (process, log_file) pair so REPL / test harnesses that call setup twice
    # don't get duplicate writes.
    log_file_abs = log_file.resolve()
    if not any(
        isinstance(h, _JsonlFileHandler) and Path(h.baseFilename).resolve() == log_file_abs
        for h in root.handlers
    ):
        file_handler = _JsonlFileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)
class _JsonlFileHandler(logging.FileHandler):
    """Writes one JSON object per line."""
    def emit(self, record: logging.LogRecord) -> None:
        try:

            entry = {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                # formatException lives on Formatter, not Handler — use traceback directly.
                # Strips trailing newline so the JSON line is self-contained.
                entry["exc"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()
            self.stream.write(json.dumps(entry) + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)
# ---------------------------------------------------------------------------
# Secrets loader
# ---------------------------------------------------------------------------
_SECRETS_PATH = Path.home() / ".config" / "lit-monitor" / "config.toml"
def _load_secrets() -> dict[str, Any]:
    if not _SECRETS_PATH.exists():
        return {}
    try:
        with _SECRETS_PATH.open("rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:
        strict_fallback(
            logger,
            f"Could not parse secrets file {_SECRETS_PATH}: {exc}",
            exc,
        )
        return {}
def _maybe_set_ollama_key(secrets: dict[str, Any]) -> None:
    """Inject OLLAMA_API_KEY from config.toml into the environment if not already set.

    Precedence: shell env var wins over config.toml. This mirrors the AWS/GCP
    SDK convention and means an explicit export always overrides the file.
    Called once per command after _load_secrets(); OllamaClient.__init__ then
    reads the env var automatically for every client created in that process.
    """
    if os.environ.get("OLLAMA_API_KEY"):
        return  # env var already set — leave it unchanged
    toml_key = secrets.get("ollama", {}).get("api_key", "")
    if toml_key:
        os.environ["OLLAMA_API_KEY"] = toml_key
def _maybe_set_s2_key(secrets: dict[str, Any]) -> None:
    """Inject S2_API_KEY from config.toml into the environment if not already set.

    Precedence: shell env var wins over config.toml — same convention as
    _maybe_set_ollama_key. Must be called BEFORE importing modules under
    scripts.search.* (semantic_scholar / citation_graph), because those
    modules capture os.environ["S2_API_KEY"] at import time into a module-level
    _DEFAULT_S2_API_KEY constant.
    """
    if os.environ.get("S2_API_KEY"):
        return  # env var already set — leave it unchanged
    toml_key = secrets.get("semantic_scholar", {}).get("api_key", "")
    if toml_key:
        os.environ["S2_API_KEY"] = toml_key
# ---------------------------------------------------------------------------
# Shared object factory helpers
# ---------------------------------------------------------------------------
def _make_config():
    from lit_monitor.core.config import get_config
    return get_config()
def _make_state_db(config):
    from lit_monitor.core.state_db import StateDB
    return StateDB(config.state_db.path)
def _make_embeddings_db(config):
    from lit_monitor.output.embeddings import EmbeddingsDB
    persist_dir = str(Path(config.state_db.path).parent / "chroma")
    # Use embeddings.ollama_host if explicitly set; fall back to brain_build host.
    ollama_host = getattr(config.embeddings, "ollama_host", None)
    if ollama_host is None:
        ollama_host = getattr(config.brain_build, "ollama_host", "http://localhost:11434")
    # Bundle F: read provider config; default to ollama/mxbai-embed-large if absent.
    emb_cfg = getattr(config, "embedding", None)
    if emb_cfg is not None:
        provider = getattr(emb_cfg, "provider", "ollama")
        provider_sub = getattr(emb_cfg, provider, None)
        model = getattr(provider_sub, "model", None) if provider_sub else None
        dim = getattr(provider_sub, "dim", None) if provider_sub else None
        # For ollama, prefer the sub-block host over the legacy brain_build host.
        if provider == "ollama" and provider_sub is not None:
            ollama_host = getattr(provider_sub, "host", ollama_host)
    else:
        provider = "ollama"
        model = getattr(config.embeddings, "model", "mxbai-embed-large")
        dim = None
    return EmbeddingsDB(
        persist_dir=persist_dir,
        ollama_host=ollama_host,
        provider=provider,
        model=model,
        dim=dim,
        state_db_path=str(Path(config.state_db.path).expanduser()),
    )
def _make_zotero_client(config, secrets: dict):
    from lit_monitor.core.zotero_client import ZoteroClient
    zot_secrets = secrets.get("zotero", {})
    return ZoteroClient(
        library_id=str(zot_secrets.get("library_id", config.zotero.library_id)),
        api_key=str(zot_secrets.get("api_key", "")),
        library_type=config.zotero.library_type,
        local_storage_path=config.zotero.local_storage_path,
    )
def _make_llm(config, mode: str, model_override: str | None = None, think: bool = True):
    from lit_monitor.llm.llm_client import (
        _OLLAMA_NUM_CTX_DEFAULTS,
        OllamaClient,
        get_clients_for_passes,
    )
    _num_ctx_for_mode = _OLLAMA_NUM_CTX_DEFAULTS

    if model_override:
        # model_override is used by compare-models — always a single Ollama client;
        # per-pass selection (F15) does not apply when an explicit override is given.
        parts = model_override.split(":", 1)
        if len(parts) == 2 and parts[0] == "ollama":
            model_name = parts[1]
        else:
            model_name = model_override
        mode_config = getattr(config, mode, config.ingestion)
        raw_timeout = getattr(mode_config, "timeout", None)
        # Honour per-mode 'think:' from extraction.yaml — same logic as get_client().
        mode_think = getattr(mode_config, "think", None)
        effective_think = mode_think if mode_think is not None else think
        return OllamaClient(
            model=model_name,
            host=getattr(mode_config, "ollama_host", "http://localhost:11434"),
            timeout=int(raw_timeout) if raw_timeout is not None else None,
            temperature=float(getattr(mode_config, "temperature", 0.1)),
            num_ctx=_num_ctx_for_mode.get(mode),
            think=effective_think,
        )
    # No model_override → use extraction.yaml config, including per-pass models if set (F15).
    return get_clients_for_passes(config, mode=mode, think=think)
# ---------------------------------------------------------------------------
# Result printer helpers
# ---------------------------------------------------------------------------
def _print_check_results(results: dict[str, Sequence[Any]]) -> bool:
    """Print check results, return True if all passed.

    Severity-aware: when a result is a ``CheckResult`` with ``severity="warn"``
    (optional source absent), render a yellow ⚠ rather than green ✓. The
    boolean return still reflects strict ok-ness — warns don't fail the run.
    """
    all_ok = True
    for name, result in results.items():
        ok = result[0]
        msg = result[1]
        sev = getattr(result, "severity", None)
        if sev == "warn":
            icon = click.style("⚠", fg="yellow")
        elif ok:
            icon = click.style("✓", fg="green")
        else:
            icon = click.style("✗", fg="red")
        click.echo(f"  {icon}  {name}: {msg}")
        if not ok:
            all_ok = False
    return all_ok
# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------
@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.option(
    "--strict", "-S",
    is_flag=True,
    default=False,
    help=(
        "Turn silent fallbacks into hard errors. "
        "Corrupt configs, unreadable files, and unexpected API responses "
        "raise RuntimeError instead of logging a warning and continuing. "
        "Also activated by LIT_MONITOR_STRICT=1 in the environment."
    ),
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, strict: bool) -> None:
    """lit-monitor — personal literature monitoring pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["strict"] = strict
    if strict:
        set_strict(True)
# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
@main.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """Verify configuration, Ollama, and Zotero are all reachable."""
    from lit_monitor.setup.health_check import run_health_check
    _setup_logging("check", verbose=ctx.obj.get("verbose", False))
    overall_ok = True
    # Single in-process call returns per-section results in the same
    # shape the original inline code produced — the web UI (Phase C/D)
    # reuses run_health_check() directly instead of shelling out.
    health = run_health_check()
    click.echo(click.style("\n── Configuration ──", bold=True))
    if not _print_check_results(health["config"]):
        overall_ok = False

    click.echo(click.style("\n── Ollama ──", bold=True))
    if not _print_check_results(health["ollama"]):
        overall_ok = False
    click.echo(click.style("\n── Zotero ──", bold=True))
    if not _print_check_results(health["zotero"]):
        overall_ok = False
    # Ollama API key visibility — show source without leaking the key value.
    click.echo(click.style("\n── Ollama API Key ──", bold=True))
    secrets = _load_secrets()
    toml_key = secrets.get("ollama", {}).get("api_key", "")
    env_key = os.environ.get("OLLAMA_API_KEY", "")
    # Determine configured Ollama host to distinguish local vs cloud.
    ollama_host = "http://localhost:11434"
    try:
        cfg = _make_config()
        ollama_host = getattr(cfg.brain_build, "ollama_host", ollama_host)
    except Exception:
        pass
    is_local = any(
        ollama_host.startswith(prefix)
        for prefix in ("http://localhost", "http://127.0.0.1", "http://0.0.0.0")
    )
    if env_key:
        click.echo(f"  {click.style('✓', fg='green')}  source: shell env var (OLLAMA_API_KEY)")
    elif toml_key:
        click.echo(f"  {click.style('✓', fg='green')}  source: config.toml ([ollama].api_key)")
    elif is_local:
        click.echo(f"  {click.style('✓', fg='green')}  no key set — local Ollama, no auth required")
    else:
        click.echo(
            f"  {click.style('!', fg='yellow')}  no key set — cloud Ollama host detected "
            f"({ollama_host}); set OLLAMA_API_KEY or add [ollama] api_key to config.toml"
        )
    click.echo()
    if overall_ok:
        click.echo(click.style("All checks passed.", fg="green", bold=True))
    else:
        click.echo(click.style("Some checks failed — see above.", fg="red", bold=True))
        sys.exit(1)
# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--config-only",
    is_flag=True,
    default=False,
    help=(
        "Only validate tracked config files (paths.yaml, extraction.yaml, schema "
        "YAMLs, prompt YAMLs). Skip service-reachability checks (Ollama, Zotero). "
        "Safe to run in CI where those services are not present."
    ),
)
@click.pass_context
def diagnose(ctx: click.Context, config_only: bool) -> None:
    """Read-only health report. Reveals every silent fallback that would have occurred.

    Activates strict mode internally so silent fallbacks become hard errors.
    Each tracked config file is loaded and reported as OK or FAIL. Then (unless
    --config-only) runs the same service checks as ``lit-monitor check``.

    Use when something feels off but ``lit-monitor check`` returns OK — for
    example, a corrupt domain_context.yaml silently becomes "" and the LLM
    gets no domain context. This command surfaces that before any pipeline run.

    Exit code 0 if all files load cleanly; 1 if any FAIL.
    """
    _setup_logging("diagnose", verbose=ctx.obj.get("verbose", False))
    from lit_monitor.setup.diagnose import (
        ABSENT_OPTIONAL_MSG,
        check_core_configs,
        check_optional_configs,
        check_prompts,
        check_schemas,
    )
    from lit_monitor.setup.health_check import run_health_check
    # Helper closure: prints a single (label, (ok, msg)) row in the same
    # format the old inline _check_yaml emitted. Mutates `all_ok` via
    # nonlocal so the section loops match the original control flow.
    all_ok = True

    def _print_file_row(label: str, status: tuple[bool, str]) -> None:
        nonlocal all_ok
        ok, msg = status
        # ABSENT_OPTIONAL_MSG is encoded as (True, ABSENT_OPTIONAL_MSG)
        # by check_optional_configs(); render the same yellow "--" row
        # the inline code produced. Not a failure.
        if ok and msg == ABSENT_OPTIONAL_MSG:
            click.echo(f"  {click.style('--', fg='yellow')}  {label}: {ABSENT_OPTIONAL_MSG}")
            return
        if ok:
            click.echo(f"  {click.style('OK', fg='green')}  {label}: {msg}")
        else:
            click.echo(f"  {click.style('FAIL', fg='red')}  {label}: {msg}")
            all_ok = False

    click.echo(click.style("\n── Core config files ──", bold=True))
    for label, status in check_core_configs().items():
        _print_file_row(label, status)

    click.echo(click.style("\n── Schema files ──", bold=True))
    for label, status in check_schemas().items():
        _print_file_row(label, status)

    click.echo(click.style("\n── Optional config files ──", bold=True))
    for label, status in check_optional_configs().items():
        _print_file_row(label, status)

    click.echo(click.style("\n── Prompt YAMLs ──", bold=True))
    for label, status in check_prompts().items():
        _print_file_row(label, status)

    if not config_only:
        click.echo(click.style("\n── Service checks ──", bold=True))
        health = run_health_check()
        click.echo(click.style("  Configuration:", bold=False))
        if not _print_check_results(health["config"]):
            all_ok = False
        click.echo(click.style("  Ollama:", bold=False))
        if not _print_check_results(health["ollama"]):
            all_ok = False
        click.echo(click.style("  Zotero:", bold=False))
        if not _print_check_results(health["zotero"]):
            all_ok = False

    click.echo()
    if all_ok:
        click.echo(click.style("All diagnostics passed.", fg="green", bold=True))
    else:
        click.echo(click.style("Some diagnostics failed — see above.", fg="red", bold=True))
        sys.exit(1)
# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@main.command("serve")
@click.option("--host", default=None,
              help="Bind host (default: [server].host from config.toml, else 127.0.0.1).")
@click.option("--port", default=None, type=int,
              help="Bind port (default: [server].port from config.toml, else 8765).")
@click.option("--reload", is_flag=True, default=False)
@click.option("--no-browser", is_flag=True, default=False,
              help="Skip auto-opening the browser even if [server].open_browser is true.")
@click.option("--dev", is_flag=True, default=False,
              help="Enable internal test-setup page at /dev. Adds dev-mode banner; mounts /dev router.")
@click.pass_context
def serve(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    reload: bool,
    no_browser: bool,
    dev: bool,
) -> None:
    """Start the lit-monitor web UI + API server.

    Precedence for host/port: CLI flag > [server] block in config.toml >
    hardcoded default (127.0.0.1:8765).
    """
    import threading
    import webbrowser

    import uvicorn

    from lit_monitor.server.config_io import load_server_config

    _setup_logging("serve", verbose=ctx.obj.get("verbose", False))
    # Propagate OLLAMA_API_KEY into the process env so embedding/LLM calls
    # made from FastAPI request handlers (dev page ingest, brain-build via
    # /api/control, etc.) authenticate against cloud Ollama. CLI subcommands
    # already do this; serve() needs it too for the spawned uvicorn worker
    # (same process, factory mode).
    _maybe_set_ollama_key(_load_secrets())
    server_cfg = load_server_config()
    final_host = host if host is not None else server_cfg.get("host", "127.0.0.1")
    final_port = port if port is not None else int(server_cfg.get("port", 8765))

    # Browser-open precedence: --no-browser CLI flag wins, else [server].open_browser,
    # else default True.
    cfg_open_browser = bool(server_cfg.get("open_browser", True))
    effective_open_browser = (not no_browser) and cfg_open_browser

    # Propagate --dev to the worker process started by uvicorn's app factory.
    # uvicorn spawns create_app() in a fresh import context, so an env var is
    # the cleanest hand-off (Python-level state would not survive the import).
    if dev:
        os.environ["LIT_MONITOR_DEV"] = "1"

    click.echo(">> lit-monitor serve")
    click.echo(f">> running at http://{final_host}:{final_port}  (Ctrl+C to stop)")
    if dev:
        click.echo(">> DEV MODE enabled — /dev test surface mounted; sandbox active.")

    if effective_open_browser:
        url = f"http://{final_host}:{final_port}"

        def _open_browser_safe() -> None:
            # Headless boxes (no DISPLAY) raise from webbrowser.open — swallow,
            # log a warning, and keep serving.
            try:
                webbrowser.open(url)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to auto-open browser at %s: %s", url, exc)

        timer = threading.Timer(1.5, _open_browser_safe)
        timer.daemon = True
        timer.start()

    uvicorn.run(
        "lit_monitor.server.app:create_app",
        factory=True,
        host=final_host,
        port=final_port,
        reload=reload,
        log_config=None,
    )


# ---------------------------------------------------------------------------
# first-run — interactive onboarding
# ---------------------------------------------------------------------------
@main.command("first-run")
@click.pass_context
def first_run(ctx: click.Context) -> None:
    """Interactive first-time setup. Credentials are write-once (skipped if
    `~/.config/lit-monitor/config.toml` already exists). The `[server]`
    block (host/port/open_browser) is re-prompted every run so the user
    can change defaults. If the chosen port is already in use, launch is
    skipped (no duplicate serve).
    """
    import socket
    import subprocess
    import time
    import webbrowser

    from lit_monitor.core.seed import seed_user_config
    from lit_monitor.server.config_io import (
        load_server_config,
        save_secrets,
        save_server_config,
    )
    from lit_monitor.setup._paths import SECRETS_PATH

    _setup_logging("first_run", verbose=ctx.obj.get("verbose", False))

    # Step 1: credentials. Only prompt if the file doesn't exist yet.
    if not SECRETS_PATH.exists():
        click.echo(click.style("\nWelcome to lit-monitor!", bold=True))
        click.echo(
            "We'll set up the minimum credentials now. You can add more "
            "(Scopus, WoS, etc.) later from the web UI."
        )
        api_key = click.prompt("Zotero API key", type=str).strip()
        library_id = click.prompt("Zotero library ID", type=str).strip()
        email = click.prompt("PubMed contact email (required by NCBI)", type=str).strip()
        secrets: dict[str, Any] = {
            "zotero": {"api_key": api_key, "library_id": library_id},
            "pubmed": {"email": email},
        }
        save_secrets(secrets)
        click.echo(f"  Wrote credentials to {SECRETS_PATH} (mode 0600).")
    else:
        click.echo(f"Found existing credentials at {SECRETS_PATH} — skipping credential prompts.")

    # Step 1b: seed the user config dir from the packaged example configs.
    # Idempotent — only writes files that don't exist yet, so re-running
    # first-run never clobbers the user's edits.
    seed_target = SECRETS_PATH.parent  # ~/.config/lit-monitor
    seeded = seed_user_config(seed_target)
    if seeded:
        click.echo(f"  Seeded {len(seeded)} example config(s) into {seed_target}:")
        for path in seeded:
            click.echo(f"    - {path.name}")
        click.echo("  Edit these (or use the web Setup wizard) to configure lit-monitor.")
    else:
        click.echo(f"  Config files already present in {seed_target} — nothing to seed.")

    # Step 2: server settings. Always prompt (re-runnable), defaults from
    # the existing [server] block if present.
    existing = load_server_config()
    host: str = click.prompt(
        "Server host", default=existing.get("host", "127.0.0.1"), show_default=True,
    )
    port: int = click.prompt(
        "Server port", default=int(existing.get("port", 8765)), type=int, show_default=True,
    )
    open_browser: bool = click.confirm(
        "Open browser to the setup page after launch?",
        default=bool(existing.get("open_browser", True)),
    )

    # Step 3: persist [server] block.
    save_server_config(host=host, port=port, open_browser=open_browser)
    click.echo(f"  Saved [server] block to {SECRETS_PATH}.")

    # Step 4: detect whether the port is already in use; if so, skip launch.
    url = f"http://{host}:{port}"
    try:
        with socket.create_connection((host, port), timeout=0.5):
            port_in_use = True
    except OSError:
        port_in_use = False

    if port_in_use:
        click.echo(
            f"  Port {port} on {host} is already in use — assuming lit-monitor serve "
            "is already running. Skipping launch."
        )
        if open_browser:
            click.echo(f"  Opening browser at {url}/setup …")
            webbrowser.open(f"{url}/setup")
        return

    # Step 5: detached launch.
    click.echo(f">> launching lit-monitor serve at {url}")
    if open_browser:
        click.echo(f"   (will open browser at {url}/setup once the server is ready)")
    # Detach stdout/stderr so the spawned serve doesn't leak uvicorn output
    # into the user's shell after first-run exits. Structured JSONL logs from
    # _setup_logging("serve", ...) still get written to disk under logs/.
    subprocess.Popen(
        [sys.executable, "-m", "lit_monitor.cli", "serve"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if open_browser:
        # Give uvicorn ~1.5s to bind before pointing the browser at it.
        time.sleep(1.5)
        webbrowser.open(f"{url}/setup")
    log_date = datetime.now().strftime("%Y-%m-%d")
    click.echo("lit-monitor serve is running in the background.")
    click.echo(f"  Logs: logs/{log_date}_serve.jsonl")
    click.echo(f"  To stop it: find the PID via `lsof -i :{port}` and `kill <pid>`.")
# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show extraction and embedding counts from the state DB."""
    _setup_logging("status", verbose=ctx.obj.get("verbose", False))
    try:
        config = _make_config()
        state_db = _make_state_db(config)
    except Exception as exc:
        click.echo(f"Error loading config/state DB: {exc}", err=True)
        sys.exit(1)
    counts_by_status = state_db.count_by_status()
    papers = state_db.get_all_by_source_type("paper")
    reviews = state_db.get_all_by_source_type("review")
    embedded_papers = sum(1 for p in papers if p.get("embeddings_indexed"))
    embedded_reviews = sum(1 for r in reviews if r.get("embeddings_indexed"))
    click.echo(click.style("\n── State DB ──", bold=True))
    click.echo(f"  DB path: {config.state_db.path}")
    click.echo(click.style("\n── Content counts ──", bold=True))
    click.echo(f"  Papers:   {len(papers):4d}  ({embedded_papers} embedded)")
    click.echo(f"  Reviews:  {len(reviews):4d}  ({embedded_reviews} embedded)")
    click.echo(click.style("\n── Status breakdown ──", bold=True))
    for status_val, n in sorted(counts_by_status.items(), key=lambda x: (x[0] is None, x[0] or "")):
        label = status_val if status_val is not None else "(none)"
        click.echo(f"  {label:<30s} {n:4d}")
    click.echo()

    # G13: append graph line when [graph] extra is present
    try:
        from lit_monitor.graph import safe_graph_db as _safe_graph_db
        _g = _safe_graph_db()
        if _g is not None:
            try:
                _conn = _g._conn
                _res = _conn.execute("MATCH (p:Paper) RETURN count(p) AS c")
                _paper_count = int(_res.get_next()[0]) if _res.has_next() else 0
                _res = _conn.execute("MATCH (e:Entity) RETURN count(e) AS c")
                _entity_count = int(_res.get_next()[0]) if _res.has_next() else 0
                # Count papers with graph_indexed=1 vs total from state_db
                with state_db._connect() as _sconn:
                    _rows = _sconn.execute(
                        "SELECT COUNT(*) FROM papers WHERE graph_indexed = 1"
                    ).fetchall()
                    _indexed = _rows[0][0] if _rows else 0
                    _rows = _sconn.execute("SELECT COUNT(*) FROM papers").fetchall()
                    _total = _rows[0][0] if _rows else 0
                click.echo(
                    f"  Graph: indexed={_indexed} / total={_total}  entities={_entity_count}"
                )
            finally:
                try:
                    _g.close()
                except Exception:  # noqa: BLE001
                    pass
    except ImportError:
        pass  # [graph] extra not installed — silently omit the line


# ---------------------------------------------------------------------------
# feedback (P4 Part B) — record one feedback event from the terminal
# ---------------------------------------------------------------------------
@main.command("feedback")
@click.argument("doi")
@click.option("--saved", is_flag=True, default=False,
              help="Mark the paper as saved (positive signal).")
@click.option("--dismissed", is_flag=True, default=False,
              help="Mark the paper as dismissed (negative signal).")
@click.option("--opened", is_flag=True, default=False,
              help="Mark the paper as opened (weak engagement signal).")
@click.option("--thumbs-up", "thumbs_up", is_flag=True, default=False,
              help="Thumbs-up the paper (positive signal).")
@click.option("--thumbs-down", "thumbs_down", is_flag=True, default=False,
              help="Thumbs-down the paper (negative signal).")
@click.option("--rating", type=int, default=None,
              help="Star rating 1-5 (records a 'rated' signal).")
@click.pass_context
def feedback_cmd(
    ctx: click.Context,
    doi: str,
    saved: bool,
    dismissed: bool,
    opened: bool,
    thumbs_up: bool,
    thumbs_down: bool,
    rating: int | None,
) -> None:
    """Record one feedback event for DOI.

    The .md-digest reader (Obsidian) can't click the web buttons, so this CLI
    is the keyboard path to the SAME feedback_events store the web UI writes to
    (P5 Rocchio active learning consumes it).

    Pass exactly one signal: --saved / --dismissed / --opened / --thumbs-up /
    --thumbs-down, OR --rating N (1-5). Validation mirrors POST /api/feedback:
    a 'rated' signal requires a rating in [1, 5], and a rating is rejected for
    any other signal — both enforced by StateDB.record_feedback_event (the
    single source of truth shared with the HTTP route).
    """
    _setup_logging("feedback", verbose=ctx.obj.get("verbose", False))

    # --- input shaping: resolve exactly one signal source ---
    # Map each chosen flag to its closed-vocabulary signal_type.
    chosen = [
        name for flag, name in (
            (saved, "saved"),
            (dismissed, "dismissed"),
            (opened, "opened"),
            (thumbs_up, "thumbs_up"),
            (thumbs_down, "thumbs_down"),
        ) if flag
    ]
    if rating is not None:
        chosen.append("rated")
    if not chosen:
        raise click.UsageError(
            "choose one signal: --saved/--dismissed/--opened/--thumbs-up/"
            "--thumbs-down or --rating N."
        )
    if len(chosen) > 1:
        raise click.UsageError(
            "exactly one signal may be given "
            f"(got {len(chosen)}: {', '.join(chosen)})."
        )
    signal = chosen[0]

    # --- rating range guard (mirrors the route's cross-field check, which runs
    #     BEFORE the DB call so a bad rating fails fast with a friendly message
    #     even though record_feedback_event would also reject it). ---
    if signal == "rated" and not (1 <= int(rating) <= 5):
        click.echo("Error: --rating must be an integer in 1-5.", err=True)
        sys.exit(1)

    # --- friendly DOI guard (the route relies on pydantic typing; the CLI
    #     adds an emptiness check so a stray whitespace arg fails loudly). ---
    doi_clean = doi.strip()
    if not doi_clean:
        click.echo("Error: DOI must not be empty.", err=True)
        sys.exit(1)

    # --- config / state-db load guard (canonical sibling-command pattern) ---
    try:
        config = _make_config()
        state_db = _make_state_db(config)
    except Exception as exc:
        click.echo(f"Error loading config/state DB: {exc}", err=True)
        sys.exit(1)

    # --- record: cross-field rating validation lives in record_feedback_event,
    #     the same ValueError the HTTP route surfaces as a 422. Surface it here
    #     as a friendly error + nonzero exit rather than a raw traceback. ---
    try:
        state_db.record_feedback_event(
            doi_clean,
            signal,
            rating=rating,
            source="cli",
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if signal == "rated":
        click.echo(f"Recorded feedback: {doi_clean} rated {rating}/5.")
    else:
        click.echo(f"Recorded feedback: {doi_clean} → {signal}.")


# ---------------------------------------------------------------------------
# _suggest_topics — helper shared by build-vocabulary
# ---------------------------------------------------------------------------
def _suggest_topics(clusters: dict, suggested_path: Path) -> None:  # noqa: F821
    """
    Write a topics_suggested.yaml from the current clustering and offer to
    append new entries to topics.yaml.

    Behavior:
    - Always writes *suggested_path* (config/topics_suggested.yaml).
    - Compares with the live config/topics.yaml by search name.
    - If new themes exist and stdin is a TTY, prompts the user to append them.
    - Non-interactive runs (cron, CI) print the diff but skip the prompt.
    - Existing manual searches in topics.yaml are never removed.
    """
    import yaml as _yaml

    from lit_monitor.core.config import config_dir as _config_dir
    from lit_monitor.vocabulary.clusterer import clusters_to_topics_yaml

    # The live topics.yaml is user config we read-and-append, so anchor it on the
    # active config dir (write location) — a CWD-relative ``config/topics.yaml``
    # would miss the user's real file in a wheel run from an arbitrary CWD.
    _topics_live = _config_dir() / "topics.yaml"

    # Always write the suggested file so it's available even if the user skips.
    suggested_yaml = clusters_to_topics_yaml(clusters)
    suggested_path.parent.mkdir(parents=True, exist_ok=True)
    suggested_path.write_text(suggested_yaml, encoding="utf-8")
    suggested_data = _yaml.safe_load(suggested_yaml) or {}
    new_suggested = suggested_data.get("searches", [])

    # Determine which suggested entries are genuinely new (not already in topics.yaml).
    if _topics_live.exists():
        live_data = _yaml.safe_load(_topics_live.read_text(encoding="utf-8")) or {}
        live_names: set[str] = {s["name"] for s in live_data.get("searches", [])}
    else:
        live_data = {}
        live_names = set()

    new_entries = [s for s in new_suggested if s["name"] not in live_names]

    click.echo(f"\nWrote {suggested_path} ({len(new_suggested)} suggested searches).")
    if not new_entries:
        click.echo("  All suggested themes already have a matching entry in topics.yaml.")
        return

    click.echo(f"  {len(new_entries)} new theme(s) not yet in topics.yaml:")
    for entry in new_entries:
        click.echo(f"    + {entry['name']}")
        click.echo(f"      query: {entry['query']}")

    if not sys.stdin.isatty():
        # Non-interactive — just print instructions.
        click.echo(
            f"\n  (Non-interactive mode) Review {suggested_path} and copy the "
            "entries you want to config/topics.yaml manually."
        )
        return

    click.echo()
    if click.confirm(
        f"Append these {len(new_entries)} search(es) to config/topics.yaml?",
        default=False,
    ):
        # Append only the new entries; preserve all existing content and order.
        # Note: yaml.dump() does not preserve YAML comments — they are stripped.
        # The user is warned below; originals can be restored from git.
        live_data.setdefault("searches", []).extend(new_entries)
        with open(_topics_live, "w", encoding="utf-8") as _f:
            _yaml.dump(
                live_data, _f,
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )
        logger.info(
            "suggest_topics: appended %d new search(es) to %s",
            len(new_entries), _topics_live,
        )
        click.echo(f"  Appended {len(new_entries)} search(es) to {_topics_live}.")
        click.echo(
            "  Note: YAML comments in topics.yaml are removed by this operation. "
            "Restore from git if needed."
        )
    else:
        logger.info(
            "suggest_topics: user declined to append %d suggestion(s)", len(new_entries)
        )
        click.echo(
            f"  Skipped. Review {suggested_path} and copy the entries you want manually."
        )


# ---------------------------------------------------------------------------
# build-vocabulary
# ---------------------------------------------------------------------------
@main.command("build-vocabulary")
@click.option(
    "--min-count",
    default=2,
    show_default=True,
    help=(
        "Minimum number of library items a keyword must appear in to be included "
        "in LLM clustering. Singletons (count=1) are skipped by default — they "
        "add noise without contributing to discoverable themes. Set to 1 to cluster "
        "every unique tag."
    ),
)
@click.option(
    "--chunk-size",
    default=0,
    show_default=True,
    help=(
        "Hard cap on keywords per LLM clustering call (N1). "
        "Default 0 = smart sizing only (chunk size is computed from the LLM's "
        "num_ctx so each prompt fills the available input budget). "
        "Pass a positive integer to clamp the smart size DOWN — useful when you "
        "want more focused theme groupings per call, or when smart sizing "
        "overshoots on a verbose model. "
        "Smart sizing typically yields 200–1000 keywords/chunk on cloud models "
        "with 128k context, and ~300 keywords/chunk on local Ollama with 8k context."
    ),
)
@click.option(
    "--remediate/--no-remediate",
    default=True,
    show_default=True,
    help=(
        "Run a remediation pass after initial clustering: sends unclustered "
        "keywords back to the LLM with the full list of identified themes and "
        "asks it to assign each keyword to the closest existing theme. "
        "On by default. Use --no-remediate for a faster dry-run."
    ),
)
@click.option(
    "--refine",
    is_flag=True,
    default=False,
    help=(
        "Run a global cross-theme refinement pass after clustering. "
        "The LLM reviews the full vocabulary in one call and identifies keywords "
        "that should appear in additional themes — e.g. 'multimodal chromatography' "
        "belonging in both Bioprocessing and Chromatography. "
        "Off by default; adds one extra LLM call over the complete vocabulary. "
        "Note: the refinement prompt grows with library size (1400+ keywords ≈ 7000 "
        "input tokens). For large libraries the num_ctx=8192 ceiling leaves little "
        "output room — consider reducing --chunk-size to 30 if refinement truncates."
    ),
)
@click.option(
    "--refine-only",
    "refine_only",
    is_flag=True,
    default=False,
    help=(
        "Skip clustering entirely and run only the cross-theme refinement pass on the "
        "existing config/concepts_draft.yaml. Use this after a successful build-vocabulary "
        "run when you want to add cross-theme keyword assignments without re-running "
        "the full pipeline. Mutually exclusive with --refine."
    ),
)
@click.pass_context
def build_vocabulary_cmd(
    ctx: click.Context,
    min_count: int,
    chunk_size: int,
    remediate: bool,
    refine: bool,
    refine_only: bool,
) -> None:
    """Extract Zotero tags, normalise, cluster into themes."""
    _setup_logging("build_vocabulary", verbose=ctx.obj.get("verbose", False))
    from collections import Counter

    from lit_monitor.core.config import config_dir as _config_dir
    from lit_monitor.vocabulary.clusterer import (
        CONCEPTS_DRAFT_PATH as _CONCEPTS_DRAFT_LOGICAL,
    )
    from lit_monitor.vocabulary.clusterer import (
        TOPICS_SUGGESTED_PATH as _TOPICS_SUGGESTED_LOGICAL,
    )
    from lit_monitor.vocabulary.clusterer import (
        _refine_clustering,
        build_vocabulary,
    )
    from lit_monitor.vocabulary.keyword_extractor import extract_all_keywords
    from lit_monitor.vocabulary.normalizer import normalise_keywords

    # Both files are GENERATED outputs (draft + suggested), so they must land in
    # the active config dir — not a CWD-relative ``config/`` path that a wheel
    # run from an arbitrary CWD would create in the wrong place. config_dir()
    # honours LIT_MONITOR_ROOT and the dev ./config fallback. The imported
    # constants stay logical (public import + documented); we resolve to the
    # write location here.
    _active_config_dir = _config_dir()
    CONCEPTS_DRAFT_PATH = _active_config_dir / _CONCEPTS_DRAFT_LOGICAL.name
    TOPICS_SUGGESTED_PATH = _active_config_dir / _TOPICS_SUGGESTED_LOGICAL.name

    if refine and refine_only:
        click.echo("Error: --refine and --refine-only are mutually exclusive.", err=True)
        sys.exit(1)

    # --refine-only: skip Zotero extraction and clustering entirely.
    # Load the existing draft, run the cross-theme pass, write back.
    if refine_only:
        import yaml as _yaml

        from lit_monitor.llm.llm_client import OllamaClient
        try:
            config = _make_config()
            _maybe_set_ollama_key(_load_secrets())
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        if not CONCEPTS_DRAFT_PATH.exists():
            click.echo(
                f"Error: {CONCEPTS_DRAFT_PATH} not found. "
                "Run build-vocabulary (without --refine-only) first.",
                err=True,
            )
            sys.exit(1)
        with open(CONCEPTS_DRAFT_PATH, encoding="utf-8") as fh:
            merged = _yaml.safe_load(fh) or {}
        if not merged.get("themes"):
            click.echo("Error: concepts_draft.yaml has no themes to refine.", err=True)
            sys.exit(1)
        n_themes = len(merged["themes"])
        click.echo(f"Loaded {n_themes} themes from {CONCEPTS_DRAFT_PATH}.")
        click.echo("Running cross-theme refinement pass…")
        # Refinement prompt is large (full vocabulary ≈ 7000 tokens), so we
        # do NOT set num_ctx — let Ollama use the model's native context window.
        bv = config.build_vocabulary
        raw_timeout = getattr(bv, "timeout", None)
        refine_llm = OllamaClient(
            model=getattr(bv, "model", "qwen2.5:3b"),
            host=getattr(bv, "ollama_host", "http://localhost:11434"),
            timeout=int(raw_timeout) if raw_timeout is not None else None,
            temperature=float(getattr(bv, "temperature", 0.1)),
            num_ctx=None,   # no ceiling — refinement input alone is ~7000 tokens
            think=False,
        )
        # max_tokens: positive ceiling. Cloud Ollama (https://ollama.com) returns
        # 400 Bad Request when num_predict=-1 is sent; local Ollama tolerates -1
        # but cloud requires a positive cap. 32768 is generous given a ~7000-token
        # refinement prompt and gemma4:31b-cloud's native context (~128k).
        # debug_output_path: always write the raw response so the user can
        # verify the pass completed and check for truncation.
        debug_path = CONCEPTS_DRAFT_PATH.with_name(
            CONCEPTS_DRAFT_PATH.stem + "_refinement_raw.txt"
        )
        _refine_clustering(
            merged,
            refine_llm,
            max_tokens=32768,
            debug_output_path=debug_path,
        )
        CONCEPTS_DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONCEPTS_DRAFT_PATH, "w", encoding="utf-8") as fh:
            _yaml.dump(merged, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        click.echo(f"Refinement complete. Updated {CONCEPTS_DRAFT_PATH}.")
        click.echo(f"  Raw LLM response saved to: {debug_path}")
        return

    try:
        config = _make_config()
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        zotero_client = _make_zotero_client(config, secrets)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo("Extracting keywords from Zotero…")
    try:
        raw_kw_pairs = extract_all_keywords(zotero_client)
    except Exception as exc:
        # extract_all_keywords retries internally; if it raises, all attempts
        # failed.  Exit non-zero so the shell / cron job sees the failure.
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    if not raw_kw_pairs:
        click.echo(
            "Warning: Zotero returned no tagged items. "
            "Add tags to items in your Zotero collection before running build-vocabulary.",
            err=True,
        )
        sys.exit(0)
    # First element of each pair is the tag; second is the Zotero item key.
    raw_keywords = [kw for kw, _ in raw_kw_pairs]
    click.echo(f"  Found {len(raw_keywords)} raw keyword occurrences.")

    # Frequency filter: drop keywords that appear in fewer than min_count items.
    # This eliminates singleton tags that don't represent meaningful themes, and
    # dramatically reduces the LLM chunk count on large libraries.
    if min_count > 1:
        kw_counts = Counter(raw_keywords)
        raw_keywords = [kw for kw in raw_keywords if kw_counts[kw] >= min_count]
        n_after = len(set(raw_keywords))
        click.echo(
            f"  Frequency filter (min_count={min_count}): "
            f"{len(kw_counts)} unique tags → {n_after} kept."
        )

    click.echo("Normalising…")
    normalised = normalise_keywords(raw_keywords)
    click.echo(f"  {sum(len(v) for v in normalised.values())} unique forms → "
               f"{len(normalised)} canonical clusters.")
    click.echo("Building vocabulary (LLM clustering)…")
    # Use the dedicated build_vocabulary config block (qwen2.5:3b, no timeout
    # during initial verification).  Thinking is OFF — keyword grouping is a
    # structured JSON task, not open-ended reasoning.
    llm = _make_llm(config, mode="build_vocabulary", think=False)
    # Read per-call token budget from config — applies to chunked clustering
    # and the remediation pass only (refinement always uses max_tokens=32768).
    max_tokens = int(getattr(config.build_vocabulary, "max_tokens_per_call", 16384))

    # Refinement needs a separate LLM with no num_ctx ceiling.  The chunking
    # LLM has num_ctx=8192 (to cap verbose chunk output), but the refinement
    # prompt includes the full merged vocabulary — ~7 000 input tokens — leaving
    # almost no room for the response under an 8192-token context window.
    refine_llm_obj = None
    if refine:
        from lit_monitor.llm.llm_client import OllamaClient
        bv = config.build_vocabulary
        raw_timeout = getattr(bv, "timeout", None)
        refine_llm_obj = OllamaClient(
            model=getattr(bv, "model", "qwen2.5:3b"),
            host=getattr(bv, "ollama_host", "http://localhost:11434"),
            timeout=int(raw_timeout) if raw_timeout is not None else None,
            temperature=float(getattr(bv, "temperature", 0.1)),
            num_ctx=None,   # no ceiling — full model context for additions JSON
            think=False,
        )

    # N1: per-mode chunk_chars override from extraction.yaml (analog to M5 for papers).
    _raw_chunk_chars = getattr(config.build_vocabulary, "chunk_chars", None)
    _chunk_chars_override = (
        int(_raw_chunk_chars)
        if isinstance(_raw_chunk_chars, (int, float)) and not isinstance(_raw_chunk_chars, bool)
        else None
    )
    merged_result = build_vocabulary(
        normalised,
        llm,
        max_tokens=max_tokens,
        chunk_size=chunk_size,
        remediate=remediate,
        refine=refine,
        refine_llm=refine_llm_obj,
        chunk_chars_override=_chunk_chars_override,
    )
    click.echo("Vocabulary build complete.")
    if refine:
        debug_path = CONCEPTS_DRAFT_PATH.with_name(
            CONCEPTS_DRAFT_PATH.stem + "_refinement_raw.txt"
        )
        click.echo(f"  Raw refinement response saved to: {debug_path}")

    _suggest_topics(merged_result, TOPICS_SUGGESTED_PATH)
# ---------------------------------------------------------------------------
# compare-models
# ---------------------------------------------------------------------------
@main.command("compare-models")
@click.option("--papers", default=3, show_default=True,
              help="Number of items to run per model.")
@click.option("--models", "models_str", default=None,
              help="Comma-separated provider:model list (default: extraction.yaml comparison_models).")
@click.pass_context
def compare_models_cmd(
    ctx: click.Context,
    papers: int,
    models_str: str | None,
) -> None:
    """Compare LLM models on extraction quality. Review outputs in comparison/."""
    _setup_logging("compare", verbose=ctx.obj.get("verbose", False))
    from lit_monitor.pipelines.model_compare import run_model_comparison
    try:
        config = _make_config()
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        state_db = _make_state_db(config)
        zotero_client = _make_zotero_client(config, secrets)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    # A4: the textbook mode was removed in R-10, leaving --mode a dead
    # single-choice no-op. The flag is gone; "paper" is the only mode.
    mode = "paper"
    models_list = [m.strip() for m in models_str.split(",")] if models_str else None
    click.echo(f"Running model comparison: mode={mode}, n={papers} items per model…")
    result = run_model_comparison(
        config=config,
        state_db=state_db,
        zotero_client=zotero_client,
        n_items=papers,
        models=models_list,
        mode=mode,
    )
    click.echo(click.style(f"\n── Model Comparison ({mode} mode) ──", bold=True))
    click.echo(f"  {'Model':<30} {'Run':>4} {'Failed':>6} {'Avg s':>7} {'JSON Err':>9} {'Missing':>8}")
    click.echo(f"  {'-'*30} {'-'*4} {'-'*6} {'-'*7} {'-'*9} {'-'*8}")
    for s in result.scores:
        click.echo(
            f"  {s.model:<30} {s.items_run:>4} {s.items_failed:>6} "
            f"{s.avg_seconds:>7.1f} {s.json_parse_errors:>9} {s.required_fields_missing:>8}"
        )
    click.echo(f"\n  Outputs written to: {result.output_dir}")
    click.echo("  Review comparison/*/summary.md for qualitative assessment.")
# ---------------------------------------------------------------------------
# brain-build
# ---------------------------------------------------------------------------
@main.command("brain-build")
@click.option("--batch-size", default=50, show_default=True,
              help="Number of papers to process per batch.")
@click.option("--max-papers", default=None, type=int,
              help="Stop after successfully processing N papers. Items without a "
                   ".md attachment are skipped and do not count toward the limit.")
@click.option("--model", default=None,
              help="Override LLM model (e.g. ollama:mistral:7b).")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip papers already fully extracted.")
@click.option("--all-library", is_flag=True, default=False,
              help="Iterate the entire Zotero library instead of the configured collection (N22). "
                   "Items without a .md attachment are still skipped silently.")
@click.option("--resolve-no-doi", is_flag=True, default=False,
              help="After the run, interactively prompt for DOIs for items that had none.")
@click.option(
    "--reset-extractions",
    "reset_extractions",
    is_flag=True,
    default=False,
    help=(
        "M8: Wipe all existing extraction_json values and re-extract from scratch. "
        "Use after a schema change (e.g. upgrading to Phase M). "
        "In non-interactive (cron/CI) runs this flag must be passed explicitly — "
        "brain-build refuses to continue silently when it detects a schema mismatch."
    ),
)
@click.pass_context
def brain_build_cmd(
    ctx: click.Context,
    batch_size: int,
    max_papers: int | None,
    model: str | None,
    resume: bool,
    all_library: bool,
    resolve_no_doi: bool,
    reset_extractions: bool,
) -> None:
    """Process all papers in Zotero 'lit-monitor' collection."""
    _setup_logging("brain_build", verbose=ctx.obj.get("verbose", False))
    # M3: hydrate S2_API_KEY before importing brain_build (which transitively
    # imports scripts.search.semantic_scholar at module-load time).
    _maybe_set_s2_key(_load_secrets())
    from lit_monitor.core.state_db import CURRENT_SCHEMA_VERSION
    from lit_monitor.output.embeddings import check_embed_model_change
    from lit_monitor.pipelines.brain_build import (
        RateLimitExhausted,
        _process_paper,
        run_brain_build,
        write_brain_build_report,
    )
    try:
        config = _make_config()
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
        # Use the model stored on the instance — single source of truth, no repeated config read.
        check_embed_model_change(state_db, embeddings_db, embeddings_db.embed_model)
        zotero_client = _make_zotero_client(config, secrets)
        llm = _make_llm(config, mode="brain_build", model_override=model)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # M8: schema-change detection — prompt to wipe old extractions before the run.
    stored_version = state_db.get_schema_version()
    if stored_version != CURRENT_SCHEMA_VERSION:
        n_extracted = state_db.count_with_extraction()
        if n_extracted > 0:
            if reset_extractions:
                n_reset = state_db.reset_extractions()
                click.echo(
                    f"Schema migration ({stored_version!r} → {CURRENT_SCHEMA_VERSION!r}): "
                    f"wiped {n_reset} extraction(s). Papers will be re-extracted on this run."
                )
            elif sys.stdin.isatty():
                click.echo(
                    f"\nSchema change detected (stored={stored_version!r}, "
                    f"current={CURRENT_SCHEMA_VERSION!r})."
                )
                if click.confirm(
                    f"Wipe {n_extracted} existing extraction(s) and re-extract on this run?",
                    default=False,
                ):
                    n_reset = state_db.reset_extractions()
                    click.echo(f"Wiped {n_reset} extraction(s) — re-extracting on this run.")
                else:
                    click.echo("Keeping existing extractions — continuing without reset.")
            else:
                click.echo(
                    f"Schema mismatch: stored={stored_version!r}, "
                    f"current={CURRENT_SCHEMA_VERSION!r}. "
                    f"{n_extracted} paper(s) have extractions under the old schema. "
                    "Pass --reset-extractions to wipe and re-extract, "
                    "or run interactively to be prompted.",
                    err=True,
                )
                sys.exit(1)
    state_db.set_schema_version(CURRENT_SCHEMA_VERSION)

    max_label = f", max_papers={max_papers}" if max_papers is not None else ""
    scope_label = ", scope=all-library" if all_library else ""
    click.echo(
        f"Starting brain build (batch_size={batch_size}, resume={resume}"
        f"{max_label}{scope_label})…"
    )
    try:
        summary = run_brain_build(
            config=config,
            state_db=state_db,
            zotero_client=zotero_client,
            embeddings_db=embeddings_db,
            llm=llm,
            batch_size=batch_size,
            resume=resume,
            max_papers=max_papers,
            show_progress=True,
            all_library=all_library,
        )
    except RateLimitExhausted as exc:
        # P4.5: preserve the historical exit-code-2 behaviour at the CLI
        # boundary; the library now raises a catchable domain exception.
        click.echo(click.style(f"\n{exc}", fg="red"), err=True)
        click.echo(
            "Resume with `lit-monitor brain-build --resume` once quota resets.",
            err=True,
        )
        sys.exit(2)
    click.echo(click.style("\n── Brain Build Summary ──", bold=True))
    click.echo(f"  Papers processed: {summary.papers_processed}")
    click.echo(f"  Papers skipped:   {summary.papers_skipped}")
    click.echo(f"  Papers failed:    {summary.papers_failed}")
    if summary.errors:
        click.echo(click.style(f"\n  {len(summary.errors)} error(s):", fg="red"))
        for e in summary.errors[:10]:
            click.echo(f"    • {e}")
    # I5: write run report to digests folder
    try:
        model_str = getattr(llm, "model", "") or ""
        report_path = write_brain_build_report(config, summary, model_str=model_str)
        if report_path:
            click.echo(f"\n  Run report: {report_path}")
    except Exception as exc:
        click.echo(click.style(f"\n  Warning: could not write run report: {exc}", fg="yellow"))
    # N20: remind the user to run relink + rebuild-citations after a successful build.
    if summary.papers_processed > 0:
        click.echo()
        click.echo(click.style("Next steps:", bold=True))
        click.echo("  lit-monitor obsidian relink               (populate ## Related Work sections)")
        click.echo("  lit-monitor obsidian rebuild-citations --scope all  (resolve citation edges)")
        click.echo()
    # I2: --resolve-no-doi interactive batch
    if resolve_no_doi and summary.no_doi_items:
        click.echo(click.style(
            f"\n── {len(summary.no_doi_items)} item(s) with no DOI ──", bold=True,
        ))
        from lit_monitor.llm.extractor import extract_paper as _extract_paper
        from lit_monitor.output.obsidian_writer import write_paper_note
        for item_info in summary.no_doi_items:
            click.echo(f"\n  Title: {item_info['title']}")
            click.echo(f"  Key:   {item_info['zotero_key']}")
            doi_input = click.prompt(
                "  Enter DOI (or press Enter to skip)", default="", show_default=False,
            ).strip()
            if not doi_input:
                continue
            try:
                _ok, _ = _process_paper(
                    doi=doi_input,
                    zotero_key=item_info["zotero_key"],
                    item=item_info["_item"],
                    config=config,
                    state_db=state_db,
                    zotero_client=zotero_client,
                    embeddings_db=embeddings_db,
                    llm=llm,
                    extract_paper_fn=_extract_paper,
                    write_paper_note_fn=write_paper_note,
                )
                click.echo(click.style(f"  Processed: {doi_input}", fg="green"))
                summary.papers_processed += 1
            except Exception as exc:
                click.echo(click.style(f"  Error processing {doi_input}: {exc}", fg="red"))
# ---------------------------------------------------------------------------
# run (discovery pipeline)
# ---------------------------------------------------------------------------
@main.command("run")
@click.option("--dry-run", is_flag=True, default=False,
              help="Discovery only — do not write state DB or notes.")
@click.option("--screen-all", is_flag=True, default=False,
              help="Send all new results through LLM rationale (not just top-K).")
@click.option("--top-k", default=20, show_default=True,
              help="Number of papers to include in LLM rationale.")
@click.option("--sim-threshold", default=0.3, show_default=True,
              help="Similarity score threshold for digest inclusion.")
@click.option(
    "--rag-mode",
    type=click.Choice(["vector", "graph", "hybrid"]),
    default=None,
    help=(
        "Retrieval mode for paper ranking. Default reads from "
        "extraction.yaml retrieval.default_mode (falls back to 'vector')."
    ),
)
@click.option(
    "--source",
    type=click.Choice(["manual", "scheduled"]),
    default="manual",
    show_default=True,
    help="How this run was triggered. The OS scheduler passes 'scheduled'.",
)
@click.pass_context
def run_cmd(
    ctx: click.Context,
    dry_run: bool,
    screen_all: bool,
    top_k: int,
    sim_threshold: float,
    rag_mode: str | None,
    source: str,
) -> None:
    """Run the discovery pipeline: search + ranking + ingest new Zotero items."""
    _setup_logging("discovery", verbose=ctx.obj.get("verbose", False))
    # M3: hydrate S2_API_KEY before importing discovery (which transitively
    # imports scripts.search.semantic_scholar at module-load time).
    _maybe_set_s2_key(_load_secrets())
    from lit_monitor.output.embeddings import check_embed_model_change
    from lit_monitor.pipelines.brain_build import RateLimitExhausted
    from lit_monitor.pipelines.discovery import run_discovery
    try:
        config = _make_config()
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
        check_embed_model_change(state_db, embeddings_db, embeddings_db.embed_model)
        zotero_client = _make_zotero_client(config, secrets)
        llm = _make_llm(config, mode="ingestion")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    # G9: resolve rag_mode — CLI flag overrides config; config falls back to "vector".
    _effective_rag = rag_mode or getattr(
        getattr(config, "retrieval", None), "default_mode", "vector"
    ) or "vector"
    # W4: explicit --rag-mode graph|hybrid without the [graph] extra is a hard
    # error — surface it before discovery searches start. Probe with
    # safe_graph_db() so we don't have to import GraphDB directly here.
    if rag_mode in ("graph", "hybrid"):
        from lit_monitor.graph import safe_graph_db as _probe_graph_db
        _probe = _probe_graph_db()
        if _probe is None:
            raise click.UsageError(
                f"--rag-mode {rag_mode} requires the [graph] extra. "
                "Install with: uv sync --extra graph"
            )
        try:
            _probe.close()
        except Exception:
            pass
    mode_label = "[DRY RUN] " if dry_run else ""
    click.echo(f"{mode_label}Starting discovery pipeline… (rag-mode: {_effective_rag})")
    try:
        summary = run_discovery(
            config=config,
            state_db=state_db,
            zotero_client=zotero_client,
            embeddings_db=embeddings_db,
            llm=llm,
            dry_run=dry_run,
            screen_all=screen_all,
            top_k=top_k,
            sim_threshold=sim_threshold,
            rag_mode=_effective_rag,
            trigger=source,
        )
    except RateLimitExhausted as exc:
        # P5.5: preserve the historical exit-code-2 behaviour at the CLI
        # boundary; the library now raises a catchable domain exception.
        click.echo(click.style(f"\n{exc}", fg="red"), err=True)
        click.echo(
            "Quota should reset shortly; the next run will resume.",
            err=True,
        )
        sys.exit(2)
    click.echo(click.style("\n── Discovery Run Summary ──", bold=True))
    click.echo(f"  New papers found: {summary.new_papers_found}")
    click.echo(f"  Papers ingested:  {summary.papers_ingested}")
    click.echo(f"  Papers failed:    {summary.papers_failed}")
    if summary.digest_path:
        click.echo(f"  Digest:           {summary.digest_path}")
    if summary.errors:
        click.echo(click.style(f"\n  {len(summary.errors)} error(s):", fg="red"))
        for e in summary.errors[:10]:
            click.echo(f"    • {e}")
# ---------------------------------------------------------------------------
# obsidian sub-group
# ---------------------------------------------------------------------------
@main.group()
def obsidian() -> None:
    """Obsidian vault management commands."""
@obsidian.command("relink")
@click.option("--doi", default=None,
              help="Relink a single note by DOI (default: all notes).")
@click.option(
    "--rag-mode",
    type=click.Choice(["vector", "graph", "hybrid"]),
    default=None,
    help=(
        "Retrieval mode for similarity expansion. Default reads from "
        "extraction.yaml retrieval.default_mode (falls back to 'vector')."
    ),
)
@click.pass_context
def obsidian_relink(ctx: click.Context, doi: str | None, rag_mode: str | None) -> None:
    """Update ## Related Work and ## Referenced By sections across notes."""
    _setup_logging("relink", verbose=ctx.obj.get("verbose", False))
    from lit_monitor.obsidian_tools.relink import relink_note
    try:
        config = _make_config()
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    # G9: resolve rag_mode FIRST so the graph_db open decision uses the
    # effective mode (CLI flag → config default → "vector"). C4 fix: previously
    # graph_db was opened only when the CLI flag was set, so an implicit
    # config-default of "graph" silently degraded to vector with graph_db=None.
    from lit_monitor.graph import safe_graph_db as _safe_graph_db
    _effective_rag = rag_mode or getattr(
        getattr(config, "retrieval", None), "default_mode", "vector"
    ) or "vector"
    _graph_db = _safe_graph_db() if _effective_rag in ("graph", "hybrid") else None
    # W4: when the user EXPLICITLY requested graph/hybrid via --rag-mode but
    # the [graph] extra is missing, surface a clear UsageError instead of
    # silently falling back to vector. Implicit config-default fallthrough is
    # still allowed (logged at INFO inside safe_graph_db when relevant).
    if rag_mode in ("graph", "hybrid") and _graph_db is None:
        raise click.UsageError(
            f"--rag-mode {rag_mode} requires the [graph] extra. "
            "Install with: uv sync --extra graph"
        )
    if (
        rag_mode is None
        and _effective_rag in ("graph", "hybrid")
        and _graph_db is None
    ):
        logger.info(
            "Config default rag-mode is %r but [graph] extra is not installed; "
            "falling back to vector for this run.", _effective_rag,
        )
        _effective_rag = "vector"
    try:
        def _relink_by_doi(d: str, *, silent_missing: bool = False) -> str | None:
            record = state_db.get_paper(d)
            if not record:
                if not silent_missing:
                    logging.getLogger(__name__).warning("No DB record for doi: %s", d)
                return None
            note_path = record.get("note_path")
            if not note_path:
                if not silent_missing:
                    logging.getLogger(__name__).warning("No note_path in DB for doi: %s", d)
                return None
            relink_note(
                note_path, embeddings_db, state_db, config=config,
                rag_mode=_effective_rag, graph_db=_graph_db,
            )
            return note_path

        if doi:
            click.echo(f"Relinking {doi}…")
            result = _relink_by_doi(doi)
            click.echo(f"Done: {result}")
        else:
            click.echo("Relinking all notes…")
            # Audit R29: pre-filter to records that actually have a note + embeddings
            # before iterating.  This silences the "No note_path in DB" warnings for
            # items that legitimately have no note (e.g. no_markdown skips) and
            # surfaces the actual relink count cleanly.
            all_records = [
                r for r in (
                    state_db.get_all_by_source_type("paper")
                    + state_db.get_all_by_source_type("review")
                )
                if r.get("note_path") and r.get("embeddings_indexed") == 1
            ]
            ok = failed = 0
            for record in all_records:
                d = record.get("doi", "")
                if not d:
                    continue
                try:
                    _relink_by_doi(d, silent_missing=True)
                    ok += 1
                except Exception as exc:
                    logging.getLogger(__name__).error("Relink failed for %s: %s", d, exc)
                    failed += 1
            click.echo(f"Relink complete: {ok} ok, {failed} failed.")
    finally:
        # G9: release graph_db connection when it was opened for this command.
        if _graph_db is not None:
            try:
                _graph_db.close()
            except Exception:  # pragma: no cover
                pass
@obsidian.command("retheme")
@click.option("--old", required=True, help="Old theme name.")
@click.option("--new", "new_theme", required=True, help="New theme name.")
@click.pass_context
def obsidian_retheme(ctx: click.Context, old: str, new_theme: str) -> None:
    """Bulk rename a theme and rewrite all wikilinks across the vault."""
    _setup_logging("retheme", verbose=ctx.obj.get("verbose", False))
    from lit_monitor.obsidian_tools.retheme import retheme
    try:
        config = _make_config()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    vault_path = config.obsidian.vault_path
    click.echo(f"Retheme: '{old}' → '{new_theme}' in {vault_path}")
    stats = retheme(vault_path, old, new_theme)
    click.echo(f"  Notes updated: {stats.get('files_modified', 0)}")
    click.echo(f"  Links rewritten: {stats.get('wikilinks_rewritten', 0)}")
    if stats.get("page_renamed"):
        click.echo("  Theme page renamed.")
@obsidian.command("rerender")
@click.option(
    "--source-type",
    default=None,
    type=click.Choice(["paper", "review"]),
    help="Limit rerender to one content type (default: all).",
)
@click.option("--doi", default=None, help="Rerender a single note by DOI.")
@click.pass_context
def obsidian_rerender(
    ctx: click.Context, source_type: str | None, doi: str | None
) -> None:
    """Regenerate Obsidian notes from extraction JSON in state DB."""
    _setup_logging("rerender", verbose=ctx.obj.get("verbose", False))
    from lit_monitor.obsidian_tools.rerender import rerender_all, rerender_note
    try:
        config = _make_config()
        state_db = _make_state_db(config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    if doi:
        click.echo(f"Rerendering {doi}…")
        note_path = rerender_note(doi, config, state_db)
        click.echo(f"Done: {note_path}")
        return
    types = [source_type] if source_type else ["paper", "review"]
    total_ok = total_failed = 0
    for st in types:
        click.echo(f"Rerendering all {st} notes…")
        stats = rerender_all(st, config, state_db)
        total_ok += stats.get("rerendered", 0)

        total_failed += stats.get("failed", 0)
    click.echo(f"Rerender complete: {total_ok} ok, {total_failed} failed.")
@obsidian.command("synthesize")
@click.option("--topic", default=None, help="Topic to synthesize notes on.")
@click.option(
    "--topics-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "YAML file with a 'topics:' list of strings to synthesize in batch. "
        "Mutually exclusive with --topic."
    ),
)
@click.option("--top-k", default=15, show_default=True,
              help="Number of related notes to include.")
@click.option(
    "--rag-mode",
    type=click.Choice(["vector", "graph", "hybrid"]),
    default=None,
    help=(
        "Retrieval mode for synthesis. Default reads from "
        "extraction.yaml retrieval.default_mode (falls back to 'vector'). "
        "Non-vector modes trigger cloud-Ollama query expansion."
    ),
)
@click.pass_context
def obsidian_synthesize(
    ctx: click.Context,
    topic: str | None,
    topics_file: str | None,
    top_k: int,
    rag_mode: str | None,
) -> None:
    """Generate a synthesis note across related notes for a topic or list of topics."""
    _setup_logging("synthesize", verbose=ctx.obj.get("verbose", False))
    # Mutual exclusion check
    if topic and topics_file:
        raise click.UsageError("--topic and --topics-file are mutually exclusive.")
    if not topic and not topics_file:
        raise click.UsageError("Provide either --topic TOPIC or --topics-file PATH.")
    from lit_monitor.obsidian_tools.synthesize import synthesize
    try:
        config = _make_config()
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
        llm = _make_llm(config, mode="brain_build")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    # G9: resolve rag_mode — CLI flag overrides config default.
    _effective_rag = rag_mode or getattr(
        getattr(config, "retrieval", None), "default_mode", "vector"
    ) or "vector"
    # G9: open graph_db once for the whole synthesize batch (None when [graph] extra absent).
    from lit_monitor.graph import safe_graph_db as _safe_graph_db
    _graph_db = _safe_graph_db() if _effective_rag in ("graph", "hybrid") else None
    # W4: explicit --rag-mode graph|hybrid without the [graph] extra is a hard
    # error (mirrors obsidian_relink). Implicit config-default fallthrough is
    # downgraded to vector with an INFO log.
    if rag_mode in ("graph", "hybrid") and _graph_db is None:
        raise click.UsageError(
            f"--rag-mode {rag_mode} requires the [graph] extra. "
            "Install with: uv sync --extra graph"
        )
    if (
        rag_mode is None
        and _effective_rag in ("graph", "hybrid")
        and _graph_db is None
    ):
        logger.info(
            "Config default rag-mode is %r but [graph] extra is not installed; "
            "falling back to vector for this run.", _effective_rag,
        )
        _effective_rag = "vector"
    try:
        # Build topic list
        if topics_file:
            import yaml
            # L5: cap topics-file size at 1 MB to avoid loading pathological
            # YAML into memory and to surface obvious user error early.
            if Path(topics_file).stat().st_size > 1_048_576:
                click.echo("Topics file >1 MB cap; aborting", err=True)
                sys.exit(1)
            with open(topics_file, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            topics_list: list[str] = raw.get("topics", []) if isinstance(raw, dict) else []
            if not topics_list:
                click.echo("No 'topics:' entries found in the file.", err=True)
                sys.exit(1)
        else:
            topics_list = [topic]

        ok = failed = 0
        for t in topics_list:
            click.echo(f"Synthesizing: '{t}' (top_k={top_k}, rag-mode: {_effective_rag})…")
            try:
                note_path = synthesize(
                    topic=t,
                    config=config,
                    state_db=state_db,
                    embeddings_db=embeddings_db,
                    llm=llm,
                    top_k=top_k,
                    rag_mode=_effective_rag,
                    graph_db=_graph_db,
                )
                if note_path:
                    click.echo(f"  → {note_path}")
                    ok += 1
                else:
                    click.echo("  No relevant notes found.")
                    failed += 1
            except Exception as exc:
                click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
                failed += 1

        if len(topics_list) > 1:
            click.echo(f"\nSynthesis complete: {ok} written, {failed} skipped/failed.")
    finally:
        # G9: release graph_db connection when it was opened for this command.
        if _graph_db is not None:
            try:
                _graph_db.close()
            except Exception:  # pragma: no cover
                pass
@obsidian.command("rechunk-all")
@click.option("--doi", default=None, help="Rechunk a single paper by DOI (default: all papers).")
@click.option("--all", "rechunk_all", is_flag=True, default=False,
              help="Rechunk every paper with a stored markdown fulltext (runs without --doi).")
@click.pass_context
def obsidian_rechunk_all(ctx: click.Context, doi: str | None, rechunk_all: bool) -> None:
    """Rebuild chunk-level ChromaDB index from stored markdown attachments.

    Useful after upgrading the chunker or re-attaching markdown files.
    Reads fulltext from the Zotero markdown attachments, re-splits into
    chunks, and re-indexes in lit_monitor_chunks_v1.
    """
    _setup_logging("rechunk_all", verbose=ctx.obj.get("verbose", False))
    if not doi and not rechunk_all:
        click.echo(
            "Specify --doi DOI to rechunk a single paper, "
            "or --all to rechunk everything.",
            err=True,
        )
        sys.exit(1)
    try:
        config = _make_config()
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        zotero_client = _make_zotero_client(config, secrets)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    from lit_monitor.core.chunker import chunk_markdown

    def _rechunk_doi(d: str) -> bool:
        record = state_db.get_paper(d)
        if not record:
            logging.getLogger(__name__).warning("No DB record for doi: %s", d)
            return False
        zkey = record.get("zotero_key")
        if not zkey:
            logging.getLogger(__name__).warning("No zotero_key for doi: %s", d)
            return False
        fulltext = zotero_client.get_markdown_attachment(zkey)
        if fulltext is None:
            logging.getLogger(__name__).debug("No markdown attachment for %s — skipping", d)
            return False
        from lit_monitor.core.markdown_processor import strip_end_matter
        chunks = chunk_markdown(strip_end_matter(fulltext), d)
        embeddings_db.add_chunks(d, chunks)
        return True

    if doi:
        click.echo(f"Rechunking {doi}…")
        ok = _rechunk_doi(doi)
        click.echo("Done." if ok else f"No markdown attachment found for {doi}.")
        return

    # --all: rechunk every paper/review in the DB
    all_records = (
        state_db.get_all_by_source_type("paper")
        + state_db.get_all_by_source_type("review")
    )
    click.echo(f"Rechunking {len(all_records)} papers/reviews…")
    ok_count = skip_count = fail_count = 0
    for record in all_records:
        d = record.get("doi", "")
        if not d:
            continue
        try:
            if _rechunk_doi(d):
                ok_count += 1
            else:
                skip_count += 1
        except Exception as exc:
            logging.getLogger(__name__).error("Rechunk failed for %s: %s", d, exc)
            fail_count += 1
    click.echo(
        f"Rechunk complete: {ok_count} indexed, {skip_count} skipped "
        f"(no attachment), {fail_count} failed."
    )
@obsidian.command("re-extract")
@click.option("--doi", required=True, help="DOI of the record to re-extract.")
@click.option(
    "--scope",
    default="doi",
    type=click.Choice(["doi", "failed"]),
    help=(
        "doi: re-extract one DOI; "
        "failed: re-extract all that previously errored on the requested phase."
    ),
)
@click.option(
    "--phase",
    "phases",
    multiple=True,
    type=click.Choice(["simple", "complex"]),
    help="Which extraction phases to re-run. Omit for both.",
)
@click.option(
    "--field",
    multiple=True,
    help=(
        "Specific fields to re-extract via a focused prompt (E3). "
        "Cannot be combined with --phase."
    ),
)
@click.option("--no-rerender", is_flag=True, default=False,
              help="Skip rerendering the Obsidian note after re-extraction.")
@click.option(
    "--rag-mode",
    type=click.Choice(["vector", "graph"]),
    default="vector",
    show_default=True,
    help=(
        "Retrieval mode for re-extraction context. "
        "graph: inject corpus-aware context (related entities, adjacent papers) "
        "into the prompt via KuzuDB — requires the [graph] extra. "
        "Falls back to vector on failure."
    ),
)
@click.pass_context
def obsidian_re_extract(
    ctx: click.Context,
    doi: str,
    scope: str,
    phases: tuple[str, ...],
    field: tuple[str, ...],
    no_rerender: bool,
    rag_mode: str,
) -> None:
    """Re-run LLM extraction for a specific DOI or set of records."""
    _setup_logging("re_extract", verbose=ctx.obj.get("verbose", False))
    from lit_monitor.obsidian_tools.re_extract import (
        re_extract,
        re_extract_all_failed_phase,
    )
    try:
        config = _make_config()
        state_db = _make_state_db(config)
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        llm = _make_llm(config, mode="brain_build")
        zotero_client = _make_zotero_client(config, secrets)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    phases_arg = list(phases) if phases else None
    fields_arg = list(field) if field else None
    rerender_flag = not no_rerender

    if scope == "failed":
        click.echo("Re-extracting all previously failed records…")
        for phase in (phases_arg or ["simple", "complex"]):
            stats = re_extract_all_failed_phase(
                phase=phase, config=config, state_db=state_db,
                llm=llm, zotero_client=zotero_client,
            )
            click.echo(
                f"  Phase {phase}: {stats.get('re_extracted', 0)} ok, "
                f"{stats.get('failed', 0)} failed."
            )
    else:
        click.echo(f"Re-extracting {doi}…")
        result = re_extract(
            doi=doi,
            config=config,
            state_db=state_db,
            llm=llm,
            phases=phases_arg,
            fields=fields_arg,
            rerender=rerender_flag,
            zotero_client=zotero_client,
            rag_mode=rag_mode,
        )
        click.echo(f"Done: {result}")


@obsidian.command("build-citation-graph")
@click.option(
    "--doi",
    required=False,
    default=None,
    help="DOI of a single paper (pass-4 data required).",
)
@click.option(
    "--scope",
    default="single",
    type=click.Choice(["single", "all"]),
    show_default=True,
    help="'single' resolves one paper; 'all' resolves every paper with pass-4 data.",
)
@click.option(
    "--max-retries",
    default=3,
    show_default=True,
    help="Maximum S2 retry attempts on rate-limit responses.",
)
@click.option(
    "--no-graph",
    is_flag=True,
    default=False,
    help="Skip mirroring resolved citations into the KuzuDB graph (G5).",
)
@click.pass_context
def obsidian_build_citation_graph(
    ctx: click.Context,
    doi: str | None,
    scope: str,
    max_retries: int,
    no_graph: bool,
) -> None:
    """Resolve key_citations to DOIs via S2 and write citation_edges rows.

    Requires complex-phase extraction (key_citations) to have run for the
    target paper(s). Run ``lit-monitor obsidian re-extract --doi <doi> --phase complex`` first.
    """
    from lit_monitor.core.config import Config
    from lit_monitor.core.state_db import StateDB

    _setup_logging("build_citation_graph", verbose=ctx.obj.get("verbose", False))

    # M3: hydrate S2_API_KEY from config.toml before importing search modules,
    # which capture the env var at import time into _DEFAULT_S2_API_KEY.
    _maybe_set_s2_key(_load_secrets())
    from lit_monitor.search.citation_graph import build_citation_graph

    config = Config()
    state_db = StateDB(config.state_db.path)

    api_key = os.environ.get("S2_API_KEY")

    if scope == "single":
        if not doi:
            raise click.UsageError("--doi is required when --scope single.")
        result = build_citation_graph(
            doi.strip(), state_db, api_key=api_key, max_retries=max_retries,
        )
        click.echo(
            f"Done: {result.n_resolved} resolved, {result.n_unresolved} unresolved "
            f"({result.s2_references_count} S2 references)"
        )
    else:
        # scope == "all" — find all papers that have key_citations in extraction_json
        papers = state_db.get_all_by_source_type("paper")
        papers += state_db.get_all_by_source_type("review")
        total = 0
        resolved_total = 0
        unresolved_total = 0
        failed = 0
        for paper in papers:
            paper_doi = paper.get("doi", "")
            if not paper_doi:
                continue
            raw = paper.get("extraction_json") or "{}"
            try:
                extraction = json.loads(raw)
            except Exception:
                continue
            if not extraction.get("key_citations"):
                continue
            try:
                r = build_citation_graph(
                    paper_doi, state_db, api_key=api_key, max_retries=max_retries,
                )
                total += 1
                resolved_total += r.n_resolved
                unresolved_total += r.n_unresolved
            except Exception as exc:
                logger.error("Citation graph failed for %s: %s", paper_doi, exc)
                failed += 1
        click.echo(
            f"Done: {total} papers, {resolved_total} resolved edges, "
            f"{unresolved_total} unresolved, {failed} failed"
        )

    # G5: mirror resolved citation_edges into Kuzu after E1 work, unless opted out.
    if not no_graph:
        from lit_monitor.graph.import_citations import mirror_citations, safe_graph_db
        graph_db = safe_graph_db()
        if graph_db is not None:
            try:
                added = mirror_citations(graph_db, state_db)
                if added:
                    click.echo(f"  Mirrored {added} CITES edges into Kuzu.")
            finally:
                graph_db.close()


@obsidian.command("rebuild-citations")
@click.option(
    "--doi",
    required=False,
    default=None,
    help="DOI of the target paper (required when --scope doi).",
)
@click.option(
    "--scope",
    default="doi",
    type=click.Choice(["doi", "all", "failed"]),
    show_default=True,
    help=(
        "doi: re-extract complex phase + resolve + relink one paper; "
        "all: resolve + relink every paper with key_citations; "
        "failed: resolve + relink papers that have key_citations but no resolved edges."
    ),
)
@click.option(
    "--max-retries",
    default=3,
    show_default=True,
    help="Maximum S2 retry attempts on rate-limit responses.",
)
@click.option(
    "--no-rerender",
    is_flag=True,
    default=False,
    help="Skip relinking the Obsidian note after resolution.",
)
@click.option(
    "--no-graph",
    is_flag=True,
    default=False,
    help="Skip mirroring resolved citations into the KuzuDB graph (G5).",
)
@click.pass_context
def obsidian_rebuild_citations(
    ctx: click.Context,
    doi: str | None,
    scope: str,
    max_retries: int,
    no_rerender: bool,
    no_graph: bool,
) -> None:
    """Re-extract complex phase, resolve citations via S2, and relink notes.

    For --scope doi, runs a full cycle: complex-phase LLM re-extraction to refresh
    key_citations, then S2 resolution, then Obsidian note relink.

    For --scope all|failed, skips re-extraction and only resolves + relinks.
    Use 'failed' to catch papers where S2 resolution produced no edges.
    """
    _setup_logging("rebuild_citations", verbose=ctx.obj.get("verbose", False))
    # M3: hydrate S2_API_KEY from config.toml before importing search modules,
    # which capture the env var at import time into _DEFAULT_S2_API_KEY.
    _maybe_set_s2_key(_load_secrets())
    from lit_monitor.obsidian_tools.relink import relink_note
    from lit_monitor.search.citation_graph import build_citation_graph

    try:
        config = _make_config()
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        zotero_client = _make_zotero_client(config, secrets)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    api_key = os.environ.get("S2_API_KEY")

    # A3: honour the config default_mode for the Pass-1 relink here, mirroring
    # obsidian_relink's G9 fix. When the effective rag_mode is graph/hybrid we
    # open a GraphDB and wire it into relink_note; default (vector) leaves
    # _graph_db=None so behaviour is unchanged. There is no --rag-mode flag on
    # rebuild-citations, so the mode comes purely from config (no explicit
    # user request → implicit fallback to vector when the [graph] extra is
    # missing, logged at INFO inside safe_graph_db).
    from lit_monitor.graph import safe_graph_db as _safe_graph_db
    _effective_rag = getattr(
        getattr(config, "retrieval", None), "default_mode", "vector"
    ) or "vector"
    _graph_db = _safe_graph_db() if _effective_rag in ("graph", "hybrid") else None
    if _effective_rag in ("graph", "hybrid") and _graph_db is None:
        # Config-default fallthrough only (no explicit flag here): degrade to
        # vector for this run rather than erroring.
        logger.info(
            "Config default rag-mode is %r but [graph] extra is not installed; "
            "relink falls back to vector for this run.", _effective_rag,
        )
        _effective_rag = "vector"

    def _resolve_and_relink(paper_doi: str) -> Any:
        """Run S2 resolution and optional note relink for one paper."""
        result = build_citation_graph(
            paper_doi, state_db, api_key=api_key, max_retries=max_retries,
        )
        if not no_rerender:
            record = state_db.get_paper(paper_doi)
            note_path = record.get("note_path") if record else None
            if note_path:
                # A3: pass the effective rag_mode + graph_db so a config
                # default_mode of graph/hybrid drives the Pass-1 relink here
                # (cf. obsidian_relink's G9 fix). Vector mode keeps graph_db
                # None, preserving the prior vector-only behaviour.
                relink_note(
                    note_path, embeddings_db, state_db, config=config,
                    rag_mode=_effective_rag, graph_db=_graph_db,
                )
        return result

    try:
        _rebuild_citations_body(
            scope=scope,
            doi=doi,
            config=config,
            state_db=state_db,
            zotero_client=zotero_client,
            max_retries=max_retries,
            no_graph=no_graph,
            resolve_and_relink=_resolve_and_relink,
        )
    finally:
        # A3: release the relink graph_db connection when it was opened.
        if _graph_db is not None:
            try:
                _graph_db.close()
            except Exception:  # pragma: no cover
                pass


def _rebuild_citations_body(
    *,
    scope: str,
    doi: str | None,
    config: Any,
    state_db: Any,
    zotero_client: Any,
    max_retries: int,
    no_graph: bool,
    resolve_and_relink: Any,
) -> None:
    """Body of obsidian_rebuild_citations, extracted so the graph_db opened for
    the Pass-1 relink (A3) can be reliably closed in a finally in the caller."""
    if scope == "doi":
        if not doi:
            raise click.UsageError("--doi is required when --scope doi.")
        from lit_monitor.obsidian_tools.re_extract import re_extract
        llm = _make_llm(config, mode="brain_build")
        click.echo(f"Re-extracting complex phase for {doi}…")
        re_extract(
            doi=doi.strip(),
            config=config,
            state_db=state_db,
            llm=llm,
            phases=["complex"],
            rerender=False,  # relink happens below via resolve_and_relink
            zotero_client=zotero_client,
        )
        click.echo("Resolving citations via S2…")
        r = resolve_and_relink(doi.strip())
        click.echo(
            f"Done: {r.n_resolved} resolved, {r.n_unresolved} unresolved "
            f"({r.s2_references_count} S2 references)"
        )
    else:
        papers = (
            state_db.get_all_by_source_type("paper")
            + state_db.get_all_by_source_type("review")
        )
        total = resolved_total = unresolved_total = failed = 0
        for paper in papers:
            paper_doi = paper.get("doi", "")
            if not paper_doi:
                continue
            try:
                extraction = json.loads(paper.get("extraction_json") or "{}")
            except Exception:
                continue
            if not extraction.get("key_citations"):
                continue
            if scope == "failed" and state_db.get_citation_edges(paper_doi):
                continue  # already has resolved edges — skip
            try:
                r = resolve_and_relink(paper_doi)
                total += 1
                resolved_total += r.n_resolved
                unresolved_total += r.n_unresolved
            except Exception as exc:
                logger.error("rebuild-citations failed for %s: %s", paper_doi, exc)
                failed += 1
        click.echo(
            f"Done: {total} papers, {resolved_total} resolved edges, "
            f"{unresolved_total} unresolved, {failed} failed."
        )

    # G5: mirror resolved citation_edges into Kuzu after E1 work, unless opted out.
    if not no_graph:
        from lit_monitor.graph.import_citations import mirror_citations, safe_graph_db
        graph_db = safe_graph_db()
        if graph_db is not None:
            try:
                added = mirror_citations(graph_db, state_db)
                if added:
                    click.echo(f"  Mirrored {added} CITES edges into Kuzu.")
            finally:
                graph_db.close()


# ---------------------------------------------------------------------------
# P10: obsidian sync — bulk note renderer for deferred write mode
# ---------------------------------------------------------------------------

@obsidian.command("sync")
@click.option(
    "--all",
    "all_flag",
    is_flag=True,
    default=False,
    help="Sync all papers where embeddings_indexed=1 and notes_synced=0.",
)
@click.option(
    "--doi",
    default=None,
    help="Sync a specific DOI (re-renders regardless of current notes_synced flag).",
)
@click.option(
    "--since",
    default=None,
    help=(
        "ISO date (YYYY-MM-DD or datetime) — sync papers updated on or after "
        "this date that have notes_synced=0."
    ),
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Maximum number of papers to process in this run.",
)
@click.pass_context
def obsidian_sync(
    ctx: click.Context,
    all_flag: bool,
    doi: str | None,
    since: str | None,
    limit: int | None,
) -> None:
    """Bulk-sync pending Obsidian notes from state.db to the vault (P10).

    Used when discovery.notes.auto_write_per_paper=false — notes accumulate
    with notes_synced=0 and this command renders them in a controlled batch.
    Re-running on already-synced papers is a no-op (they are filtered out
    unless --doi is used to force a specific paper).

    Exits with code 1 if any note failed to render.
    """
    if not all_flag and not doi and not since:
        click.echo(
            "Error: specify at least one of --all, --doi, or --since to scope the sync.",
            err=True,
        )
        raise click.exceptions.Exit(2)

    _setup_logging("obsidian_sync", verbose=ctx.obj.get("verbose", False))
    # Guard config/state-db load with the canonical friendly-error pattern used
    # by sibling commands (e.g. db_cleanup_stale): a malformed config should
    # exit cleanly with a message, not dump a traceback.
    try:
        config = _make_config()
        state_db = _make_state_db(config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    from lit_monitor.obsidian_tools.sync import sync_notes

    result = sync_notes(state_db, config, doi=doi, since=since, limit=limit)
    click.echo(
        f"processed={result['processed']} "
        f"succeeded={result['succeeded']} "
        f"failed={result['failed']}"
    )
    if result["failed"] > 0:
        raise click.exceptions.Exit(1)


# ---------------------------------------------------------------------------
# db — database maintenance commands
# ---------------------------------------------------------------------------

@main.group("db")
def db_group() -> None:
    """Database maintenance commands."""


@db_group.command("cleanup-stale")
@click.pass_context
def db_cleanup_stale(ctx: click.Context) -> None:
    """Remove stale book/chapter rows left from the pre-R-10 textbook-build era.

    These rows were written by the old textbook-build pipeline (removed in R-10,
    2026-05-14) and are no longer processed by any active pipeline.  This command
    is idempotent: re-running it after a clean database is a no-op.
    """
    _setup_logging("db_cleanup_stale", verbose=ctx.obj.get("verbose", False))
    try:
        config = _make_config()
        state_db = _make_state_db(config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    n = state_db.cleanup_stale_item_types()
    if n:
        click.echo(f"Removed {n} stale row(s) (book / chapter / textbook_chapter).")
    else:
        click.echo("Nothing to clean up — no stale rows found.")


# ---------------------------------------------------------------------------
# reset — destructive wipe of pipeline state and/or generated vault notes
# ---------------------------------------------------------------------------
@main.group("reset")
def reset_group() -> None:
    """Destructive: wipe pipeline state and/or generated Obsidian notes.

    Each subcommand requires the user to type the exact subcommand phrase
    (case-sensitive) before any deletion happens. Credentials in
    ``~/.config/lit-monitor/config.toml`` are NEVER touched — delete those
    by hand if you really want them gone.
    """


def _render_targets(targets: list[ResetTarget], header: str) -> None:
    """Print the ``About to PERMANENTLY DELETE`` block for a reset target list."""
    click.echo()
    click.echo(click.style(header, bold=True, fg="red"))
    for tgt in targets:
        if not tgt.exists:
            click.echo(f"  - {tgt.label}: {tgt.path}  (not present)")
            continue
        if tgt.file_count > 0:
            size_str = _format_size_bytes(tgt.size_bytes)
            click.echo(
                f"  - {tgt.label}: {tgt.path}  "
                f"({tgt.file_count} files, {size_str})"
            )
        else:
            size_str = _format_size_bytes(tgt.size_bytes)
            click.echo(f"  - {tgt.label}: {tgt.path}  ({size_str})")


def _render_preserved(items: list[str]) -> None:
    """Print the NOT-touched callout."""
    click.echo()
    click.echo(click.style("NOT touched:", bold=True, fg="green"))
    for item in items:
        click.echo(f"  - {item}")


def _confirm_phrase(expected: str) -> bool:
    """Prompt for the confirmation phrase; return True iff it matches exactly.

    Case-sensitive, no whitespace stripping beyond a trailing newline. We use
    ``input()`` rather than ``click.prompt`` because click.prompt strips the
    trailing newline but normalises in ways we don't want for an
    exact-match confirmation gate.
    """
    click.echo()
    try:
        typed = input(f"Type '{expected}' (exactly) to confirm: ")
    except EOFError:
        click.echo("(no TTY for confirmation — aborting)", err=True)
        return False
    return typed == expected


def _render_results(results: list[ResetResult]) -> None:
    """Print per-target ``✓ deleted`` / ``- skipped`` lines."""
    for r in results:
        if r.deleted:
            click.echo(
                click.style(f"  ✓ deleted {r.label}: {r.path}", fg="green")
            )
        else:
            reason = r.skipped_reason or "skipped"
            click.echo(f"  - skipped {r.label} ({reason}): {r.path}")


_STATE_PRESERVED = [
    "credentials (~/.config/lit-monitor/config.toml)",
    "dev sandbox (~/.config/lit-monitor/chroma_dev, state_dev.db)",
    "run logs (~/.config/lit-monitor/logs/)",
    "tracked configs (paths.yaml, topics.yaml, concepts.yaml, …)",
    "any *.example.yaml files",
]

_VAULT_PRESERVED = [
    "credentials (~/.config/lit-monitor/config.toml)",
    "books_folder (user-managed; book/bookSection items skip the pipeline)",
    "Literature/_Dev/ (dev sandbox)",
    "vault root theme pages",
    "every vault subdirectory not listed above",
]


def _run_state_reset(ctx: click.Context) -> None:
    """Shared body for ``reset state`` and the state half of ``reset all``.

    Handles its own exit semantics: returns on either a successful wipe or a
    user abort, and calls ``sys.exit(1)`` on config-load failure. The return
    value is intentionally ``None`` — callers (including ``reset all``) do
    not branch on success; aborting one half does not abort the other.

    Confirms via the typed phrase ``reset state`` before deleting state
    targets — the same phrase is required when this is invoked as the state
    half of ``reset all`` so the prompt remains predictable for muscle memory.
    """
    try:
        config = _make_config()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        click.echo(f"Error loading config: {exc}", err=True)
        click.echo(
            "Run 'lit-monitor first-run' or 'lit-monitor diagnose' first.",
            err=True,
        )
        sys.exit(1)
    targets = state_targets(config)
    _render_targets(targets, "About to PERMANENTLY DELETE (pipeline state):")
    _render_preserved(_STATE_PRESERVED)
    if not _confirm_phrase("reset state"):
        click.echo("Aborted — nothing deleted.")
        return
    click.echo()
    results = perform_state_reset(targets)
    _render_results(results)
    click.echo()
    click.echo(
        "Next: run 'lit-monitor brain-build' to rebuild from your Zotero "
        "collection."
    )


def _run_vault_reset(ctx: click.Context) -> None:
    """Shared body for ``reset vault`` and the vault half of ``reset all``.

    Handles its own exit semantics: returns on either a successful wipe or a
    user abort, and calls ``sys.exit(1)`` on config-load failure. The return
    value is intentionally ``None`` — callers do not branch on success.

    Confirms via the typed phrase ``reset vault`` before deleting markdown
    files in the Papers, Digests, and Synthesis folders.
    """
    try:
        config = _make_config()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        click.echo(f"Error loading config: {exc}", err=True)
        click.echo(
            "Run 'lit-monitor first-run' or 'lit-monitor diagnose' first.",
            err=True,
        )
        sys.exit(1)
    targets = vault_targets(config)
    click.echo()
    click.echo(
        click.style(
            "WARNING: user edits inside generated notes (persist-zone blocks) "
            "will be destroyed.",
            bold=True,
            fg="yellow",
        )
    )
    _render_targets(targets, "About to PERMANENTLY DELETE (vault notes):")
    _render_preserved(_VAULT_PRESERVED)
    if not _confirm_phrase("reset vault"):
        click.echo("Aborted — nothing deleted.")
        return
    click.echo()
    # Audit-2 B2: pass the resolved vault root so perform_vault_reset can
    # refuse to wipe *.md outside it (corrupted *_folder config guard).
    vault_root = Path(config.obsidian.vault_path).expanduser()
    results = perform_vault_reset(targets, vault_root=vault_root)
    _render_results(results)
    click.echo()
    click.echo(
        "Next: run 'lit-monitor obsidian rerender --all' to regenerate notes "
        "from the state DB."
    )


@reset_group.command("state")
@click.pass_context
def reset_state(ctx: click.Context) -> None:
    """Wipe durable pipeline state (state.db + chroma + auto-generated drafts).

    Requires typing ``reset state`` exactly (case-sensitive) to proceed.
    """
    _setup_logging("reset_state", verbose=ctx.obj.get("verbose", False))
    _run_state_reset(ctx)


@reset_group.command("vault")
@click.pass_context
def reset_vault(ctx: click.Context) -> None:
    """Wipe Obsidian notes generated by lit-monitor (Papers/Reviews/Digests/Synthesis).

    Requires typing ``reset vault`` exactly (case-sensitive) to proceed.
    Dev sandbox folder, vault root pages, and other subdirectories are preserved.
    """
    _setup_logging("reset_vault", verbose=ctx.obj.get("verbose", False))
    _run_vault_reset(ctx)


@reset_group.command("all")
@click.pass_context
def reset_all(ctx: click.Context) -> None:
    """Run ``reset state`` and ``reset vault`` in sequence.

    Each half requires its own typed confirmation phrase. Aborting one half
    does not abort the other — declining ``reset state`` still lets you
    proceed with ``reset vault``.
    """
    _setup_logging("reset_all", verbose=ctx.obj.get("verbose", False))
    _run_state_reset(ctx)
    _run_vault_reset(ctx)


def _run_vectors_reset(ctx: click.Context) -> None:
    """Shared body for ``reset vectors``.

    Wipes the ChromaDB persist directory, then resets the
    ``embeddings_indexed`` flag on every paper row so the next
    ``embeddings rebuild`` re-embeds from scratch.
    """
    try:
        config = _make_config()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        click.echo(f"Error loading config: {exc}", err=True)
        click.echo(
            "Run 'lit-monitor first-run' or 'lit-monitor diagnose' first.",
            err=True,
        )
        sys.exit(1)
    targets = vectors_targets(config)
    _render_targets(targets, "About to PERMANENTLY DELETE (vector store):")
    _render_preserved([
        "state DB — your extractions (vectors rebuild from it)",
        "knowledge graph",
        "vault notes",
        "credentials (~/.config/lit-monitor/config.toml)",
    ])
    if not _confirm_phrase("reset vectors"):
        click.echo("Aborted — nothing deleted.")
        return
    click.echo()
    results = perform_state_reset(targets)
    state_db = _make_state_db(config)
    state_db.reset_embeddings_indexed()
    _render_results(results)
    click.echo()
    click.echo("Next: run 'lit-monitor embeddings rebuild --confirm' to re-embed.")


def _run_graph_reset(ctx: click.Context) -> None:
    """Shared body for ``reset graph``.

    Wipes the KuzuDB knowledge graph, then resets the graph-stamp columns on
    every paper row so the next ``graph backfill`` re-indexes from scratch.
    """
    try:
        config = _make_config()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        click.echo(f"Error loading config: {exc}", err=True)
        click.echo(
            "Run 'lit-monitor first-run' or 'lit-monitor diagnose' first.",
            err=True,
        )
        sys.exit(1)
    targets = graph_targets(config)
    _render_targets(targets, "About to PERMANENTLY DELETE (knowledge graph):")
    _render_preserved([
        "state DB — your extractions (graph rebuilds from it)",
        "vector store",
        "vault notes",
        "credentials (~/.config/lit-monitor/config.toml)",
    ])
    if not _confirm_phrase("reset graph"):
        click.echo("Aborted — nothing deleted.")
        return
    click.echo()
    results = perform_state_reset(targets)
    state_db = _make_state_db(config)
    state_db.reset_graph_stamps()
    _render_results(results)
    click.echo()
    click.echo("Next: run 'lit-monitor graph backfill --all' to rebuild the graph.")


@reset_group.command("vectors")
@click.pass_context
def reset_vectors(ctx: click.Context) -> None:
    """Wipe the ChromaDB vector store (rebuild with ``embeddings rebuild``)."""
    _setup_logging("reset_vectors", verbose=ctx.obj.get("verbose", False))
    _run_vectors_reset(ctx)


@reset_group.command("graph")
@click.pass_context
def reset_graph(ctx: click.Context) -> None:
    """Wipe the KuzuDB knowledge graph (rebuild with ``graph backfill --all``)."""
    _setup_logging("reset_graph", verbose=ctx.obj.get("verbose", False))
    _run_graph_reset(ctx)


# ---------------------------------------------------------------------------
# graph — knowledge graph operator commands (G10)
# ---------------------------------------------------------------------------
@main.group("graph")
def graph_cmd() -> None:
    """Knowledge graph operator commands."""


@graph_cmd.command("backfill")
@click.option(
    "--all", "all_papers",
    is_flag=True, default=False,
    help="Process every paper where graph_indexed=0.",
)
@click.option("--doi", default=None, help="Process this single DOI.")
@click.option(
    "--since", default=None,
    help="Only process papers updated since YYYY-MM-DD.",
)
@click.option(
    "--limit", default=None, type=int,
    help="Cap the number of papers processed in this run.",
)
@click.option(
    "--ner", is_flag=True, default=False,
    help=(
        "Run NER-augmented entity pipeline (BioBERT + schema) over existing "
        "papers. Requires the [nlp] extra: uv sync --extra nlp"
    ),
)
@click.option(
    "--ner-with-llm", "ner_with_llm", is_flag=True, default=False,
    help=(
        "Same as --ner plus cloud-Ollama long-tail NER. "
        "Requires the [nlp] extra and OLLAMA_API_KEY."
    ),
)
@click.option(
    "--relationships", "do_relationships", is_flag=True, default=False,
    help=(
        "Run G4 schema-source relationship extraction over existing papers. "
        "Requires the [graph] extra. Add --relationships-with-llm for "
        "cloud-Ollama LLM relationship extraction."
    ),
)
@click.option(
    "--relationships-with-llm", "relationships_with_llm", is_flag=True, default=False,
    help=(
        "Same as --relationships plus R2 cloud-Ollama LLM relationship extraction. "
        "Requires OLLAMA_API_KEY. Implies --relationships."
    ),
)
def graph_backfill(
    all_papers: bool,
    doi: str | None,
    since: str | None,
    limit: int | None,
    ner: bool,
    ner_with_llm: bool,
    do_relationships: bool,
    relationships_with_llm: bool,
) -> None:
    """Backfill the knowledge graph from existing state.db papers.

    Schema-only mode (default): re-indexes papers with graph_indexed=0.

    NER mode (--ner / --ner-with-llm): runs the N1+N2+N3+N4 BioBERT entity
    pipeline over papers that have not yet been NER-processed
    (ner_processed_at IS NULL).  Requires the [nlp] extra.

    Relationship mode (--relationships / --relationships-with-llm): runs G4
    schema-source relationship extraction over papers that have not yet been
    relationship-processed (rel_processed_at IS NULL).  Add
    --relationships-with-llm to also call R2 cloud-Ollama; requires
    OLLAMA_API_KEY.  Requires the [graph] extra.
    """
    from datetime import datetime as dt

    from rich.progress import Progress

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.graph import safe_graph_db

    # --ner-with-llm implies --ner.
    if ner_with_llm:
        ner = True

    # --relationships-with-llm implies --relationships.
    if relationships_with_llm:
        do_relationships = True

    if not (all_papers or doi or since or ner or ner_with_llm or do_relationships or relationships_with_llm):
        raise click.UsageError(
            "Must specify --all, --doi, --since, --ner, --ner-with-llm, "
            "--relationships, or --relationships-with-llm."
        )

    config = get_config()
    state_db = StateDB(config.state_db.path)
    graph_db = safe_graph_db()
    if graph_db is None:
        raise click.UsageError(
            "[graph] extra not installed. Install with: uv sync --extra graph"
        )

    since_dt = dt.fromisoformat(since) if since else None

    try:
        if ner:
            # N5: NER-augmented backfill path.
            from lit_monitor.graph.backfill import backfill_ner  # noqa: PLC0415

            with Progress() as progress:
                task = progress.add_task("NER backfilling...", total=None)

                def _ner_cb(d: str, done: int, total: int) -> None:  # noqa: ANN001
                    progress.update(task, completed=done, total=total)

                summary = backfill_ner(
                    state_db, graph_db,
                    with_llm=ner_with_llm,
                    limit=limit,
                    since=since_dt,
                    progress_callback=_ner_cb,
                )

            click.echo(
                f"NER backfill: {summary['papers_processed']} papers, "
                f"{summary['edges_added']} edges added, "
                f"{summary['failures']} failures."
            )
            if summary["failures"] > 0:
                # Partial success — non-zero exit so callers can detect it.
                click.get_current_context().exit(1)

        elif do_relationships:
            # R5: relationship backfill path.
            from lit_monitor.graph.backfill import backfill_relationships  # noqa: PLC0415

            with Progress() as progress:
                task = progress.add_task("Relationship backfilling...", total=None)

                def _rel_cb(d: str, done: int, total: int) -> None:  # noqa: ANN001
                    progress.update(task, completed=done, total=total)

                rel_summary = backfill_relationships(
                    state_db, graph_db,
                    with_llm=relationships_with_llm,
                    limit=limit,
                    since=since_dt,
                    progress_callback=_rel_cb,
                )

            click.echo(
                f"Relationship backfill: {rel_summary['papers_processed']} papers, "
                f"{rel_summary['edges_added']} edges added, "
                f"{rel_summary['failures']} failures."
            )
            if rel_summary["failures"] > 0:
                # Partial success — non-zero exit so callers can detect it.
                click.get_current_context().exit(1)

        else:
            # G10: schema-only backfill path (unchanged).
            from lit_monitor.graph.backfill import backfill_papers  # noqa: PLC0415

            filter_doi = doi if not all_papers else None

            with Progress() as progress:
                task = progress.add_task("Backfilling...", total=None)

                def _cb(d: str, done: int, total: int) -> None:  # noqa: ANN001
                    progress.update(task, completed=done, total=total)

                count = backfill_papers(
                    state_db, graph_db,
                    filter_doi=filter_doi,
                    since=since_dt,
                    progress_callback=_cb,
                )
            click.echo(f"Backfilled {count} papers.")
    finally:
        try:
            graph_db.close()
        except Exception:  # noqa: BLE001
            pass


@graph_cmd.command("rebuild")
@click.option(
    "--all", "all_data",
    is_flag=True, default=False,
    help="Drop all graph data and re-backfill from state.db.",
)
@click.option(
    "--aliases-only",
    is_flag=True, default=False,
    help="Re-normalize existing Entity nodes against the current alias YAML.",
)
@click.confirmation_option(prompt="This is destructive — continue?")
def graph_rebuild(all_data: bool, aliases_only: bool) -> None:
    """Rebuild the knowledge graph (destructive)."""
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.graph import safe_graph_db
    from lit_monitor.graph.backfill import rebuild_aliases_only, rebuild_all

    if not (all_data or aliases_only):
        raise click.UsageError("Must specify --all or --aliases-only.")
    if all_data and aliases_only:
        raise click.UsageError("--all and --aliases-only are mutually exclusive.")

    config = get_config()
    state_db = StateDB(config.state_db.path)
    graph_db = safe_graph_db()
    if graph_db is None:
        raise click.UsageError(
            "[graph] extra not installed. Install with: uv sync --extra graph"
        )

    try:
        if all_data:
            count = rebuild_all(state_db, graph_db, confirm=True)
            click.echo(f"Rebuilt graph from scratch; {count} papers re-indexed.")
        else:
            count = rebuild_aliases_only(graph_db)
            click.echo(f"Re-normalized {count} Entity nodes.")
    finally:
        try:
            graph_db.close()
        except Exception:  # noqa: BLE001
            pass


@graph_cmd.command("propose-aliases")
@click.option("--min-ratio", default=80, type=int,
              help="Fuzzy match threshold (0-100). Higher = stricter clusters. Default: 80.")
@click.option("--out", default=None,
              help="Output YAML path. Default: <config-dir>/entity_aliases.suggested.yaml.")
@click.option("--with-llm", is_flag=True, default=False,
              help="N6: validate fuzzy clusters via cloud-Ollama before writing. "
                   "Requires OLLAMA_API_KEY. Falls back to fuzzy-only on any LLM failure.")
def graph_propose_aliases(min_ratio: int, out: str | None, with_llm: bool) -> None:
    """Propose new aliases via fuzzy clustering (optional --with-llm consensus)."""
    from pathlib import Path

    from lit_monitor.core.config import config_dir as _config_dir
    from lit_monitor.graph import safe_graph_db
    from lit_monitor.graph.propose_aliases import propose_aliases, write_proposal_file

    # Default the GENERATED output to the active config dir (write location) so a
    # wheel run doesn't drop it in a CWD-relative ``config/``. An explicit --out
    # is honoured verbatim. config_dir() honours LIT_MONITOR_ROOT / dev ./config.
    if out is None:
        out = str(_config_dir() / "entity_aliases.suggested.yaml")

    graph_db = safe_graph_db()
    if graph_db is None:
        raise click.UsageError(
            "[graph] extra not installed. Install with: uv sync --extra graph"
        )

    try:
        proposals = propose_aliases(graph_db, min_ratio=min_ratio, with_llm=with_llm)
        if not proposals:
            click.echo("No alias proposals (no clusters found at the given threshold).")
            return
        out_path = Path(out)
        write_proposal_file(out_path, proposals)
        total = sum(len(v) for v in proposals.values())
        click.echo(f"Wrote {total} alias proposals across {len(proposals)} types → {out}")
        click.echo("Review and merge by hand into config/entity_aliases.yaml,")
        click.echo("then run `lit-monitor graph rebuild --aliases-only` to apply.")
    finally:
        try:
            graph_db.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# G13 / N8: graph status subcommand
# ---------------------------------------------------------------------------
@graph_cmd.command("status")
@click.option(
    "--by-source", "by_source",
    is_flag=True, default=False,
    help=(
        "Also show MENTIONS counts grouped by source "
        "(schema / biobert / llm_cloud)."
    ),
)
def graph_status(by_source: bool) -> None:
    """Show Kuzu node + edge counts as a rich.Table (G13).

    With --by-source, also prints a second table grouping MENTIONS edges by
    their m.source value ({schema, biobert, llm_cloud}) — N8 Phase 2 observability.
    """
    from rich.console import Console
    from rich.table import Table

    try:
        from lit_monitor.graph import safe_graph_db
    except ImportError:
        click.echo("[graph] extra not installed.")
        click.echo("Install with: uv sync --extra graph")
        return  # exit 0 — not an error

    graph_db = safe_graph_db()
    if graph_db is None:
        click.echo("[graph] extra installed but no DB found.")
        click.echo("Run: lit-monitor graph backfill --all")
        return  # exit 0 — not an error

    try:
        conn = graph_db._conn
        rows: list[tuple[str, int]] = []

        # Node counts
        for node_type in ("Paper", "Entity"):
            try:
                res = conn.execute(f"MATCH (n:{node_type}) RETURN count(n) AS c")
                row = res.get_next() if res.has_next() else None
                rows.append((node_type, int(row[0]) if row else 0))
            except Exception:  # noqa: BLE001
                rows.append((node_type, 0))

        # Edge counts for all schema-defined relationship types
        # R1 added EXTENDS + CONTRADICTS (Phase 3 LLM-only predicates)
        for rel in (
            "MENTIONS", "CITES", "COMPARES_TO", "DEPENDS_ON",
            "PROPOSES", "LIMITED_BY", "INTRODUCES", "RAISES_QUESTION",
            "EXTENDS", "CONTRADICTS",
        ):
            try:
                res = conn.execute(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
                row = res.get_next() if res.has_next() else None
                rows.append((rel, int(row[0]) if row else 0))
            except Exception:  # noqa: BLE001
                rows.append((rel, 0))

        table = Table(title="lit-monitor Graph Status")
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right", style="green")
        for name, count in rows:
            table.add_row(name, str(count))
        Console().print(table)

        # N8: --by-source breakdown of MENTIONS edges
        # R6: also show typed-predicate (Phase 3) counts broken down by prompt_version
        if by_source:
            try:
                res = conn.execute(
                    "MATCH (p:Paper)-[m:MENTIONS]->(e:Entity) "
                    "RETURN m.source AS source, count(m) AS edge_count, "
                    "count(DISTINCT e) AS entity_count "
                    "ORDER BY source"
                )
                by_source_rows: list[tuple[str, int, int]] = []
                while res.has_next():
                    row = res.get_next()
                    by_source_rows.append((str(row[0]), int(row[1]), int(row[2])))
                src_table = Table(title="MENTIONS by source")
                src_table.add_column("Source")
                src_table.add_column("MENTIONS", justify="right")
                src_table.add_column("Distinct entities", justify="right")
                for source, mc, ec in by_source_rows:
                    src_table.add_row(source, str(mc), str(ec))
                Console().print(src_table)
            except Exception:  # noqa: BLE001
                click.echo("(by-source query failed — MENTIONS edges may not carry a source property)")

            # R6: typed-predicate breakdown by prompt_version (schema vs LLM split)
            _TYPED_PREDICATES = [
                "COMPARES_TO", "DEPENDS_ON", "PROPOSES", "LIMITED_BY",
                "INTRODUCES", "RAISES_QUESTION", "EXTENDS", "CONTRADICTS",
            ]
            typed_by_pv: list[tuple[str, str, int]] = []
            for pred in _TYPED_PREDICATES:
                try:
                    res = conn.execute(
                        f"MATCH ()-[r:{pred}]->() "
                        "RETURN r.prompt_version AS pv, count(r) AS n "
                        "ORDER BY pv"
                    )
                    while res.has_next():
                        row = res.get_next()
                        typed_by_pv.append((pred, str(row[0]) if row[0] else "(none)", int(row[1])))
                except Exception:  # noqa: BLE001
                    pass  # table may not exist yet (pre-R1 DB)
            if typed_by_pv:
                pv_table = Table(title="Typed predicates by prompt_version (Phase 3 split)")
                pv_table.add_column("Predicate", style="cyan")
                pv_table.add_column("prompt_version")
                pv_table.add_column("Count", justify="right", style="green")
                for pred, pv, cnt in typed_by_pv:
                    pv_table.add_row(pred, pv, str(cnt))
                Console().print(pv_table)
    finally:
        try:
            graph_db.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# A5: lit-monitor ask — NL → Cypher → execute → summarize
# ---------------------------------------------------------------------------
@main.command("ask")
@click.argument("question_words", nargs=-1, required=True)
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Print stage labels (Generating Cypher, Executing, Summarizing).",
)
@click.option(
    "--cypher-only",
    is_flag=True, default=False,
    help="Print only the generated Cypher; skip execution + summary.",
)
@click.option(
    "--rag-mode",
    type=click.Choice(["auto", "graph"]),
    default="graph",
    help="Backend mode. Only 'graph' is supported for ask.",
)
@click.option(
    "--max-rows",
    type=int, default=20,
    help="Max rows to render in the markdown table.",
)
def ask_command(
    question_words: tuple[str, ...],
    verbose: bool,
    cypher_only: bool,
    rag_mode: str,
    max_rows: int,
) -> None:
    """Ask a natural-language question against the literature graph.

    Translates your question to Cypher, executes it against the KuzuDB
    knowledge graph, and summarizes the results in prose.

    The question is passed as one or more words — no quotes needed:

        lit-monitor ask how many papers are from 2023
    """
    # Lazy imports — keeps CLI startup fast for non-ask commands.
    from lit_monitor.core.config import get_config
    from lit_monitor.graph import safe_graph_db
    from lit_monitor.graph.ask import (
        execute_cypher,
        generate_cypher,
        render_rows,
        summarize_results,
    )
    from lit_monitor.graph.schema_describer import describe_schema

    question = " ".join(question_words).strip()
    if not question:
        click.echo("No question provided.", err=True)
        raise click.exceptions.Exit(1)

    cfg = get_config()
    graph_db = safe_graph_db()
    if graph_db is None:
        click.echo(
            "Graph backend unavailable. Run 'lit-monitor graph build' first.",
            err=True,
        )
        raise click.exceptions.Exit(1)

    # === A1: introspect the live graph schema ===
    if verbose:
        click.echo(">> Describing schema…", err=True)
    schema_text = describe_schema(graph_db)
    if not schema_text or not schema_text.strip():
        click.echo("Failed to introspect graph schema.", err=True)
        raise click.exceptions.Exit(1)

    # === A2: translate natural-language question → Cypher ===
    if verbose:
        click.echo(">> Generating Cypher…", err=True)
    cypher = generate_cypher(question, schema_text, cfg=cfg)
    if cypher is None:
        click.echo(
            "I couldn't translate your question into a query. "
            "Try rephrasing, or use --verbose to see why.",
            err=True,
        )
        raise click.exceptions.Exit(1)

    if verbose or cypher_only:
        click.echo(f"\nCypher:\n  {cypher}\n")

    if cypher_only:
        return

    # === A3: execute the Cypher and render rows as a markdown table ===
    if verbose:
        click.echo(">> Executing Cypher…", err=True)
    rows = execute_cypher(graph_db, cypher)
    if rows is None:
        click.echo(
            f"I generated a query but it failed to execute.\n"
            f"  Cypher: {cypher}\n"
            "Try rephrasing.",
            err=True,
        )
        raise click.exceptions.Exit(1)

    rendered = render_rows(rows, max_rows=max_rows)

    # === A4: prose summarization ===
    if verbose:
        click.echo(">> Summarizing…", err=True)
    prose = summarize_results(question, cypher, rendered, cfg=cfg)

    if prose is None:
        # We still have data — show the table with a note and exit clean.
        click.echo(rendered)
        click.echo("\n(Summarization failed — raw results above.)")
        return

    # Happy path: prose summary followed by the raw table.
    click.echo(prose)
    click.echo()
    click.echo(rendered)


# ---------------------------------------------------------------------------
# B4: mcp group — lit-monitor mcp serve
# ---------------------------------------------------------------------------
# Note: The original Phase 4b brief described an SSE-over-HTTP transport with
# --host / --port flags. We deliberately chose stdio transport instead because:
#   (a) Claude Desktop / Continue / Cursor all expect stdio out of the box.
#   (b) SSE would require uvicorn + a separate /sse endpoint — unnecessary
#       complexity for the current consumer surface.
# If SSE is needed later, add a second subcommand (e.g. `lit-monitor mcp sse`).
# ---------------------------------------------------------------------------

@main.group("mcp")
def mcp_group() -> None:
    """Manage the MCP server (knowledge-graph + vector RAG over Model Context Protocol)."""


@mcp_group.command("serve")
def mcp_serve_command() -> None:
    """Run the lit-monitor MCP server (stdio transport).

    Starts the MCP server using stdio transport so any MCP client
    (Claude Desktop, Continue, Cursor, …) can launch this command as a
    subprocess and communicate over stdin/stdout.

    IMPORTANT: stdout is the MCP wire protocol. Do NOT redirect stdout to a
    file or pipe it into a pager — that breaks the client handshake.  All
    human-readable output goes to stderr.

    Add to your MCP client config:

    \b
        "lit-monitor-graph": {
          "command": "lit-monitor",
          "args": ["mcp", "serve"]
        }
    """
    # Probe for the [mcp] extra before doing anything else.
    # Lazy check — the mcp SDK is only required when this subcommand runs.
    try:
        import mcp  # noqa: F401
    except ImportError:
        click.echo(
            "MCP server requires the [mcp] extra. "
            "Install with: uv sync --extra mcp",
            err=True,
        )
        raise click.exceptions.Exit(1)

    # -----------------------------------------------------------------------
    # Banner — goes to STDERR.
    # With stdio transport stdout carries the MCP wire protocol; printing
    # anything there would corrupt the client handshake.
    # -----------------------------------------------------------------------
    click.echo(">> lit-monitor MCP server starting (stdio transport)", err=True)
    click.echo(
        ">> Register in your MCP client config (Claude Desktop / Continue / Cursor):",
        err=True,
    )
    click.echo('>>   "lit-monitor-graph": {', err=True)
    click.echo('>>     "command": "lit-monitor",', err=True)
    click.echo('>>     "args": ["mcp", "serve"]', err=True)
    click.echo(">>   }", err=True)
    click.echo("", err=True)

    # Lazy import — only triggered when running `mcp serve`, so CLI startup
    # for all other commands is unaffected by the [mcp] extra being present
    # or absent.
    from lit_monitor.mcp.graph_server import main as server_main  # noqa: PLC0415

    try:
        server_main()
    except KeyboardInterrupt:
        click.echo(">> MCP server shutting down (Ctrl-C)", err=True)


# ---------------------------------------------------------------------------
# chunks — chunk-level ChromaDB embedding operator commands (CB1)
# ---------------------------------------------------------------------------
@main.group("chunks")
def chunks_cmd() -> None:
    """Manage chunk-level (passage) ChromaDB embeddings."""


@chunks_cmd.command("backfill")
@click.option(
    "--all", "all_flag",
    is_flag=True, default=False,
    help="Backfill all papers with chunks_indexed=0 AND fully_complete=1.",
)
@click.option("--doi", default=None, help="Backfill only this DOI.")
@click.option(
    "--since", default=None,
    help="Backfill only papers with last_updated >= DATE (ISO format).",
)
@click.option(
    "--limit", type=int, default=None,
    help="Cap the number of papers processed in this run.",
)
def chunks_backfill_cmd(
    all_flag: bool,
    doi: str | None,
    since: str | None,
    limit: int | None,
) -> None:
    """Re-chunk + re-index papers whose chunks have not been indexed (CB1).

    Walks state.db for papers WHERE chunks_indexed=0 AND fully_complete=1,
    reads their stored fulltext_path, runs chunk_markdown, and calls
    EmbeddingsDB.add_chunks.  No LLM calls — pure re-chunking.

    Exit code 1 if any papers failed; 0 if all succeeded.
    """
    if not all_flag and not doi:
        click.echo(
            "Specify --all or --doi to scope the backfill.", err=True
        )
        raise click.exceptions.Exit(2)

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.pipelines.chunks_backfill import backfill_chunks

    config = get_config()
    state_db = StateDB(config.state_db.path)
    embeddings_db = _make_embeddings_db(config)

    summary = backfill_chunks(state_db, embeddings_db, doi=doi, since=since, limit=limit)
    click.echo(
        f"processed={summary['processed']} "
        f"succeeded={summary['succeeded']} "
        f"failed={summary['failed']}"
    )
    if summary["failed"] > 0:
        raise click.exceptions.Exit(1)


# ---------------------------------------------------------------------------
# discovery sub-group — P6 view, P7 export-md
# ---------------------------------------------------------------------------


@main.group("discovery")
def discovery_group() -> None:
    """View and export discovery run results."""


@discovery_group.command("view")
@click.option(
    "--run",
    "run_selector",
    default="latest",
    type=click.Choice(["latest"]),
    show_default=True,
)
@click.option("--run-id", default=None, type=int)
@click.option("--top-k", default=20, show_default=True, type=int)
def discovery_view(run_selector: str, run_id: int | None, top_k: int) -> None:
    """Print discovery run results as a Rich table (P6)."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from lit_monitor.api.queries import get_discovery_run, get_discovery_run_papers
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    console = Console()

    if run_id is None:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT id FROM discovery_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            console.print("[yellow]No discovery runs found in state.db.[/yellow]")
            return
        run_id = row[0]

    run = get_discovery_run(db, run_id)
    if run is None:
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise click.exceptions.Exit(1)

    papers = get_discovery_run_papers(db, run_id, top_k=top_k)
    if not papers:
        console.print(f"[yellow]No papers in run #{run_id}.[/yellow]")
        return

    date_str = (run.get("started_at") or "")[:10] or "unknown"
    table = Table(
        title=f"Discovery Run #{run_id} — {date_str}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", width=4, justify="right")
    table.add_column("Title", min_width=30, max_width=60)
    table.add_column("DOI", max_width=25)
    table.add_column("Score", width=6, justify="right")
    table.add_column("Rationale", max_width=50)

    for i, p in enumerate(papers, 1):
        rationale = p.get("rationale") or ""
        if len(rationale) > 80:
            rationale = rationale[:79] + "…"
        table.add_row(
            str(i),
            p.get("title") or "",
            p.get("doi") or "",
            f"{p.get('score') or 0:.3f}",
            rationale,
        )

    console.print(table)


@discovery_group.command("export-md")
@click.option(
    "--run",
    "run_selector",
    default="latest",
    type=click.Choice(["latest"]),
    show_default=True,
)
@click.option("--run-id", default=None, type=int)
@click.option(
    "--to",
    "output_path",
    default=None,
    type=click.Path(),
    help="Output file path. Default: Discovery_{date}.md in cwd.",
)
def discovery_export_md(
    run_selector: str, run_id: int | None, output_path: str | None
) -> None:
    """Export a discovery run as a Markdown digest file (P7)."""
    from pathlib import Path

    from lit_monitor.api.queries import get_discovery_run, get_discovery_run_papers
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.output.digest_renderer import render_digest

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())

    if run_id is None:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT id FROM discovery_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            click.echo("No discovery runs found.", err=True)
            raise click.exceptions.Exit(1)
        run_id = row[0]

    run = get_discovery_run(db, run_id)
    if run is None:
        click.echo(f"Run {run_id} not found.", err=True)
        raise click.exceptions.Exit(1)

    papers = get_discovery_run_papers(db, run_id, top_k=1000)
    md = render_digest(run, papers)

    date_str = (run.get("started_at") or "")[:10] or "unknown"
    dest = Path(output_path) if output_path else Path(f"Discovery_{date_str}.md")
    dest.write_text(md, encoding="utf-8")
    click.echo(f"Written to {dest}")


# ---------------------------------------------------------------------------
# Bundle F: embeddings management commands
# ---------------------------------------------------------------------------
@main.group("embeddings")
def embeddings_group() -> None:
    """Manage the embedding store — provider, rebuild, and status."""


@embeddings_group.command("status")
def embeddings_status_cmd() -> None:
    """List all embedding collections and their recorded provenance.

    Shows: collection name, provider, model, dimension, creation date,
    and whether it is the currently active collection.
    """
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    rows = db.list_embedding_provenance()

    if not rows:
        click.echo("No embedding collections recorded.")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title="Embedding collections",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Collection", style="bold")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Dim", justify="right")
        table.add_column("Created", justify="center")
        table.add_column("Active", justify="center")
        for r in rows:
            table.add_row(
                r["collection_name"],
                r["provider"],
                r["model"],
                str(r["dim"]),
                (r["created_at"] or "")[:10],
                "✓" if r["is_current"] else "",
            )
        Console().print(table)
    except ImportError:
        # Fallback: plain-text table when rich is not installed.
        header = f"{'Collection':<30} {'Provider':<10} {'Model':<30} {'Dim':>6}  {'Active'}"
        click.echo(header)
        click.echo("-" * len(header))
        for r in rows:
            active = "*" if r["is_current"] else " "
            click.echo(
                f"{r['collection_name']:<30} {r['provider']:<10} "
                f"{r['model']:<30} {r['dim']:>6}  {active}"
            )


@embeddings_group.command("switch")
@click.option("--provider", required=True, type=click.Choice(["ollama", "litellm"]),
              help="Embedding provider to switch to.")
@click.option("--model", required=True,
              help="Model identifier string.")
@click.option("--dim", type=int, default=None,
              help="Embedding dimension (uses provider default if omitted).")
@click.option("--keep-old", is_flag=True, default=False,
              help="Preserve the old collection; build a new one alongside it. "
                   "Recommended for production: safe, reversible.")
@click.option("--confirm", is_flag=True, default=False,
              help="Required when not using --keep-old (destructive: deletes old collection).")
def embeddings_switch_cmd(
    provider: str,
    model: str,
    dim: int | None,
    keep_old: bool,
    confirm: bool,
) -> None:
    """Switch the active embedding provider/model.

    \b
    Safe path (recommended):
        lit-monitor embeddings switch --provider litellm \\
            --model text-embedding-3-large --keep-old

    \b
    Destructive path (requires explicit --confirm):
        lit-monitor embeddings switch --provider ollama \\
            --model nomic-embed-text --confirm
    """
    from lit_monitor.pipelines.embeddings_migration import switch_provider
    switch_provider(
        provider=provider,
        model=model,
        dim=dim,
        keep_old=keep_old,
        confirm=confirm,
    )


@embeddings_group.command("rebuild")
@click.option("--keep-old", is_flag=True, default=False,
              help="Preserve the current collection; build a fresh one alongside.")
@click.option("--confirm", is_flag=True, default=False,
              help="Required when not using --keep-old.")
def embeddings_rebuild_cmd(keep_old: bool, confirm: bool) -> None:
    """Re-embed the active collection using the current config.

    Use this after manually editing the embedding model in extraction.yaml
    to rebuild the ChromaDB collection with the new model.
    """
    from lit_monitor.pipelines.embeddings_migration import rebuild_active
    rebuild_active(keep_old=keep_old, confirm=confirm)


# ---------------------------------------------------------------------------
# Bundle G (v0.9): `lit-monitor domain` — LLM extraction of structured
# focus areas from config/domain_context.yaml.
# ---------------------------------------------------------------------------
@main.group("domain")
def domain_group() -> None:
    """Analyze and review structured domain focus areas (Bundle G)."""


def _load_domain_text() -> str | None:
    """Read domain_focus from config/domain_context.yaml; None when empty."""
    from pathlib import Path

    import yaml

    from lit_monitor.core.path_utils import resolve_path as _resolve_path

    # Route the read through resolve_path so a wheel run resolves the user's
    # config dir / packaged default (dev/editable still resolves ./config);
    # resolve_path raises when nothing matches, which maps to "no domain text".
    try:
        path = _resolve_path(Path("config/domain_context.yaml"))
    except FileNotFoundError:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if isinstance(data, dict):
        text = data.get("domain_focus") or data.get("domain_context") or ""
    elif isinstance(data, str):
        text = data
    else:
        return None
    text = (text or "").strip()
    return text or None


@domain_group.command("analyze")
def domain_analyze_cmd() -> None:
    """Run LLM extraction on config/domain_context.yaml; persist to state.db."""
    from pathlib import Path

    from rich.console import Console

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.domain.extract import analyze_domain

    text = _load_domain_text()
    if text is None:
        click.echo(
            "config/domain_context.yaml not found or empty. Fill in "
            "'domain_focus' before running this command.",
            err=True,
        )
        raise click.exceptions.Exit(1)

    click.echo(">> Analyzing domain focus via LLM (single call)...")
    extraction = analyze_domain(text)
    if extraction is None:
        click.echo(
            "Extraction failed. Run with --verbose for the breadcrumb log.",
            err=True,
        )
        raise click.exceptions.Exit(1)

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    db.save_domain_extraction(extraction)

    # Print rich summary
    console = Console()
    for field, items in extraction.items():
        if items:
            console.print(f"[bold]{field}:[/bold] {', '.join(items)}")
        else:
            console.print(f"[dim]{field}: (none)[/dim]")
    click.echo(">> Saved to state.db::domain_focus_extracted")


@domain_group.command("view")
def domain_view_cmd() -> None:
    """Print the current domain extraction as a Rich table."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    extraction = db.list_domain_extraction()

    any_items = any(items for items in extraction.values())
    console = Console()
    if not any_items:
        console.print(
            "[yellow]No domain extraction recorded.[/yellow] Run "
            "`lit-monitor domain analyze` first."
        )
        return

    table = Table(
        title="Domain focus extraction",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Field")
    table.add_column("Value")
    table.add_column("Confirmed", justify="center")
    for field, items in extraction.items():
        for item in items:
            table.add_row(
                field,
                item["value"],
                "[green]✓[/green]" if item["user_confirmed"] else "",
            )
    console.print(table)


@domain_group.command("clear")
@click.confirmation_option(
    prompt="Wipe all rows from domain_focus_extracted?",
)
def domain_clear_cmd() -> None:
    """Wipe the domain extraction table — useful before re-analysis."""
    from pathlib import Path

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    n = db.clear_domain_extraction()
    click.echo(f"Cleared {n} rows from domain_focus_extracted.")


# ---------------------------------------------------------------------------
# Bundle C: cluster commands
# ---------------------------------------------------------------------------

@main.group("cluster")
def cluster_group() -> None:
    """Manage library theme clustering (Bundle C)."""


@cluster_group.command("recompute")
@click.option(
    "--threshold",
    default=None,
    type=int,
    help="Override min_papers_threshold from config (default: use config value).",
)
@click.pass_context
def cluster_recompute_cmd(ctx: click.Context, threshold: int | None) -> None:
    """Recompute k-means clusters from all library embeddings.

    Silhouette score automatically selects K in [k_min, k_max].
    Below min_papers_threshold the command exits with a message.
    """
    from pathlib import Path

    from lit_monitor.clustering.recompute import recompute_clusters
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    edb = _make_embeddings_db(cfg)

    clustering_cfg = getattr(cfg, "clustering", None)
    if clustering_cfg is None or not getattr(clustering_cfg, "enabled", True):
        click.echo("Clustering is disabled in config (clustering.enabled=false).")
        return

    effective_threshold = threshold or int(getattr(clustering_cfg, "min_papers_threshold", 100))
    n_indexed = edb._collection.count()

    if n_indexed < effective_threshold:
        click.echo(
            f"Library has {n_indexed} indexed papers (threshold={effective_threshold}). "
            "Clustering requires more papers — import more and run brain-build first."
        )
        return

    click.echo(f"Recomputing clusters for {n_indexed} papers…")
    n = recompute_clusters(db, edb, cfg)
    click.echo(f"Done — {n} theme clusters created/updated.")
    click.echo("Run `lit-monitor cluster view` to inspect them.")


@cluster_group.command("view")
@click.pass_context
def cluster_view_cmd(ctx: click.Context) -> None:
    """Display active library clusters in a rich table."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    clusters = db.list_active_clusters()

    if not clusters:
        click.echo(
            "No clusters found. Run `lit-monitor cluster recompute` to build them."
        )
        return

    console = Console()
    table = Table(title="Library Theme Clusters", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", width=6)
    table.add_column("Theme", min_width=20)
    table.add_column("Papers", justify="right", width=8)
    table.add_column("Cohesion", justify="right", width=10)
    table.add_column("Computed", width=20)

    for c in clusters:
        cohesion = c.get("cohesion_score")
        cohesion_str = f"{cohesion:.3f}" if cohesion is not None else "—"
        computed = (c.get("computed_at") or "")[:16]
        table.add_row(
            str(c["id"]),
            c.get("display_name") or "(unnamed)",
            str(c.get("n_papers", 0)),
            cohesion_str,
            computed,
        )

    console.print(table)


@cluster_group.command("assign")
@click.pass_context
def cluster_assign_cmd(ctx: click.Context) -> None:
    """Re-assign all papers to clusters without recomputing centroids.

    Useful after adding new papers to the library — existing centroids are
    kept but assignments are refreshed to include the new papers.
    """
    from pathlib import Path

    import numpy as np

    from lit_monitor.clustering.assign import assign_papers_to_clusters
    from lit_monitor.clustering.kmeans import Cluster
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    edb = _make_embeddings_db(cfg)

    active = db.list_active_clusters()
    if not active:
        click.echo("No clusters found. Run `lit-monitor cluster recompute` first.")
        return

    clusters: list[Cluster] = []
    for row in active:
        blob = row.get("centroid_blob")
        if not blob:
            continue
        try:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            clusters.append(Cluster(
                id=row["id"],
                display_name=row.get("display_name"),
                centroid_vec=vec,
                members=[],
                cohesion_score=row.get("cohesion_score") or 0.0,
            ))
        except Exception as exc:
            click.echo(f"Warning: could not parse centroid for cluster {row['id']}: {exc}")

    click.echo(f"Assigning papers to {len(clusters)} existing clusters…")
    n = assign_papers_to_clusters(db, edb, clusters)
    click.echo(f"Done — {n} assignments written.")


@cluster_group.group("write-back")
def cluster_write_back_group() -> None:
    """Write cluster themes back to Zotero as tags or sub-collections."""


@cluster_write_back_group.command("tags")
@click.option(
    "--dry-run/--confirm",
    default=True,
    help="Preview changes (--dry-run, default) or apply them (--confirm).",
)
@click.pass_context
def cluster_writeback_tags_cmd(ctx: click.Context, dry_run: bool) -> None:
    """Add lm/<theme-name> tags to Zotero items for each cluster.

    Default: dry-run mode. Pass --confirm to actually write.
    Strictly additive — never removes existing tags.
    """
    from pathlib import Path

    from lit_monitor.clustering.write_back import push_tags_to_zotero
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    secrets = _load_secrets()
    zot = _make_zotero_client(cfg, secrets)

    if dry_run:
        click.echo("Dry-run mode — no changes will be made. Pass --confirm to write.")

    report = push_tags_to_zotero(db, zot, dry_run=dry_run)

    if dry_run:
        click.echo(
            f"Would add {report['tags_added']} tags across {report['papers_processed']} papers."
        )
    else:
        click.echo(
            f"Added {report['tags_added']} tags across {report['papers_processed']} papers."
        )
        if report.get("papers_skipped"):
            click.echo(f"  (Skipped {report['papers_skipped']} papers — no Zotero key.)")


@cluster_write_back_group.command("collections")
@click.option(
    "--dry-run/--confirm",
    default=True,
    help="Preview changes (--dry-run, default) or apply them (--confirm).",
)
@click.option(
    "--parent",
    default="lit-monitor",
    show_default=True,
    help="Name of the top-level Zotero collection to create sub-collections under.",
)
@click.pass_context
def cluster_writeback_collections_cmd(
    ctx: click.Context, dry_run: bool, parent: str
) -> None:
    """Create Zotero sub-collections under parent for each cluster theme.

    Default: dry-run mode. Pass --confirm to write.
    Strictly additive — existing collections and memberships are preserved.
    """
    from pathlib import Path

    from lit_monitor.clustering.write_back import push_collections_to_zotero
    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    secrets = _load_secrets()
    zot = _make_zotero_client(cfg, secrets)

    if dry_run:
        click.echo("Dry-run mode — no changes will be made. Pass --confirm to write.")

    report = push_collections_to_zotero(db, zot, parent_name=parent, dry_run=dry_run)

    if dry_run:
        click.echo(
            f"Would create {report['collections_to_create']} collections, "
            f"add {report['items_to_add']} items."
        )
    else:
        click.echo(
            f"Created {report['collections_to_create']} collections, "
            f"added {report['items_to_add']} items."
        )


# ---------------------------------------------------------------------------
# Bundle E (v0.9): trending concept suggestion commands
# ---------------------------------------------------------------------------

@main.group("trending")
def trending_group() -> None:
    """Concepts recently active in your library (ingest-recency, not field trends).

    Surfaces entities you've been adding to your library most actively lately,
    based on when papers were INGESTED (extracted_at), not when they were
    published. This is library-activity recency, not a field-level publication
    trend.
    """


@trending_group.command("suggest")
def trending_suggest_cmd() -> None:
    """Detect concepts recently active in your library; persist new suggestions.

    Buckets MENTIONS edges by ingest time (extracted_at), so this measures what
    you've been adding to your library lately — not field-level publication trends.
    """
    from pathlib import Path

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.graph.trending import detect_and_persist_trending

    cfg = get_config()
    if not cfg.trending_concepts.enabled:
        click.echo(
            "Trending concepts are disabled. "
            "Set trending_concepts.enabled: true in extraction.yaml to use this feature."
        )
        return

    db = StateDB(Path(cfg.state_db.path).expanduser())
    try:
        # Shared CLI/route code path (DRY): opens the graph, runs detection,
        # persists each suggestion, returns the count.
        persisted = detect_and_persist_trending(cfg, db)
    except Exception as exc:
        click.echo(f"Could not open graph DB: {exc}", err=True)
        return

    if not persisted:
        click.echo("No trending concepts detected above the configured thresholds.")
        return

    click.echo(f"Detected and persisted {persisted} trending concept(s).")
    click.echo("Run `lit-monitor trending view` to review them.")


@trending_group.command("view")
def trending_view_cmd() -> None:
    """List all trending-concept suggestions (pending, accepted, dismissed)."""
    from pathlib import Path

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    cfg = get_config()
    db = StateDB(Path(cfg.state_db.path).expanduser())
    rows = db.list_all_trending_suggestions()
    if not rows:
        click.echo("No trending concept suggestions found.")
        return

    click.echo(f"{'ID':>4}  {'Status':>10}  {'Growth':>7}  {'New':>5}  {'Prev':>5}  Concept")
    click.echo("-" * 70)
    for r in rows:
        action = r.get("user_action") or "pending"
        growth_pct = f"{r['growth_rate'] * 100:+.0f}%"
        click.echo(
            f"{r['id']:>4}  {action:>10}  {growth_pct:>7}  "
            f"{r['n_mentions_new']:>5}  {r['n_mentions_prev']:>5}  {r['concept_text']}"
        )


@trending_group.command("accept")
@click.argument("suggestion_id", type=int)
def trending_accept_cmd(suggestion_id: int) -> None:
    """Accept a suggestion (SUGGESTION_ID); add the concept to topics.yaml."""
    from pathlib import Path

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.server.config_io import safe_save_topics

    # Guard config/state-db load with the canonical friendly-error pattern used
    # by sibling main-level commands: a malformed config should exit cleanly
    # with a message rather than surfacing a raw traceback.
    try:
        cfg = get_config()
        db = StateDB(Path(cfg.state_db.path).expanduser())
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    row = db.get_trending_suggestion_by_id(suggestion_id)
    if row is None:
        click.echo(f"Error: no suggestion with id={suggestion_id}", err=True)
        raise SystemExit(1)

    db.update_trending_action(suggestion_id, "accepted")
    new_topic = {
        "name": row["concept_text"],
        "query": row["concept_text"],
        "databases": ["pubmed", "arxiv"],
        "source": f"trending_suggestion:{suggestion_id}",
    }
    try:
        safe_save_topics(new_topic)
        click.echo(
            f"Accepted: {row['concept_text']!r} added to config/topics.yaml."
        )
    except Exception as exc:
        click.echo(f"Warning: accepted in state.db but failed to save to topics.yaml: {exc}", err=True)


@trending_group.command("dismiss")
@click.argument("suggestion_id", type=int)
def trending_dismiss_cmd(suggestion_id: int) -> None:
    """Dismiss a suggestion (SUGGESTION_ID); suppresses re-suggestion for the cooldown window."""
    from pathlib import Path

    from lit_monitor.core.config import get_config
    from lit_monitor.core.state_db import StateDB

    # Guard config/state-db load with the canonical friendly-error pattern used
    # by sibling main-level commands: a malformed config should exit cleanly
    # with a message rather than surfacing a raw traceback.
    try:
        cfg = get_config()
        db = StateDB(Path(cfg.state_db.path).expanduser())
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    row = db.get_trending_suggestion_by_id(suggestion_id)
    if row is None:
        click.echo(f"Error: no suggestion with id={suggestion_id}", err=True)
        raise SystemExit(1)

    db.update_trending_action(suggestion_id, "dismissed")
    click.echo(
        f"Dismissed: {row['concept_text']!r} will not be re-suggested "
        f"for {getattr(getattr(get_config(), 'trending_concepts', None), 'cooldown_days_after_dismiss', 60)} days."
    )


# ---------------------------------------------------------------------------
# Bundle E (v0.9): topics expansions subcommand
# ---------------------------------------------------------------------------

@main.group("topics")
def topics_group() -> None:
    """Topic management commands."""


@topics_group.group("expansions")
def topics_expansions_group() -> None:
    """Graph-driven query expansion suggestions for topics."""


@topics_expansions_group.command("suggest")
@click.argument("topic_name")
@click.option("--top-k", default=3, show_default=True, type=int)
def topics_expansions_suggest_cmd(topic_name: str, top_k: int) -> None:
    """Suggest top-K co-occurring entities for TOPIC_NAME's primary entity."""
    from pathlib import Path

    from lit_monitor.core.config import get_config
    from lit_monitor.graph.query_expansion import suggest_expansions

    cfg = get_config()
    try:
        from lit_monitor.graph.db import GraphDB

        graph_path = Path(cfg.retrieval.graph_db.persist_dir).expanduser()
        graph_db = GraphDB(str(graph_path))
    except Exception as exc:
        click.echo(f"Could not open graph DB: {exc}", err=True)
        return

    with graph_db:
        suggestions = suggest_expansions(topic_name, graph_db, top_k=top_k)

    if not suggestions:
        click.echo(f"No expansion suggestions found for topic {topic_name!r}.")
        return

    click.echo(f"Expansion suggestions for {topic_name!r} (top {top_k}):")
    for i, cid in enumerate(suggestions, 1):
        click.echo(f"  {i}. {cid}")
    click.echo(
        "\nTo apply: add these as OR-expansions to the topic query in config/topics.yaml."
    )


# ---------------------------------------------------------------------------
# Bundle J2 (v1.0 active-learning): interest-vector learning commands
# ---------------------------------------------------------------------------

@main.group("learning")
def learning_group() -> None:
    """Active-learning: build/inspect the Rocchio interest vector (Bundle J2).

    The interest vector turns your saved/dismissed feedback into a single
    "interest direction" that re-ranks discoveries toward what you like. It is
    INERT until BOTH (a) you set ranking.weights.feedback > 0 in config AND
    (b) you have accumulated at least 10 feedback events.
    """


# Cap on how many feedback events we pull for a recompute. The interest vector
# is a coarse direction, so this is generous headroom rather than a true limit.
_LEARNING_MAX_EVENTS = 100_000


@learning_group.command("recompute")
@click.pass_context
def learning_recompute_cmd(ctx: click.Context) -> None:
    """Rebuild the global interest vector from all feedback events.

    Steps: load every feedback event → look up each DOI's stored paper
    embedding → compute the library centroid (mean of all paper embeddings) →
    run the Rocchio update → store the result under scope 'global'. Below 10
    feedback events the stored vector stays inert (the rank-time soft-gate is
    0.0), and the command says so.
    """
    _setup_logging("learning", verbose=ctx.obj.get("verbose", False))
    from datetime import datetime

    import numpy as np

    from lit_monitor.learning.rocchio import (
        COLD_START_FLOOR,
        build_interest_inputs,
        compute_interest_vector,
        soft_gate,
    )

    try:
        config = _make_config()
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    try:
        events = state_db.list_feedback_events(limit=_LEARNING_MAX_EVENTS)
        if not events:
            click.echo(
                "No feedback events recorded yet — nothing to learn from. "
                f"Save or dismiss papers (need ≥{COLD_START_FLOOR}) and re-run."
            )
            return

        # Fetch ALL stored paper embeddings once: serves both the per-event
        # embed_lookup AND the library centroid (mean over all vectors).
        all_embeddings = embeddings_db.get_paper_embeddings()
        if not all_embeddings:
            click.echo(
                "No paper embeddings found in the library — run "
                "`lit-monitor brain-build` first so feedback can be grounded."
            )
            return

        def embed_lookup(doi: str) -> np.ndarray | None:
            return all_embeddings.get(doi)

        # Library centroid = mean of all stored paper vectors (the Rocchio prior).
        library_centroid = np.mean(
            np.stack(list(all_embeddings.values())), axis=0
        ).astype(np.float32)

        positive, negative, n_events = build_interest_inputs(
            events, embed_lookup, now=datetime.now()
        )
        interest_vec = compute_interest_vector(library_centroid, positive, negative)

        state_db.store_interest_vector("global", interest_vec, n_events)

        gate = soft_gate(n_events)
        click.echo(
            f"Interest vector recomputed: {n_events} contributing event(s) "
            f"(of {len(events)} total), dim={interest_vec.shape[0]}, "
            f"soft_gate={gate:.3f}."
        )
        if gate <= 0.0:
            click.echo(
                f"INERT — need ≥{COLD_START_FLOOR} contributing feedback events "
                "before the interest vector affects ranking (soft_gate is 0.0)."
            )
        else:
            click.echo(
                "Active once ranking.weights.feedback > 0 in config. "
                "Run `lit-monitor learning view` to inspect what it learned."
            )
    finally:
        # StateDB / EmbeddingsDB use per-call connections (no long-lived handle
        # to close); kept as an explicit finally for symmetry with other groups.
        pass


@learning_group.command("view")
@click.option("--top-k", default=10, show_default=True, type=int,
              help="Show the K library papers most aligned with the interest vector.")
@click.option("--per-cluster", "per_cluster", is_flag=True, default=False,
              help="Show each active cluster's atrophy feedback_weight (Bundle K-a) "
                   "instead of the global interest vector.")
@click.pass_context
def learning_view_cmd(ctx: click.Context, top_k: int, per_cluster: bool) -> None:
    """Inspect the stored global interest vector and what it has learned.

    With ``--per-cluster``, instead shows each active cluster's atrophy
    ``feedback_weight`` (Bundle K-a): a value in ``[floor, 1.0]`` that decays
    only on active dismissal within that cluster and floors at
    ``feedback.minimum_cluster_floor`` so an un-engaged theme never vanishes.
    """
    _setup_logging("learning", verbose=ctx.obj.get("verbose", False))
    import numpy as np

    from lit_monitor.learning.rocchio import COLD_START_FLOOR, soft_gate

    try:
        config = _make_config()
        state_db = _make_state_db(config)
        embeddings_db = _make_embeddings_db(config)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Bundle K-a: per-cluster atrophy feedback weights view.
    if per_cluster:
        _learning_view_per_cluster(config, state_db)
        return

    try:
        stored = state_db.get_interest_vector("global")
        if stored is None:
            click.echo(
                "No interest vector stored yet. Run "
                "`lit-monitor learning recompute` to build one."
            )
            return
        interest_vec, n_events = stored

        # computed_at lives on the row; read it directly.
        with state_db._connect() as conn:
            row = conn.execute(
                "SELECT computed_at FROM interest_vectors WHERE scope = ?",
                ("global",),
            ).fetchone()
        computed_at = (row["computed_at"] if row else "") or "unknown"

        gate = soft_gate(n_events)
        click.echo("Global interest vector:")
        click.echo(f"  events contributing : {n_events}")
        click.echo(f"  soft_gate           : {gate:.3f}"
                   + ("  (INERT — below cold-start floor)" if gate <= 0.0 else ""))
        click.echo(f"  cold-start floor    : {COLD_START_FLOOR}")
        click.echo(f"  computed_at         : {computed_at}")
        click.echo(f"  dimensionality      : {interest_vec.shape[0]}")

        # Nice-to-have: top-K library papers most aligned with the interest vec.
        # CU-1: ranking computation is shared with the /insights page via
        # rank_papers_by_interest; the CLI keeps its own cosmetic rendering
        # (signed +.3f cosine, 60-char title truncation) at the render site.
        from lit_monitor.learning.ranking import rank_papers_by_interest

        i_norm = float(np.linalg.norm(interest_vec))
        if i_norm < 1e-9 or not np.isfinite(i_norm):
            click.echo("\n(Interest vector is degenerate; no alignment ranking.)")
            return
        all_embeddings = embeddings_db.get_paper_embeddings()
        if not all_embeddings:
            click.echo("\n(No library embeddings to rank by alignment.)")
            return

        ranked = rank_papers_by_interest(
            interest_vec, embeddings_db, state_db, top_k=top_k
        )
        if ranked:
            click.echo(f"\nTop {len(ranked)} library papers by interest alignment:")
            for row in ranked:
                doi, title, cos = row["doi"], row["title"], row["cosine"]
                # title is "title or doi" from the ranker; when it equals the
                # bare DOI (no resolvable title), show the full DOI untruncated,
                # else truncate the real title to 60 chars (unchanged behaviour).
                label = doi if title == doi else f"{title[:60]}"
                click.echo(f"  {cos:+.3f}  {label}")
    finally:
        pass


def _learning_view_per_cluster(config, state_db) -> None:  # noqa: ANN001 — duck-typed
    """Print each active cluster's atrophy feedback_weight (Bundle K-a).

    Reads the floor from ``config.feedback.minimum_cluster_floor`` and computes
    weights via the pure atrophy core over all feedback events + active cluster
    assignments. Clusters are listed sorted by weight ascending (most-atrophied
    first) so an un-engaged theme is easy to spot.
    """
    from lit_monitor.clustering.atrophy import (
        compute_cluster_feedback_weights_from_db,
    )

    floor = float(
        getattr(getattr(config, "feedback", None), "minimum_cluster_floor", 0.1)
    )

    clusters = state_db.list_active_clusters()
    if not clusters:
        click.echo(
            "No active clusters yet. Run `lit-monitor cluster recompute` first."
        )
        return

    try:
        weights = compute_cluster_feedback_weights_from_db(state_db, floor=floor)
    except Exception as exc:  # noqa: BLE001 — view must never crash the CLI
        click.echo(f"Error computing per-cluster weights: {exc}", err=True)
        return

    click.echo("Per-cluster atrophy feedback weights (Bundle K-a):")
    click.echo(f"  floor: {floor:.2f}  (a weight never decays below this)")
    click.echo("  weight decays only on active dismissal within a cluster.\n")

    # Sort by weight ascending (most atrophied first), then by id for stability.
    rows = []
    for cluster in clusters:
        cid = int(cluster["id"])
        name = cluster.get("display_name") or f"Cluster {cid}"
        # A cluster with no assignments has no entry in `weights` → full weight.
        weight = weights.get(cid, 1.0)
        rows.append((weight, cid, name))
    rows.sort(key=lambda r: (r[0], r[1]))

    for weight, cid, name in rows:
        at_floor = "  (at floor)" if abs(weight - floor) < 1e-6 else ""
        click.echo(f"  {weight:.3f}  [{cid}] {name}{at_floor}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
