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

from scripts.core.strict_mode import set_strict, strict_fallback
from scripts.setup.reset import (
    ResetResult,
    ResetTarget,
    _format_size_bytes,
    perform_state_reset,
    perform_vault_reset,
    state_targets,
    vault_targets,
)

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# __file__ is scripts/cli.py; parent.parent is the project root
_LOG_DIR = Path(__file__).parent.parent / "logs"
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
    effective_dir = log_dir or _LOG_DIR
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
    from scripts.core.config import get_config
    return get_config()
def _make_state_db(config):
    from scripts.core.state_db import StateDB
    return StateDB(config.state_db.path)
def _make_embeddings_db(config):
    from scripts.output.embeddings import EmbeddingsDB
    persist_dir = str(Path(config.state_db.path).parent / "chroma")
    # Use embeddings.ollama_host if explicitly set; fall back to brain_build host.
    ollama_host = getattr(config.embeddings, "ollama_host", None)
    if ollama_host is None:
        ollama_host = getattr(config.brain_build, "ollama_host", "http://localhost:11434")
    embed_model = getattr(config.embeddings, "model", "mxbai-embed-large")
    return EmbeddingsDB(persist_dir=persist_dir, ollama_host=ollama_host, embed_model=embed_model)
def _make_zotero_client(config, secrets: dict):
    from scripts.core.zotero_client import ZoteroClient
    zot_secrets = secrets.get("zotero", {})
    return ZoteroClient(
        library_id=str(zot_secrets.get("library_id", config.zotero.library_id)),
        api_key=str(zot_secrets.get("api_key", "")),
        library_type=config.zotero.library_type,
        local_storage_path=config.zotero.local_storage_path,
    )
def _make_llm(config, mode: str, model_override: str | None = None, think: bool = True):
    from scripts.llm.llm_client import (
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
    from scripts.setup.health_check import run_health_check
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
    from scripts.setup.diagnose import (
        ABSENT_OPTIONAL_MSG,
        check_core_configs,
        check_optional_configs,
        check_prompts,
        check_schemas,
    )
    from scripts.setup.health_check import run_health_check
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

    from scripts.server.config_io import load_server_config

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
        "scripts.server.app:create_app",
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

    from scripts.server.config_io import (
        load_server_config,
        save_secrets,
        save_server_config,
    )
    from scripts.setup._paths import SECRETS_PATH

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
        [sys.executable, "-m", "scripts.cli", "serve"],
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

    from scripts.vocabulary.clusterer import clusters_to_topics_yaml

    _topics_live = Path("config/topics.yaml")

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

    from scripts.vocabulary.clusterer import (
        CONCEPTS_DRAFT_PATH,
        TOPICS_SUGGESTED_PATH,
        _refine_clustering,
        build_vocabulary,
    )
    from scripts.vocabulary.keyword_extractor import extract_all_keywords
    from scripts.vocabulary.normalizer import normalise_keywords

    if refine and refine_only:
        click.echo("Error: --refine and --refine-only are mutually exclusive.", err=True)
        sys.exit(1)

    # --refine-only: skip Zotero extraction and clustering entirely.
    # Load the existing draft, run the cross-theme pass, write back.
    if refine_only:
        import yaml as _yaml

        from scripts.llm.llm_client import OllamaClient
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
        from scripts.llm.llm_client import OllamaClient
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
@click.option("--mode", default="paper", show_default=True,
              type=click.Choice(["paper"]),
              help="Test on journal papers and reviews.")
@click.pass_context
def compare_models_cmd(
    ctx: click.Context,
    papers: int,
    models_str: str | None,

    mode: str,
) -> None:
    """Compare LLM models on extraction quality. Review outputs in comparison/."""
    _setup_logging("compare", verbose=ctx.obj.get("verbose", False))
    from scripts.pipelines.model_compare import run_model_comparison
    try:
        config = _make_config()
        secrets = _load_secrets()
        _maybe_set_ollama_key(secrets)
        state_db = _make_state_db(config)
        zotero_client = _make_zotero_client(config, secrets)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
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
    from scripts.core.state_db import CURRENT_SCHEMA_VERSION
    from scripts.output.embeddings import check_embed_model_change
    from scripts.pipelines.brain_build import (
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
        from scripts.llm.extractor import extract_paper as _extract_paper
        from scripts.output.obsidian_writer import write_paper_note
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
@click.pass_context
def run_cmd(
    ctx: click.Context,
    dry_run: bool,
    screen_all: bool,
    top_k: int,
    sim_threshold: float,
    rag_mode: str | None,
) -> None:
    """Run the discovery pipeline: search + ranking + ingest new Zotero items."""
    _setup_logging("discovery", verbose=ctx.obj.get("verbose", False))
    # M3: hydrate S2_API_KEY before importing discovery (which transitively
    # imports scripts.search.semantic_scholar at module-load time).
    _maybe_set_s2_key(_load_secrets())
    from scripts.output.embeddings import check_embed_model_change
    from scripts.pipelines.discovery import run_discovery
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
        from scripts.graph import safe_graph_db as _probe_graph_db
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
    )
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
    from scripts.obsidian_tools.relink import relink_note
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
    from scripts.graph import safe_graph_db as _safe_graph_db
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
    from scripts.obsidian_tools.retheme import retheme
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
    from scripts.obsidian_tools.rerender import rerender_all, rerender_note
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
    from scripts.obsidian_tools.synthesize import synthesize
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
    from scripts.graph import safe_graph_db as _safe_graph_db
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
    from scripts.core.chunker import chunk_markdown

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
        from scripts.core.markdown_processor import strip_end_matter
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
@click.pass_context
def obsidian_re_extract(
    ctx: click.Context,
    doi: str,
    scope: str,
    phases: tuple[str, ...],
    field: tuple[str, ...],
    no_rerender: bool,
) -> None:
    """Re-run LLM extraction for a specific DOI or set of records."""
    _setup_logging("re_extract", verbose=ctx.obj.get("verbose", False))
    from scripts.obsidian_tools.re_extract import (
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
    default=4,
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

    Requires pass-4 extraction to have run for the target paper(s).
    Use ``lit-monitor obsidian re-extract --pass 4 --scope all`` first.
    """
    from scripts.core.config import Config
    from scripts.core.state_db import StateDB

    _setup_logging("build_citation_graph", verbose=ctx.obj.get("verbose", False))

    # M3: hydrate S2_API_KEY from config.toml before importing search modules,
    # which capture the env var at import time into _DEFAULT_S2_API_KEY.
    _maybe_set_s2_key(_load_secrets())
    from scripts.search.citation_graph import build_citation_graph

    config = Config()
    state_db = StateDB(config.paths.state_db)

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
        from scripts.graph.import_citations import mirror_citations, safe_graph_db
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
    default=4,
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
    from scripts.obsidian_tools.relink import relink_note
    from scripts.search.citation_graph import build_citation_graph

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

    def _resolve_and_relink(paper_doi: str) -> Any:
        """Run S2 resolution and optional note relink for one paper."""
        result = build_citation_graph(
            paper_doi, state_db, api_key=api_key, max_retries=max_retries,
        )
        if not no_rerender:
            record = state_db.get_paper(paper_doi)
            note_path = record.get("note_path") if record else None
            if note_path:
                relink_note(note_path, embeddings_db, state_db, config=config)
        return result

    if scope == "doi":
        if not doi:
            raise click.UsageError("--doi is required when --scope doi.")
        from scripts.obsidian_tools.re_extract import re_extract
        llm = _make_llm(config, mode="brain_build")
        click.echo(f"Re-extracting complex phase for {doi}…")
        re_extract(
            doi=doi.strip(),
            config=config,
            state_db=state_db,
            llm=llm,
            phases=["complex"],
            rerender=False,  # relink happens below via _resolve_and_relink
            zotero_client=zotero_client,
        )
        click.echo("Resolving citations via S2…")
        r = _resolve_and_relink(doi.strip())
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
                r = _resolve_and_relink(paper_doi)
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
        from scripts.graph.import_citations import mirror_citations, safe_graph_db
        graph_db = safe_graph_db()
        if graph_db is not None:
            try:
                added = mirror_citations(graph_db, state_db)
                if added:
                    click.echo(f"  Mirrored {added} CITES edges into Kuzu.")
            finally:
                graph_db.close()


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
    results = perform_vault_reset(targets)
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
def graph_backfill(all_papers: bool, doi: str | None, since: str | None) -> None:
    """Backfill the knowledge graph from existing state.db papers."""
    from datetime import datetime as dt

    from rich.progress import Progress

    from scripts.core.config import get_config
    from scripts.core.state_db import StateDB
    from scripts.graph import safe_graph_db
    from scripts.graph.backfill import backfill_papers

    if not (all_papers or doi or since):
        raise click.UsageError("Must specify --all, --doi, or --since.")

    config = get_config()
    state_db = StateDB(config.state_db.path)
    graph_db = safe_graph_db()
    if graph_db is None:
        raise click.UsageError(
            "[graph] extra not installed. Install with: uv sync --extra graph"
        )

    since_dt = dt.fromisoformat(since) if since else None
    filter_doi = doi if not all_papers else None

    try:
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
    from scripts.core.config import get_config
    from scripts.core.state_db import StateDB
    from scripts.graph import safe_graph_db
    from scripts.graph.backfill import rebuild_aliases_only, rebuild_all

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
            count = rebuild_all(state_db, graph_db)
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
@click.option("--out", default="config/entity_aliases.suggested.yaml",
              help="Output YAML path.")
def graph_propose_aliases(min_ratio: int, out: str) -> None:
    """Propose new aliases via fuzzy clustering (no LLM)."""
    from pathlib import Path

    from scripts.graph import safe_graph_db
    from scripts.graph.propose_aliases import propose_aliases, write_proposal_file

    graph_db = safe_graph_db()
    if graph_db is None:
        raise click.UsageError(
            "[graph] extra not installed. Install with: uv sync --extra graph"
        )

    try:
        proposals = propose_aliases(graph_db, min_ratio=min_ratio)
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
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
