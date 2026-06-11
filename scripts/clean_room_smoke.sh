#!/usr/bin/env bash
# clean_room_smoke.sh — maintainer / CI fidelity check for the built wheel.
#
# Installs the freshly built wheel into a throwaway virtualenv with an isolated
# HOME, then runs from an unrelated tmp dir (NO repo on disk, NO editable
# install, NO config/ on CWD) and asserts that everything a pip user needs
# actually ships in the wheel: the CLI entry point, first-run config seeding,
# the FastAPI templates, the vendored findpapers library, the packaged
# Obsidian writer + prompt registry, and the MCP tools module.
#
# This is the gate that catches "works on my checkout, 500s on pip install"
# regressions — package-data omissions, hardcoded repo-relative paths, etc.
#
# NOT shipped in the wheel; lives under scripts/ for maintainers + CI.
#
# Usage (build first, then run):
#     uv build && ./scripts/clean_room_smoke.sh
#
# Exits non-zero on the first failed check. Prints
# "ALL CLEAN-ROOM CHECKS PASSED" on success.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"

# --- locate the wheel (build is assumed already done) -----------------------
WHEEL="$(ls -t "${DIST_DIR}"/lit_monitor-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "${WHEEL}" ]]; then
    echo "ERROR: no wheel found in ${DIST_DIR} — run 'uv build' first." >&2
    exit 1
fi
echo "Wheel under test: ${WHEEL}"

# --- throwaway sandbox: isolated HOME + venv + unrelated CWD -----------------
SANDBOX="$(mktemp -d "${TMPDIR:-/tmp}/lit_clean_room.XXXXXX")"
# Best-effort cleanup; never fail the script on cleanup.
trap 'rm -rf "${SANDBOX}" 2>/dev/null || true' EXIT

export HOME="${SANDBOX}/home"          # isolate ~/.config/lit-monitor seeding
mkdir -p "${HOME}"
VENV="${SANDBOX}/venv"
WORKDIR="${SANDBOX}/unrelated"         # a dir with NO repo / no config/ in it
mkdir -p "${WORKDIR}"

# Build the venv with the same interpreter that launched the script's python,
# falling back to python3. We want a *vanilla* pip install — no uv, no
# workaround, no --no-deps, no chromadb-first ordering.
PYBIN="$(command -v python3.11 || command -v python3)"
echo "Creating throwaway venv with ${PYBIN} ..."
"${PYBIN}" -m venv "${VENV}"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

"${PIP}" install --quiet --upgrade pip
echo "Installing the wheel (vanilla pip, no workaround) ..."
"${PIP}" install --quiet "${WHEEL}"

CLI="${VENV}/bin/lit-monitor"

# --- run every assertion from an unrelated CWD ------------------------------
cd "${WORKDIR}"
echo "Running checks from unrelated CWD: $(pwd)"

# 1. CLI entry point works (no repo, no config).
echo "[1/6] lit-monitor --help"
"${CLI}" --help >/dev/null

# 2. First-run config seeding writes paths.yaml into the isolated HOME.
echo "[2/6] seed_user_config writes paths.yaml"
"${PY}" - <<'PYEOF'
from pathlib import Path
from lit_monitor.core.seed import seed_user_config

target = Path.home() / ".config" / "lit-monitor"
seed_user_config(target)
assert (target / "paths.yaml").exists(), f"paths.yaml not seeded into {target}"
print("    seeded:", target / "paths.yaml")
PYEOF

# 3. The FastAPI app serves "/" without a 500 — proves templates ship.
echo "[3/6] TestClient(create_app()).get('/') status != 500"
"${PY}" - <<'PYEOF'
from starlette.testclient import TestClient
from lit_monitor.server.app import create_app

resp = TestClient(create_app(), raise_server_exceptions=False).get("/")
assert resp.status_code != 500, f"GET / returned {resp.status_code} (templates likely missing)"
print("    GET / ->", resp.status_code)
PYEOF

# 4. Vendored findpapers is importable and exposes a callable search().
echo "[4/6] vendored findpapers.search is callable"
"${PY}" - <<'PYEOF'
from lit_monitor._vendor import findpapers
assert callable(findpapers.search), "findpapers.search is not callable"
print("    findpapers.search OK")
PYEOF

# 5. Obsidian writer imports + packaged prompt registry loads a prompt.
echo "[5/6] obsidian_writer import + load_prompt('rationale')"
"${PY}" - <<'PYEOF'
import lit_monitor.output.obsidian_writer  # noqa: F401
from lit_monitor.llm.prompt_registry import load_prompt

prompt = load_prompt("rationale")  # must resolve from packaged _data, not repo config/
assert prompt is not None, "load_prompt('rationale') returned None"
print("    obsidian_writer + rationale prompt OK")
PYEOF

# 6. MCP tools module imports.
echo "[6/6] import lit_monitor.mcp.tools"
"${PY}" - <<'PYEOF'
from lit_monitor.mcp import tools  # noqa: F401
print("    mcp.tools OK")
PYEOF

echo "ALL CLEAN-ROOM CHECKS PASSED"
