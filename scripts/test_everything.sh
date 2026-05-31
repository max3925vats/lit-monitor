#!/usr/bin/env bash
# lit-monitor release-gate test script.
#
# Runs four tiers of validation in order. Tiers 1-3 are hard gates
# (script exits 1 on any failure). Tier 4 is a soft gate — writes a
# Markdown report you review by eye before tagging.
#
# Usage:
#   ./scripts/test_everything.sh            # run all four tiers
#   ./scripts/test_everything.sh --tier 1   # run just one tier (1-4)
#   ./scripts/test_everything.sh --skip-4   # skip the soft tier
#
# Time budget: ~25-35 min for the full run. Tier 1: ~30 sec. Tier 2:
# ~2 min. Tier 3: ~15 min (real LLM tokens). Tier 4: ~10 min.
#
# Configuration (env vars or edit the constants below):
#   REFERENCE_DOI     — a DOI already in your state DB with a written
#                       note. Used by Tier 2 (rerender) and Tier 3 (relink).
#   REFERENCE_DOIS    — comma-separated DOIs for Tier 4 quality table.
#   See docs/internal/TESTING_STRATEGY.md for the operator's manual.

set -euo pipefail

# ───────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REFERENCE_DOI="${REFERENCE_DOI:-}"           # fill in before first use
REFERENCE_DOIS="${REFERENCE_DOIS:-}"          # comma-separated
EXPECTED_MIN_TESTS="${EXPECTED_MIN_TESTS:-829}"
REPORT_PATH="${REPORT_PATH:-/tmp/lit-monitor-release-quality-report.md}"

# Leakage patterns are defined in scripts/check_leakage.sh (called from Tier 1
# below). Same script is invoked by CI on every push/PR so leakage is caught
# whether you run the release-gate locally or open a PR.

# ───────────────────────────────────────────────────────────────────
# Output helpers
# ───────────────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
    BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

banner() { printf "\n${BOLD}=== %s ===${NC}\n" "$1"; }
pass()   { printf "${GREEN}  ✓ %s${NC}\n" "$1"; }
fail()   { printf "${RED}  ✗ %s${NC}\n" "$1"; exit 1; }
warn()   { printf "${YELLOW}  ⚠ %s${NC}\n" "$1"; }
info()   { printf "    %s\n" "$1"; }

require_uv() {
    command -v uv >/dev/null 2>&1 || fail "uv not installed; run ./install.sh first"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

# ───────────────────────────────────────────────────────────────────
# Tier 1 — Code correctness (~30 sec)
# ───────────────────────────────────────────────────────────────────

tier_1() {
    banner "Tier 1 — Code correctness"

    require_uv
    require_cmd ruff
    require_cmd git

    info "ruff check ."
    if ! ruff check . >/dev/null; then
        ruff check .  # re-run to show the failures
        fail "ruff found issues"
    fi
    pass "ruff clean"

    info "uv run pytest tests/unit/ tests/llm/ -q"
    local output
    output="$(uv run pytest tests/unit/ tests/llm/ -q --no-header 2>&1)" || {
        echo "$output" | tail -20
        fail "unit/llm test suite failed"
    }
    local count
    count="$(echo "$output" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+')" || count=0
    if (( count < EXPECTED_MIN_TESTS )); then
        echo "$output" | tail -5
        fail "test count $count < expected minimum $EXPECTED_MIN_TESTS (silent test deletion?)"
    fi
    pass "$count tests passed (>= $EXPECTED_MIN_TESTS)"

    info "leakage grep on tracked tree (delegates to scripts/check_leakage.sh)"
    if ! bash scripts/check_leakage.sh; then
        fail "leakage detected in tracked files (see paths above)"
    fi

    # G13: graph smoke check — gated on REFERENCE_DOI and GRAPH_ENABLED=1
    if [[ -n "${REFERENCE_DOI:-}" ]] && [[ "${GRAPH_ENABLED:-}" == "1" ]]; then
        info "[Tier 1 Graph Smoke] backfill 1 paper + verify status output"
        uv run lit-monitor graph backfill --doi "$REFERENCE_DOI" || { fail "graph backfill --doi failed"; }
        uv run lit-monitor graph status | grep -q "Paper" || { fail "graph status output missing 'Paper' row"; }
        pass "graph smoke ok"
    fi

    # N8: Phase 2 NER smoke (gated)
    if [[ -n "${REFERENCE_DOI:-}" ]] && [[ "${GRAPH_ENABLED:-}" == "1" ]] && [[ "${NER_ENABLED:-}" == "1" ]]; then
        echo ">>> [Tier 1 NER Smoke] graph backfill --ner --limit 1"
        uv run lit-monitor graph backfill --ner --limit 1 \
          || { echo "FAIL: backfill --ner"; exit 1; }
        uv run lit-monitor graph status --by-source | grep -qE "biobert|schema" \
          || { echo "FAIL: status --by-source"; exit 1; }
        echo "  ok"
    fi

    # R6: Phase 3 relationship smoke (gated)
    if [[ -n "${REFERENCE_DOI:-}" ]] && [[ "${GRAPH_ENABLED:-}" == "1" ]] && [[ "${RELATIONSHIPS_ENABLED:-}" == "1" ]]; then
        echo ">>> [Tier 1 Relationship Smoke] graph backfill --relationships --limit 1"
        uv run lit-monitor graph backfill --relationships --limit 1 \
            || { echo "FAIL: backfill --relationships"; exit 1; }
        uv run lit-monitor graph status | grep -E "EXTENDS|CONTRADICTS|COMPARES_TO" \
            || { echo "FAIL: status missing Phase 3 predicates"; exit 1; }
        echo "  ok"
    fi

    # P11: Phase 5 discovery CLI smoke (non-fatal — no runs in sandbox is expected)
    info "[smoke] lit-monitor discovery view"
    uv run lit-monitor discovery view --run latest --top-k 5 || true

    info "[smoke] lit-monitor discovery export-md"
    uv run lit-monitor discovery export-md --run latest --to /tmp/test_p11_digest.md || true

    # --help smoke confirms the subcommand is registered (fatal if missing)
    info "[smoke] lit-monitor obsidian sync --help"
    uv run lit-monitor obsidian sync --help > /dev/null

    pass "Tier 1 OK"
}

# ───────────────────────────────────────────────────────────────────
# Tier 2 — End-to-end smoke (~2 min)
# ───────────────────────────────────────────────────────────────────

tier_2() {
    banner "Tier 2 — End-to-end smoke"

    [[ -d ".venv" ]] || fail ".venv missing; run ./install.sh first"
    # shellcheck disable=SC1091
    source .venv/bin/activate

    info "lit-monitor check"
    local check_out
    check_out="$(lit-monitor check 2>&1)" || { echo "$check_out"; fail "lit-monitor check failed (Ollama/Zotero/config issue)"; }
    pass "check passed"

    # Optional-key severity regression — if scopus.api_key is absent, the row must
    # render yellow (⚠), not green (✓). If user has scopus configured, skip.
    if echo "$check_out" | grep -q "scopus.api_key not set"; then
        if ! echo "$check_out" | grep -qE '⚠.*scopus\.api_key'; then
            echo "$check_out"
            fail "scopus.api_key row should render yellow (⚠) when absent, not green"
        fi
        pass "optional-key severity (scopus = yellow)"
    else
        info "scopus.api_key appears configured — skipping yellow-row check"
    fi

    info "lit-monitor status — assert paper/embedding counts > 0"
    local status_out
    status_out="$(lit-monitor status 2>&1)" || { echo "$status_out"; fail "status failed"; }
    if echo "$status_out" | grep -qE 'Papers:[[:space:]]+0|Embeddings:[[:space:]]+0'; then
        echo "$status_out"
        fail "state DB has 0 papers or 0 embeddings (wiped or never built?)"
    fi
    pass "state DB populated"

    info "install.sh prompt-seeding round-trip"
    local tmp_prompts="/tmp/lit-monitor-test-prompts-$$"
    mkdir -p "$tmp_prompts"
    # Move real prompt yaml files aside (they're gitignored, .example.yaml stay).
    shopt -s nullglob
    local real_prompts=(config/prompts/*.yaml)
    shopt -u nullglob
    # Exclude .example.yaml entries from the list.
    local non_example=()
    for f in "${real_prompts[@]}"; do
        [[ "$f" == *.example.yaml ]] || non_example+=("$f")
    done
    if (( ${#non_example[@]} > 0 )); then
        mv "${non_example[@]}" "$tmp_prompts/"
    fi
    # Re-run just the prompt-seeding block from install.sh.
    for example in config/prompts/*.example.yaml; do
        [[ -f "$example" ]] || continue
        real="${example%.example.yaml}.yaml"
        [[ -f "$real" ]] || cp "$example" "$real"
    done
    # Verify each .example.yaml has a matching .yaml now.
    local missing=0
    for example in config/prompts/*.example.yaml; do
        real="${example%.example.yaml}.yaml"
        [[ -f "$real" ]] || { warn "seed missed: $real"; missing=1; }
    done
    # Restore the user's real files.
    if (( ${#non_example[@]} > 0 )); then
        # Remove the freshly-seeded versions first so mv-back doesn't fail.
        for f in "${non_example[@]}"; do rm -f "$f"; done
        mv "$tmp_prompts"/*.yaml config/prompts/
    fi
    rmdir "$tmp_prompts" 2>/dev/null || true
    (( missing == 0 )) || fail "install.sh prompt seeding incomplete"
    pass "install.sh seeding round-trip OK"

    if [[ -z "$REFERENCE_DOI" ]]; then
        warn "REFERENCE_DOI not set — skipping rerender smoke test"
        warn "  (export REFERENCE_DOI=10.xxxx/... to enable)"
    else
        info "lit-monitor obsidian rerender --doi $REFERENCE_DOI"
        if ! lit-monitor obsidian rerender --doi "$REFERENCE_DOI" >/dev/null 2>&1; then
            lit-monitor obsidian rerender --doi "$REFERENCE_DOI"
            fail "rerender failed for $REFERENCE_DOI — is it in state DB with a note?"
        fi
        pass "rerender OK"
    fi

    info "web UI boot + /api/health/badge probe"
    local probe_port=18997
    # Defensive: clean up any leftover process on the probe port.
    pkill -f "lit-monitor serve --port $probe_port" 2>/dev/null || true
    sleep 0.5
    lit-monitor serve --port $probe_port --no-browser >/tmp/lit-monitor-tier2-probe.log 2>&1 &
    local serve_pid=$!
    # Wait up to 5 sec for the server to bind.
    local boot_ok=0
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$probe_port/api/health" 2>/dev/null | grep -q '^200$'; then
            boot_ok=1
            break
        fi
        sleep 0.5
    done
    if (( boot_ok != 1 )); then
        kill "$serve_pid" 2>/dev/null || true
        echo "--- /tmp/lit-monitor-tier2-probe.log (tail) ---"
        tail -30 /tmp/lit-monitor-tier2-probe.log 2>/dev/null || true
        fail "lit-monitor serve did not become reachable on port $probe_port within 5s"
    fi
    local badge_body
    badge_body="$(curl -s "http://127.0.0.1:$probe_port/api/health/badge" 2>/dev/null || true)"
    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
    if ! echo "$badge_body" | grep -q 'class="status-badge '; then
        echo "$badge_body"
        fail "/api/health/badge did not return a status-badge HTML fragment"
    fi
    if ! echo "$badge_body" | grep -qE 'status-badge (healthy|degraded|misconfigured|unconfigured)'; then
        echo "$badge_body"
        fail "/api/health/badge state is not one of {healthy,degraded,misconfigured,unconfigured}"
    fi
    pass "web UI boot + badge probe OK"

    pass "Tier 2 OK"
}

# ───────────────────────────────────────────────────────────────────
# Tier 3 — Live pipeline (~15 min, costs LLM tokens)
# ───────────────────────────────────────────────────────────────────

tier_3() {
    banner "Tier 3 — Live pipeline (costs LLM tokens)"

    [[ -d ".venv" ]] || fail ".venv missing"
    # shellcheck disable=SC1091
    source .venv/bin/activate

    info "lit-monitor brain-build --max-papers 1 --resume"
    local bb_out
    bb_out="$(lit-monitor brain-build --max-papers 1 --resume 2>&1)" || {
        echo "$bb_out" | tail -30
        fail "brain-build --max-papers 1 failed"
    }
    if ! echo "$bb_out" | grep -qE 'processed.*1|1 paper'; then
        echo "$bb_out" | tail -10
        warn "brain-build ran but no paper was processed (collection exhausted?)"
    else
        pass "brain-build processed 1 paper"
    fi

    info "lit-monitor run --dry-run"
    local state_db_path
    state_db_path="$(uv run python -c 'from scripts.core.config import get_config; print(get_config().state_db.path)' 2>/dev/null || echo '')"
    local state_mtime_before=0
    if [[ -n "$state_db_path" && -f "${state_db_path/#\~/$HOME}" ]]; then
        state_mtime_before="$(stat -f %m "${state_db_path/#\~/$HOME}" 2>/dev/null || echo 0)"
    fi
    if ! timeout 300 lit-monitor run --dry-run >/dev/null 2>&1; then
        lit-monitor run --dry-run
        fail "lit-monitor run --dry-run failed or timed out (>5 min)"
    fi
    if [[ -n "$state_db_path" && -f "${state_db_path/#\~/$HOME}" ]]; then
        local state_mtime_after
        state_mtime_after="$(stat -f %m "${state_db_path/#\~/$HOME}" 2>/dev/null || echo 0)"
        if (( state_mtime_after != state_mtime_before )); then
            warn "state.db was modified by --dry-run (mtime changed)"
        fi
    fi
    pass "discovery dry-run OK"

    if [[ -z "$REFERENCE_DOI" ]]; then
        warn "REFERENCE_DOI not set — skipping relink test"
    else
        info "lit-monitor obsidian relink --doi $REFERENCE_DOI"
        if ! lit-monitor obsidian relink --doi "$REFERENCE_DOI" >/dev/null 2>&1; then
            lit-monitor obsidian relink --doi "$REFERENCE_DOI"
            fail "relink failed for $REFERENCE_DOI"
        fi
        pass "relink OK"
    fi

    pass "Tier 3 OK"
}

# ───────────────────────────────────────────────────────────────────
# Tier 4 — LLM quality + safety regressions (~10 min, soft gate)
# ───────────────────────────────────────────────────────────────────

tier_4() {
    banner "Tier 4 — LLM quality + safety (soft gate)"

    [[ -d ".venv" ]] || fail ".venv missing"
    # shellcheck disable=SC1091
    source .venv/bin/activate

    {
        echo "# lit-monitor release-gate quality report"
        echo
        echo "Generated: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
        echo "Commit: $(git rev-parse --short HEAD)"
        echo
    } > "$REPORT_PATH"

    info "compare-models on 2 papers"
    if lit-monitor compare-models --papers 2 --mode paper >/dev/null 2>&1; then
        local latest_dir
        latest_dir="$(ls -dt comparison/*_paper 2>/dev/null | head -1 || true)"
        if [[ -n "$latest_dir" && -f "$latest_dir/summary.md" ]]; then
            {
                echo "## Model comparison summary ($latest_dir)"
                echo
                cat "$latest_dir/summary.md"
                echo
            } >> "$REPORT_PATH"
            pass "compare-models output captured to report"
        else
            warn "compare-models ran but no summary.md found"
        fi
    else
        warn "compare-models failed (non-blocking)"
    fi

    info "{domain_focus} rendering regression (A6 guard)"
    local rendered_ok
    rendered_ok="$(uv run python -c "
from scripts.llm.prompt_registry import load_prompt
p = load_prompt('rationale')
print('OK' if '{domain_focus}' not in p.system else 'FAIL')
" 2>/dev/null)" || rendered_ok="ERROR"
    if [[ "$rendered_ok" == "OK" ]]; then
        pass "{domain_focus} rendered correctly"
        echo "## A6 guard: {domain_focus} rendering — PASS" >> "$REPORT_PATH"
    else
        warn "{domain_focus} placeholder NOT rendered (A6 regression!)"
        echo "## A6 guard: {domain_focus} rendering — **FAIL**" >> "$REPORT_PATH"
    fi

    info "prompt-injection sanitization regression (B1/B2 guard)"
    if uv run pytest tests/unit/test_prompt_safety.py -q --no-header >/dev/null 2>&1; then
        pass "sanitization tests pass"
        echo "## B1/B2 guard: prompt-injection sanitization — PASS" >> "$REPORT_PATH"
    else
        warn "sanitization tests failed (B1/B2 regression!)"
        echo "## B1/B2 guard: prompt-injection sanitization — **FAIL**" >> "$REPORT_PATH"
    fi

    info "reference-paper quality table"
    local ref_yaml="docs/internal/reference_papers.yaml"
    if [[ ! -f "$ref_yaml" ]]; then
        warn "no reference_papers.yaml — skipping quality table"
        warn "  (create $ref_yaml with your reference DOIs to enable)"
    else
        {
            echo
            echo "## Reference-paper quality table"
            echo
            uv run python -c "
import sqlite3, yaml, json
from pathlib import Path
from scripts.core.config import get_config

db_path = Path(get_config().state_db.path).expanduser()
with open('$ref_yaml') as fh:
    refs = yaml.safe_load(fh) or {}

if not refs:
    print('_no reference DOIs configured in $ref_yaml — add some_')
else:
    conn = sqlite3.connect(db_path)
    for doi, expected in (refs.get('papers') or {}).items():
        row = conn.execute(
            'SELECT extraction_json FROM papers WHERE doi = ?', (doi,)
        ).fetchone()
        print(f'### {doi}')
        print()
        if row is None or not row[0]:
            print('_not in state DB — re-run brain-build_')
            continue
        ex = json.loads(row[0])
        for field in ('core_finding', 'methods_summary', 'relevance_to_domain'):
            print(f'**{field}**')
            print()
            print(f'- Current: {ex.get(field, \"_(missing)_\")}')
            print(f'- Expected: {(expected or {}).get(field, \"_(not specified)_\")}')
            print()
" 2>&1
        } >> "$REPORT_PATH"
        pass "quality table written to $REPORT_PATH"
    fi

    banner "Report written to $REPORT_PATH"
    info "Review the report by eye, then proceed with tagging if it looks good."
}

# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

main() {
    local only_tier=""
    local skip_4=0
    while (( $# > 0 )); do
        case "$1" in
            --tier) only_tier="$2"; shift 2 ;;
            --skip-4) skip_4=1; shift ;;
            -h|--help)
                sed -n '/^# Usage:/,/^# Configuration/p' "$0" | sed 's/^# \{0,1\}//'
                exit 0 ;;
            *) fail "unknown arg: $1" ;;
        esac
    done

    if [[ -n "$only_tier" ]]; then
        case "$only_tier" in
            1) tier_1 ;; 2) tier_2 ;; 3) tier_3 ;; 4) tier_4 ;;
            *) fail "--tier must be 1-4" ;;
        esac
    else
        tier_1
        tier_2
        tier_3
        (( skip_4 == 1 )) || tier_4
    fi

    banner "ALL TIERS COMPLETE"
}

main "$@"
