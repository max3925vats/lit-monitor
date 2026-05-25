#!/usr/bin/env bash
# Personal-content leakage check (LOCAL-ONLY).
#
# Reads grep patterns from docs/internal/leak_patterns.txt (gitignored).
# The patterns themselves never appear in this script — that's the whole
# point. The script ships in git as generic infrastructure; the patterns
# stay on the author's machine.
#
# On a fresh public clone (no patterns file), this script is a no-op.
# That's correct: a stranger cloning the repo has no personal data to
# check for. The check matters only on the author's machine, where the
# patterns file exists.
#
# Used by:
#   - scripts/test_everything.sh (Tier 1 of the release-gate)
#
# NOT used by CI — CI can't read the patterns file, and shipping the
# patterns to make the CI step work would defeat the purpose.
#
# Add a new line to docs/internal/leak_patterns.txt whenever you scrub a
# new identifier from the codebase.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PATTERNS_FILE="${LIT_MONITOR_LEAK_PATTERNS_FILE:-docs/internal/leak_patterns.txt}"

# Files where personal-looking patterns may appear legitimately and must
# NOT trigger failure. Add to this list if a tracked file (LICENSE, etc.)
# intentionally contains a pattern.
EXCLUDE_REGEX='^(LICENSE)$'

if [[ ! -f "$PATTERNS_FILE" ]]; then
    echo "ℹ️  No patterns file at $PATTERNS_FILE — leakage check skipped."
    echo "   (Expected on a fresh clone. On the author's machine, create the"
    echo "    file with one regex pattern per line.)"
    exit 0
fi

# Build the alternation regex from the file: strip comments, strip blanks,
# join with '|'. Bash 3 / macOS compatible.
ALTS=()
while IFS= read -r line; do
    # Strip leading/trailing whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    # Skip blanks and comments.
    [[ -z "$line" ]] && continue
    [[ "$line" == \#* ]] && continue
    ALTS+=("$line")
done < "$PATTERNS_FILE"

if [[ ${#ALTS[@]} -eq 0 ]]; then
    echo "ℹ️  $PATTERNS_FILE has no active patterns — leakage check skipped."
    exit 0
fi

LEAK_REGEX="$(IFS='|'; echo "${ALTS[*]}")"

hits="$(git ls-files \
    | xargs grep -l -iE "$LEAK_REGEX" 2>/dev/null \
    | grep -vE "$EXCLUDE_REGEX" \
    || true)"

if [[ -n "$hits" ]]; then
    echo "✗ Personal-content leakage detected in tracked files:"
    echo "$hits" | sed 's/^/    /'
    echo
    echo "Either scrub the value before committing, or add the file to EXCLUDE_REGEX"
    echo "in scripts/check_leakage.sh if it intentionally contains the pattern."
    exit 1
fi

echo "✓ no personal identifiers in tracked files"
exit 0
