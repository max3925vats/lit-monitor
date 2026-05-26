#!/usr/bin/env bash
# lit-monitor install script.
#
# Usage:
#   git clone https://github.com/max3925vats/lit-monitor.git
#   cd lit-monitor
#   ./install.sh
#
# What it does:
#   1. Installs uv (Astral's Python package manager) if not present.
#   2. Creates a project-local .venv via uv.
#   3. Resolves and installs all dependencies (including the cloud extras).
#      uv handles the findpapers <-> chromadb typer conflict automatically
#      via the [tool.uv] override-dependencies block in pyproject.toml.
#   4. Copies each config/*.example.yaml into its real config/*.yaml on first
#      run, leaving any pre-existing real configs untouched.
#   5. Prints the remaining first-time-setup steps.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ">> lit-monitor install"
echo "   repo: $REPO_DIR"
echo

# --- 1. uv ---
if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer writes uv to $HOME/.local/bin (or $CARGO_HOME/bin). Source
    # the post-install env script so uv is on PATH for the rest of this session.
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    fi
fi
echo "   uv: $(uv --version)"
echo

# --- 2. venv ---
if [ ! -d ".venv" ]; then
    echo ">> creating .venv"
    uv venv
fi

# --- 3. dependencies ---
echo ">> installing dependencies (this can take a few minutes on first run)"
uv sync --extra dev --extra cloud --extra server
echo

# --- 4. seed configs from examples ---
declare -a CONFIGS=(concepts topics domain_context researchers paths)
for name in "${CONFIGS[@]}"; do
    real="config/${name}.yaml"
    example="config/${name}.example.yaml"
    if [ ! -f "$real" ] && [ -f "$example" ]; then
        cp "$example" "$real"
        echo "   seeded $real from $example"
    fi
done

# --- seed prompt files from examples ---
for example in config/prompts/*.example.yaml; do
    [ -f "$example" ] || continue
    real="${example%.example.yaml}.yaml"
    if [ ! -f "$real" ]; then
        cp "$example" "$real"
        echo "   seeded $real from $example"
    fi
done
echo

# --- 5. next steps ---
cat <<'EOF'
>> install complete.

Next steps:

  1. Create your credentials file at ~/.config/lit-monitor/config.toml
     (see README.md "First-time setup" for the template).

  2. Edit the seeded configs in ./config/ for your own research area:
     - config/paths.yaml          (Zotero library_id, Obsidian vault path)
     - config/topics.yaml         (weekly search queries)
     - config/concepts.yaml       (theme vocabulary; or generate via
                                   `lit-monitor build-vocabulary`)
     - config/domain_context.yaml (one-paragraph description of your field)
     - config/researchers.yaml    (tracked authors; optional)

  3. Verify the setup:
     source .venv/bin/activate
     lit-monitor check

  4. First run:
     lit-monitor build-vocabulary
     lit-monitor brain-build --resume

EOF
