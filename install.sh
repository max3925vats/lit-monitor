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
#   3. Resolves and installs all dependencies (dev tooling + LiteLLM cloud
#      routing). findpapers is vendored into lit_monitor/_vendor (its old
#      typer<0.4 metadata pin conflicted with click>=8 / chromadb), so there is
#      no resolver conflict to work around. (BioBERT NER is the heavy [nlp]
#      extra — opt in with `uv sync --extra nlp` only if you want it.)
#   4. Hands off to `lit-monitor first-run`, which seeds your config files
#      (from the packaged examples under lit_monitor/_data/config_examples/)
#      and opens the setup wizard.

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
uv sync --extra dev --extra litellm
echo

# --- 4. next steps ---
# Config seeding is handled by `lit-monitor first-run` (below), which copies the
# packaged example configs into ~/.config/lit-monitor/. We intentionally do NOT
# seed repo-relative ./config/ here: once first-run creates ~/.config/lit-monitor/
# config_dir() resolves there, so a ./config/ copy would just be shadowed.
echo ">> install complete."
echo
echo "Launch lit-monitor now?"
echo "This will run \`lit-monitor first-run\` to configure credentials,"
echo "pick a host/port, and open the setup wizard in your browser."
read -r -p "Launch now? [Y/n] " ans
ans="${ans:-Y}"
if [[ "$ans" =~ ^[Yy] ]]; then
    echo
    # Activate the venv inside this subshell so `lit-monitor` resolves
    # without `uv run`. The user must source the venv themselves in their
    # own shell for subsequent invocations.
    # shellcheck disable=SC1091
    . .venv/bin/activate
    lit-monitor first-run
else
    cat <<'EOF'

Skipping interactive launch. When you're ready:

    source .venv/bin/activate
    lit-monitor first-run     # interactive setup + launch
  or
    lit-monitor serve         # if already configured

EOF
fi
