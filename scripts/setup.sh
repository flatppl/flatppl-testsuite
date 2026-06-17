#!/usr/bin/env bash
# Install the pinned rust converter and fetch the pinned JS engine.
# Run via `pixi run setup`.
#
# Both pins are bumpable: change FLATPPL_RUST_REF / FLATPPL_JS_REF (or the
# defaults below) and re-run to pull changes. The whole point of the harness is
# to iterate on the converter and the engine, so updating a pin and re-running is
# the normal workflow.
#
# The JS engine is resolved at scoring time from FLATPPL_JS_DIR (pixi sets this to
# ../flatppl-js by default). This script CLONES that location from GitHub when it
# is missing, and leaves an existing checkout untouched — so a co-development
# sibling checkout (with your own engine changes) is never clobbered. Node 24's
# native TypeScript stripping loads the engine's .ts directly — no build step.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Rust converter
# ---------------------------------------------------------------------------

FLATPPL_RUST_REF="${FLATPPL_RUST_REF:-main}"

echo ">> installing flatppl CLI from flatppl-rust@${FLATPPL_RUST_REF} (with hs3 feature)"
cargo install \
  --git https://github.com/flatppl/flatppl-rust \
  --branch "${FLATPPL_RUST_REF}" \
  --features hs3 \
  --root .pixi-bin \
  --locked \
  flatppl-cli

echo ">> done. flatppl binary at .pixi-bin/bin/flatppl"

# ---------------------------------------------------------------------------
# 2. JS engine
# ---------------------------------------------------------------------------

FLATPPL_JS_REF="${FLATPPL_JS_REF:-main}"
FLATPPL_JS_REPO="https://github.com/flatppl/flatppl-js"
JS_DIR="${FLATPPL_JS_DIR:-$PROJECT_ROOT/../flatppl-js}"

if [ -f "$JS_DIR/packages/engine/index.ts" ]; then
  echo ">> JS engine already present at $JS_DIR — leaving it untouched"
else
  echo ">> fetching flatppl-js@${FLATPPL_JS_REF} into $JS_DIR"
  # Blobless clone: fast, but keeps refs so any branch/tag/SHA can be checked out.
  git clone --filter=blob:none "$FLATPPL_JS_REPO" "$JS_DIR"
  git -C "$JS_DIR" checkout "$FLATPPL_JS_REF"
fi

# Install engine deps if absent. npm workspaces hoist all deps (incl. @stdlib/*)
# to the repo-root node_modules/, which the engine's require() chain finds when
# walking up from packages/engine/.
if [ ! -d "$JS_DIR/node_modules/@stdlib" ]; then
  echo ">> installing JS engine dependencies (npm install in $JS_DIR)"
  ( cd "$JS_DIR" && npm install --prefer-offline --loglevel warn )
fi

echo ">> done. JS engine at $JS_DIR (FLATPPL_JS_DIR)"
