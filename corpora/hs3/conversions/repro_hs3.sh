#!/usr/bin/env bash
# Reproduction harness for the HS3 examples.
#
# Runs two independent checks in sequence:
#   1. FlatPPL JS engine  (repro_hs3_js.cjs)
#   2. ROOT / PyROOT      (src/hs3/*_root.py)  — skipped if PyROOT unavailable
#
# SETUP (done automatically on first run):
#   Clones https://github.com/flatppl/flatppl-js into ~/.cache/flatppl-js and
#   runs `npm install` there. Subsequent runs reuse the cached clone.
#
# OVERRIDES:
#   FLATPPL_JS_DIR=/path/to/flatppl-js  — use an existing checkout instead
#   NODE=/path/to/node                  — override Node binary
#
# REQUIREMENTS:
#   Node >= 24  (native TypeScript type-stripping; see note below)
#   git, npm    (for engine setup)
#   python3 + ROOT >= 6.30 with RooFit JSON support (optional; for ROOT checks)
#
# Node 24 note: the engine ships TypeScript source only — no separate compile
# step. Node 24's built-in type-stripping makes `require('./x.ts')` work
# without any build step. If you can only run Node 22.x, set
#   NODE_OPTIONS=--experimental-strip-types
# before calling this script (Node 22.6+). Node 18/20 are not supported.
#
# ALTERNATIVE (no clone required):
#   If the flatppl-js team publishes an `engine-node.cjs` bundle artifact
#   (Node 18+, no TypeScript stripping needed), set:
#     FLATPPL_ENGINE_BUNDLE=/path/to/engine-node.cjs
#   and the JS check will use it instead. See repro_hs3_js.cjs for details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HS3_DIR="$SCRIPT_DIR"
FLATPPL_JS_REPO="https://github.com/flatppl/flatppl-js"
CACHE_DIR="$HOME/.cache/flatppl-js"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { printf '==> %s\n' "$*" >&2; }
warn()  { printf 'warn: %s\n' "$*" >&2; }
die()   { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Locate or fetch flatppl-js
# ---------------------------------------------------------------------------

find_checkout() {
  # Candidates, in order:
  #  a) FLATPPL_JS_DIR env var (explicit override)
  #  b) local cache at ~/.cache/flatppl-js (cloned from GitHub)
  local candidates=(
    "${FLATPPL_JS_DIR:-}"
    "$CACHE_DIR"
  )
  for c in "${candidates[@]}"; do
    [ -n "$c" ] && [ -f "$c/packages/engine/index.ts" ] && printf '%s' "$c" && return 0
  done
  return 1
}

if ! FLATPPL_JS="$(find_checkout)"; then
  info "flatppl-js not found locally — cloning into $CACHE_DIR"
  git clone --depth=1 "$FLATPPL_JS_REPO" "$CACHE_DIR"
  FLATPPL_JS="$CACHE_DIR"
else
  info "Using flatppl-js at $FLATPPL_JS"
fi

ENGINE_DIR="$FLATPPL_JS/packages/engine"

# Install engine dependencies if node_modules are absent.
# Running `npm install` at the repo root uses npm workspaces to hoist all
# deps (including @stdlib/*) to node_modules/, which the engine's require()
# chain will find when walking up from packages/engine/.
if [ ! -d "$FLATPPL_JS/node_modules/@stdlib" ]; then
  info "Installing engine dependencies (npm install in $FLATPPL_JS)"
  ( cd "$FLATPPL_JS" && npm install --prefer-offline --loglevel warn )
fi

# ---------------------------------------------------------------------------
# 2. Node version check
# ---------------------------------------------------------------------------

NODE="${NODE:-node}"
NODE_VER="$("$NODE" --version | sed 's/^v//')"
NODE_MAJOR="$(printf '%s' "$NODE_VER" | cut -d. -f1)"

if [ "$NODE_MAJOR" -lt 24 ]; then
  if [ "$NODE_MAJOR" -eq 22 ]; then
    NODE_OPTIONS="${NODE_OPTIONS:-} --experimental-strip-types"
    export NODE_OPTIONS
    warn "Node $NODE_VER: adding --experimental-strip-types (upgrade to Node 24 for stable support)"
  else
    die "Node >= 22.6 required (found v$NODE_VER). Install from https://nodejs.org"
  fi
fi

# ---------------------------------------------------------------------------
# 3. JS engine repro
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Per-model: FlatPPL JS then ROOT
# ---------------------------------------------------------------------------

JS_EXIT=0
ROOT_EXIT=0
HAS_ROOT=false
python3 -c "import ROOT" 2>/dev/null && HAS_ROOT=true

for model in gaussian histfactory product; do
  FLATPPL_JS_DIR="$FLATPPL_JS" "$NODE" "$SCRIPT_DIR/repro_hs3_js.cjs" --model "$model" \
    || JS_EXIT=$?
  if $HAS_ROOT; then
    python3 "$HS3_DIR/$model/${model}_root.py" || ROOT_EXIT=$?
  fi
  printf '\n'
done

if ! $HAS_ROOT; then
  printf '(ROOT oracle skipped — PyROOT not available; install ROOT >= 6.30)\n' >&2
fi

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

printf '%-30s %s\n' 'JS engine (flatppl-js):' "$([ "$JS_EXIT" -eq 0 ] && echo PASS || echo "FAIL (exit $JS_EXIT)")" >&2
printf '%-30s %s\n' 'ROOT oracle:' "$([ "$ROOT_EXIT" -eq 0 ] && echo "PASS (or skipped)" || echo "FAIL (exit $ROOT_EXIT)")" >&2

[ "$JS_EXIT" -eq 0 ] && [ "$ROOT_EXIT" -eq 0 ]
