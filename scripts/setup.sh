#!/usr/bin/env bash
# Install the pinned rust converter and fetch the pinned JS engine.
# Run via `pixi run setup`.
#
# Both pins are bumpable: change FLATPPL_RUST_REF / FLATPPL_JS_REF (or the
# defaults below) and re-run to pull changes. The whole point of the harness is
# to iterate on the converter and the engine, so updating a pin and re-running is
# the normal workflow. Either ref may be a branch or a 40-hex commit.
#
# CI leaves both at `main`, on purpose: an upstream regression should meet the
# frozen values on the merge that introduces it. Pass a commit to reproduce a
# frozen table instead — the one a verdict table's metadata records.
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

# `cargo install --git` takes a commit through `--rev` and a branch through
# `--branch`, so the form of the ref decides the flag. The 40-hex form is what
# reproduces a frozen verdict table: its metadata records the determiniser
# commit its rows were generated against.
if [[ "$FLATPPL_RUST_REF" =~ ^[0-9a-f]{40}$ ]]; then
  REF_FLAG=(--rev "$FLATPPL_RUST_REF")
else
  REF_FLAG=(--branch "$FLATPPL_RUST_REF")
fi

echo ">> installing flatppl CLI from flatppl-rust@${FLATPPL_RUST_REF} (hs3 + stablehlo + default verbs incl. determinize)"
# `--features hs3,stablehlo` ADDS to the default feature set (no --no-default-features),
# so the installed binary carries the default verbs — including `determinize` (which
# the det-js scoring engine shells out to) and `stablehlo` (which the StableHLO
# numeric-execution gate emits with). The `determinize` verb landed in flatppl-rust
# #61, `stablehlo` in #70; `FLATPPL_RUST_REF=main` (the default) tracks both.
cargo install \
  --git https://github.com/flatppl/flatppl-rust \
  "${REF_FLAG[@]}" \
  --features hs3,stablehlo \
  --root .pixi-bin \
  --locked \
  flatppl-cli

echo ">> done. flatppl binary at .pixi-bin/bin/flatppl"

# Record the commit `cargo install` actually resolved `--branch` to, so the
# density sweep's CI gate (sweep/table.py::check_provenance) can tell whether
# a committed verdict table was generated against the SAME determinizer this
# run just installed, rather than silently diffing against a different one.
# `cargo install --git` writes the resolved commit into `.crates2.json`'s
# install key, e.g. `flatppl-cli 0.1.0 (git+https://...?branch=main#<40-hex>)`
# -- extracted here rather than re-resolving the ref ourselves, since this is
# the exact commit that was actually built, not a second (possibly racing)
# `git ls-remote`. `|| true` on the grep: an absent/reshaped `.crates2.json`
# must leave provenance representable as "not recorded", never abort setup.
COMMIT="$(grep -oE '#[0-9a-f]{40}\)' .pixi-bin/.crates2.json 2>/dev/null | head -1 | tr -d '#)' || true)"
if [ -n "$COMMIT" ]; then
  echo "$COMMIT" > .pixi-bin/flatppl-rust.commit
  echo ">> recorded determinizer commit $COMMIT"
else
  echo ">> could not resolve a determinizer commit from .pixi-bin/.crates2.json" >&2
fi

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
