#!/usr/bin/env bash
# Install the pinned rust converter. Run via `pixi run setup`.
#
# The pin is bumpable: change FLATPPL_RUST_REF (or the default below) and re-run
# to pull converter changes. The whole point of the harness is to iterate on the
# converter, so updating the pin and re-running is the normal workflow.
#
# The JS engine is NOT installed here. It is resolved at scoring time from a
# flatppl-js checkout via FLATPPL_JS_DIR (pixi sets this to ../flatppl-js by
# default; override to a pinned clone such as ~/.cache/flatppl-js). Node 24's
# native TypeScript stripping loads the engine's .ts directly — no build step.
set -euo pipefail

# Pin for the rust converter. A branch name or a commit SHA both work.
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
echo ">> JS engine will be resolved from FLATPPL_JS_DIR=${FLATPPL_JS_DIR:-../flatppl-js}"
