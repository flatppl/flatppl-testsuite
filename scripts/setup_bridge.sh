#!/usr/bin/env bash
# Build the black-box row's pieces into .bridge/ (or $FLATPPL_BRIDGE_BUILD).
# Run via `pixi run -e bridge bridge-setup`.
#
# - the flatppl Python package from a flatppl-python checkout
#   ($FLATPPL_PYTHON_DIR, default ../flatppl-python);
# - the flatppl_bridge Python package and the BridgeStan plugin from a
#   flatppl-bridge checkout ($FLATPPL_BRIDGE_DIR, default ../flatppl-bridge);
# - the Stan model of tests/bridge/normal.stan, with BridgeStan at $BRIDGESTAN
#   or the copy that flatppl-bridge's build-model.sh fetches.
#
# Existing checkouts are used as they are, so a co-development branch is what
# the row tests. provenance.json records the commits.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="${FLATPPL_PYTHON_DIR:-$PROJECT_ROOT/../flatppl-python}"
BRIDGE_DIR="${FLATPPL_BRIDGE_DIR:-$PROJECT_ROOT/../flatppl-bridge}"
BUILD="${FLATPPL_BRIDGE_BUILD:-$PROJECT_ROOT/.bridge}"
mkdir -p "$BUILD/wheels" "$BUILD/plugin"
# Build outside the checkouts, which other work may share.
export CARGO_TARGET_DIR="$BUILD/target"

echo ">> flatppl from $PYTHON_DIR"
(cd "$PYTHON_DIR" && maturin build --release --locked --interpreter python --out "$BUILD/wheels")
echo ">> flatppl_bridge from $BRIDGE_DIR/python"
(cd "$BRIDGE_DIR/python" && maturin build --release --interpreter python --out "$BUILD/wheels")
newest() { ls -t "$BUILD"/wheels/"$1"-*.whl | head -n 1; }
python -m pip install --force-reinstall --no-deps "$(newest flatppl)" "$(newest flatppl_bridge)"

echo ">> BridgeStan plugin"
cargo build --release --manifest-path "$BRIDGE_DIR/Cargo.toml" -p fpb-bridgestan
cp "$BUILD"/target/release/libfpb_bridgestan.* "$BUILD/plugin/"

echo ">> Stan model"
"$BRIDGE_DIR/plugins/bridgestan/build-model.sh" "$PROJECT_ROOT/tests/bridge/normal.stan" "$BUILD"

commit() { git -C "$1" rev-parse HEAD 2>/dev/null || echo unknown; }
cat > "$BUILD/provenance.json" <<JSON
{"flatppl_python_commit": "$(commit "$PYTHON_DIR")", "flatppl_bridge_commit": "$(commit "$BRIDGE_DIR")"}
JSON
echo ">> done: $BUILD"
