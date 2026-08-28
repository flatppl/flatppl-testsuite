#!/usr/bin/env python3
"""Print the engine pins the committed verdict tables were frozen against.

    python3 scripts/engine-pins.py >> "$GITHUB_ENV"

CI evaluates this into the environment before `scripts/setup.sh` runs, so the
engines that run the gates are the ones the frozen tables can be diffed
against. Building both engines at `main` instead makes the two provenance
gates (`tests/sweep/test_gate.py`, `tests/sweep/test_sampler_gate.py`) red on
every unrelated engine merge, which says nothing about this repo's numbers.
Move a pin with `pixi run repin`, never by editing a table by hand.

Deliberately stdlib-only and outside pixi: it runs before the environment
exists.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (env var setup.sh reads, table, metadata key holding the commit)
PINS = (
    ("FLATPPL_RUST_REF", "verdicts/density-sweep.json", "determinizer_commit"),
    ("FLATPPL_JS_REF", "verdicts/sampler-sweep.json", "engine_commit"),
)


def resolve() -> tuple[list[str], list[str]]:
    lines: list[str] = []
    problems: list[str] = []
    for var, rel, key in PINS:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel}: no such table")
            continue
        commit = json.loads(path.read_text()).get("metadata", {}).get(key)
        # A table frozen without provenance records "unknown". Passing that on
        # as a ref resolves to a branch name that does not exist, or worse to
        # `main` -- exactly the unpinned build this script replaces.
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            problems.append(f"{rel}: metadata.{key} is {commit!r}, not a 40-hex commit")
            continue
        lines.append(f"{var}={commit}")
    return lines, problems


def main() -> int:
    lines, problems = resolve()
    if problems:
        print("cannot resolve the engine pins:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
