#!/usr/bin/env python3
"""Print the engine pins the committed verdict tables were frozen against.

A LOCAL reproduction aid, for the question "which engines produced these rows?":

    eval "$(python3 scripts/engine-pins.py)" && pixi run setup

That installs the determiniser at the commit `verdicts/density-sweep.json`
records and clones flatppl-js at the one `verdicts/sampler-sweep.json` records,
which is what makes a frozen row reproducible byte for byte -- useful when a row
diff is confusing and the first question is whether both sides are even the same
build. `pixi run provenance` names the same skew without rebuilding anything.

CI does NOT use this. It builds both engines at main on purpose, so that an
upstream regression meets the frozen values on the merge that introduces it, and
the resulting pin skew is reported rather than blocking (see pytest.ini). A
short-lived arrangement where CI built these pins instead was reverted for
exactly that reason; do not wire this back into the workflow.

Move a pin with `pixi run repin`, never by editing a table by hand.

Deliberately stdlib-only and outside pixi, so it works before the environment
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
