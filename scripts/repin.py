#!/usr/bin/env python3
"""Re-pin both sweep verdict tables to the engines currently checked out.

    pixi run repin

Run this after a flatppl-js or flatppl-rust merge moves an engine past the
commit a table was frozen against. CI builds both engines at the pins these
tables carry (`scripts/engine-pins.py`), so a merge elsewhere can never redden
this repo -- but the pins then only advance deliberately, here.

It regenerates against the engines CURRENTLY configured and never installs one
itself: the JS engine is the sibling checkout at its HEAD, and the determiniser
is the binary `pixi run setup` last installed. So run `pixi run setup` first
when the determiniser pin is the one to move -- rebuilding that binary from
inside this script would restart it under any other run sharing the checkout.
Both transitions are printed, so an unmoved pin is visible.

A re-pin refreshes provenance. It is NOT a way to freeze new behaviour, so it
regenerates both tables against the current engines and stops without
committing if a verdict moved. Judge a moved verdict on its merits (the spec
plus an independent oracle) and freeze it deliberately with
`pixi run sweep-regen` / `pixi run sampler-sweep-regen` instead.

WHAT COUNTS AS A MOVED VERDICT. Each sweep's own `table.diff` -- the comparison
its CI gate runs -- not byte equality of the rows. The sampler table freezes
Monte-Carlo estimates whose last digits move whenever the sampler's RNG
consumption does, and the density table freezes values compared against their
ORACLE rather than against the previous run; `diff` is what both gates call
drift, so it is what blocks a re-pin. Field-level movement is reported too, as
context on what the commit contains.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The regens are subprocesses writing straight to the terminal, so a block-
# buffered parent would print its own progress after theirs.
sys.stdout.reconfigure(line_buffering=True)

COMMIT_MESSAGE = "verdicts: re-pin sweep tables to current engines"

# Metadata a re-pin may change: provenance itself, plus anything derived from
# the rows. A change anywhere else (seed, draw count, sigma, the CI slice
# definition) redefines what the table measures, which is not a re-pin.
_REPINNABLE_METADATA = frozenset({
    "generated_at", "engine_resolved_by", "determinizer_commit", "engine_commit",
    "probe_count", "outcome_counts", "failing_rows",
})

# The fast table first: its drift shows in ~15 s, the density sweep's in
# minutes, and either one alone stops the re-pin.
TABLES = (
    ("sampler", Path("verdicts/sampler-sweep.json"),
     ("python", "-m", "flatppl_testsuite.sampler_sweep.cli", "regen")),
    ("density", Path("verdicts/density-sweep.json"),
     ("python", "-m", "flatppl_testsuite.sweep.cli", "regen", "--full")),
)


def git(*args: str) -> str:
    out = subprocess.run(("git", "-C", str(ROOT)) + args,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def field_movement(old: dict, new: dict) -> list[str]:
    """One line per changed row field. Reported, never blocking on its own."""
    a = {r["probe_id"]: r for r in old["rows"]}
    b = {r["probe_id"]: r for r in new["rows"]}
    lines = []
    for pid in sorted(set(a) | set(b)):
        if pid not in a:
            lines.append(f"  + {pid}: new row, not in the committed table")
        elif pid not in b:
            lines.append(f"  - {pid}: gone from the live sweep")
        else:
            for k in sorted(set(a[pid]) | set(b[pid])):
                if a[pid].get(k) != b[pid].get(k):
                    lines.append(f"  ~ {pid}.{k}")
    return lines


def metadata_drift(old: dict, new: dict) -> list[str]:
    a, b = old.get("metadata", {}), new.get("metadata", {})
    return [f"  metadata.{k}: {a.get(k)!r} -> {b.get(k)!r}"
            for k in sorted(set(a) | set(b))
            if k not in _REPINNABLE_METADATA and a.get(k) != b.get(k)]


def gate_drift(name: str, old_path: Path, new_path: Path) -> list[str]:
    """The sweep's own gate comparison, committed table vs live regen."""
    if name == "density":
        from flatppl_testsuite.sweep import table as t

        return t.diff(t.load(old_path), t.load(new_path))
    from flatppl_testsuite.sampler_sweep import table as t

    return t.diff(t.load(old_path)[1], t.load(new_path)[1])


def preflight():
    """Read the committed tables and name the engines this run pins to.

    Returns None (after explaining) when the re-pin must not start at all.
    """
    dirty = git("status", "--porcelain", "--", "verdicts")
    if dirty:
        print("verdicts/ has uncommitted changes; commit or discard them first:\n"
              + dirty, file=sys.stderr)
        return None

    from flatppl_testsuite.sampler_sweep import engine as js_engine
    from flatppl_testsuite.sampler_sweep import table as sampler_table
    from flatppl_testsuite.sweep import table as density_table

    try:
        js_root, why = js_engine.resolve_engine_dir()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return None
    # A pin is a claim about the tree CI will check out, and a branch worktree is
    # a private tree whose commits nobody else can fetch.
    if ".worktrees" in js_root.resolve().parts:
        print(f"the JS engine resolved to {js_root} ({why}), inside a worktree; "
              f"re-pin against the sibling checkout", file=sys.stderr)
        return None
    # Modified tracked files mean the rows describe a tree no commit names, so
    # CI could not reproduce them from the pin. Untracked files are noted rather
    # than blocked: scratch models and notes sit in these checkouts routinely.
    if subprocess.run(["git", "-C", str(js_root), "diff", "--quiet", "HEAD"]).returncode:
        print(f"the JS engine at {js_root} has uncommitted changes to tracked "
              f"files; its rows would not be reproducible from any commit",
              file=sys.stderr)
        return None
    untracked = subprocess.run(
        ["git", "-C", str(js_root), "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True).stdout.split()
    if untracked:
        print(f"note: {len(untracked)} untracked file(s) in {js_root}")

    rust_commit = density_table.resolved_commit()
    js_commit = sampler_table.engine_commit(js_root)
    if not rust_commit:
        print("the running flatppl binary has no recorded commit, so the re-pinned "
              "table would record \"unknown\" and the gate would fail; run "
              "`pixi run setup`", file=sys.stderr)
        return None
    if js_commit == "unknown":
        print(f"the JS engine at {js_root} is not a git checkout, so its commit "
              f"cannot be pinned", file=sys.stderr)
        return None

    saved = {}
    keep = Path(tempfile.mkdtemp(prefix="repin-"))
    for name, path, _cmd in TABLES:
        full = ROOT / path
        if not full.exists():
            print(f"no committed table at {path}; a re-pin refreshes an existing one",
                  file=sys.stderr)
            return None
        copy = keep / path.name
        copy.write_bytes(full.read_bytes())
        saved[name] = copy

    print(f"determiniser  {density_table.load_metadata(ROOT / TABLES[1][1]).get('determinizer_commit')}"
          f" -> {rust_commit}")
    print(f"JS engine     {sampler_table.load(ROOT / TABLES[0][1])[0].get('engine_commit')}"
          f" -> {js_commit}   ({js_root})")
    # CI checks out the JS pin from GitHub, so an unpushed commit is a pin that
    # cannot be fetched. Reported, not blocked: this reads the local remote refs
    # without fetching, so it is stale rather than wrong.
    contains = subprocess.run(
        ["git", "-C", str(js_root), "branch", "-r", "--contains", js_commit],
        capture_output=True, text=True)
    if contains.returncode or not contains.stdout.strip():
        print(f"WARNING: no remote branch in {js_root} contains {js_commit[:12]} "
              f"(local remote refs may be stale). CI clones from GitHub, so push "
              f"it before this pin lands.")
    return saved


def main() -> int:
    saved = preflight()
    if saved is None:
        return 2

    blocking: list[str] = []
    movement: list[str] = []
    for name, path, cmd in TABLES:
        print(f"\n>> regenerating {path}")
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            blocking.append(f"{path}: regen failed")
            break
        problems = gate_drift(name, saved[name], ROOT / path)
        old = json.loads(saved[name].read_text())
        new = json.loads((ROOT / path).read_text())
        problems += metadata_drift(old, new)
        if problems:
            blocking.append(f"{path}: the gate's own diff is not silent:\n"
                            + "\n".join(f"  {p}" for p in problems))
            break
        moved = field_movement(old, new)
        if moved:
            movement.append(f"{path}: {len(moved)} row field(s) moved within the "
                            f"gate's tolerance:\n" + "\n".join(moved[:40])
                            + ("\n  ..." if len(moved) > 40 else ""))

    if blocking:
        for name, path, _cmd in TABLES:
            (ROOT / path).write_bytes(saved[name].read_bytes())
        print("\n" + "\n\n".join(blocking), file=sys.stderr)
        print("\nnothing committed, both tables restored. A moved verdict is a "
              "behavioural change: judge it against the spec and an independent "
              "oracle, then freeze it deliberately with `pixi run sweep-regen` / "
              "`pixi run sampler-sweep-regen`.", file=sys.stderr)
        return 1

    for m in movement:
        print("\n" + m)

    paths = [str(p) for _n, p, _c in TABLES]
    if not git("status", "--porcelain", "--", *paths):
        print("\nno verdict moved and no metadata changed: already pinned to these "
              "engines, nothing to commit.")
        return 0
    git("add", *paths)
    git("commit", "-m", COMMIT_MESSAGE)
    print(f"\nno verdict moved. Committed {git('rev-parse', '--short', 'HEAD')} "
          f"({COMMIT_MESSAGE}). Not pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
