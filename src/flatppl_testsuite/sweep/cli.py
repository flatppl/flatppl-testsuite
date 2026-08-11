"""CLI for the density sweep: `regen` writes the committed verdict table
(density-sweep.json); `report` reads it back and prints the sized defect
list a human triages from.

    python -m flatppl_testsuite.sweep.cli regen [--full] [--commit SHA]
    python -m flatppl_testsuite.sweep.cli report
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from flatppl_testsuite.sweep import table
from flatppl_testsuite.sweep.space import enumerate_probes, is_shared_latent


def _regen(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="sweep regen")
    ap.add_argument("--full", action="store_true",
                     help="sweep the whole probe space (the scalar axes' "
                          "cross-product plus the targeted vector and "
                          "shared-latent families); default is the CI slice "
                          "(see table.SLICE_DESCRIPTION)")
    ap.add_argument("--commit", default=None,
                     help="the flatppl determinizer commit this table is generated against; "
                          "recorded in the table's metadata. Defaults to "
                          "table.resolved_commit() (FLATPPL_RUST_COMMIT, or the "
                          "sidecar scripts/setup.sh writes) -- pass explicitly only "
                          "when neither is available")
    args = ap.parse_args(argv)

    commit = args.commit or table.resolved_commit()
    rows = table.sweep(slice_only=not args.full)
    table.save(table.DEFAULT_PATH, rows, commit=commit)
    counts = Counter(r.outcome for r in rows)
    print(f"wrote {len(rows)} rows to {table.DEFAULT_PATH} "
          f"({'full space' if args.full else 'CI slice'}, determinizer "
          f"{commit or 'unknown'}): "
          f"{counts.get('LOWERS', 0)} LOWERS, {counts.get('REFUSES', 0)} REFUSES, "
          f"{counts.get('MALFORMED', 0)} MALFORMED")
    return 0


def _report(argv: list[str]) -> int:
    argparse.ArgumentParser(prog="sweep report").parse_args(argv)

    rows = list(table.load(table.DEFAULT_PATH).values())
    if not rows:
        print("no committed verdict table -- run `pixi run sweep-regen`", file=sys.stderr)
        return 1
    # A shared-latent probe has no wrap stack; its analogous grouping key is the
    # ancestry graph, so it is reported by `shape` in the same column.
    wrap_of = {p.id: (f"shared:{p.shape}" if is_shared_latent(p) else p.wraps[0].kind)
               for p in enumerate_probes()}

    lowers = [r for r in rows if r.outcome == "LOWERS"]
    lowers_wrong = [r for r in lowers if r.known_defect]
    lowers_ok = [r for r in lowers if not r.known_defect]
    refuses = [r for r in rows if r.outcome == "REFUSES"]
    refuses_justified = [r for r in refuses if r.spec_justified]
    refuses_unjustified = [r for r in refuses if not r.spec_justified]
    malformed = [r for r in rows if r.outcome == "MALFORMED"]
    unvalidated = [r for r in rows if r.oracle_unvalidated]

    meta = table.load_metadata(table.DEFAULT_PATH)

    print(f"density-sweep.json: {len(rows)} probes "
          f"(determinizer {meta.get('determinizer_commit', 'unknown')}, "
          f"generated {meta.get('generated_at', 'unknown')})")
    print(f"  LOWERS      {len(lowers)}  ({len(lowers_ok)} correct, {len(lowers_wrong)} known-wrong)")
    if lowers_wrong:
        by_wrap = Counter(wrap_of.get(r.probe_id, "?") for r in lowers_wrong)
        for wrap, c in sorted(by_wrap.items()):
            print(f"    known-wrong by wrap: {wrap}: {c}")
    print(f"  REFUSES     {len(refuses)}  "
          f"({len(refuses_justified)} spec-justified, {len(refuses_unjustified)} over-refusal)")
    print(f"  MALFORMED   {len(malformed)}")
    print(f"  oracle-unvalidated  {len(unvalidated)}")

    return 0


_COMMANDS = {"regen": _regen, "report": _report}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in _COMMANDS:
        print(f"usage: python -m flatppl_testsuite.sweep.cli {{{'|'.join(_COMMANDS)}}} [args]",
              file=sys.stderr)
        return 2
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
