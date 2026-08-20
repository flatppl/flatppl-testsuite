"""`python -m flatppl_testsuite.sampler_sweep.cli {regen,report}`.

Mirrors the density sweep's CLI split: `regen` runs live against the engine and
overwrites the frozen table, `report` reads the committed table and prints a
sized breakdown without touching the engine.
"""
from __future__ import annotations

import argparse
import sys

from flatppl_testsuite.sampler_sweep import space, table


def _regen(args) -> int:
    rows = table.sweep()
    table.store(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    failing = [r for r in rows if r.failed]
    print(f"wrote {table.DEFAULT_PATH} — {len(rows)} rows")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  rows with a failing check: {len(failing)}")
    for r in failing:
        for c in r.failed:
            print(f"    {r.probe_id}  {c['name']}: {c['detail']}")
    return 0


def _report(args) -> int:
    meta, rows = table.load()
    if not rows:
        print("no committed sampler table — run `pixi run sampler-sweep-regen`", file=sys.stderr)
        return 1
    print(f"sampler sweep — {meta.get('probe_count')} rows, "
          f"n={meta.get('n_draws')} draws/row, {meta.get('sigma')}-sigma bands")
    print(f"  engine {meta.get('engine_commit', '?')[:12]}  generated {meta.get('generated_at')}")

    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    print("\noutcomes: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    refuses = [r for r in rows.values() if r.outcome == "REFUSES"]
    if refuses:
        print(f"\nREFUSES ({len(refuses)}) — by reason:")
        by_marker: dict[str, list[str]] = {}
        for r in refuses:
            by_marker.setdefault(r.marker or "?", []).append(r.probe_id)
        for marker, ids in sorted(by_marker.items()):
            print(f"  {marker}: {', '.join(sorted(ids))}")

    malformed = [r for r in rows.values() if r.outcome == "MALFORMED"]
    if malformed:
        print(f"\nMALFORMED ({len(malformed)}) — always a defect:")
        for r in malformed:
            print(f"  {r.probe_id}: {r.error}")

    failing = [r for r in rows.values() if r.failed]
    print(f"\nDRAWS rows with a failing check: {len(failing)}")
    for r in sorted(failing, key=lambda r: -(r.worst_sigma or 0)):
        head = f"  {r.probe_id}"
        if r.worst_sigma is not None:
            head += f"  (worst {r.worst_sigma:.1f} sigma)"
        print(head)
        for c in r.failed:
            print(f"    {c['name']}: {c['detail']}")

    checked = sum(1 for r in rows.values() for c in r.checks if c["status"] != "skipped")
    skipped = sum(1 for r in rows.values() for c in r.checks if c["status"] == "skipped")
    fallback = sum(1 for r in rows.values() for c in r.checks if c.get("fallback"))
    print(f"\nchecks: {checked} run, {skipped} skipped, {fallback} on a non-sigma fallback band")
    print(f"density-only REGISTRY entries with no sampler (out of scope by construction): "
          f"{len(_density_only())}")
    return 0


def _density_only():
    from flatppl_testsuite.sampler_sweep.oracle import DENSITY_ONLY

    return DENSITY_ONLY


def _roster(args) -> int:
    """The roster itself, no engine calls: families x wraps x checks."""
    from flatppl_testsuite.sampler_sweep.oracle import DENSITY_ONLY, FAMILIES

    probes = space.enumerate_probes()
    print(f"{len(FAMILIES)} sampleable families, {len(probes)} probe rows, "
          f"n={space.N_DRAWS} draws each")
    for p in probes:
        ks = "ks" if p.ks else "--"
        cov = f"cov(k={p.k})" if p.cov is not None else "--"
        tm = "mass" if p.logtotalmass is not None else "--"
        print(f"  {p.id:<34} {ks:<3} {cov:<10} {tm}")
    print(f"\n{len(DENSITY_ONLY)} density-only REGISTRY entries have no sampler:")
    for name, why in DENSITY_ONLY:
        print(f"  {name:<26} {why}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sampler_sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regen", help="run live against the engine and refreeze the table")
    sub.add_parser("report", help="print the committed table's breakdown (no engine calls)")
    sub.add_parser("roster", help="print the probe roster (no engine calls)")
    args = ap.parse_args(argv)
    return {"regen": _regen, "report": _report, "roster": _roster}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
