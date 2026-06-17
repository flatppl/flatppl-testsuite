#!/usr/bin/env python3
"""Run the HS3 corpus comparison tests and print formatted tables.

Two comparisons, one per corpus flavour:

  - Fixtures (corpora/hs3/fixtures): FlatPPL 2DeltaNLL vs the frozen RooFit
    expected vector, per scan point.
  - Conversions (corpora/hs3/conversions): FlatPPL JS engine vs the ROOT oracle
    (delegates to conversions/repro_hs3_js.cjs, which prints its own tables).

Requires the converter (FLATPPL_BIN) and the JS engine (FLATPPL_JS_DIR); both
have defaults in config.py. Run via `pixi run comparisons`, or directly:
`python corpora/hs3/run_comparisons.py [--engine <flatppl-js/packages/engine>]`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HS3_ROOT = Path(__file__).resolve().parent       # corpora/hs3
REPO = HS3_ROOT.parents[1]                        # repo root
sys.path.insert(0, str(REPO / "src"))

from flatppl_testsuite.config import CONFIG  # noqa: E402
from flatppl_testsuite.suites.hs3_import import score_scan, HS3_CORPUS, HS3_MANIFEST  # noqa: E402
from flatppl_testsuite.formats.hs3.importer import SkipUnimplemented  # noqa: E402


def _fixtures_tables() -> bool:
    print("=" * 78)
    print("FIXTURES — 2DeltaNLL: FlatPPL vs frozen RooFit   (corpora/hs3/fixtures)")
    print("=" * 78)
    manifest = json.loads(HS3_MANIFEST.read_text())
    ok_all = True
    for fx in manifest["fixtures"]:
        fdir = HS3_CORPUS / fx["path"]
        hs3_path = fdir / "hs3.json"
        hs3_doc = json.loads(hs3_path.read_text())
        expected_doc = json.loads((fdir / "expected.json").read_text())
        for check in expected_doc["checks"]:
            if check["kind"] != "twice_delta_nll_scan":
                continue
            sp, pts = check["scan_parameter"], check["scan_points"]
            exp, tol = check["expected"], check["tolerance"]
            print(f"\n{fx['test_id']} :: {check['id']}   (scan {sp})")
            try:
                got = score_scan(hs3_doc, hs3_path, check)
            except SkipUnimplemented as e:
                print(f"  SKIPPED — unimplemented HS3 construct: {e.hs3_type}")
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  UNSCOREABLE — {e}")
                ok_all = False
                continue
            print(f"  {sp:>8} | {'expected':>18} | {'got':>18} | {'|delta|':>10} | result")
            print(f"  {'-'*8}-+-{'-'*18}-+-{'-'*18}-+-{'-'*10}-+-------")
            check_ok = True
            for v, e, g in zip(pts, exp, got):
                d = abs(g - e)
                hit = d <= tol["atol"] + tol["rtol"] * abs(e)
                check_ok &= hit
                print(f"  {v:>8} | {e:>18.10g} | {g:>18.10g} | {d:>10.3e} | "
                      f"{'PASS' if hit else 'FAIL'}")
            ok_all &= check_ok
            print(f"  => {'PASS' if check_ok else 'FAIL'}  "
                  f"(atol={tol['atol']:g}, rtol={tol['rtol']:g})")
    return ok_all


def _conversions_tables(engine: str | None) -> bool:
    print("\n" + "=" * 78)
    print("CONVERSIONS — FlatPPL JS vs ROOT oracle   (corpora/hs3/conversions)")
    print("=" * 78)
    repro = HS3_ROOT / "conversions" / "repro_hs3_js.cjs"
    eng = engine or str(CONFIG.flatppl_js_dir / "packages" / "engine")
    proc = subprocess.run([CONFIG.node_bin, str(repro), "--engine", eng],
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.returncode == 0


def main() -> int:
    engine = None
    argv = sys.argv[1:]
    if "--engine" in argv:
        engine = argv[argv.index("--engine") + 1]
    fixtures_ok = _fixtures_tables()
    conversions_ok = _conversions_tables(engine)
    return 0 if (fixtures_ok and conversions_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
