#!/usr/bin/env python3
"""Render the HS3 corpus scoring results as a Markdown report.

Scores every numeric check in the manifest — the `fixtures/` 2DeltaNLL scans and
the `conversions/` point clouds — with the FlatPPL engine, compares against the
frozen ROOT vectors, and writes a single Markdown file: per check, a sparkline of
the 2DeltaNLL trace and a table of expected vs the engine's values.

    pixi run report                      # write corpora/hs3/report.md
    pixi run report -o /tmp/r.md         # choose the output path
    pixi run report --open               # open it afterwards (macOS `open`)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HS3_ROOT = Path(__file__).resolve().parent       # corpora/hs3
REPO = HS3_ROOT.parents[1]                        # repo root
sys.path.insert(0, str(REPO / "src"))

from flatppl_testsuite.suites.hs3_import import (  # noqa: E402
    score_scan, score_points, HS3_CORPUS, HS3_MANIFEST)


# ---------------------------------------------------------------------------
# Gather: score each numeric check, keep the full vectors.
# ---------------------------------------------------------------------------

def _within(got: float, exp: float, tol: dict) -> bool:
    return abs(got - exp) <= tol["atol"] + tol["rtol"] * abs(exp)


def gather() -> tuple[dict, list[dict]]:
    manifest = json.loads(HS3_MANIFEST.read_text())
    rows: list[dict] = []

    for fx in manifest.get("fixtures", []):
        fdir = HS3_CORPUS / fx["path"]
        hs3_path = fdir / "hs3.json"
        hs3_doc = json.loads(hs3_path.read_text())
        expected_doc = json.loads((fdir / "expected.json").read_text())
        for check in expected_doc["checks"]:
            if check["kind"] != "twice_delta_nll_scan":
                continue
            row = _row("fixtures", fx["test_id"], check,
                       axis=check["scan_parameter"], xs=check["scan_points"])
            try:
                row["got"] = score_scan(hs3_doc, hs3_path, check)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
            rows.append(row)

    for cv in manifest.get("conversions", []):
        cdir = HS3_CORPUS / cv["path"]
        expected_doc = json.loads((cdir / "expected.json").read_text())
        model = cdir / expected_doc["model"]
        for check in expected_doc["checks"]:
            if check["kind"] != "twice_delta_nll_points":
                continue
            row = _row("conversions", cv["test_id"], check,
                       axis=check["binding"], xs=list(range(len(check["expected"]))))
            try:
                row["got"] = score_points(model, check)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
            rows.append(row)

    return manifest, rows


def _row(corpus: str, test_id: str, check: dict, axis: str, xs: list) -> dict:
    return {
        "corpus": corpus,
        "test_id": test_id,
        "check_id": check["id"],
        "axis": axis,
        "xs": xs,
        "expected": check["expected"],
        "tol": check["tolerance"],
        "got": None,
        "error": None,
    }


def _passed(row: dict) -> bool:
    return (not row["error"]
            and all(_within(g, e, row["tol"])
                    for g, e in zip(row["got"], row["expected"])))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """Map a vector to unicode block levels — the 2DeltaNLL trace at a glance."""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(_SPARK[min(len(_SPARK) - 1, int((v - lo) / span * (len(_SPARK) - 1) + 0.5))]
                   for v in values)


def _g(v: float) -> str:
    return f"{v:.10g}"


def check_block(row: dict) -> str:
    head = f"### `{row['test_id']}` · {row['check_id']} {'✅' if _passed(row) else '❌'}"
    tol = row["tol"]
    meta = (f"scan `{row['axis']}`" if row["corpus"] == "fixtures"
            else f"binding `{row['axis']}`")
    meta += f" · atol {tol['atol']:g} / rtol {tol['rtol']:g}"

    if row["error"]:
        return f"{head}\n\n{meta}\n\n> **UNSCOREABLE** — {row['error']}\n"

    spark = sparkline(row["expected"])
    lines = [
        head, "",
        meta, "",
        f"`{spark}`  2ΔNLL trace ({len(row['expected'])} pts)", "",
        "| pt | expected | got | \\|Δ\\| | tol | |",
        "|---:|---:|---:|---:|---:|:--:|",
    ]
    for i, (x, e) in enumerate(zip(row["xs"], row["expected"])):
        g = row["got"][i]
        d = abs(g - e)
        t = tol["atol"] + tol["rtol"] * abs(e)
        mark = "✓" if d <= t else "✗"
        lines.append(f"| {_g(float(x))} | {_g(e)} | {_g(g)} | {d:.2e} | {t:.1e} | {mark} |")
    return "\n".join(lines) + "\n"


def render(manifest: dict, rows: list[dict]) -> str:
    passed = sum(1 for r in rows if _passed(r))
    total = len(rows)
    failed = total - passed
    n_fix = len(manifest.get("fixtures", []))
    n_conv = len(manifest.get("conversions", []))
    backend = manifest.get("reference_backend", "ROOT/RooFit")
    status = "✅ NOMINAL" if failed == 0 else f"❌ {failed} FAILING"

    out = [
        "# FlatPPL × HS3 — 2ΔNLL conformance report",
        "",
        "Every converted model, scored by the FlatPPL engine and compared "
        "point-by-point against the frozen RooFit / ROOT 2ΔNLL vector. The "
        "sparkline is the likelihood scan; the table holds the engine's values.",
        "",
        f"**Status: {status}**",
        "",
        "| checks | pass | fail | fixtures | conversions | oracle |",
        "|---:|---:|---:|---:|---:|:--|",
        f"| {total} | {passed} | {failed} | {n_fix} | {n_conv} | {backend} |",
        "",
    ]

    for corpus, title in (("fixtures", "Fixtures — RooFit tutorials"),
                          ("conversions", "Conversions — HS3 paper appendix")):
        sub = [r for r in rows if r["corpus"] == corpus]
        if not sub:
            continue
        out += [f"## {title}", ""]
        out += [check_block(r) for r in sub]

    out += ["---", "", "_generated by `pixi run report` (corpora/hs3/report.py)_", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=str(HS3_ROOT / "report.md"),
                    help="output Markdown path (default: corpora/hs3/report.md)")
    ap.add_argument("--open", action="store_true", help="open the report afterwards")
    args = ap.parse_args()

    manifest, rows = gather()
    out = Path(args.out)
    out.write_text(render(manifest, rows))

    failed = [r for r in rows if not _passed(r)]
    print(f"wrote {out}  ({len(rows) - len(failed)}/{len(rows)} checks pass)")
    if args.open:
        subprocess.run(["open", str(out)], check=False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
