#!/usr/bin/env python3
"""Phase 4b refuse-reason report: run the HS3 corpus under the `det-js` engine
and print a `model -> PASS(vs ROOT) / SKIP(reason) / MISMATCH` table.

`det-js` is the convert-free path: `flatppl determinize` lowers a FlatPPL
model straight to the deterministic FlatPDL profile (no measure layer), which
is then scored via `score_flatpdl.cjs`. Any construct the determiniser can't
yet legalize raises `DeterminizeRefused` (CLI exit 3) — that is a SKIP, not a
failure: the model is outside the determiniser's current density fragment.
A `twice_delta_nll` mismatch against the frozen ROOT vector, in contrast, is a
real numeric bug and is reported as MISMATCH.

This is the coverage-gap output for the numeric gate: which HS3 models the
determiniser currently handles, and why the rest refuse. It always runs under
`FLATPPL_ENGINE=det-js`, independent of the ambient `FLATPPL_ENGINE` (if any).

    pixi run gate
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HS3_ROOT = Path(__file__).resolve().parent       # corpora/hs3
REPO = HS3_ROOT.parents[1]                        # repo root
sys.path.insert(0, str(REPO / "src"))

os.environ["FLATPPL_ENGINE"] = "det-js"

from flatppl_testsuite.suites.hs3_import import (  # noqa: E402
    score_scan, score_points, HS3_CORPUS, HS3_MANIFEST)
from flatppl_testsuite.scoring.compare import compare_vectors  # noqa: E402
from flatppl_testsuite.scoring.engine import DeterminizeRefused  # noqa: E402


def _row(test_id: str, outcome: str, detail: str = "") -> dict:
    return {"test_id": test_id, "outcome": outcome, "detail": detail}


def gather() -> list[dict]:
    """Score every manifest entry's numeric check under det-js; classify each
    as PASS (matches the frozen ROOT vector), SKIP (determinizer refused), or
    MISMATCH (determinized + scored, but diverges from ROOT beyond tolerance)."""
    rows: list[dict] = []
    manifest = json.loads(HS3_MANIFEST.read_text())

    for fx in manifest.get("fixtures", []):
        fdir = HS3_CORPUS / fx["path"]
        hs3_path = fdir / "hs3.json"
        hs3_doc = json.loads(hs3_path.read_text())
        expected_doc = json.loads((fdir / "expected.json").read_text())
        for check in expected_doc["checks"]:
            if check["kind"] != "twice_delta_nll_scan":
                continue
            test_id = f"{fx['test_id']}::{check['id']}"
            try:
                got = score_scan(hs3_doc, hs3_path, check)
            except DeterminizeRefused as e:
                rows.append(_row(test_id, "SKIP", str(e)))
                continue
            except Exception as e:  # noqa: BLE001
                rows.append(_row(test_id, "ERROR", str(e)))
                continue
            try:
                compare_vectors(got, check["expected"], check["tolerance"])
            except AssertionError as e:
                rows.append(_row(test_id, "MISMATCH", str(e)))
                continue
            rows.append(_row(test_id, "PASS"))

    for cv in manifest.get("conversions", []):
        cdir = HS3_CORPUS / cv["path"]
        expected_doc = json.loads((cdir / "expected.json").read_text())
        model = cdir / expected_doc["model"]
        for check in expected_doc["checks"]:
            if check["kind"] != "twice_delta_nll_points":
                continue
            test_id = f"{cv['test_id']}::{check['id']}"
            try:
                got = score_points(model, check)
            except DeterminizeRefused as e:
                rows.append(_row(test_id, "SKIP", str(e)))
                continue
            except Exception as e:  # noqa: BLE001
                rows.append(_row(test_id, "ERROR", str(e)))
                continue
            try:
                compare_vectors(got, check["expected"], check["tolerance"])
            except AssertionError as e:
                rows.append(_row(test_id, "MISMATCH", str(e)))
                continue
            rows.append(_row(test_id, "PASS"))

    return rows


def render(rows: list[dict]) -> str:
    width = max((len(r["test_id"]) for r in rows), default=8)
    lines = [
        "=" * 78,
        "HS3 CORPUS UNDER det-js — determinize+score vs frozen ROOT",
        "=" * 78,
        "",
        f"  {'model :: check':<{width}}  outcome   detail",
        f"  {'-' * width}  -------   ------",
    ]
    for r in rows:
        detail = r["detail"].splitlines()[0] if r["detail"] else ""
        if len(detail) > 90:
            detail = detail[:87] + "..."
        lines.append(f"  {r['test_id']:<{width}}  {r['outcome']:<7}   {detail}")
    n_pass = sum(1 for r in rows if r["outcome"] == "PASS")
    n_skip = sum(1 for r in rows if r["outcome"] == "SKIP")
    n_mismatch = sum(1 for r in rows if r["outcome"] == "MISMATCH")
    n_error = sum(1 for r in rows if r["outcome"] == "ERROR")
    lines.append("")
    lines.append(
        f"  {n_pass} PASS, {n_skip} SKIP, {n_mismatch} MISMATCH, {n_error} ERROR "
        f"(of {len(rows)} checks)"
    )
    if n_mismatch:
        lines.append("")
        lines.append(
            "  MISMATCH means the determiniser lowered + scored the model, but the "
            "result diverges from the independent ROOT oracle beyond tolerance — a "
            "real numeric bug (escalate; do not silently accept)."
        )
    return "\n".join(lines)


def main() -> int:
    rows = gather()
    print(render(rows))
    # The gate script's job is to report coverage, not to fail CI on expected
    # refusals — only a MISMATCH or ERROR (a real bug or unclassified failure)
    # trips a nonzero exit; SKIP is expected coverage-gap output.
    return 1 if any(r["outcome"] in ("MISMATCH", "ERROR") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
