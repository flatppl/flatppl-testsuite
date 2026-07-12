#!/usr/bin/env python3
"""Bayesian-inference corpus gate: run every posterior under the det-js
engine and print a `test_id::check -> PASS/SKIP/MISMATCH` table.

Like `corpora/fragment/gate.py` (and unlike HS3's `gate.py`, a parameterized
2DeltaNLL scan vs the ROOT oracle), each posterior is a self-contained model
already ending in a fixed-point `lp = logdensityof(posterior, <point>)`
binding — no theta append, scored directly and compared to the frozen scipy
oracle in its `expected.json` (see `gen_expected.py`).

    pixi run bayesian_inference
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BAYESIAN_INFERENCE_ROOT = Path(__file__).resolve().parent   # corpora/bayesian_inference
REPO = BAYESIAN_INFERENCE_ROOT.parents[1]                    # repo root
sys.path.insert(0, str(REPO / "src"))

os.environ["FLATPPL_ENGINE"] = "det-js"

from flatppl_testsuite.suites.bayesian_inference_gate import PosteriorGateSuite  # noqa: E402

_OUTCOME_OF = {"passed": "PASS", "skipped": "SKIP", "failed": "MISMATCH"}


def render(results) -> str:
    labels = [f"{r.test_id}::{r.check_id}" for r in results]
    width = max((len(label) for label in labels), default=8)
    lines = [
        "=" * 78,
        "BAYESIAN_INFERENCE CORPUS UNDER det-js — determinize+score vs frozen scipy oracle",
        "=" * 78,
        "",
        f"  {'test_id :: check':<{width}}  outcome    detail",
        f"  {'-' * width}  -------    ------",
    ]
    for r, label in zip(results, labels):
        outcome = _OUTCOME_OF.get(r.status, r.status.upper())
        detail = r.message.splitlines()[0] if r.message else ""
        if len(detail) > 90:
            detail = detail[:87] + "..."
        lines.append(f"  {label:<{width}}  {outcome:<7}    {detail}")
    n_pass = sum(1 for r in results if r.status == "passed")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_mismatch = sum(1 for r in results if r.status == "failed")
    lines.append("")
    lines.append(
        f"  {n_pass} PASS, {n_skip} SKIP, {n_mismatch} MISMATCH (of {len(results)} checks)"
    )
    return "\n".join(lines)


def main() -> int:
    results = PosteriorGateSuite().run()
    print(render(results))
    # Coverage report, not a CI trip wire on expected refusals: only a real
    # MISMATCH (a determinized+scored posterior diverging from the frozen
    # oracle) trips a nonzero exit.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
