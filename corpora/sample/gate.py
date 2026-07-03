#!/usr/bin/env python3
"""Sample corpus gate: seed-sweep the hierarchical-Normal sample-path model
under the det-js engine and print a
``test_id::check -> PASS/SKIP/MISMATCH`` table.

Unlike the fragment corpus's gate (one frozen scalar per model), the sample
corpus's model ends in ``rand(rng, lawof(record(...)))`` — a single seed
gives ONE random realization, not a reproducible scalar — so this gate
sweeps N seeds (see ``scoring/sample_sweep.cjs``) and compares the empirical
mean/var/cov to the model's closed-form structural moments
(``corpora/sample/oracle.py`` / ``gen_expected.py``).

    pixi run sample
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SAMPLE_ROOT = Path(__file__).resolve().parent      # corpora/sample
REPO = SAMPLE_ROOT.parents[1]                      # repo root
sys.path.insert(0, str(REPO / "src"))

os.environ["FLATPPL_ENGINE"] = "det-js"

from flatppl_testsuite.suites.sample_gate import SampleGateSuite  # noqa: E402

_OUTCOME_OF = {"passed": "PASS", "skipped": "SKIP", "failed": "MISMATCH"}


def render(results) -> str:
    labels = [f"{r.test_id}::{r.check_id}" for r in results]
    width = max((len(label) for label in labels), default=8)
    lines = [
        "=" * 78,
        "SAMPLE CORPUS UNDER det-js — seed-sweep vs closed-form oracle (shared-ancestor cov catch)",
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
    results = SampleGateSuite().run()
    print(render(results))
    # Coverage report, not a CI trip wire on expected refusals: only a real
    # MISMATCH (a seed-swept realization set diverging from the closed-form
    # oracle) trips a nonzero exit.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
