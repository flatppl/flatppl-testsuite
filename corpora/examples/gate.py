#!/usr/bin/env python3
"""Examples corpus gate: run every flatppl-examples posterior query under the
det-js engine and print a `test_id::check -> LOWERS/REFUSE/MISMATCH` table.

Like `corpora/bayesian_inference/gate.py`, each manifest entry is scored via
`DetJsScoreEngine.log_density` (append `__score__ = logdensityof(binding,
theta)`, determinize, score) — but unlike bayesian_inference/fragment, the
flatppl-examples models carry no fixed-point query themselves (they end in
`posterior = bayesupdate(L, prior)`), so the query's theta grid comes from
this corpus's own manifest, and each entry additionally records whether the
determinizer is expected to lower the query (`status: "lowers"`, checked
against a frozen oracle) or refuse it (`status: "refuses"`, checked against a
required substring of the refusal message). See
`flatppl_testsuite.suites.examples_gate` for the full schema and outcome
mapping, and `corpora/examples/README.md` for the corpus's scope.

    pixi run examples

The manifest starts empty (Task 1: scaffold only) — Task 2 populates it, at
which point this prints a real table instead of the empty one.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent   # corpora/examples
REPO = EXAMPLES_ROOT.parents[1]                    # repo root
sys.path.insert(0, str(REPO / "src"))

os.environ["FLATPPL_ENGINE"] = "det-js"

from flatppl_testsuite.suites.examples_gate import (  # noqa: E402
    ExamplesGateSuite, EXAMPLES_MANIFEST)

_OUTCOME_OF = {"passed": "?", "skipped": "SKIP", "failed": "MISMATCH"}


def render(results, status_of: dict[str, str]) -> str:
    labels = [f"{r.test_id}::{r.check_id}" for r in results]
    width = max((len(label) for label in labels), default=8)
    lines = [
        "=" * 78,
        "EXAMPLES CORPUS UNDER det-js — determinize+score a constructed posterior query",
        "=" * 78,
        "",
        f"  {'test_id :: check':<{width}}  outcome    detail",
        f"  {'-' * width}  -------    ------",
    ]
    for r, label in zip(results, labels):
        if r.status == "passed":
            # A "passed" lowers entry means it lowered+matched; a "passed"
            # refuses entry means it refused as expected — distinguish them
            # for the reader rather than collapsing both to a bare PASS.
            outcome = "LOWERS" if status_of.get(r.test_id) == "lowers" else "REFUSE"
        else:
            outcome = _OUTCOME_OF.get(r.status, r.status.upper())
        detail = r.message.splitlines()[0] if r.message else ""
        if len(detail) > 90:
            detail = detail[:87] + "..."
        lines.append(f"  {label:<{width}}  {outcome:<7}    {detail}")
    n_lowers = sum(1 for r in results if r.status == "passed" and status_of.get(r.test_id) == "lowers")
    n_refuse = sum(1 for r in results if r.status == "passed" and status_of.get(r.test_id) == "refuses")
    n_mismatch = sum(1 for r in results if r.status == "failed")
    n_skip = sum(1 for r in results if r.status == "skipped")
    lines.append("")
    lines.append(
        f"  {n_lowers} LOWERS, {n_refuse} REFUSE, {n_skip} SKIP, "
        f"{n_mismatch} MISMATCH (of {len(results)} checks)"
    )
    return "\n".join(lines)


def main() -> int:
    manifest = json.loads(EXAMPLES_MANIFEST.read_text())
    status_of = {ex["test_id"]: ex["status"] for ex in manifest.get("examples", [])}

    results = ExamplesGateSuite().run()
    print(render(results, status_of))
    # Coverage report, not a CI trip wire on expected refusals: only a real
    # MISMATCH (a regression against status "lowers"/"refuses") trips a
    # nonzero exit.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
