#!/usr/bin/env python3
"""Examples corpus gate: run every flatppl-examples posterior query under the
det-js engine and print a `test_id::check -> LOWERS/UNSCOREABLE/REFUSE/MISMATCH`
table.

Like `corpora/bayesian_inference/gate.py`, each manifest entry is scored via
`DetJsScoreEngine.log_density` (append `__score__ = logdensityof(binding,
theta)`, determinize, score) — but unlike bayesian_inference/fragment, the
flatppl-examples models carry no fixed-point query themselves (they end in
`posterior = bayesupdate(L, prior)`), so the query's theta grid comes from
this corpus's own manifest, and each entry additionally records whether the
determinizer is expected to lower the query (`status: "lowers"`, checked
against a frozen oracle), refuse it (`status: "refuses"`, checked against a
required substring of the refusal message), or lower it but crash at the
score stage on a named, documented engine/determiniser gap (`status:
"unscoreable"`, checked against a required substring of the crash message —
see the manifest's `category`/`note` for the human-readable gap). See
`flatppl_testsuite.suites.examples_gate` for the full schema and outcome
mapping, and `corpora/examples/README.md` for the corpus's scope.

    pixi run examples
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


def _outcome(r, status_of: dict[str, str], category_of: dict[str, str]) -> str:
    if r.status != "passed":
        return _OUTCOME_OF.get(r.status, r.status.upper())
    # A "passed" lowers entry means it lowered+matched the oracle; a "passed"
    # refuses entry means it refused as expected; a "passed" unscoreable
    # entry means it lowered and then crashed at the score stage exactly as
    # documented — distinguish all three for the reader rather than
    # collapsing them to a bare PASS.
    manifest_status = status_of.get(r.test_id)
    if manifest_status == "lowers":
        return "LOWERS"
    if manifest_status == "unscoreable":
        gap = category_of.get(r.test_id, "gap")
        return f"LOWERS(unscoreable: {gap})"
    return "REFUSE"


def render(results, status_of: dict[str, str], category_of: dict[str, str]) -> str:
    labels = [f"{r.test_id}::{r.check_id}" for r in results]
    width = max((len(label) for label in labels), default=8)
    outcomes = [_outcome(r, status_of, category_of) for r in results]
    outcome_width = max((len(o) for o in outcomes), default=7)
    lines = [
        "=" * 78,
        "EXAMPLES CORPUS UNDER det-js — determinize+score a constructed posterior query",
        "=" * 78,
        "",
        f"  {'test_id :: check':<{width}}  {'outcome':<{outcome_width}}  detail",
        f"  {'-' * width}  {'-' * outcome_width}  ------",
    ]
    for r, label, outcome in zip(results, labels, outcomes):
        detail = r.message.splitlines()[0] if r.message else ""
        if len(detail) > 90:
            detail = detail[:87] + "..."
        lines.append(f"  {label:<{width}}  {outcome:<{outcome_width}}  {detail}")

    n_lowers = sum(1 for r in results if r.status == "passed" and status_of.get(r.test_id) == "lowers")
    n_unscoreable = sum(1 for r in results if r.status == "passed" and status_of.get(r.test_id) == "unscoreable")
    n_refuse = sum(1 for r in results if r.status == "passed" and status_of.get(r.test_id) == "refuses")
    n_mismatch = sum(1 for r in results if r.status == "failed")
    n_skip = sum(1 for r in results if r.status == "skipped")
    lines.append("")
    lines.append(
        f"  {n_lowers} LOWERS, {n_unscoreable} UNSCOREABLE, {n_refuse} REFUSE, "
        f"{n_skip} SKIP, {n_mismatch} MISMATCH (of {len(results)} checks)"
    )
    return "\n".join(lines)


def main() -> int:
    manifest = json.loads(EXAMPLES_MANIFEST.read_text())
    examples = manifest.get("examples", [])
    status_of = {ex["test_id"]: ex["status"] for ex in examples}
    category_of = {ex["test_id"]: ex["category"] for ex in examples if ex.get("category")}

    results = ExamplesGateSuite().run()
    print(render(results, status_of, category_of))
    # Coverage report, not a CI trip wire on expected refusals or documented
    # unscoreable entries: only a real MISMATCH trips a nonzero exit — that
    # covers a "lowers"/"refuses" regression AND an "unscoreable" entry that
    # regresses (DeterminizeRefused) or improves (scores cleanly), both of
    # which examples_gate.py tags "failed" precisely so they surface here.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
