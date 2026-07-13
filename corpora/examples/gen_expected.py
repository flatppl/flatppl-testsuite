#!/usr/bin/env python3
"""Generate ``corpora/examples/<test_id>/expected.json`` from an INDEPENDENT
oracle (scipy / Julia Distributions.jl — never the sibling FlatPPL engine),
one per ``status: "lowers"`` entry in ``corpora/examples/manifest.json``.

Each flatppl-examples model already ends in ``posterior =
bayesupdate(L, prior)`` with no query, so the oracle here must reproduce the
SAME posterior log-density this corpus's manifest constructs — the prior
log-density plus the likelihood log-density, at each point in that entry's
theta grid — exactly like
``corpora/bayesian_inference/gen_expected.py``'s ``oracle_bi_posterior``/
``oracle_eight_schools``, just parameterized over one oracle function per
example instead of writing the model out again.

Stub only (Task 1: scaffold): ``ORACLES`` is empty because
``corpora/examples/manifest.json`` has no ``"lowers"`` entries yet (Task 2
populates the manifest); Task 3 fills in the per-model oracle functions here.
Not on the default test path (``pixi run test`` does not import this
module). Run it manually to verify / regenerate:

    pixi run python corpora/examples/gen_expected.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent

# test_id -> (oracle_fn, list-of-tolerances-per-theta-grid-point). oracle_fn
# takes the manifest entry's theta grid (list[dict]) and returns a
# list[float] of log-densities, one per theta point, in grid order.
ORACLES: dict[str, Callable[[list[dict]], list[float]]] = {}


def gen(test_id: str, model: str, values: list[float],
        tolerance: dict[str, float] | None = None) -> None:
    tolerance = tolerance or {"atol": 1e-9, "rtol": 1e-9}
    doc = {
        "schema_version": 1,
        "test_id": test_id,
        "model": model,
        "reference_backend": "scipy 1.18",
        "checks": [
            {
                "id": f"theta_{i}",
                "kind": "logdensity_value",
                "index": i,
                "binding": "posterior",
                "expected": value,
                "tolerance": tolerance,
            }
            for i, value in enumerate(values)
        ],
    }
    out_dir = HERE / test_id
    out_dir.mkdir(exist_ok=True)
    (out_dir / "expected.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{test_id}: {len(values)} value(s) written")


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    for ex in manifest.get("examples", []):
        if ex["status"] != "lowers":
            continue
        test_id = ex["test_id"]
        oracle_fn = ORACLES[test_id]
        values = oracle_fn(ex["theta"])
        gen(test_id, ex["model"], values)


if __name__ == "__main__":
    main()
