"""StableHLO numeric-execution gate as pytest.

Emits StableHLO from the local `flatppl` binary, executes it under Enzyme-JAX,
and asserts every fixture's value / gradient / sample-distribution /
rng-independence check matches the frozen scipy oracle (`corpora/stablehlo/`).

Skips cleanly when the executor stack (jax + enzyme_ad, only in the `stablehlo`
pixi env) or a `stablehlo`-capable `flatppl` binary (`FLATPPL_BIN`) is absent —
so the base-env `pixi run test` collects but skips it. Run for real with:

    FLATPPL_BIN=/path/to/target/release/flatppl \\
        pixi run -e stablehlo python -m pytest tests/test_stablehlo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Executor stack lives only in the `stablehlo` env; skip the whole module
# elsewhere (base `pixi run test` has neither jax nor enzyme_ad).
pytest.importorskip("jax", reason="StableHLO gate needs the `stablehlo` pixi env")
pytest.importorskip("enzyme_ad", reason="StableHLO gate needs Enzyme-JAX")

_CORPUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CORPUS))

import executor  # noqa: E402
import gate  # noqa: E402
import oracle  # noqa: E402

pytestmark = pytest.mark.skipif(
    not executor.binary_supports_stablehlo(),
    reason="FLATPPL_BIN must point at a `flatppl` built with the `stablehlo` feature",
)

# Run the whole gate ONCE (the 100k-draw sample checks are the costly part),
# lazily and cached — never at collection time, so a missing binary/executor
# skips (via pytestmark) rather than erroring during collection.
_RESULTS: dict | None = None


def _results() -> dict:
    global _RESULTS
    if _RESULTS is None:
        _RESULTS = {(r.test_id, r.check_id): r for r in gate.run()}
    return _RESULTS


# Static parametrization (no execution at collection): every (fixture, check).
_CHECKS = (
    "logdensity_value", "logdensity_gradient",
    "sample_distribution", "sample_independence",
)
_PARAMS = [(fx.key, chk) for fx in oracle.FIXTURES for chk in _CHECKS]


def test_manifest_spans_required_surface():
    keys = {fx.key for fx in oracle.FIXTURES}
    required = {
        "normal", "gamma", "beta", "studentt", "poisson", "binomial",
        "bernoulli", "lognormal", "exponential", "uniform", "mvnormal", "dirichlet",
    }
    missing = required - keys
    assert not missing, f"gate is missing required distributions: {missing}"
    assert len(keys) >= 12


@pytest.mark.parametrize("test_id,check_id", _PARAMS, ids=lambda v: v)
def test_gate_check(test_id, check_id):
    r = _results().get((test_id, check_id))
    assert r is not None, f"no result for {test_id}::{check_id}"
    assert r.status != "failed", f"{test_id}::{check_id}: {r.message}"


def test_no_mismatches_overall():
    bad = [f"{tid}::{cid}: {r.message}"
           for (tid, cid), r in _results().items() if r.status == "failed"]
    assert not bad, "gate MISMATCHes:\n" + "\n".join(bad)
