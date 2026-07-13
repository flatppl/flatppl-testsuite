"""Examples corpus numeric gate.

Every entry in `corpora/examples/manifest.json` names a flatppl-examples
model, a binding to query (typically `posterior`), and a theta grid; this
test builds `logdensityof(binding, theta_i)` for each grid point via the
convert-free det-js path (`flatppl determinize` -> `score_flatpdl.cjs`) and
either compares it to a frozen oracle (`status: "lowers"`), asserts the
determinizer refuses it (`status: "refuses"`), or — for a query that DOES
determinize but crashes at the score stage on a named, documented
engine/determiniser gap — asserts just the lowering half and checks the
crash matches the documented one (`status: "unscoreable"`) — see
`flatppl_testsuite.suites.examples_gate` for the full schema and outcome
mapping. All three statuses are represented in `ExamplesGateSuite.run`'s
results by `CheckResult.status`, so `test_example_numeric_check_passes`
below needs no per-status branching: a documented `unscoreable` crash is
`"passed"`, exactly like a matched `"lowers"` oracle or an expected
`"refuses"` refusal — only a regression or an unexpected change reports
`"failed"`.
"""
from __future__ import annotations

import json
import shutil

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.suites.examples_gate import (
    ExamplesGateSuite, EXAMPLES_MANIFEST)


def _flatppl_bin_available() -> bool:
    return CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)) is not None


pytestmark = pytest.mark.skipif(
    not _flatppl_bin_available()
    or not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists()
    or not (CONFIG.examples_dir / "examples").is_dir(),
    reason="requires a determinize-capable flatppl binary, a flatppl-js checkout, "
           "and a flatppl-examples checkout",
)

_MANIFEST = json.loads(EXAMPLES_MANIFEST.read_text())
_TEST_IDS = [ex["test_id"] for ex in _MANIFEST.get("examples", [])]

# The full current manifest roster (7 "lowers", 4 "unscoreable", 3
# "refuses") — a literal set, not derived from `_MANIFEST` itself, so
# `test_all_examples_are_gated` actually guards against a flatppl-examples
# posterior silently dropping out of (or an extra one sneaking into) the
# manifest, rather than trivially checking the manifest against itself.
_EXPECTED_TEST_IDS = {
    "ex_bayesian_inference_1",
    "ex_bayesian_inference_2",
    "ex_best_estimation",
    "ex_capture_recapture",
    "ex_eight_schools",
    "ex_gamma_reparam",
    "ex_hierarchical_logistic",
    "ex_partial_pooling",
    "ex_poisson_glm_link",
    "ex_poisson_model",
    "ex_rasch_1pl",
    "ex_dissimilar_mixture",
    "ex_linear_regression",
    "ex_zero_inflated_binomial",
}


@pytest.mark.parametrize("test_id", _TEST_IDS)
def test_example_numeric_check_passes(test_id):
    results = ExamplesGateSuite().run(selected={test_id})
    assert results, f"no check ran for {test_id}"
    for r in results:
        assert r.status == "passed", f"{r.test_id}::{r.check_id}: {r.status} {r.tag} {r.message}"


def test_all_examples_are_gated():
    """Guard against a flatppl-examples posterior silently dropping out of
    the manifest (or an extra, un-triaged one sneaking in)."""
    if not _TEST_IDS:
        pytest.skip("manifest.json has no examples yet (Task 2 populates it)")
    assert set(_TEST_IDS) == _EXPECTED_TEST_IDS
