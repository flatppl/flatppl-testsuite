"""Tests for the runner module (Task 5).

compare_vectors: pointwise tolerance check.
run: fixture dispatch — rf101_basics passes end-to-end; rf207_comptools skips.
run + oracles: oracle annotation attached to passing NLL check without altering pass/fail.
"""

import pytest

from flatppl_testsuite.scoring.compare import compare_vectors
from flatppl_testsuite.scoring.result import CheckResult
from flatppl_testsuite.runner import run


def test_compare_vectors_within_tol():
    compare_vectors([1.0, 2.0], [1.0, 2.0], {"atol": 1e-7, "rtol": 1e-8})


def test_compare_vectors_mismatch():
    with pytest.raises(AssertionError):
        compare_vectors([1.0], [2.0], {"atol": 1e-7, "rtol": 1e-8})


def test_run_rf101_passes():
    results = run(selected={"rf101_basics"})
    nll = [r for r in results if r.check_id == "twice_delta_nll_scan"]
    assert nll and nll[0].status == "passed"


def test_run_unimplemented_skips(monkeypatch):
    # run() maps a SkipUnimplemented (from scoring) to a CONVERT_SKIP result.
    # Force the skip via the scoring path so no unimplemented fixture is needed.
    from flatppl_testsuite.suites import hs3_import
    from flatppl_testsuite.formats.hs3.importer import SkipUnimplemented

    def _skip(*_a, **_k):
        raise SkipUnimplemented(hs3_type="chebychev_dist")

    monkeypatch.setattr(hs3_import, "score_scan", _skip)
    results = run(selected={"rf101_basics"})
    skips = [r for r in results if r.status == "skipped" and r.tag == "CONVERT_SKIP"]
    assert skips and skips[0].message == "chebychev_dist"


def test_run_rf101_oracle_annotation():
    """Oracle annotation must not flip a passing check to failed; message notes the oracle."""
    results = run(selected={"rf101_basics"}, oracles=("roofit",))
    nll = [r for r in results if r.check_id == "twice_delta_nll_scan"]
    assert nll, "no twice_delta_nll_scan result for rf101_basics"
    r = nll[0]
    # Pass/fail must not be changed by oracle plumbing.
    assert r.status == "passed"
    # Whether oracle env is live or absent, the message must mention "oracle[roofit]".
    assert "oracle[roofit]" in r.message
