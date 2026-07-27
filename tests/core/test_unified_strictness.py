"""An unexpected SKIP must fail the unified harness, not pass silently.

The legacy gates were strict: `corpora/examples/tests/` asserted
`r.status == "passed"` for all 14 examples, and `corpora/hs3/tests/` treated
`status != "passed"` as a failure for every numeric check. The unified harness
replaced both with "no result has status == failed", which quietly makes a
`DETERMINIZE_SKIP` / `CONVERT_SKIP` / `NO_RUNNER` a pass. Under that rule a
determiniser regression that turns 14 posteriors into refusals ships green --
the run is all-skips and reports success.

Strict is therefore the default, and a directory that legitimately cannot be
scored yet opts out explicitly via `"allow_skip": true` in its `test.json`,
which is visible in review rather than invisible in a status field.
"""
from __future__ import annotations

import pytest

from flatppl_testsuite.scoring.result import CheckResult
from tests.test_unified import assert_results_acceptable


def _r(status: str, tag: str = "", cid: str = "c") -> CheckResult:
    return CheckResult("t", cid, status, tag, "detail")


def test_all_passed_is_accepted():
    assert_results_acceptable([_r("passed"), _r("passed")], allow_skip=False)


def test_a_failure_raises():
    with pytest.raises(AssertionError, match="failed"):
        assert_results_acceptable([_r("passed"), _r("failed", "NUMERIC_MISMATCH")],
                                  allow_skip=False)


def test_an_unexpected_skip_raises():
    """The regression this module exists for."""
    with pytest.raises(AssertionError, match="skipped"):
        assert_results_acceptable([_r("skipped", "DETERMINIZE_SKIP")], allow_skip=False)


def test_an_all_skip_run_raises():
    """The specific silent-green scenario: everything refuses."""
    with pytest.raises(AssertionError):
        assert_results_acceptable(
            [_r("skipped", "DETERMINIZE_SKIP", f"c{i}") for i in range(14)],
            allow_skip=False,
        )


def test_a_skip_is_accepted_when_the_dir_opts_in():
    assert_results_acceptable([_r("skipped", "DETERMINIZE_SKIP")], allow_skip=True)


def test_no_runner_is_a_skip_and_still_caught():
    """A `(test_type, engine)` pair with no registered runner must not pass."""
    with pytest.raises(AssertionError):
        assert_results_acceptable([_r("skipped", "NO_RUNNER")], allow_skip=False)


def test_an_empty_result_set_raises():
    with pytest.raises(AssertionError, match="no results"):
        assert_results_acceptable([], allow_skip=False)
