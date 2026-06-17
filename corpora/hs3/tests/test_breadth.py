"""Breadth run over all HS3TestSuite fixtures — coverage invariant (Task 7).

Runs the harness over the full 19-fixture suite and asserts that every result
is either:
  - passed, OR
  - skipped with tag="CONVERT_SKIP" and a non-empty message naming the HS3 type, OR
  - failed with a non-empty stage tag (UNSCOREABLE or NUMERIC_MISMATCH).

The key invariant: no result is failed with an empty tag — no silently unclassified
failure exits the runner.
"""

from flatppl_testsuite.runner import run


def test_breadth_coverage_invariant():
    results = run()

    # Coverage report must not be empty.
    assert results, "run() returned no results — manifest empty or fixture directory missing"

    untagged_failures = [
        r for r in results if r.status == "failed" and not r.tag
    ]
    assert not untagged_failures, (
        f"Untagged failures detected (runner bug — every failure must carry a stage tag):\n"
        + "\n".join(f"  {r.test_id}::{r.check_id}: {r.message}" for r in untagged_failures)
    )

    # Every CONVERT_SKIP must name the unimplemented HS3 type (non-empty message).
    unnamed_skips = [
        r for r in results if r.tag == "CONVERT_SKIP" and not r.message
    ]
    assert not unnamed_skips, (
        f"CONVERT_SKIP results with no type name:\n"
        + "\n".join(f"  {r.test_id}::{r.check_id}" for r in unnamed_skips)
    )
