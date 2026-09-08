"""Numeric vector gates preserve finite tolerances and support infinities."""

import math

import pytest

from flatppl_testsuite.scoring.compare import compare_vectors


@pytest.mark.parametrize("actual,expected", [
    (math.nan, 1.0),
    (1.0, math.nan),
    (1.0, -math.inf),
    (math.inf, -math.inf),
])
def test_vector_nonfinite_mismatch_fails(actual, expected):
    with pytest.raises(AssertionError):
        compare_vectors([actual], [expected], {"atol": 1e-9, "rtol": 1e-9})


def test_vector_support_infinities_and_finite_tolerance():
    compare_vectors(
        [-math.inf, math.inf, 2.000001], [-math.inf, math.inf, 2.0],
        {"atol": 1e-9, "rtol": 1e-6},
    )
    with pytest.raises(AssertionError):
        compare_vectors([2.01], [2.0], {"atol": 1e-9, "rtol": 1e-6})
