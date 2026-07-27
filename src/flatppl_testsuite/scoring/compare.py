"""Pointwise vector comparison for twice_delta_nll checks, and a scalar
counterpart for fragment corpus logdensity_value checks."""

from __future__ import annotations

import math


def compare_scalar(actual: float, expected: float,
                   tolerance: dict[str, float]) -> None:
    """|actual - expected| <= atol + rtol * |expected|; raise on mismatch.

    An infinite `expected` (a truncation gate scored out of support) has no
    finite tolerance band, so it is compared with exact `==` instead.
    """
    if math.isnan(actual) or math.isnan(expected):
        raise AssertionError(f"got {actual!r}, expected {expected!r} (NaN never matches)")
    if math.isinf(expected):
        if actual != expected:
            raise AssertionError(f"got {actual!r}, expected {expected!r}")
        return
    atol = tolerance["atol"]
    rtol = tolerance["rtol"]
    if abs(actual - expected) > atol + rtol * abs(expected):
        raise AssertionError(
            f"got {actual!r}, expected {expected!r} "
            f"(diff={abs(actual - expected)!r}, tol={atol + rtol * abs(expected)!r})"
        )


def compare_vectors(actual: list[float], expected: list[float],
                    tolerance: dict[str, float]) -> None:
    """Pointwise |actual - expected| <= atol + rtol * |expected|; raise on mismatch."""
    if len(actual) != len(expected):
        raise AssertionError(
            f"length mismatch: got {len(actual)}, expected {len(expected)}"
        )
    atol = tolerance["atol"]
    rtol = tolerance["rtol"]
    for i, (got, want) in enumerate(zip(actual, expected)):
        if abs(got - want) > atol + rtol * abs(want):
            raise AssertionError(
                f"index {i}: got {got!r}, expected {want!r} "
                f"(diff={abs(got - want)!r}, tol={atol + rtol * abs(want)!r})"
            )
