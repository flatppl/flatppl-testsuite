"""Pointwise vector comparison for twice_delta_nll checks."""

from __future__ import annotations


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
