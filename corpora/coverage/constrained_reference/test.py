"""Closed-form score for a Gaussian observation with a constrained Lebesgue prior.

The Lebesgue factor contributes only its support indicator. In particular, the
equal-x points at widths 0.75 and 1.5 must differ only by the Exponential
prior contribution, not by an additional -log(2).
"""
import math


def oracle(point: dict) -> float:
    bound, x = float(point["bound_v"]), float(point["x_v"])
    if x < 0 or x > bound:
        return -math.inf
    return (
        math.log(1.3) - 1.3 * bound
        - math.log(0.6) - math.log(2 * math.pi) / 2
        - ((0.4 - x) / 0.6) ** 2 / 2
    )
