"""Exact support indicators for two constant-density affine priors.

Decimal coordinates are interpreted as rationals. This independently verifies
their intended constraints without copying an engine's floating-point tolerance.
In particular, [0.1, 0.2, -0.3] is a valid coordinate-chart image even though
its floating-point sum can round away from zero. Tiny same-sign coordinates
are off the zero-sum plane regardless of their absolute magnitude.
"""
from fractions import Fraction
import math


def oracle(point: dict) -> float:
    effects = [Fraction(str(value)) for value in point["effects_v"]]
    weights = [Fraction(str(value)) for value in point["weights_v"]]
    if sum(effects) != 0 or sum(weights) != 1 or any(value < 0 for value in weights):
        return -math.inf
    return 0.0
