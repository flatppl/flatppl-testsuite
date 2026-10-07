"""Closed-form exponential-tilt normalizer used in two scopes."""

import math


def oracle(point):
    a = point["a"]
    z = math.expm1(a)/a
    density = math.exp(a*0.25)/z
    return a*0.7 - 3*math.log(z) + math.log1p(density) - math.log(2)
