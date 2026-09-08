"""§13 fixed-input promotion, checked with the explicit Normal logdensity."""

import math


def oracle(point):
    sigma, mu = point["sigma"], point["mu"]
    return -0.5 * math.log(2 * math.pi) - math.log(sigma) - (1 - mu)**2 / (2 * sigma**2)
