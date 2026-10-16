"""Normal( a + 2*b, 2 ) at 15, derived directly from its density."""

import math


def oracle(point):
    residual = (15 - point["a"] - 2 * point["b"]) / 2
    return -0.5 * math.log(2 * math.pi) - math.log(2) - 0.5 * residual**2
