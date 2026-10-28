"""A half-normal tail above 1 has mass erfc(1/(s sqrt(2))).

The Normal mass on [0, infinity] is 1/2 for every s, so normalization
divides the upper tail by 1/2. A zero count has score minus that mass.
"""

import math


def oracle(point):
    return -math.erfc(1 / (point["s"] * math.sqrt(2)))


def grad_oracle(point):
    s = point["s"]
    phi = math.exp(-0.5 / s**2) / math.sqrt(2 * math.pi)
    return {"s": -2 * phi / s**2}
