"""Zero counts have log probability minus the sum of the bin masses.

The bins partition [0, 1]; integrating a^2 exp(x) gives a^2 (e - 1).
"""

import math


def oracle(point):
    return -(point["a"] ** 2) * math.expm1(1.0)


def grad_oracle(point):
    return {"a": -2 * point["a"] * math.expm1(1.0)}
