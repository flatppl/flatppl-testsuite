"""Zero counts have log probability minus the sum of the bin masses.

The bins partition [0, 1]; integrating a exp(x) gives a (e - 1).
"""

import math


def oracle(point):
    return -point["a"] * math.expm1(1.0)


def grad_oracle(point):
    return {"a": -math.expm1(1.0)}
