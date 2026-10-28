"""Restriction discards the singular region before numerical integration.

The retained mass is integral_.5^1 a/x dx = a log(2). The first bin is
empty; at a=0 the zero-count score still has derivative -log(2).
"""

import math


def oracle(point):
    return -point["a"] * math.log(2)


def grad_oracle(point):
    return {"a": -math.log(2)}
