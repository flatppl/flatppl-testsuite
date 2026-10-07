"""Differentiate the closed-form Gaussian product integral, including bounds."""

import math


def grad_oracle(point):
    lo, hi, x, y = (point[name] for name in ("lo", "hi", "px", "py"))
    mid = (x+y)/2
    mass = (math.erf(mid-lo)-math.erf(mid-hi))/2
    lower = math.exp(-(mid-lo)**2)/(math.sqrt(math.pi)*mass)
    upper = math.exp(-(mid-hi)**2)/(math.sqrt(math.pi)*mass)
    shared = (lower-upper)/2
    return {"lo": 1/(hi-lo)-lower, "hi": upper-1/(hi-lo),
            "px": -(x-y)/2+shared, "py": (x-y)/2+shared}
