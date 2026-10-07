"""Integrate the Gaussian product over the uniform latent in closed form."""

import math


def oracle(point):
    lo, hi, x, y = (point[name] for name in ("lo", "hi", "px", "py"))
    mid = (x+y)/2
    mass = (math.erf(mid-lo)-math.erf(mid-hi))/2
    return -(x-y)**2/4 - math.log(2*math.sqrt(math.pi)) + math.log(mass) - math.log(hi-lo)
