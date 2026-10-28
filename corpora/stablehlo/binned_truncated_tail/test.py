"""Gamma(1, rate) is exponential, so its tail ratio is exp(-rate).

Both unnormalized tails underflow at these cutoffs. The full-support
normalizer leaves a finite bin mean; zero counts have score minus the mean.
"""

import math


def oracle(point):
    return -math.exp(-point["rate"])


def grad_oracle(point):
    return {"rate": math.exp(-point["rate"])}
