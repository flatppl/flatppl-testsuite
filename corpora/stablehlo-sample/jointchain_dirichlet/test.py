"""The issue model: a Dirichlet base, then a Normal whose scale is the first share.

v ~ Dirichlet([2, 3, 5]) and y | v ~ Normal(0, v[1]) (spec §06 jointchain: draw the
base, then the kernel at it). The pivot z = y / v[1] is Normal(0, 1) for every v.
A pivot alone cannot see a misthreaded key (z can be N(0, 1) and still be built from the
bits of v), so the moment check reads u = z^2 v[1]: under the model's independence of z
and v, E[u] = E[v[1]] = 2/10 = 0.2.
"""
import math

PARAMS = {'alpha': [2.0, 3.0, 5.0]}


def stat() -> dict:
    return {"distribution": {"family": "norm", "kwargs": {"loc": 0.0, "scale": 1.0}}, "discrete": False, "moment": {"mean": 0.2}}
