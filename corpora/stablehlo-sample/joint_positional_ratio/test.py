"""Independent closed-form reference: the share of the second component of a
positional `joint` of two Gammas.

Spec §06 `joint(M1, M2)` is the independent product, and its positional variate is
`cat` of the component variates, so `v = [g1, g2]` with g1 ~ Gamma(a, 1) and
g2 ~ Gamma(3, 1) independent. Then g2 / (g1 + g2) ~ Beta(3, a). A positional order
swapped by the lowering gives Beta(a, 3), and two components drawn from one key
are not independent, so the KS check separates both.
"""

PARAMS = {"a": 2.0}


def stat() -> dict:
    return {"distribution": {"family": "beta", "kwargs": {"a": 3.0, "b": PARAMS["a"]}}, "discrete": False}
