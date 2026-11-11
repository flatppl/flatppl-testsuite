"""Independent closed-form reference: the keyword form of `joint`.

Spec §06 `joint(x = M1, w = M2)` is a measure over records with independent
fields x ~ Gamma(a, 1) and w ~ Gamma(3, 1), so w / (x + w) ~ Beta(3, a). Swapped
fields give Beta(a, 3), and one key reused for both fields breaks independence.
"""

PARAMS = {"a": 2.0}


def stat() -> dict:
    return {"distribution": {"family": "beta", "kwargs": {"a": 3.0, "b": PARAMS["a"]}}, "discrete": False}
