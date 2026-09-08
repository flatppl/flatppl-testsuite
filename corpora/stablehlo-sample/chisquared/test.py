"""Independent SciPy reference for ChiSquared sampling.

Spec §08 ChiSquared(k) is Gamma(k/2, rate=1/2), matching scipy chi2(df=k).
"""
PARAMS = {"k":4.5}


def stat() -> dict:
    return {
        "distribution": {
            "family": "chi2",
            "kwargs": {"df": PARAMS["k"]},
        },
        "discrete": False,
    }
