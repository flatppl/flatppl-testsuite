"""Independent SciPy reference for GeneralizedNormal sampling.

Spec §08 density beta/(2*alpha*Gamma(1/beta))*exp(-abs((x-mean)/alpha)**beta)
matches scipy gennorm(beta=beta, loc=mean, scale=alpha). Beta 1.5 exercises
neither the Normal nor the Laplace special case.
"""
PARAMS = {"mean":0.3,"alpha":1.2,"beta":1.5}


def stat() -> dict:
    return {
        "distribution": {
            "family": "gennorm",
            "kwargs": {"beta": PARAMS["beta"], "loc": PARAMS["mean"], "scale": PARAMS["alpha"]},
        },
        "discrete": False,
    }
