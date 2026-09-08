"""§08 GeneralizedNormal(mean, alpha, beta), independently scored by SciPy.

The log density is log(beta) - log(2*alpha) - lgamma(1/beta)
- (abs(x-mean)/alpha)**beta. SciPy's gennorm shape is beta, loc is mean,
and scale is alpha. Points vary all parameters and straddle beta=1 and 2.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.gennorm.logpdf(
        point["xobs"], point["beta"], loc=point["mean"], scale=point["alpha"],
    ))
