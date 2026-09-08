"""§08 VonMises(mu, kappa), independently scored by SciPy.

The specified periodic log density is kappa*cos(x-mu)-log(2*pi)-log(I0(kappa)).
SciPy evaluates the Bessel normalizer stably; positive and negative period
offsets exercise periodicity outside the canonical fundamental interval.
This fixture makes no normalization claim over the whole real line.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.vonmises.logpdf(
        point["xobs"], point["kappa"], loc=point["mu"],
    ))
