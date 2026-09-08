"""§08 ChiSquared(k), independently scored by SciPy chi2.

For x>0, log f = (k/2-1)*log(x)-x/2-(k/2)*log(2)-lgamma(k/2).
Degrees of freedom are positive reals, not only integers. Negative values
are outside support. The unsettled x=0 policy is deliberately not pinned.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.chi2.logpdf(point["xobs"], point["k"]))
