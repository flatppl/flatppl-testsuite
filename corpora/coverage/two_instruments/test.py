"""Independent oracle for cov_two_instruments (closed form).

joint_likelihood sums the two instrument log-likelihoods (§06
"Combining likelihoods") and bayesupdate adds the prior term:

    lp(mu) = Normal(1.5 | mu, 1).logpdf
           + Normal(3.2 | 2 mu, 0.5).logpdf
           + Normal(mu | 0, 2).logpdf

Instrument B's mean is the DERIVED expression `2 mu` — the audit
H3/H5 class — and the prior is not the law of a like-named draw (the
audit H1 off-idiom shape); both must score, post-CLM.
"""
from scipy import stats

OBS_A, OBS_B = 1.5, 3.2


def oracle(point: dict) -> float:
    mu = float(point["mu"])
    return float(
        stats.norm.logpdf(OBS_A, mu, 1.0)
        + stats.norm.logpdf(OBS_B, 2.0 * mu, 0.5)
        + stats.norm.logpdf(mu, 0.0, 2.0)
    )
