"""Independent oracle for cov_censored_lifetimes (scipy).

Un-normalized `truncate(Exponential(rate), interval(0, tau))` has
density indicator * base density (§06 `truncate`, no mass term), so:

    lp(rate, tau) = -inf                        if tau < max(t) = 2.4
                  = sum_i Expon(t_i | rate).logpdf
                  + Gamma(rate | 2, 1).logpdf + Gamma(tau | 4, 1).logpdf

The tau = 2.0 point freezes "-inf" (exact-equality compare, the
trunc_out precedent): it is the point with teeth against the audit
C1-secondary defect, where an unresolvable parametric truncation set
silently read as in-support.
"""
import numpy as np
from scipy import stats

T_OBS = np.array([0.3, 1.1, 0.7, 2.4, 0.2, 1.8])


def oracle(point: dict) -> float:
    rate, tau = float(point["rate"]), float(point["tau"])
    if tau < T_OBS.max():
        return float("-inf")
    lik = stats.expon.logpdf(T_OBS, scale=1.0 / rate).sum()
    prior = stats.gamma.logpdf(rate, a=2.0, scale=1.0) + stats.gamma.logpdf(tau, a=4.0, scale=1.0)
    return float(lik + prior)
