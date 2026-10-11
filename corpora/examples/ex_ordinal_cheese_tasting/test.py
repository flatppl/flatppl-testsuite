"""Independent oracle for ex_ordinal_cheese_tasting.

Log-posterior ``logdensityof(Pi_post, point)``: the log-likelihood below plus
the normalized prior of eight sorted ``Normal(0, 5)`` draws on the cut points
and independent ``Normal(0, 5)`` priors on the three locations.

Cumulative-logit ordinal regression on McCullagh's cheese-tasting counts:
``P(Y <= j | a) = invlogit(cuts[j] - eta[a])`` for ``j = 1..8`` with
``eta = (effect[1], effect[2], effect[3], 0)``; each additive's nine rating
counts are one ``Multinomial(52, p_a)`` draw, where ``p_a`` is the difference of
neighbouring cumulative probabilities padded with 0 and 1.
"""
import math

from scipy.special import expit
from scipy.stats import multinomial, norm

_COUNTS = [
    [0, 0, 1, 7, 8, 8, 19, 8, 1],
    [6, 9, 12, 11, 7, 6, 1, 0, 0],
    [1, 1, 6, 8, 23, 7, 5, 1, 0],
    [0, 0, 0, 1, 3, 7, 14, 16, 11],
]


def oracle(point: dict) -> float:
    cuts = point["cuts"]
    eta = list(point["effect"]) + [0.0]
    lp = 0.0
    for counts, e in zip(_COUNTS, eta):
        cumulative = [0.0] + [float(expit(c - e)) for c in cuts] + [1.0]
        p = [cumulative[j + 1] - cumulative[j] for j in range(9)]
        lp += multinomial.logpmf(counts, n=52, p=p)
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        return -math.inf
    lp += sum(norm.logpdf(c, 0.0, 5.0) for c in cuts)
    lp += math.lgamma(len(cuts) + 1)
    lp += sum(norm.logpdf(e, 0.0, 5.0) for e in point["effect"])
    return float(lp)
