"""Independent oracle for ex_rasch_1pl.

Rasch 1PL: ``theta ~ iid(Normal(0, 1.5), 4)`` (abilities), ``b ~ iid(Normal(0,
1.5), 5)`` (difficulties); ``prob_k = invlogit(theta[person_k] - b[item_k])``,
``y_k ~ Bernoulli(prob_k)`` against the fixed long-format placeholder
responses (``person``/``item`` are 1-based).
"""
import math

from scipy.stats import bernoulli, norm

_PERSON = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
_ITEM = [1, 2, 3, 4, 5] * 4
_Y = [
    True, True, False, False, False,
    True, True, True, False, False,
    True, False, True, True, False,
    True, True, True, True, True,
]


def _invlogit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def oracle(point: dict) -> float:
    theta = point["theta"]
    b = point["b"]
    lp = sum(norm.logpdf(tp, 0.0, 1.5) for tp in theta) + sum(
        norm.logpdf(bi, 0.0, 1.5) for bi in b
    )
    for p, i, y in zip(_PERSON, _ITEM, _Y):
        eta = theta[p - 1] - b[i - 1]
        lp += bernoulli.logpmf(int(y), _invlogit(eta))
    return float(lp)
