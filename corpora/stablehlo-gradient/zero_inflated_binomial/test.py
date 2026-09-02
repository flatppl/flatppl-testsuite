"""Independent analytic gradient oracle for zero_inflated_binomial.

Same model as ``corpora/examples/ex_zero_inflated_binomial`` (that dir scores
the VALUE against scipy; this one scores the DERIVATIVE). The model is copied
because the harness reads ``model.flatppl`` from the test dir itself;
``tests/core/test_corpus_roster.py`` pins that the copies stay byte-identical.

Derived by hand, not differentiated from the value oracle.

* ``Beta(1.5,1.5)`` on both ``p`` and ``psi``:
  ``logpdf = 0.5 log x + 0.5 log(1-x) - log B(1.5,1.5)``, so
  ``d/dx = 0.5/x - 0.5/(1-x)``;
* per observation, ``la = log psi + Binomial.logpmf(y; K, p)`` and
  ``lb = log(1-psi) + Dirac(0).logdensity(y)``, the latter being 0 at ``y == 0``
  and -inf otherwise. With softmax weights ``sa``, ``sb``:
  ``d/dp   = sum_y sa (y/p - (K-y)/(1-p))`` and
  ``d/dpsi = sum_y (sa/psi - sb/(1-psi))``.

Seven of the ten observations are non-zero, so ``lb`` is -inf and ``sb`` is
exactly 0 there; the ``sb/(1-psi)`` term is masked rather than evaluated, since
its limit is 0. That also means those observations pin ``d/dpsi`` at exactly
``1/psi`` each, which is a useful independent sanity check on the softmax
arithmetic: three observations are zero, so
``d/dpsi = 7/psi + sum_{y=0} (sa/psi - sb/(1-psi))``.

The mixture is NOT wrapped in ``normalize`` in this model --- the weights sum to
one by construction --- so there is no total-mass term to differentiate.

Verification, before the ``expected_grad`` vectors were frozen: this oracle
agrees with ``jax.grad`` at x64 of an independent f64 re-implementation of the
log-density to 4.5e-16 worst relative error over all eight points, and with f64
central differences on that formula to 3.4e-09.

``grad_atol`` basis: the worst absolute error of Enzyme's f32 gradient against
these frozen f64 vectors is 5.85e-04, at ``p = 0.95``, where ``d/dp`` reaches
1.9e+03 --- so 3.0e-07 relative, which is f32 round-off. Worst RELATIVE error
over all eight points is 6.1e-07. ``grad_atol`` is 2.5e-03, about a 4x margin.

Why the boundary points are 0.95/0.05/0.01 and not 0.99: the runner's
``grad_atol`` is an ABSOLUTE bound and the f32 error tracks the gradient
magnitude, so a single extreme point loosens the bound for every other point.
Measured: ``p = 0.99`` makes ``d/dp`` 9.8e+03 and its absolute error 1.04e-02,
which would force ``grad_atol`` about 20x looser and stop the interior points
from gating anything. ``psi = 0.99`` does the same (``d/dpsi`` 3.4e+02, error
3.46e-04). The 0.95 and 0.01 points probe the same region at a bound that still
bites. A relative tolerance alongside the absolute one would remove the
trade-off; that is a harness change, not made here.
"""
import numpy as np

_K = 20
_Y = np.array([7, 0, 5, 8, 0, 6, 4, 0, 9, 3], dtype=float)


def grad_oracle(point: dict) -> dict:
    p = float(point["p"])
    psi = float(point["psi"])

    # Beta(1.5, 1.5) priors.
    dp = 0.5 / p - 0.5 / (1.0 - p)
    dpsi = 0.5 / psi - 0.5 / (1.0 - psi)

    from scipy.special import gammaln

    log_choose = (gammaln(_K + 1.0) - gammaln(_Y + 1.0)
                  - gammaln(_K - _Y + 1.0))
    log_binom = log_choose + _Y * np.log(p) + (_K - _Y) * np.log1p(-p)

    is_zero = _Y == 0.0
    la = np.log(psi) + log_binom
    lb = np.where(is_zero, np.log1p(-psi), -np.inf)

    m = np.maximum(la, lb)
    ea = np.exp(la - m)
    eb = np.where(np.isfinite(lb), np.exp(lb - m), 0.0)
    tot = ea + eb
    sa, sb = ea / tot, eb / tot

    dp += float((sa * (_Y / p - (_K - _Y) / (1.0 - p))).sum())
    # sb == 0 at every non-zero observation; the limit of the term is 0.
    dpsi += float((sa / psi).sum())
    dpsi -= float(np.where(sb == 0.0, 0.0, sb / (1.0 - psi)).sum())

    return {"p": dp, "psi": dpsi}
