"""Independent analytic gradient oracle for dissimilar_mixture.

Same model as ``corpora/examples/ex_dissimilar_mixture`` (that dir scores the
VALUE against scipy; this one scores the DERIVATIVE). The model is copied
because the harness reads ``model.flatppl`` from the test dir itself;
``tests/core/test_corpus_roster.py`` pins that the copies stay byte-identical.

Derived by hand, not differentiated from the value oracle. The ABI parameter is
``sigma``, not ``sigma2``: the model draws ``sigma2 ~ InverseGamma(2,2)`` and
binds ``sigma = sqrt(sigma2)``, so the density carries the ``pushfwd``
log-volume ``log(2*sigma)`` (see the value oracle's docstring). Priors:

* ``Beta(2,2)``: ``logpdf = log 6 + log p + log(1-p)``, so
  ``d/dp = 1/p - 1/(1-p)``;
* ``Normal(0,1)`` on ``mu``: ``d/dmu = -mu``;
* the reparametrised ``sigma``: with ``x = sigma^2``, ``a = 2``, ``scale = 2``,
  ``InverseGamma.logpdf = a log(scale) - lgamma(a) - (a+1) log x - scale/x``, so
  ``d/dx = -(a+1)/x + scale/x^2`` and ``dx/dsigma = 2 sigma``; the ``log(2*sigma)``
  volume term adds ``1/sigma``;
* half-``Normal(0,5)`` on ``shape`` and on ``rate``: ``d/dtheta = -theta/25``
  (the ``log 2`` normaliser is constant).

Per observation the mixture is contracted in LOG space via its softmax weights
``sa`` (normal mixand) and ``sb`` (gamma mixand), with
``la = log p + Normal.logpdf(y; mu, sigma)`` and
``lb = log(1-p) + Gamma.logpdf(y; shape, rate)``:

* ``d/dp   = sum_y (sa/p - sb/(1-p))``
* ``d/dmu  = sum_y sa (y-mu)/sigma^2``
* ``d/dsigma += sum_y sa (-1/sigma + (y-mu)^2/sigma^3)``
* ``d/dshape = sum_y sb (log rate - digamma(shape) + log y)``
* ``d/drate  = sum_y sb (shape/rate - y)``

The log-space form is not cosmetic here. Four of the twenty observations are
NEGATIVE (-0.33, -0.22, -0.75, -0.76), where the Gamma mixand has log-density
-inf, so ``sb`` is exactly 0 and ``log y`` is nan. Those terms are masked on
``sb == 0`` rather than evaluated, since the limit is 0. A density-space
oracle would produce nan for ``d/dshape`` at every such observation.

Verification, before the ``expected_grad`` vectors were frozen: this oracle
agrees with ``jax.grad`` at x64 of an independent f64 re-implementation of the
log-density to 6.4e-15 worst relative error over all eight points, and with f64
central differences on that formula to 8.6e-09.

``grad_atol`` basis: the worst absolute error of Enzyme's f32 gradient against
these frozen f64 vectors is 5.65e-05, at point 3 (``p = 0.95``) in ``d/dp``,
where the gradient reaches -1.54e+02 --- so 3.7e-07 relative, which is f32
round-off. Worst RELATIVE error over all eight points is 5.8e-06, at point 1 in
``d/dshape``, where the gradient is only -0.44 and a round-off-sized absolute
error therefore reads large relatively. ``grad_atol`` is an ABSOLUTE bound, so
it has to cover the largest gradient in the set rather than the largest
relative slip: it is 2.5e-04, about a 4x margin over the measured worst, the
same margin the two sibling cases use. No point has a non-finite component.

This case was authored earlier and could not be shipped: Enzyme 0.0.14 did not
terminate differentiating the emitted module with respect to ``rate``, killed
at 4 and at 25 minutes, while the other four parameters were fine. That was
never the ``stablehlo.power`` in the Gamma normaliser --- the op is still there,
exactly one of it. It was the two-element ``logsumexp``, which the emitter used
to lower as a ``concatenate`` plus a ``reduce ... maximum``; the argmax the
adjoint of that reduce has to recover is what did not terminate in combination
with the ``power`` adjoint. Once ``lower_logsumexp`` emits the scalar
``logaddexp`` form instead, every parameter including ``rate`` differentiates in
about 0.1 s.
"""
import numpy as np
from scipy.special import digamma

_Y = np.array([
    7.23, 5.13, 1.20, -0.33, 0.23,
    -0.22, 1.34, 0.80, 0.50, -0.75,
    3.79, 0.01, -0.76, 0.21, 1.48,
    1.21, 12.11, 15.96, 9.83, 3.92,
])
_IG_A = 2.0
_IG_SCALE = 2.0
_HN_SIGMA = 5.0


def grad_oracle(point: dict) -> dict:
    p = float(point["p"])
    mu = float(point["mu"])
    sigma = float(point["sigma"])
    shape = float(point["shape"])
    rate = float(point["rate"])

    # Priors.
    dp = 1.0 / p - 1.0 / (1.0 - p)
    dmu = -mu
    x = sigma**2
    dsigma = (-(_IG_A + 1.0) / x + _IG_SCALE / x**2) * 2.0 * sigma + 1.0 / sigma
    dshape = -shape / _HN_SIGMA**2
    drate = -rate / _HN_SIGMA**2

    # Mixture, in log space.
    pos = _Y > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.where(pos, np.log(np.abs(_Y)), 0.0)

    ln = (-np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
          - 0.5 * ((_Y - mu) / sigma) ** 2)
    lg = np.where(
        pos,
        (shape * np.log(rate) - _lgamma(shape) + (shape - 1.0) * log_y
         - rate * np.where(pos, _Y, 0.0)),
        -np.inf,
    )
    la = np.log(p) + ln
    lb = np.log1p(-p) + lg

    m = np.maximum(la, lb)
    ea = np.exp(la - m)
    eb = np.where(np.isfinite(lb), np.exp(lb - m), 0.0)
    tot = ea + eb
    sa, sb = ea / tot, eb / tot

    dp += float((sa / p - sb / (1.0 - p)).sum())
    dmu += float((sa * (_Y - mu) / sigma**2).sum())
    dsigma += float((sa * (-1.0 / sigma + (_Y - mu) ** 2 / sigma**3)).sum())
    # sb == 0 at the negative observations, where `log y` is undefined; the
    # limit of the term is 0, so mask instead of evaluating it.
    dshape += float(np.where(sb == 0.0, 0.0,
                             sb * (np.log(rate) - digamma(shape) + log_y)).sum())
    drate += float(np.where(sb == 0.0, 0.0, sb * (shape / rate - _Y)).sum())

    return {"p": dp, "mu": dmu, "sigma": dsigma,
            "shape": dshape, "rate": drate}


def _lgamma(v):
    from scipy.special import gammaln

    return float(gammaln(v))
