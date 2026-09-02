"""Independent analytic gradient oracle for signal-background-counting.

Same model as ``corpora/examples/ex_signal_background_counting`` (that dir
scores the VALUE against scipy; this one scores the DERIVATIVE). The model is
copied rather than shared because the harness reads ``model.flatppl`` from the
test dir itself; ``tests/core/test_corpus_roster.py`` pins that the two copies
stay byte-identical.

The gradient is derived by hand, not differentiated from the value oracle, so
it is independent of any autodiff implementation. Writing ``e`` for exposure,
``f`` for efficiency, ``k`` for the observed counts, ``a = e*B`` (nu_B),
``b = e*f*S`` (nu_S), ``nu = a + b``, ``w = a/nu``, ``s = sigma_B`` and
``mu = log(m_B) - s^2/2``:

* the four uniform priors are constant on their support, so they contribute
  nothing (every authored point is inside the support);
* LogNormal prior, per component, with ``z = log B``:
  ``d/dB = -1/B - (z - mu)/(s^2 B)``,
  ``d/dm_B = (1/m_B) * sum_j (z_j - mu)/s^2``,
  ``d/ds = sum_j [-1/s + (z_j - mu)^2/s^3] - s * sum_j (z_j - mu)/s^2``
  (the second term is the chain rule through ``mu``, whose ``d mu/ds = -s``);
* Poisson: ``d/dnu = k/nu - 1``, and ``dnu/dB = e``, ``dnu/dS = e*f``;
* the per-event two-component mixture is differentiated in LOG space via its
  softmax weights, which is what keeps this oracle valid where the model's own
  arithmetic is not. With ``la = log w + log g(E)``,
  ``lb = log(1-w) + log h(E)``, ``sa = exp(la - logsumexp)``, ``sb = 1 - sa``:
  ``dT/dlam = sum_E sa * (E/lam^2 - 1/lam)`` and
  ``dT/dw = sum_E (sa/w - sb/(1-w))``, then ``dw/da = b/nu^2`` and
  ``dw/db = -a/nu^2``.

Two places need an explicit guard, and both are the point of doing it this way:

* ``sb == 0`` exactly (reached when ``S`` is small enough that ``1 - w``
  underflows) would give ``0/0`` in ``sb/(1-w)``. The limit is 0, so the term is
  dropped. The emitted StableHLO has no such guard and returns nan there --- a
  separately recorded defect, which is why no point with ``S <= 1e-6`` is
  frozen here.
* ``log g`` is written as ``-log(lam) - E/lam`` rather than by taking the log of
  a density, so ``lam`` near zero underflows to a large negative log-density
  instead of to zero. A density-space oracle returns -inf at ``lam = 1e-8``
  where the module is correctly finite.

Verification, before the ``expected_grad`` vectors were frozen: this oracle
agrees with ``jax.grad`` at x64 of an independent f64 re-implementation of the
log-density to 4.3e-15 worst relative error over all nine points, and with f64
central differences on that formula to 8.8e-08. One coordinate is not
finite-difference-checkable --- ``lam`` at the ``1e-8`` point, where a centred
step of any usable size leaves the positive domain --- and is covered by the
autodiff comparison alone.

``grad_atol`` basis: the worst absolute error of Enzyme's f32 gradient against
these frozen f64 vectors, over all nine points and every component, is 2.59e-05
(at the ``lam = 1e-8`` point, where ``dS`` is 33.8, so 8.6e-07 relative --- f32
round-off). ``grad_atol`` is set to 1e-04, about a 4x margin. It is an ABSOLUTE
bound in the runner, so it has to cover the largest component: ``sigma_B``
reaches 73.1 at point 1 and ``dS`` reaches 33.8 at point 7.

Deliberately NOT frozen here: any point with ``S <= 1e-6``. The value is correct
there but the emitted gradient is nan, because ``w`` rounds to exactly 1.0 in
f32 and the adjoint of ``log(1 - w)`` at ``1 - w = 0`` produces ``0 * inf``.
That is a recorded defect of the model's ``weighted(1.0 - w[i], ...)`` spelling
rather than anything this case should assert, and freezing it would make the
case red for a reason unrelated to what it gates. Add such a point once that is
fixed --- this oracle already returns the correct finite gradient there.
"""
import numpy as np

_EXPOSURE = np.array([1.6, 1.3, 1.0, 0.7, 0.4])
_EFFICIENCY = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
_COUNTS = np.array([13, 10, 10, 2, 2], dtype=float)
_S_MU = 100.0
_S_SIGMA = 2.0
_EVENTS = [
    np.array([32.29945024522565, 69.08580773988763, 25.440423777510436,
              75.61403615267723, 7.9267656229908585, 3.3354153534688638,
              20.670985493456392, 82.75167461000609, 136.75349754794027,
              19.976541400061382, 34.57232691499431, 52.32632653325007,
              4.75519376172241]),
    np.array([63.76604456320073, 4.602635553779128, 68.24639884216056,
              4.272813246216692, 49.33863066627453, 41.748541234687956,
              47.49935392156085, 37.52247790527123, 26.040570121268903,
              58.08667451701596]),
    np.array([58.822595816291454, 8.785379233761681, 55.58320441541379,
              63.51718319418958, 58.297943138713215, 40.37315193306145,
              7.875521943121733, 139.17144907577784, 100.79360983559722,
              98.9325347685139]),
    np.array([31.878682258259104, 21.28067021826852]),
    np.array([53.041794807908694, 98.55886779280806]),
]


def grad_oracle(point: dict) -> dict:
    S = float(point["S"])
    s = float(point["sigma_B"])
    m_B = float(point["m_B"])
    lam = float(point["lam"])
    B = np.asarray(point["B"], dtype=float)

    mu = np.log(m_B) - s**2 / 2.0
    z = np.log(B)
    resid = (z - mu) / s**2

    # LogNormal prior.
    dB = -1.0 / B - resid / B
    dm_B = resid.sum() / m_B
    ds = (-1.0 / s + (z - mu) ** 2 / s**3).sum() - s * resid.sum()

    a = _EXPOSURE * B
    b = _EXPOSURE * _EFFICIENCY * S
    nu = a + b

    # Poisson counts.
    dnu = _COUNTS / nu - 1.0
    dB = dB + dnu * _EXPOSURE
    dS = float((dnu * _EXPOSURE * _EFFICIENCY).sum())

    # Per-event mixture, in log space.
    w = a / nu
    dlam = 0.0
    log_h_const = -np.log(_S_SIGMA) - 0.5 * np.log(2.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = np.log(w)
        log_1mw = np.log1p(-w)
    for i, E in enumerate(_EVENTS):
        log_g = -np.log(lam) - E / lam
        log_h = log_h_const - 0.5 * ((E - _S_MU) / _S_SIGMA) ** 2
        la = log_w[i] + log_g
        lb = log_1mw[i] + log_h
        m = np.maximum(la, lb)
        # Both terms -inf would make m - m indeterminate; not reachable at any
        # frozen point (it needs w == 0 and 1 - w == 0 at once).
        ea, eb = np.exp(la - m), np.exp(lb - m)
        tot = ea + eb
        sa, sb = ea / tot, eb / tot

        dlam += float((sa * (E / lam**2 - 1.0 / lam)).sum())

        # sb == 0 exactly: the limit of sb/(1-w) is 0, so drop the term rather
        # than evaluate 0/0.
        term_a = sa / w[i] if w[i] != 0.0 else np.zeros_like(sa)
        one_mw = 1.0 - w[i]
        term_b = np.where(sb == 0.0, 0.0,
                          sb / (one_mw if one_mw != 0.0 else np.inf))
        dT_dw = float((term_a - term_b).sum())

        dw_da = b[i] / nu[i] ** 2
        dw_db = -a[i] / nu[i] ** 2
        dB[i] += dT_dw * dw_da * _EXPOSURE[i]
        dS += dT_dw * dw_db * _EXPOSURE[i] * _EFFICIENCY[i]

    return {
        "S": float(dS),
        "sigma_B": float(ds),
        "m_B": float(dm_B),
        "lam": float(dlam),
        "B": [float(x) for x in dB],
    }
