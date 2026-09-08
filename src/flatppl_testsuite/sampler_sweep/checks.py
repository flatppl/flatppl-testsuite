"""The checks, and where every tolerance comes from.

TOLERANCE DISCIPLINE. Moment bands use five asymptotic standard errors, derived
from the estimator's closed-form variance, never fitted to engine output.
They are not finite-sample Gaussian tail guarantees, especially for heavy tails
or importance weights. The symmetric two-point variance case and the KS
threshold use their exact null laws.

    mean       SE = sqrt(var / n)                      band = 5 SE
    variance   SE = sqrt((mu4 - var^2) / n)            band = 5 SE
    covariance SE = var / sqrt(n)                      band = 5 SE
    KS         D_crit = kstwo.isf(2 * Phi(-5), n_ks)   = 0.019398 at n_ks = 20000
    totalmass  atol 1e-12 / rtol 1e-9                  (not Monte Carlo — see below)

The default covariance SE assumes independent coordinates: for independent
centred X, Y with variance v each, Var(XY) = v^2. Shared-parameter rows supply
their own estimator coefficient, which includes the nonzero covariance.

The KS threshold is the EXACT null distribution (`scipy.stats.kstwo`), not the
asymptotic approximation — they agree to 5e-6 here, but the exact one costs
nothing. 5 sigma two-sided is p = 5.733e-07.

WEIGHTED MOMENTS NEED ESTIMATOR-SPECIFIC VARIANCES. For a self-normalised
importance estimator with influence g and proposal weight w, the coefficient
in SE = sqrt(A / ESS) is A = E_q[w² g²] / E_q[w²]. Substituting ESS into the
target-law variance alone is incorrect when weights depend on the estimand.
The probe supplies `estimator_var` for the known weighted proposal laws.

`totalmass` is NOT a Monte-Carlo quantity. `logTotalmass` is a deterministic
closed-form number the engine computes from the measure algebra, so it gets a
float-precision band, matching the density sweep's `_TOLERANCE` of 1e-9.

WHY THE VARIANCE BAND SOMETIMES FALLS BACK. Two roster rows have no finite
fourth central moment: `Pareto(shape = 4)` needs shape > 4 for one, and it
diverges. Such a row cannot state a sigma band for its variance, so it takes a
RELATIVE band instead (`VAR_REL_FALLBACK`), and the row is marked
`tolerance_fallback` in the table so the weaker check is visible rather than
silently equivalent to the others. This fallback has no calibrated tail
probability, and it is flagged per row.

DEGENERATE ROWS. `Dirac(value)` has variance exactly 0, so a relative band is
undefined and the sigma band collapses to 0. Both its mean and variance take an
absolute float band (`DEGENERATE_ATOL`).

BIAS. The driver accumulates raw sums, so the variance below is the population
form (ddof = 0), whose bias is -var/n. At n = 200000 that is 2e-5 on the Normal
row against a 5-sigma band of 0.063 — three orders of magnitude under, so no
ddof correction is applied and none is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

SIGMA = 5.0
"""Moment bands use this many asymptotic standard errors."""

VAR_REL_FALLBACK = 0.03
"""Relative variance band for a row whose fourth central moment diverges."""

DEGENERATE_ATOL = 1e-12
"""Absolute band for a row whose closed-form variance is exactly 0."""

TOTALMASS_ATOL = 1e-12
TOTALMASS_RTOL = 1e-9

KS_SIGMA_P = 5.733031437583869e-07
"""Two-sided 5-sigma tail probability, `2 * scipy.stats.norm.sf(5.0)`."""


def ks_critical(n: int) -> float:
    """Exact KS critical statistic at the 5-sigma two-sided level.

    0.019397791904131653 at n = 20000. A coordinate pinned to one component of
    the roster's mixture sits at D = 0.4987 against this — 25x the threshold.
    """
    from scipy.stats import kstwo

    return float(kstwo.isf(KS_SIGMA_P, n))


@dataclass(frozen=True)
class Check:
    """One check's outcome on one row."""

    name: str
    status: str
    """`passed`, `failed`, or `skipped`."""
    detail: str
    got: float | None = None
    want: float | None = None
    band: float | None = None
    sigma: float | None = None
    """How many standard errors out the observation landed, when a sigma band
    applies. This is the number worth reading on a failure: it separates a
    marginal row from a defect."""
    fallback: bool = False
    """True when the band is not 5-sigma-derived (see VAR_REL_FALLBACK)."""


def _verdict(name, got, want, band, se, detail_fmt, fallback=False) -> Check:
    delta = abs(got - want)
    ok = delta <= band
    sigma = (delta / se) if (se and se > 0) else None
    detail = detail_fmt.format(
        got=got, want=want, delta=delta, band=band,
        sigma=("n/a" if sigma is None else f"{sigma:.2f}"),
    )
    return Check(name, "passed" if ok else "failed", detail,
                 got=got, want=want, band=band, sigma=sigma, fallback=fallback)


def require_power(check: Check, fault: float | None) -> Check:
    """A band must exclude the fault value the probe exists to distinguish."""
    if (fault is not None and check.want is not None and fault != check.want
            and check.band is not None and abs(fault - check.want) <= check.band):
        return replace(check, status="failed", detail=(
            f"insufficient effective sample size/power: band {check.band:.6g} "
            f"around {check.want:.6g} admits fault value {fault:.6g}"))
    return check


def check_mean(coord: int, emp: float, want: float | None, var: float | None, n: float,
               why: str | None = None, *, estimator_var: float | None = None) -> Check:
    name = f"mean[{coord}]"
    if want is None:
        return Check(name, "skipped", why or "distribution has no mean (Cauchy)")
    if var is None:
        return Check(name, "skipped", "no closed-form variance, so no standard error to band with")
    if var == 0.0:
        return _verdict(name, emp, want, DEGENERATE_ATOL, None,
                        "degenerate: {got:.15g} vs {want:.15g} (|d|={delta:.3g}, atol {band:.1g})")
    se = math.sqrt((var if estimator_var is None else estimator_var) / n)
    return _verdict(name, emp, want, SIGMA * se, se,
                    "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 5 SE, {sigma} sigma)")


def check_var(coord: int, emp: float, want: float | None, fourth: float | None, n: float,
              why: str | None = None, *, estimator_var: float | None = None) -> Check:
    name = f"var[{coord}]"
    if want is None:
        return Check(name, "skipped", why or "distribution has no variance (Cauchy)")
    if want == 0.0:
        return _verdict(name, emp, want, DEGENERATE_ATOL, None,
                        "degenerate: {got:.15g} vs {want:.15g} (|d|={delta:.3g}, atol {band:.1g})")
    if fourth is None and estimator_var is None:
        band = VAR_REL_FALLBACK * abs(want)
        return _verdict(name, emp, want, band, None,
                        "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 3% rel; "
                        "fourth moment diverges, so no sigma band)", fallback=True)
    spread = fourth - want * want if estimator_var is None else estimator_var
    if spread < 0:
        return Check(name, "failed",
                     f"oracle inconsistent: variance coefficient {spread:.6g} < 0")
    if spread == 0:
        if estimator_var is not None or not math.isfinite(n) or n < 1 or n != int(n):
            return Check(name, "failed",
                         "zero influence variance requires an unweighted integer draw count")
        # Equality means X=mu±sqrt(var), with equal probabilities. For K~Bin(n,1/2),
        # the ddof=0 variance is var*(1-(2*K/n-1)^2). Its error is not Gaussian.
        from scipy.stats import binom
        upper = binom.isf(KS_SIGMA_P / 2, int(n), 0.5)
        band = want * ((2 * upper - n) / n)**2
        return _verdict(name, emp, want, band, None,
                        "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g}; "
                        "exact symmetric two-point variance law)")
    se = math.sqrt(spread / n)
    return _verdict(name, emp, want, SIGMA * se, se,
                    "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 5 SE, {sigma} sigma)")


def check_cov(coord: int, emp: float, want: float | None, var: float | None, n: float,
              *, estimator_var: float | None = None) -> Check:
    """Cross-coordinate covariance against coordinate 0.

    THE branch-pinning check. §06 "Joint composition" makes `iid(M, size)` the
    product measure, so distinct coordinates are independent and the oracle is
    exactly 0 on every row whose parameters are fixed.

    It is NOT 0 when the coordinates share a stochastic parameter: `iid` makes
    them independent GIVEN that parameter, so they share its whole variance and
    the oracle is that variance. Such a row states its own `cov`, and 0 is then
    the failure it exists to catch -- a parameter re-drawn per coordinate.
    """
    name = f"cov[0,{coord}]"
    if want is None:
        return Check(name, "skipped", "single-coordinate row")
    if var is None or var == 0.0:
        return Check(name, "skipped", "no closed-form variance, so no standard error to band with")
    se = math.sqrt((var * var if estimator_var is None else estimator_var) / n)
    return _verdict(name, emp, want, SIGMA * se, se,
                    "{got:+.6f} vs {want:+.6f} (|d|={delta:.3g}, band {band:.3g} = 5 SE, {sigma} sigma)")


def check_ks(sample, ks_spec, n_ks: int) -> Check:
    """Goodness of fit against the closed-form cdf."""
    if ks_spec is None:
        return Check("ks", "skipped", "no continuous cdf reference for this row")
    from scipy.stats import kstest

    cdf = build_cdf(ks_spec)
    crit = ks_critical(n_ks)
    d = float(kstest(sample, cdf).statistic)
    ok = d <= crit
    return Check("ks", "passed" if ok else "failed",
                 f"D={d:.6f} vs critical {crit:.6f} (5 sigma, n={n_ks})",
                 got=d, want=0.0, band=crit)


def check_support(outside: int | None) -> Check:
    """Exact support membership over every coordinate, independent of moments."""
    if outside is None:
        return Check("support", "failed", "driver reported no support count")
    return Check("support", "passed" if outside == 0 else "failed",
                 f"{outside} values outside the distribution's support",
                 got=outside, want=0, band=0)


def build_cdf(spec):
    """A callable cdf from a `space.Probe.ks` recipe.

    Rebuilding a scipy frozen distribution from a recipe is a deterministic
    library call, not an oracle computation — the same convention
    `unified/sample_checks.py` uses and for the same reason: a KS test needs a
    live `.cdf`, which cannot be frozen into JSON.
    """
    from scipy import stats

    kind = spec[0]
    if kind == "dist":
        _, name, args, kwargs = spec
        return getattr(stats, name)(*args, **dict(kwargs)).cdf
    if kind == "affine":
        _, a, b, name, args, kwargs = spec
        base = getattr(stats, name)(*args, **dict(kwargs))
        # y = a x + b, a > 0  =>  F_Y(y) = F_X((y - b) / a)
        if a <= 0:
            raise ValueError("affine KS reference assumes a > 0 (orientation-preserving)")
        return lambda y: base.cdf((y - b) / a)
    if kind == "mix":
        _, weights, comps = spec
        frozen = [getattr(stats, n)(*a, **dict(k)) for n, a, k in comps]
        ws = list(weights)
        if abs(sum(ws) - 1.0) > 1e-12:
            raise ValueError(f"mixture KS reference weights sum to {sum(ws)}, not 1")
        return lambda y: sum(w * f.cdf(y) for w, f in zip(ws, frozen))
    raise ValueError(f"unknown KS reference kind {kind!r}")


def check_latent_mean(emp: float | None, want: float | None, var: float | None,
                      n_eff: float | None, *, estimator_var: float | None = None) -> Check:
    """The WEIGHTED mean of a latent that drives a `normalize`'s mass.

    §06 `normalize` makes every theta-slice of the measure a probability measure,
    so the theta-marginal of the sampled joint is the PRIOR, unchanged. That is
    the oracle, and it is exact -- no quadrature enters. The failing hypothesis
    has its own closed form (the prior tilted by Z(theta)), which
    `space.Probe.latent_tilt` records and `tests/sweep/test_sampler_gate.py`
    asserts this band rejects.

    The band is `SIGMA * sqrt(A / n_eff)`. A is the supplied squared-weight
    influence coefficient, or the prior variance when weights are constant.
    ESS is a run-reported diagnostic, not a target-law oracle.
    """
    name = "latent_mean"
    if want is None:
        return Check(name, "skipped", "row names no latent")
    if emp is None:
        return Check(name, "skipped", "driver reported no latent mean")
    if var is None or var <= 0.0:
        return Check(name, "skipped", "no closed-form prior variance to band with")
    if not n_eff or n_eff <= 0:
        return Check(name, "skipped", "no effective sample size reported")
    se = math.sqrt((var if estimator_var is None else estimator_var) / n_eff)
    return _verdict(name, emp, want, SIGMA * se, se,
                    "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 5 SE, "
                    "{sigma} sigma)")


def check_latent_cov(emp: float | None, want: float | None,
                     cov_var: float | None, n_eff: float | None) -> Check:
    """The WEIGHTED covariance of a latent with the variate's coordinate 0.

    The discriminating moment for a latent that reaches the variate only through
    a mixture's component choice. §06 `normalize`'s recommended mixture spelling
    `normalize(superpose(weighted(w1, M1), weighted(w2, M2)))` mixes atom i at
    its own w(theta_i), and a lift that pools the proportion leaves BOTH
    marginals correct -- the latent's is its prior, the variate's is linear in it
    -- so `check_mean` and `check_latent_mean` are both blind to it. The failing
    hypothesis is cov = 0 (a decoupled proportion makes the two independent),
    recorded as `space.Probe.latent_cov_null` and rejected by this band in
    `tests/sweep/test_sampler_gate.py`.

    The band is `SIGMA * sqrt(cov_var / n_eff)`, with `cov_var` the closed-form
    squared-weight influence coefficient. With constant weights it reduces to
    `E[a^2 b^2] - cov^2` over the two centred variables.
    """
    name = "latent_cov"
    if want is None:
        return Check(name, "skipped", "row names no latent covariance")
    if emp is None:
        return Check(name, "skipped", "driver reported no latent covariance")
    if cov_var is None or cov_var <= 0.0:
        return Check(name, "skipped", "no closed-form estimator variance to band with")
    if not n_eff or n_eff <= 0:
        return Check(name, "skipped", "no effective sample size reported")
    se = math.sqrt(cov_var / n_eff)
    return _verdict(name, emp, want, SIGMA * se, se,
                    "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 5 SE, "
                    "{sigma} sigma)")


def check_totalmass(emp: float | None, want: float | None) -> Check:
    """The engine's reported `logTotalmass` against the closed form.

    Deterministic, not Monte Carlo: this is the measure algebra's own bookkeeping
    (mass 2 for `weighted(2.0, ...)`, exactly 1 for a normalized measure,
    2*Phi(1)-1 for the roster's truncation), so it gets a float band.
    """
    if want is None:
        return Check("totalmass", "skipped", "no closed-form total mass for this row")
    if emp is None:
        return Check("totalmass", "skipped", "engine reported no logTotalmass")
    band = TOTALMASS_ATOL + TOTALMASS_RTOL * abs(want)
    return _verdict("totalmass", emp, want, band, None,
                    "log mass {got:.15g} vs {want:.15g} (|d|={delta:.3g}, band {band:.3g})")
