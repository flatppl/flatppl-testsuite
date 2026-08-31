"""The checks, and where every tolerance comes from.

TOLERANCE DISCIPLINE. The brief for this gate is that a stated tolerance must be
at least 5 sigma of the estimator it bounds, so a passing row is evidence and a
failing row is not noise. Each band below is therefore derived from the
estimator's own standard error at the row's `n`, using the row's CLOSED-FORM
moments — never fitted to what the engine happens to produce.

    mean       SE = sqrt(var / n)                      band = 5 SE
    variance   SE = sqrt((mu4 - var^2) / n)            band = 5 SE
    covariance SE = var / sqrt(n)                      band = 5 SE
    KS         D_crit = kstwo.isf(2 * Phi(-5), n_ks)   = 0.019398 at n_ks = 20000
    totalmass  atol 1e-12 / rtol 1e-9                  (not Monte Carlo — see below)

The covariance SE is the independent-coordinate case, which is exactly the null
being tested: for independent centred X, Y with variance v each,
Var(XY) = v^2, so the sample covariance has SE = v / sqrt(n).

The KS threshold is the EXACT null distribution (`scipy.stats.kstwo`), not the
asymptotic approximation — they agree to 5e-6 here, but the exact one costs
nothing. 5 sigma two-sided is p = 5.733e-07.

WEIGHTED MOMENTS TAKE AN EFFECTIVE `n`. A `space.Probe.weighted_variate` row's
moments are self-normalised importance estimators, so `n` above is the
ensemble's effective sample size rather than the draw count -- the same
substitution `check_latent_mean` makes. The formulas are unchanged; only the
count they divide by is.

`totalmass` is NOT a Monte-Carlo quantity. `logTotalmass` is a deterministic
closed-form number the engine computes from the measure algebra, so it gets a
float-precision band, matching the density sweep's `_TOLERANCE` of 1e-9.

WHY THE VARIANCE BAND SOMETIMES FALLS BACK. Two roster rows have no finite
fourth central moment: `Pareto(shape = 4)` needs shape > 4 for one, and it
diverges. Such a row cannot state a sigma band for its variance, so it takes a
RELATIVE band instead (`VAR_REL_FALLBACK`), and the row is marked
`tolerance_fallback` in the table so the weaker check is visible rather than
silently equivalent to the others. This is the one place a band is not
5-sigma-derived, and it is flagged per row.

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
from dataclasses import dataclass

SIGMA = 5.0
"""Every Monte-Carlo band is this many standard errors of its estimator."""

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


def check_mean(coord: int, emp: float, want: float | None, var: float | None, n: float,
               why: str | None = None) -> Check:
    name = f"mean[{coord}]"
    if want is None:
        return Check(name, "skipped", why or "distribution has no mean (Cauchy)")
    if var is None:
        return Check(name, "skipped", "no closed-form variance, so no standard error to band with")
    if var == 0.0:
        return _verdict(name, emp, want, DEGENERATE_ATOL, None,
                        "degenerate: {got:.15g} vs {want:.15g} (|d|={delta:.3g}, atol {band:.1g})")
    se = math.sqrt(var / n)
    return _verdict(name, emp, want, SIGMA * se, se,
                    "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 5 SE, {sigma} sigma)")


def check_var(coord: int, emp: float, want: float | None, fourth: float | None, n: float,
              why: str | None = None) -> Check:
    name = f"var[{coord}]"
    if want is None:
        return Check(name, "skipped", why or "distribution has no variance (Cauchy)")
    if want == 0.0:
        return _verdict(name, emp, want, DEGENERATE_ATOL, None,
                        "degenerate: {got:.15g} vs {want:.15g} (|d|={delta:.3g}, atol {band:.1g})")
    if fourth is None:
        band = VAR_REL_FALLBACK * abs(want)
        return _verdict(name, emp, want, band, None,
                        "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 3% rel; "
                        "fourth moment diverges, so no sigma band)", fallback=True)
    spread = fourth - want * want
    if spread <= 0:
        # mu4 <= var^2 is impossible for a non-degenerate law (Cauchy-Schwarz),
        # so this can only mean a frozen oracle value is inconsistent. Refuse to
        # invent a band rather than pass the row on a nonsense one.
        return Check(name, "skipped",
                     f"oracle inconsistent: mu4 {fourth:.6g} <= var^2 {want * want:.6g}")
    se = math.sqrt(spread / n)
    return _verdict(name, emp, want, SIGMA * se, se,
                    "{got:.6f} vs {want:.6f} (|d|={delta:.3g}, band {band:.3g} = 5 SE, {sigma} sigma)")


def check_cov(coord: int, emp: float, want: float | None, var: float | None, n: float) -> Check:
    """Cross-coordinate covariance against coordinate 0.

    THE branch-pinning check. §06 "Joint composition" makes `iid(M, size)` the
    product measure, so distinct coordinates are independent and every one of
    these is a test against exactly 0.
    """
    name = f"cov[0,{coord}]"
    if want is None:
        return Check(name, "skipped", "single-coordinate row")
    if var is None or var == 0.0:
        return Check(name, "skipped", "no closed-form variance, so no standard error to band with")
    se = var / math.sqrt(n)
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
                      n_eff: float | None) -> Check:
    """The WEIGHTED mean of a latent that drives a `normalize`'s mass.

    §06 `normalize` makes every theta-slice of the measure a probability measure,
    so the theta-marginal of the sampled joint is the PRIOR, unchanged. That is
    the oracle, and it is exact -- no quadrature enters. The failing hypothesis
    has its own closed form (the prior tilted by Z(theta)), which
    `space.Probe.latent_tilt` records and `tests/sweep/test_sampler_gate.py`
    asserts this band rejects.

    The band is `SIGMA * sqrt(var / n_eff)`: `var` is the prior's closed-form
    variance and `n_eff` the effective sample size the run's own weights give.
    A self-normalised importance estimator's variance depends on the weights, so
    the ESS cannot come from a closed form -- it is a run-reported diagnostic,
    not an oracle, and it is the only part of the band that is.
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
    se = math.sqrt(var / n_eff)
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
    `n * Var(cov_hat)` = `E[a^2 b^2] - cov^2` over the two centred variables. As
    with `check_latent_mean`, the ESS is the only part of the band the run
    supplies; everything else is an oracle.
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
