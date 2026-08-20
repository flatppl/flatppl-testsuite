"""Closed-form sampling oracles, and the record of how each was verified.

ORACLE PROVENANCE. Every mean/variance below was checked THREE ways before it
was frozen here, and the three agreed to 1e-7 or better:

1. the closed form written out by hand (the expressions in this file),
2. numerical quadrature of the distribution's density AS TRANSCRIBED BY HAND
   FROM `flatppl-design/docs/08-distributions.md` — `scipy.integrate.quad` of
   `p(x)`, `x p(x)` and `x^2 p(x)` over the spec's stated support (a summation
   over the support for the discrete families),
3. `scipy.stats`'s own `.mean()` / `.var()` on the frozen distribution named in
   `SCIPY` below.

Leg 2 is the one that matters and the reason this file is not just a list of
textbook formulas. It is what pins the PARAMETERIZATION: it proves that the
closed form belongs to the spec's argument convention, not to some other
convention of the same distribution. Three of these mappings are not the
obvious ones and leg 2 is what settled them:

- `Geometric(p)` counts FAILURES before the first success, so its support
  starts at 0 (§08 "Geometric": "The number of failures until this success is
  geometrically distributed", density `p(1-p)^k`). That is
  `scipy.stats.geom(p, loc=-1)`, NOT `geom(p)`. Mean `(1-p)/p`.
- `NegativeBinomial(alpha, beta)` is the shape/rate form of §08
  "NegativeBinomial", `(beta/(beta+1))^alpha (1/(beta+1))^k`. That is
  `nbinom(n=alpha, p=beta/(beta+1))`. Mean `alpha/beta`, variance
  `alpha(beta+1)/beta^2`.
- `NegativeBinomial2(mu, psi)` is `nbinom(n=psi, p=psi/(mu+psi))`. Mean `mu`,
  variance `mu + mu^2/psi`.

`VonMises` also needed leg 2. §08 "VonMises" gives a `2*pi`-periodic density on
`reals` whose "canonical fundamental domain is [mu - pi, mu + pi]", so the
LINEAR variance of a draw is the quadrature of `x^2 p(x)` over that interval,
not the circular variance `1 - I1/I0`. Quadrature over `[-pi, pi]` at
`kappa = 2` gives 0.7644619, which is also what `scipy.stats.vonmises(2).var()`
returns — so scipy's `vonmises` is the wrapped linear moment and the two agree.

Nothing here is taken from a FlatPPL engine. The engines are the SUBJECT of the
sweep; using either as the source of an expected value is the contamination the
whole harness exists to avoid.

WHAT HAS NO CLOSED-FORM MOMENT. `Cauchy` has no mean and no variance (§08
"Cauchy" is the standard Lorentzian), so its moment check is skipped by
construction and only its KS test runs. That is a property of the distribution,
not a coverage gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

# ---------------------------------------------------------------------------
# Gamma-function values used by the Weibull and GeneralizedNormal moments.
# Spelled as `math.gamma` calls rather than decimal literals so the derivation
# stays readable; they are exact function calls, not approximations.
# ---------------------------------------------------------------------------
_G = math.gamma


@dataclass(frozen=True)
class Family:
    """One base distribution row: the surface spelling plus its closed forms."""

    slug: str
    """Row-id component. Unique across the roster."""

    measure: str
    """FlatPPL surface text for the measure, e.g. `Normal(mu = 1.0, sigma = 2.0)`."""

    mean: float | None
    """Closed-form mean, or None when the distribution has none (Cauchy)."""

    var: float | None
    """Closed-form variance, or None when the distribution has none."""

    discrete: bool
    """Counting-measure family. Suppresses the KS test (which needs a continuous cdf)."""

    scipy: tuple[str, tuple, dict] | None = None
    """`(name, args, kwargs)` recipe for `scipy.stats.<name>(*args, **kwargs)` — a
    LIVE frozen distribution for the KS test's cdf. None means no KS test for
    this row (no scipy counterpart, or a discrete family). This is a recipe and
    not a frozen number for the same reason `unified/sample_checks.py` keeps one:
    a KS test needs a callable `.cdf`, and rebuilding a scipy frozen
    distribution from a recipe is a deterministic library call, not an oracle
    computation."""

    fourth: float | None = None
    """`E[(X - mean)^4]`, the central fourth moment, when it is finite. It sets
    the standard error of a sample VARIANCE — `sqrt((mu4 - var^2)/n)` — so a
    variance tolerance can be stated in sigma rather than as a blanket
    percentage. None means the tolerance falls back to the relative band in
    `checks.py`; see that module for what it does then."""

    note: str = ""


# ---------------------------------------------------------------------------
# BASE FAMILIES — every entry of flatppl-js's `sampler-registry.ts` REGISTRY
# that has a sampler, i.e. every entry NOT marked `densityOnly: true`.
#
# The eight density-only REGISTRY entries have no sampler at all and are
# therefore out of this sweep by construction, not by omission:
#   ContinuedPoisson, CrystalBall, DoubleSidedCrystalBall, Argus,
#   RelativisticBreitWigner, Voigtian, Landau, BifurcatedNormal
# (`sampler-registry.ts` gives each `randFn: { factory: _hepDensityOnly(...) }`
# or `_continuedPoissonNonGenerative`, which throw when called).
#
# Parameters are chosen so the tails are light enough that a 200k-draw variance
# lands inside a 5-sigma band. Where a moment needs a parameter constraint the
# choice respects it with margin: Pareto needs shape > 2 for a finite variance
# (shape = 4), InverseGamma needs shape > 2 (shape = 5), StudentT needs nu > 4
# for a finite fourth moment (nu = 5 -> mu4 finite but large; see `fourth`).
# ---------------------------------------------------------------------------

def _weibull_moments(k: float, lam: float) -> tuple[float, float]:
    g1, g2 = _G(1 + 1 / k), _G(1 + 2 / k)
    return lam * g1, lam * lam * (g2 - g1 * g1)


_WEI_MEAN, _WEI_VAR = _weibull_moments(2.0, 1.5)

FAMILIES: tuple[Family, ...] = (
    Family("normal", "Normal(mu = 1.0, sigma = 2.0)", 1.0, 4.0, False,
           ("norm", (1.0, 2.0), {}), fourth=3 * 4.0 ** 2),
    Family("exponential", "Exponential(rate = 2.0)", 0.5, 0.25, False,
           ("expon", (), {"scale": 0.5}), fourth=9 * 0.5 ** 4),
    Family("uniform", "Uniform(interval(-1.0, 3.0))", 1.0, 16 / 12, False,
           ("uniform", (-1.0, 4.0), {}), fourth=(4.0 ** 4) / 80),
    Family("logistic", "Logistic(mu = 1.0, s = 2.0)", 1.0, 4.0 * math.pi ** 2 / 3, False,
           ("logistic", (1.0, 2.0), {}), fourth=(7 * math.pi ** 4 / 15) * 2.0 ** 4),
    Family("weibull", "Weibull(shape = 2.0, scale = 1.5)", _WEI_MEAN, _WEI_VAR, False,
           ("weibull_min", (2.0,), {"scale": 1.5}), fourth=0.756586447403),
    Family("pareto", "Pareto(shape = 4.0, scale = 1.0)", 4 / 3, 4 / 18, False,
           ("pareto", (4.0,), {"scale": 1.0}),
           note="shape = 4: variance finite (needs > 2), fourth moment DIVERGES "
                "(needs > 4), so the variance band is the relative one"),
    Family("lognormal", "LogNormal(mu = 0.0, sigma = 0.5)",
           math.exp(0.125), (math.exp(0.25) - 1) * math.exp(0.25), False,
           ("lognorm", (0.5,), {"scale": 1.0}), fourth=1.183520556659),
    # Beta(2,2) is a REGRESSION row, not just a roster entry. `@stdlib`
    # random-base-beta@0.2.2 draws every symmetric Beta(a,a) with a > 1.5 with a
    # variance biased LOW (-3.09% at a = 2, mean unaffected); flatppl-js works
    # around it in `sampler-registry.ts` by routing that region through two
    # gammas (`randBetaFixed`). This row is what fails if that wiring is ever
    # reverted before the dependency ships a fix.
    Family("beta_2_2", "Beta(alpha = 2.0, beta = 2.0)", 0.5, 4 / (16 * 5), False,
           ("beta", (2.0, 2.0), {}), fourth=0.005357142857,
           note="regression row for the @stdlib symmetric-Beta variance defect"),
    Family("beta_2_5", "Beta(alpha = 2.0, beta = 5.0)", 2 / 7, 10 / (49 * 8), False,
           ("beta", (2.0, 5.0), {}), fourth=0.001874219075),
    Family("gamma", "Gamma(shape = 3.0, rate = 2.0)", 1.5, 0.75, False,
           ("gamma", (3.0,), {"scale": 0.5}), fourth=3 * 3.0 * (3.0 + 2) / 2.0 ** 4),
    Family("cauchy", "Cauchy(location = 0.0, scale = 1.0)", None, None, False,
           ("cauchy", (0.0, 1.0), {}),
           note="no mean, no variance: moment check skipped by construction, KS only"),
    Family("studentt", "StudentT(nu = 5.0)", 0.0, 5 / 3, False,
           ("t", (5.0,), {}), fourth=25.0,
           note="nu = 5: variance 5/3; fourth moment 3nu^2/((nu-2)(nu-4)) = 25, finite "
                "but heavy — the variance SE is correspondingly wide"),
    Family("gennormal", "GeneralizedNormal(mean = 0.0, alpha = 1.0, beta = 3.0)",
           0.0, _G(3 / 3) / _G(1 / 3), False, ("gennorm", (3.0, 0.0, 1.0), {}),
           fourth=0.336978725437),
    Family("invgamma", "InverseGamma(shape = 5.0, scale = 2.0)", 0.5, 4 / (16 * 3), False,
           ("invgamma", (5.0,), {"scale": 2.0}), fourth=0.3125),
    Family("chisquared", "ChiSquared(k = 4.0)", 4.0, 8.0, False,
           ("chi2", (4.0,), {}), fourth=12 * 4.0 * (4.0 + 4)),
    # See the module docstring: the variance is the LINEAR moment on the
    # fundamental domain [mu - pi, mu + pi], confirmed by quadrature of §08's
    # density and by scipy.stats.vonmises(2.0).var().
    Family("vonmises", "VonMises(mu = 0.0, kappa = 2.0)", 0.0, 0.7644618686336627, False,
           ("vonmises", (2.0,), {"loc": 0.0}), fourth=2.270241855362,
           note="linear variance on the [-pi, pi] fundamental domain, not circular variance"),
    Family("laplace", "Laplace(location = 1.0, scale = 2.0)", 1.0, 8.0, False,
           ("laplace", (1.0, 2.0), {}), fourth=24 * 2.0 ** 4),
    Family("dirac", "Dirac(value = 3.0)", 3.0, 0.0, False, None, fourth=0.0,
           note="degenerate: variance exactly 0, so the variance band is absolute"),

    # --- discrete: no KS (a continuous cdf does not exist), moments only
    Family("bernoulli", "Bernoulli(p = 0.3)", 0.3, 0.3 * 0.7, True,
           ("bernoulli", (0.3,), {}), fourth=0.0777),
    Family("binomial", "Binomial(n = 10, p = 0.3)", 3.0, 10 * 0.3 * 0.7, True,
           ("binom", (10, 0.3), {}), fourth=12.684),
    Family("geometric", "Geometric(p = 0.3)", 0.7 / 0.3, 0.7 / 0.09, True,
           ("geom", (0.3,), {"loc": -1}), fourth=552.222222222223,
           note="failures-before-success convention: scipy needs loc = -1"),
    Family("negbinomial", "NegativeBinomial(alpha = 3.0, beta = 2.0)", 1.5, 3.0 * 3.0 / 4.0, True,
           ("nbinom", (3.0, 2.0 / 3.0), {}), fourth=27.5625,
           note="shape/rate form: scipy p = beta/(beta+1)"),
    Family("negbinomial2", "NegativeBinomial2(mu = 4.0, psi = 3.0)", 4.0, 4.0 + 16.0 / 3.0, True,
           ("nbinom", (3.0, 3.0 / 7.0), {}), fourth=444.888888888889,
           note="scipy n = psi, p = psi/(mu+psi)"),
    Family("categorical", "Categorical(p = [0.2, 0.3, 0.5])", 2.3, 0.61, True, None,
           fourth=0.6937, note="1-based support per §08"),
    Family("categorical0", "Categorical0(p = [0.2, 0.3, 0.5])", 1.3, 0.61, True, None,
           fourth=0.6937, note="0-based support per §08"),
    Family("poisson", "Poisson(rate = 4.0)", 4.0, 4.0, True,
           ("poisson", (4.0,), {}), fourth=4.0 + 3 * 4.0 ** 2),
)

# The REGISTRY entries with `densityOnly: true` — no sampler exists, so there is
# nothing for this sweep to draw. Recorded so the roster is auditable as
# COMPLETE against `sampler-registry.ts` rather than merely long.
DENSITY_ONLY: tuple[tuple[str, str], ...] = (
    ("ContinuedPoisson", "density-only per §09; sampler stub throws (not generative)"),
    ("CrystalBall", "density-only HEP shape; `_hepDensityOnly` stub throws"),
    ("DoubleSidedCrystalBall", "density-only HEP shape; `_hepDensityOnly` stub throws"),
    ("Argus", "density-only HEP shape; `_hepDensityOnly` stub throws"),
    ("RelativisticBreitWigner", "density-only HEP shape; `_hepDensityOnly` stub throws"),
    ("Voigtian", "density-only HEP shape; `_hepDensityOnly` stub throws"),
    ("Landau", "density-only HEP shape; `_hepDensityOnly` stub throws"),
    ("BifurcatedNormal", "density-only HEP shape; `_hepDensityOnly` stub throws"),
)
