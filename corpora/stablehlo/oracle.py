#!/usr/bin/env python3
"""INDEPENDENT scipy / closed-form oracle for the StableHLO numeric-EXECUTION
gate.

This module is the single source of truth for the gate's fixtures: for each
distribution it fixes the FlatPPL model source, the exact parameter *values*
(in the order the emitter turns them into `func.func` args), the pinned
observation (the variate the log-density is evaluated at), and a pure-scipy
``logdensity`` function. Nothing here ever calls the FlatPPL engine or the
StableHLO executor (Enzyme-JAX) — that is the whole point: the emitted
StableHLO is checked *against* this, so the oracle must share no lineage with
it (maths > spec > code).

The frozen expected VALUES are scipy's; a lineage-independent second oracle
(Julia ``Distributions.jl``) can reproduce every one of them from the
``(distribution, params, variate)`` documented in ``manifest.json`` — see
``gen_expected.py``, which writes that manifest.

Gradient oracle
---------------
The gate validates Enzyme's ``jax.grad`` (the HMC path) against a *central
finite difference of this scipy log-density* w.r.t. each continuous parameter
— again independent of the executor's own autodiff. ``fd_gradient`` computes
it, structured exactly like Enzyme's per-argument gradient (scalar arg ->
scalar; vector arg -> vector) so the two can be compared component-wise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from scipy.stats import (
    bernoulli,
    beta as beta_dist,
    binom,
    dirichlet,
    expon,
    gamma as gamma_dist,
    geom,
    laplace as laplace_dist,
    lognorm,
    multivariate_normal,
    nbinom,
    norm,
    poisson,
    rv_discrete,
    t as studentt_dist,
    uniform,
)


@dataclass(frozen=True)
class Fixture:
    """One distribution's gate fixture.

    ``params`` maps each free FlatPPL parameter to its frozen value, IN THE
    ORDER the emitter lays them out as ``func.func`` arguments (``%arg0`` is
    the first entry, ``%arg1`` the second, ...). Scalars are ``float``,
    vectors are ``list[float]``, matrices are ``list[list[float]]``.

    ``logdensity`` takes those parameter values positionally and returns the
    scipy / closed-form log-density at ``variate``.

    ``grad_params`` names the subset of ``params`` that are continuous and
    thus gradient-testable (integer / support parameters are excluded).
    """

    key: str
    distribution: str
    flatppl: str
    params: dict[str, object]
    variate: object
    variate_repr: str
    logdensity: Callable[..., float]
    scipy_note: str
    grad_params: tuple[str, ...] = ()
    # @sample: scipy comparison for the distributional test. Returns a
    # frozen scipy distribution object exposing cdf/ppf/mean/var (continuous)
    # or pmf-bearing object (discrete); None if not sample-tested.
    sample_ref: Callable[[], object] | None = None
    sample_discrete: bool = False
    # @sample model source (params baked to match sample_ref) + any free args
    # (e.g. Dirichlet's alpha vector, passed as a func arg).
    sample_flatppl: str = ""
    sample_args: tuple = ()
    # @sample independence subject ("beta" | "dirichlet" | None).
    independence: str | None = None
    # Fan-out (`iid(K, n)`): a SEPARATE `.iid.sample.flatppl` model baking (or,
    # for a free-param kernel like MvNormal, threading) the same params into a
    # fixed-size batched draw, one rng_bit_generator advance per call. Covers
    # Tier 1 (straight-line: Normal/Exponential/Uniform) AND Tier 2 (batched
    # rejection: Gamma/Beta/StudentT; batched multivariate: MvNormal). Empty
    # for dists without a landed fan-out lowering.
    fanout_flatppl: str = ""
    fanout_n: int = 0
    # 0 for a scalar-lane fan-out (Tier 1 + Tier 2 rejection: draws [n]); the
    # per-draw dimension `d` for a Tier-2 MULTIVARIATE fan-out (draws [n, d],
    # e.g. MvNormal) — tells the gate to reshape+check mean/covariance instead
    # of running the scalar KS/moment check against a 1-d `sample_ref.cdf`.
    fanout_dim: int = 0
    # True for a SIMPLEX-valued multivariate fan-out (Dirichlet): like
    # `fanout_dim`, draws [n, d], but the gate's simplex check (row-sum==1 +
    # per-component mean/variance + component correlations) replaces the
    # Gaussian mean/covariance check `fanout_dim` alone selects.
    fanout_simplex: bool = False
    # The minimum integer in a discrete kernel's support (0 for every
    # zero-based counting distribution; 1 for 1-based `Categorical`) — the
    # gate's discrete fan-out check bins a chi-square goodness-of-fit from
    # here upward against `sample_ref`'s pmf.
    fanout_discrete_kmin: int = 0
    notes: str = ""

    def param_values(self) -> list:
        return [self.params[name] for name in self.params]


# ---------------------------------------------------------------------------
# Closed-form / scipy log-densities. Each is the §08 distribution scored at the
# fixture's pinned variate; the parameter->scipy-convention mapping is spelled
# out in ``scipy_note`` so the Julia cross-check is unambiguous.
# ---------------------------------------------------------------------------
_X = 0.5  # the pinned continuous observation shared by the scalar fixtures


def _normal(mu, sigma):
    return float(norm.logpdf(_X, loc=mu, scale=sigma))


def _exponential(rate):
    # FlatPPL Exponential(rate): pdf = rate*exp(-rate*x) => scipy scale = 1/rate.
    return float(expon.logpdf(_X, scale=1.0 / rate))


def _gamma(shape, rate):
    # FlatPPL Gamma(shape, rate): scipy a=shape, scale=1/rate.
    return float(gamma_dist.logpdf(_X, a=shape, scale=1.0 / rate))


def _lognormal(mu, sigma):
    # FlatPPL LogNormal(mu, sigma): scipy s=sigma, scale=exp(mu).
    return float(lognorm.logpdf(_X, s=sigma, scale=math.exp(mu)))


def _uniform():
    # Uniform over interval(-1, 3): constant density -log(length) = -log(4).
    return float(uniform.logpdf(_X, loc=-1.0, scale=4.0))


def _beta(alpha, beta):
    return float(beta_dist.logpdf(_X, alpha, beta))


def _studentt(nu):
    # FlatPPL StudentT(nu): standard (loc 0, scale 1).
    return float(studentt_dist.logpdf(_X, df=nu))


def _bernoulli(p):
    return float(bernoulli.logpmf(1, p))  # observed a = 1


def _poisson(rate):
    return float(poisson.logpmf(3, rate))  # observed a = 3


def _binomial(n, p):
    return float(binom.logpmf(2, int(round(n)), p))  # observed a = 2


def _mvnormal(mu, cov):
    return float(multivariate_normal.logpdf([0.2, 0.1], mean=mu, cov=cov))


def _dirichlet(alpha):
    return float(dirichlet.logpdf([0.2, 0.3, 0.5], alpha))


def dirichlet_theo_corr(alpha, a0, i, j):
    """The closed-form Dirichlet component correlation `Corr(X_i, X_j) =
    -sqrt(alpha_i*alpha_j / ((a0-alpha_i)*(a0-alpha_j)))` (always negative —
    components of a simplex-valued draw trade off against each other). Shared
    by the scalar `@sample` independence check and the fanned `iid` simplex
    check in ``gate.py``, both of which validate the SAME closed form against
    two different samplers (one draw per rng stream vs. one batched draw)."""
    return -math.sqrt(alpha[i] * alpha[j] / ((a0 - alpha[i]) * (a0 - alpha[j])))


def _laplace(location, scale):
    return float(laplace_dist.logpdf(_X, loc=location, scale=scale))


def _geometric(p):
    # FlatPPL Geometric(p): pmf = p*(1-p)^k for k in {0,1,...} (# FAILURES
    # before a success). scipy.stats.geom counts TRIALS (k in {1,2,...}) with
    # pmf = p*(1-p)^(k-1); geom(p, loc=-1) shifts that back to the {0,1,...}
    # failure-count convention (geom(p, loc=-1).pmf(k) = geom.pmf(k+1, p) =
    # p*(1-p)^k, matching the spec exactly).
    return float(geom.logpmf(3, p, loc=-1))  # observed a = 3


def _categorical_ref(p, base):
    """A closed-form `rv_discrete` built directly from `p` over
    `{base, ..., base+len(p)-1}` — scipy ships no built-in categorical
    distribution, so this constructs one directly from the same `p` the
    FlatPPL model is baked with (an honest closed-form reference, not a
    library call, matching this module's `_uniform`-style bespoke oracles)."""
    n = len(p)
    return rv_discrete(values=(list(range(base, base + n)), p))


def _categorical(p, base):
    return float(_categorical_ref(p, base).logpmf(2))  # observed a = 2 (1-based)


def _categorical0(p, base):
    return float(_categorical_ref(p, base).logpmf(1))  # observed a = 1 (0-based)


def _negbinomial(alpha, beta):
    # FlatPPL NegativeBinomial(alpha, beta): pmf = C(k+alpha-1,alpha-1) *
    # (beta/(beta+1))^alpha * (1/(beta+1))^k. Matching term-by-term against
    # scipy.stats.nbinom.pmf(k,n,p) = C(k+n-1,k) p^n (1-p)^k (C(k+n-1,k) =
    # C(k+n-1,n-1)): n = alpha, p = beta/(beta+1) (so 1-p = 1/(beta+1),
    # exactly the spec's k-exponent factor).
    return float(nbinom.logpmf(4, alpha, beta / (beta + 1.0)))  # observed a = 4


def _negbinomial2(mu, psi):
    # FlatPPL NegativeBinomial2(mu, psi): pmf = C(k+psi-1,k) *
    # (mu/(mu+psi))^k * (psi/(mu+psi))^psi. Matching against scipy's
    # nbinom.pmf(k,n,p) = C(k+n-1,k) p^n (1-p)^k: n = psi, p = psi/(mu+psi)
    # (1-p = mu/(mu+psi), the spec's k-exponent factor).
    return float(nbinom.logpmf(4, psi, psi / (mu + psi)))  # observed a = 4


# ---------------------------------------------------------------------------
# Fixture table. Parameter order MUST match the emitter's %arg order (verified
# by emitting each model and reading its `func.func @logdensity(...)` line).
# ---------------------------------------------------------------------------
def _src(body: str) -> str:
    return body.lstrip("\n")


FIXTURES: list[Fixture] = [
    Fixture(
        key="normal",
        distribution="Normal(mu, sigma)",
        flatppl=_src("""
mu = elementof(reals)
sigma = elementof(posreals)
a = draw(Normal(mu = mu, sigma = sigma))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={"mu": 0.3, "sigma": 1.2},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_normal,
        scipy_note="scipy.stats.norm.logpdf(0.5, loc=mu, scale=sigma)",
        grad_params=("mu", "sigma"),
        sample_ref=lambda: norm(loc=0.3, scale=1.2),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Normal(mu = 0.3, sigma = 1.2))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Normal(mu = 0.3, sigma = 1.2), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
    ),
    Fixture(
        key="exponential",
        distribution="Exponential(rate)",
        flatppl=_src("""
rate = elementof(posreals)
a = draw(Exponential(rate = rate))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={"rate": 1.5},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_exponential,
        scipy_note="scipy.stats.expon.logpdf(0.5, scale=1/rate)",
        grad_params=("rate",),
        sample_ref=lambda: expon(scale=1.0 / 1.5),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Exponential(rate = 1.5))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Exponential(rate = 1.5), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
    ),
    Fixture(
        key="gamma",
        distribution="Gamma(shape, rate)",
        flatppl=_src("""
shape = elementof(posreals)
rate = elementof(posreals)
a = draw(Gamma(shape = shape, rate = rate))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={"shape": 2.5, "rate": 1.5},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_gamma,
        scipy_note="scipy.stats.gamma.logpdf(0.5, a=shape, scale=1/rate) [CHLO lgamma]",
        grad_params=("shape", "rate"),
        sample_ref=lambda: gamma_dist(a=2.5, scale=1.0 / 1.5),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Gamma(shape = 2.5, rate = 1.5))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Gamma(shape = 2.5, rate = 1.5), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        notes="exercises chlo.lgamma; fan-out exercises the batched Marsaglia-Tsang rejection while",
    ),
    Fixture(
        key="lognormal",
        distribution="LogNormal(mu, sigma)",
        flatppl=_src("""
mu = elementof(reals)
sigma = elementof(posreals)
a = draw(LogNormal(mu = mu, sigma = sigma))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={"mu": 0.0, "sigma": 0.75},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_lognormal,
        scipy_note="scipy.stats.lognorm.logpdf(0.5, s=sigma, scale=exp(mu))",
        grad_params=("mu", "sigma"),
        sample_ref=lambda: lognorm(s=0.75, scale=math.exp(0.0)),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(LogNormal(mu = 0.0, sigma = 0.75))
draws = rand(s, lawof(x))
"""),
    ),
    Fixture(
        key="uniform",
        distribution="Uniform(support = interval(-1, 3))",
        flatppl=_src("""
a = draw(Uniform(support = interval(-1.0, 3.0)))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_uniform,
        scipy_note="scipy.stats.uniform.logpdf(0.5, loc=-1, scale=4) = -log(4)",
        grad_params=(),  # no free parameters
        sample_ref=lambda: uniform(loc=-1.0, scale=4.0),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Uniform(support = interval(-1.0, 3.0)))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Uniform(support = interval(-1.0, 3.0)), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
    ),
    Fixture(
        key="beta",
        distribution="Beta(alpha, beta)",
        flatppl=_src("""
alpha = elementof(posreals)
beta = elementof(posreals)
a = draw(Beta(alpha = alpha, beta = beta))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={"alpha": 2.0, "beta": 3.0},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_beta,
        scipy_note="scipy.stats.beta.logpdf(0.5, alpha, beta) [CHLO lgamma]",
        grad_params=("alpha", "beta"),
        sample_ref=lambda: beta_dist(2.0, 3.0),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Beta(alpha = 2.0, beta = 3.0))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Beta(alpha = 2.0, beta = 3.0), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        independence="beta",
        notes=(
            "exercises chlo.lgamma; @sample uses two internal Gamma rng streams; "
            "fan-out exercises TWO batched Marsaglia-Tsang rejection whiles (X/(X+Y))"
        ),
    ),
    Fixture(
        key="studentt",
        distribution="StudentT(nu)",
        flatppl=_src("""
nu = elementof(posreals)
a = draw(StudentT(nu = nu))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={"nu": 4.0},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_studentt,
        scipy_note="scipy.stats.t.logpdf(0.5, df=nu) [CHLO lgamma]",
        grad_params=("nu",),
        sample_ref=lambda: studentt_dist(df=4.0),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(StudentT(nu = 4.0))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(StudentT(nu = 4.0), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        notes="exercises chlo.lgamma; fan-out exercises the reducer composing batched Gamma rejection",
    ),
    Fixture(
        key="bernoulli",
        distribution="Bernoulli(p)",
        flatppl=_src("""
p = elementof(unitinterval)
a = draw(Bernoulli(p = p))
lp = logdensityof(lawof(record(a = a)), record(a = 1))
"""),
        params={"p": 0.3},
        variate=1, variate_repr="a = 1",
        logdensity=_bernoulli,
        scipy_note="scipy.stats.bernoulli.logpmf(1, p) = log(p)",
        grad_params=("p",),
        sample_ref=lambda: bernoulli(0.3),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Bernoulli(p = 0.3))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Bernoulli(p = 0.3), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        notes="elementwise select(U < p, 1, 0) fan-out — no while, no new primitive",
    ),
    Fixture(
        key="poisson",
        distribution="Poisson(rate)",
        flatppl=_src("""
rate = elementof(nonnegreals)
a = draw(Poisson(rate = rate))
lp = logdensityof(lawof(record(a = a)), record(a = 3))
"""),
        params={"rate": 2.5},
        variate=3, variate_repr="a = 3",
        logdensity=_poisson,
        scipy_note="scipy.stats.poisson.logpmf(3, rate) [CHLO lgamma]",
        grad_params=("rate",),
        sample_ref=lambda: poisson(2.5),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Poisson(rate = 2.5))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Poisson(rate = 2.5), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        notes="exercises chlo.lgamma (log factorial); fan-out exercises the batched inverse-CDF while",
    ),
    Fixture(
        key="binomial",
        distribution="Binomial(n, p)",
        flatppl=_src("""
n = elementof(posintegers)
p = elementof(unitinterval)
a = draw(Binomial(n = n, p = p))
lp = logdensityof(lawof(record(a = a)), record(a = 2))
"""),
        params={"n": 10.0, "p": 0.4},
        variate=2, variate_repr="a = 2",
        logdensity=_binomial,
        scipy_note="scipy.stats.binom.logpmf(2, n=10, p=0.4) [CHLO lgamma]",
        grad_params=("p",),  # n is a (discrete) count -> not gradient-tested
        sample_ref=lambda: binom(10, 0.4),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Binomial(n = 10, p = 0.4))
draws = rand(s, lawof(x))
"""),
        # n MUST be a compile-time-known local constant here (not an
        # `elementof` free param, unlike the logdensity model above): the
        # fanned draw's inner axis (n Bernoulli trials per lane) has to be a
        # STATIC shape, so this is a deliberately separate model, not a
        # baked-literal copy of `flatppl` above.
        fanout_flatppl=_src("""
s = rnginit(0)
n = 10
xs ~ iid(Binomial(n = n, p = 0.4), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        notes=(
            "n is a count parameter (arg0), differentiated only w.r.t. p (arg1); "
            "fan-out exercises the rank-2 [200, 10] uniform + inner-axis reduce"
        ),
    ),
    Fixture(
        key="mvnormal",
        distribution="MvNormal(mu[2], cov[2x2])",
        flatppl=_src("""
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [2, 2]))
a = draw(MvNormal(mu = mu, cov = cov))
lp = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.1]))
"""),
        params={"mu": [0.5, -0.3], "cov": [[1.2, 0.3], [0.3, 0.8]]},
        variate=[0.2, 0.1], variate_repr="a = [0.2, 0.1]",
        logdensity=_mvnormal,
        scipy_note="scipy.stats.multivariate_normal.logpdf([0.2,0.1], mean=mu, cov=cov)",
        grad_params=(),  # Enzyme cannot differentiate stablehlo.triangular_solve
        # Tier-2 multivariate fan-out: mu/cov stay FREE params (elementof), fed
        # as the @sample func args in `param_values()` order — matching the
        # emitter's `@sample(%key, %arg0=mu, %arg1=cov)` layout. Draws [n, d];
        # the gate checks the per-component mean AND the full sample covariance
        # against these SAME mu/cov (a wrong `dot_general` contraction, e.g.
        # z.L instead of z.L^T, would show up as a wrong covariance here).
        fanout_flatppl=_src("""
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [2, 2]))
s = rnginit(0)
xs ~ iid(MvNormal(mu = mu, cov = cov), 20000)
draws = rand(s, lawof(xs))
"""),
        # 20000, not the Tier-1/rejection fixtures' 5000: the covariance
        # check's asymptotic SE shrinks as 1/sqrt(n), and at n=5000 the
        # z@L-vs-z@L^T transpose bug's worst-entry z-score (~5sigma) sits too
        # close to a naive 6sigma gate to reliably trip it (Monte Carlo:
        # ~85% pass rate for the BUGGY sampler). At n=20000 the buggy
        # worst-z clears 7.5sigma in every trial while the correct sampler
        # never exceeds ~4sigma — see check_mvnormal_fanout_distribution.
        fanout_n=20000,
        fanout_dim=2,
        notes=(
            "matrix path: stablehlo.cholesky + triangular_solve. VALUE-only: "
            "Enzyme-JAX cannot compute the adjoint of triangular_solve, so the "
            "gradient is not executable here (executor limitation, not an "
            "emitter bug). Fan-out exercises the batched [n,d] cholesky-affine "
            "draw (one shared cholesky, one rng_bit_generator advance, a "
            "batched dot_general for the row-wise mu + L.z)."
        ),
    ),
    Fixture(
        key="dirichlet",
        distribution="Dirichlet(alpha[3])",
        flatppl=_src("""
alpha = elementof(cartpow(posreals, 3))
a = draw(Dirichlet(alpha = alpha))
lp = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.3, 0.5]))
"""),
        params={"alpha": [2.0, 3.0, 5.0]},
        variate=[0.2, 0.3, 0.5], variate_repr="a = [0.2, 0.3, 0.5]",
        logdensity=_dirichlet,
        scipy_note="scipy.stats.dirichlet.logpdf([0.2,0.3,0.5], alpha=[2,3,5]) [CHLO lgamma]",
        grad_params=("alpha",),
        # No sample_ref: Dirichlet's @sample is exercised by the independence
        # path, which also checks each component's Beta(a_i, a0-a_i) marginal.
        sample_flatppl=_src("""
alpha = elementof(cartpow(posreals, 3))
s = rnginit(0)
x = draw(Dirichlet(alpha = alpha))
draws = rand(s, lawof(x))
"""),
        sample_args=([2.0, 3.0, 5.0],),
        independence="dirichlet",
        # Simplex fan-out: alpha stays a FREE param (elementof), matching
        # MvNormal's Tier-2 approach, so the gate can feed it via
        # `fx.param_values()` — one rng_bit_generator advance drawing the
        # WHOLE [20000, 3] batch (one call, not chained), the same n as
        # MvNormal's covariance check for the same statistical-power reason
        # (see fanout_n's comment there).
        fanout_flatppl=_src("""
alpha = elementof(cartpow(posreals, 3))
s = rnginit(0)
xs ~ iid(Dirichlet(alpha = alpha), 20000)
draws = rand(s, lawof(xs))
"""),
        fanout_n=20000,
        fanout_dim=3,
        fanout_simplex=True,
        notes="vector path + chlo.lgamma; @sample uses one Gamma rng stream per component",
    ),
    Fixture(
        key="laplace",
        distribution="Laplace(location, scale)",
        flatppl=_src("""
location = elementof(reals)
scale = elementof(posreals)
a = draw(Laplace(location = location, scale = scale))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
"""),
        params={"location": 0.0, "scale": 1.0},
        variate=0.5, variate_repr="a = 0.5",
        logdensity=_laplace,
        scipy_note="scipy.stats.laplace.logpdf(0.5, loc=location, scale=scale)",
        grad_params=("location", "scale"),
        sample_ref=lambda: laplace_dist(loc=0.0, scale=1.0),
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Laplace(location = 0.0, scale = 1.0))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Laplace(location = 0.0, scale = 1.0), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        notes="elementwise compare/select sgn(U-1/2) fan-out — no while, no new primitive",
    ),
    Fixture(
        key="geometric",
        distribution="Geometric(p)",
        flatppl=_src("""
p = elementof(unitinterval)
a = draw(Geometric(p = p))
lp = logdensityof(lawof(record(a = a)), record(a = 3))
"""),
        params={"p": 0.3},
        variate=3, variate_repr="a = 3",
        logdensity=_geometric,
        scipy_note="scipy.stats.geom.logpmf(3+1, p) [FlatPPL counts failures, scipy counts trials]",
        grad_params=("p",),
        sample_ref=lambda: geom(0.3, loc=-1),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Geometric(p = 0.3))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Geometric(p = 0.3), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        fanout_discrete_kmin=0,
        notes="elementwise floor(log(U)/log(1-p)) fan-out — no while, no new primitive",
    ),
    Fixture(
        key="categorical",
        distribution="Categorical(p)",
        # p is baked as a literal (no `elementof(stdsimplex(3))` free param,
        # same no-free-param shape as the `uniform` fixture above) — the fixed
        # p is what both the logdensity oracle and the fan-out sampler share.
        flatppl=_src("""
a = draw(Categorical(p = [0.2, 0.3, 0.5]))
lp = logdensityof(lawof(record(a = a)), record(a = 2))
"""),
        params={},
        variate=2, variate_repr="a = 2",
        logdensity=lambda: _categorical([0.2, 0.3, 0.5], base=1),
        scipy_note="closed-form rv_discrete(values=([1,2,3],[0.2,0.3,0.5])).logpmf(2) [1-based]",
        grad_params=(),  # p is a fixed literal vector, not a free continuous param
        sample_ref=lambda: _categorical_ref([0.2, 0.3, 0.5], base=1),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Categorical(p = [0.2, 0.3, 0.5]))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Categorical(p = [0.2, 0.3, 0.5]), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        fanout_discrete_kmin=1,
        notes="inverse-CDF compare/select unroll fan-out — no while, no new primitive",
    ),
    Fixture(
        key="categorical0",
        distribution="Categorical0(p)",
        flatppl=_src("""
a = draw(Categorical0(p = [0.2, 0.3, 0.5]))
lp = logdensityof(lawof(record(a = a)), record(a = 1))
"""),
        params={},
        variate=1, variate_repr="a = 1",
        logdensity=lambda: _categorical0([0.2, 0.3, 0.5], base=0),
        scipy_note="closed-form rv_discrete(values=([0,1,2],[0.2,0.3,0.5])).logpmf(1) [0-based]",
        grad_params=(),
        sample_ref=lambda: _categorical_ref([0.2, 0.3, 0.5], base=0),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(Categorical0(p = [0.2, 0.3, 0.5]))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(Categorical0(p = [0.2, 0.3, 0.5]), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        fanout_discrete_kmin=0,
        notes="same inverse-CDF unroll as Categorical, only the initial count constant differs",
    ),
    Fixture(
        key="negativebinomial",
        distribution="NegativeBinomial(alpha, beta)",
        flatppl=_src("""
alpha = elementof(posreals)
beta = elementof(posreals)
a = draw(NegativeBinomial(alpha = alpha, beta = beta))
lp = logdensityof(lawof(record(a = a)), record(a = 4))
"""),
        params={"alpha": 5.0, "beta": 2.0},
        variate=4, variate_repr="a = 4",
        logdensity=_negbinomial,
        scipy_note="scipy.stats.nbinom.logpmf(4, n=alpha, p=beta/(beta+1)) [CHLO lgamma]",
        grad_params=("alpha", "beta"),
        sample_ref=lambda: nbinom(5.0, 2.0 / 3.0),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(NegativeBinomial(alpha = 5.0, beta = 2.0))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(NegativeBinomial(alpha = 5.0, beta = 2.0), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        fanout_discrete_kmin=0,
        notes=(
            "Gamma(shape=alpha, rate=beta) mixed into Poisson; fan-out exercises "
            "the batched Gamma while feeding the batched Poisson while (two whiles)"
        ),
    ),
    Fixture(
        key="negativebinomial2",
        distribution="NegativeBinomial2(mu, psi)",
        flatppl=_src("""
mu = elementof(posreals)
psi = elementof(posreals)
a = draw(NegativeBinomial2(mu = mu, psi = psi))
lp = logdensityof(lawof(record(a = a)), record(a = 4))
"""),
        params={"mu": 3.0, "psi": 5.0},
        variate=4, variate_repr="a = 4",
        logdensity=_negbinomial2,
        scipy_note="scipy.stats.nbinom.logpmf(4, n=psi, p=psi/(mu+psi)) [CHLO lgamma]",
        grad_params=("mu", "psi"),
        sample_ref=lambda: nbinom(5.0, 5.0 / 8.0),
        sample_discrete=True,
        sample_flatppl=_src("""
s = rnginit(0)
x = draw(NegativeBinomial2(mu = 3.0, psi = 5.0))
draws = rand(s, lawof(x))
"""),
        fanout_flatppl=_src("""
s = rnginit(0)
xs ~ iid(NegativeBinomial2(mu = 3.0, psi = 5.0), 200)
draws = rand(s, lawof(xs))
"""),
        fanout_n=200,
        fanout_discrete_kmin=0,
        notes=(
            "Gamma(shape=psi, rate=psi/mu) mixed into Poisson (mean mu); "
            "fan-out exercises the batched Gamma while feeding the batched Poisson while"
        ),
    ),
]

FIXTURES_BY_KEY = {f.key: f for f in FIXTURES}


def value(fx: Fixture) -> float:
    """The frozen scipy log-density at the fixture's pinned variate."""
    return fx.logdensity(*fx.param_values())


def _perturb(value, path, eps, sign):
    """Return a copy of ``value`` (float / list / nested list) with the single
    scalar at ``path`` (a tuple of indices, empty for a bare float) nudged by
    ``sign*eps``."""
    if not path:
        return value + sign * eps
    import copy

    out = copy.deepcopy(value)
    ref = out
    for i in path[:-1]:
        ref = ref[i]
    ref[path[-1]] += sign * eps
    return out


def _leaf_paths(value):
    """Enumerate index paths to every scalar leaf of a float / list / nested."""
    if isinstance(value, (int, float)):
        yield ()
        return
    for i, sub in enumerate(value):
        for tail in _leaf_paths(sub):
            yield (i,) + tail


def fd_gradient(fx: Fixture, eps: float = 1e-3) -> dict[str, object]:
    """Central finite-difference gradient of the scipy log-density w.r.t. each
    continuous parameter, structured like Enzyme's per-argument gradient:
    a scalar param -> a float, a vector param -> a list of per-component
    partials. Only ``fx.grad_params`` are included."""
    names = list(fx.params)
    base = fx.param_values()
    out: dict[str, object] = {}
    for name in fx.grad_params:
        j = names.index(name)
        pval = base[j]
        grad_leaf: list[float] = []
        paths = list(_leaf_paths(pval))
        for path in paths:
            plus = list(base)
            minus = list(base)
            plus[j] = _perturb(pval, path, eps, +1)
            minus[j] = _perturb(pval, path, eps, -1)
            g = (fx.logdensity(*plus) - fx.logdensity(*minus)) / (2 * eps)
            grad_leaf.append(g)
        if paths == [()]:
            out[name] = grad_leaf[0]  # scalar param
        else:
            out[name] = grad_leaf  # vector param
    return out
