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
    lognorm,
    multivariate_normal,
    norm,
    poisson,
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
    # Tier-1 fan-out (`iid(K, n)`, straight-line kernels only): a SEPARATE
    # `.iid.sample.flatppl` model baking the same params into a fixed-size
    # batched draw, one rng_bit_generator advance per call. Empty for dists
    # without a landed fan-out lowering (rejection/multivariate kernels are
    # Tier 2, not yet emitted).
    fanout_flatppl: str = ""
    fanout_n: int = 0
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
        notes="exercises chlo.lgamma",
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
        independence="beta",
        notes="exercises chlo.lgamma; @sample uses two internal Gamma rng streams",
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
        notes="exercises chlo.lgamma",
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
        notes="exercises chlo.lgamma (log factorial)",
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
        notes="n is a count parameter (arg0), differentiated only w.r.t. p (arg1)",
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
        notes=(
            "matrix path: stablehlo.cholesky + triangular_solve. VALUE-only: "
            "Enzyme-JAX cannot compute the adjoint of triangular_solve, so the "
            "gradient is not executable here (executor limitation, not an "
            "emitter bug)."
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
        notes="vector path + chlo.lgamma; @sample uses one Gamma rng stream per component",
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
